"""
GRPO (Group Relative Policy Optimization) Training Script.

This script implements RL-based fine-tuning of language models using GRPO,
a variant of reinforcement learning that optimizes policy models by comparing
group-relative rewards across multiple generated completions.

Pipeline:
    1. Load a pre-trained causal LM and its tokenizer.
    2. Read training data (JSONL), where each sample contains multi-turn
       conversation history (field "completion") and a ground-truth answer.
    3. Apply chat template to produce prompts; also build a "single-turn eval
       prompt" by merging all user turns into one (used internally for eval).
    4. Define a reward function that verifies model completions against
       ground-truth answers using math verification.
    5. Train using TRL's GRPOTrainer with DeepSpeed ZeRO-3.

Reward function:
    - Extract numerical / symbolic answers from the completion
      and compare against ground truth using both exact matching and
      math_verify symbolic checking.

Dependencies:
    - trl, transformers, datasets, peft, deepspeed, math_verify
"""

import re
import math
import json
import os
import sys
import logging
import argparse

import torch
import transformers
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from math_verify import parse, verify

# Local self-contained imports (no dependency on grpo package)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data import stream_jsonl

# RLSTA trainer (self-verification through eval_prompt likelihood)
from rlsta_trainer import RLSTATrainer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Math Evaluation
# ---------------------------------------------------------------------------

def is_number(s):
    """Check whether a string can be interpreted as a float."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def eval_math(response: str, answer: str) -> bool:
    """
    Verify a math completion against a ground-truth answer.

    Strategy:
        1. If the answer is a plain number, extract candidate numbers from
           the response and check if any is within 1e-3 of the gold value.
        2. Otherwise, use symbolic parsing (math_verify) on each line of the
           response to check equivalence with the gold expression.

    Args:
        response: Model-generated completion text.
        answer:   Ground-truth answer string (may be numeric or symbolic).

    Returns:
        True if any extracted answer matches the gold answer.
    """
    split_response = response.split("\n")
    if "=" not in answer:
        split_response = sum([r.split("=") for r in split_response], [])

    if is_number(answer):
        def skip_special_tokens(extracted_answer):
            regexes_to_ignore = [",", "\\$", "(?s).*#### ", "\\.$"]
            for regex in regexes_to_ignore:
                extracted_answer = re.sub(regex, "", extracted_answer)
            return extracted_answer

        def safe_float_convert(text):
            try:
                return float(skip_special_tokens(text))
            except (ValueError, TypeError):
                return None

        try:
            gold = float(answer)
            extracted_answer = response.strip()
            extracted_answer = re.findall(
                r"(-?[0-9.,]{2,})|(-?[0-9]+)", extracted_answer
            )
            extracted_answer = [
                skip_special_tokens(m[0] if m[0] else m[1])
                for m in extracted_answer if m
            ][-3:]
            extracted_answer = [
                num for num in map(safe_float_convert, extracted_answer)
                if num is not None
            ]
        except Exception:
            return False
        return any(abs(test_answer - gold) < 1e-3 for test_answer in extracted_answer)
    else:
        gold = parse(f"${answer}$")
        for r in split_response:
            r = r.strip().strip(".")
            pred1 = parse(r)
            pred2 = parse(f"${r}$")
            if verify(gold, pred1) or verify(gold, pred2):
                return True
        return False


# ---------------------------------------------------------------------------
# Reward Function
# ---------------------------------------------------------------------------

def reward_fn(completions, prompts, **kwargs):
    """
    Compute per-sample reward for a batch of GRPO completions.

    For each (completion, answer) pair:
        - eval_math(completion, answer) -> 0.0 or 1.0

    Args:
        completions: List[str] of model-generated text.
        prompts:     List[str] of input prompts (unused but required by API).
        **kwargs:    Must contain 'answer' list.

    Returns:
        List[float]: reward for each sample.
    """
    answers = kwargs["answer"]
    rewards = []
    for completion, answer in zip(completions, answers):
        rewards.append(float(eval_math(completion, json.loads(answer))))
    return rewards


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model with GRPO")
    parser.add_argument("--model_name_or_path", type=str,
                        default="/root/models/Qwen2.5-7B-Instruct",
                        help="Path to pre-trained model")
    parser.add_argument("--data_path", type=str,
                        default="../data/test_data.jsonl",
                        help="Path to training data JSONL file")
    parser.add_argument("--test_data_path", type=str,
                        default="../data/test_data.jsonl",
                        help="Path to evaluation data JSONL file")
    parser.add_argument("--output_dir", type=str,
                        default="./Qwen2-7B-GRPO",
                        help="Output directory for checkpoints")
    parser.add_argument("--train_mode", type=str, default="base",
                        help="Training mode: 'base' (standard GRPOTrainer) or 'rlsta' (RLSTATrainer with self-verification)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_lora", action="store_true",
                        help="Enable LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=1e-4,
                        help="KL penalty coefficient")
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompt_length", type=int, default=32768)
    parser.add_argument("--max_completion_length", type=int, default=8196)
    parser.add_argument("--num_generations", type=int, default=8,
                        help="Number of completions per prompt for GRPO")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def _build_dataset(data_path: str, tokenizer, max_prompt_length: int):
    """
    Load a JSONL training file and build a HuggingFace Dataset.

    Each JSONL record is expected to have:
        - "completion": list of message dicts (multi-turn conversation history)
        - "answer":     ground-truth answer
        - "task_id":    unique problem identifier

    For every sample we produce:
        - prompt:      chat-template-formatted multi-turn prompt
        - eval_prompt: single-turn version (all user turns merged)
        - answer:      JSON-serialized ground-truth
        - task_id:     problem id

    Samples exceeding max_prompt_length tokens are dropped.
    """
    raw_data = list(stream_jsonl(data_path))
    processed = []

    for item in raw_data:
        token_len = len(
            tokenizer.apply_chat_template(
                item["completion"], tokenize=True, return_dict=True
            )["input_ids"]
        )
        if token_len > max_prompt_length - 10:
            continue

        # Build a single-turn eval prompt by merging all user utterances
        user_shards = [
            m["content"] for m in item["completion"] if m["role"] == "user"
        ]
        if item["completion"][0]["role"] == "system":
            eval_msgs = [
                item["completion"][0],
                {"role": "user", "content": " ".join(user_shards)},
            ]
        else:
            eval_msgs = [{"role": "user", "content": " ".join(user_shards)}]

        processed.append({
            "task_id": item["task_id"],
            "prompt": tokenizer.apply_chat_template(
                item["completion"], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            ),
            "answer": json.dumps(item["answer"]),
            "eval_prompt": tokenizer.apply_chat_template(
                eval_msgs, tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            ),
        })

    return Dataset.from_list(processed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    transformers.set_seed(args.seed)

    # ---- Tokenizer ----
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if not tokenizer.pad_token:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer has no EOS token to use as pad token")

    # ---- Datasets ----
    logger.info("Loading training data from %s (mode: %s)", args.data_path, args.train_mode)
    train_dataset = _build_dataset(args.data_path, tokenizer, args.max_prompt_length)
    eval_dataset = _build_dataset(args.test_data_path, tokenizer, args.max_prompt_length)

    # ---- Precision ----
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16

    # ---- Quantization (optional) ----
    model_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if args.use_8bit:
        model_kwargs["load_in_8bit"] = True
    if args.use_4bit:
        model_kwargs["load_in_4bit"] = True
        model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # ---- GRPO Config ----
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        warmup_steps=args.warmup_steps,
        beta=args.beta,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        seed=args.seed,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        use_vllm=True,
        vllm_mode="colocate",
        save_only_model=True,
        eval_on_start=True,
        eval_steps=200,
        eval_strategy="steps",
        per_device_eval_batch_size=8,
        vllm_gpu_memory_utilization=0.3,
        do_eval=True,
    )

    # ---- LoRA (optional) ----
    peft_config = None
    if args.use_lora:
        logger.info("Using LoRA for parameter-efficient fine-tuning")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )

    # ---- Trainer ----
    if args.train_mode == "base":
        logger.info("Using standard GRPOTrainer")
        trainer = GRPOTrainer(
            model=args.model_name_or_path,
            peft_config=peft_config,
            reward_funcs=reward_fn,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    elif args.train_mode == "rlsta":
        logger.info("Using RLSTATrainer (self-verification through eval_prompt likelihood)")
        trainer = RLSTATrainer(
            model=args.model_name_or_path,
            peft_config=peft_config,
            reward_funcs=reward_fn,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    else:
        raise ValueError(
            f"Unsupported train_mode='{args.train_mode}'. "
            "Supported modes: 'base' (standard GRPOTrainer), 'rlsta' (RLSTATrainer)."
        )

    # ---- Train ----
    logger.info("Starting GRPO training")
    trainer.train()


if __name__ == "__main__":
    main()
