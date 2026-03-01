"""
Multiturn Evaluation for Latent Capability Filtering (LCF).

Reads all trail JSONL files from a directory produced by Step 1 (mt_add / mt_refine),
then for **each** conversation history (each line in the JSONL files) re-generates
the final turn with N candidates (precontext approach) and evaluates each candidate
against the ground-truth answer.

This is the "E[Ver(mn | H)]" side of the LCF condition:
    E_m~pi(·|i_full)[Ver(m)] > E_mn~pi(·|H)[Ver(mn)]

Input  (--input-dir): directory containing trail JSONL files (e.g.
       eval_responses/<model>/mt_add_math_train/)
       Each JSONL line: {"task_id": "math/0", "full_messages": [{...}, ...]}

Output (--output-file-name, written to acc_rawdata/<model_name>/):
    A JSON **list** where each element corresponds to one conversation history:
    [
      {
        "task_id": "gsm8k/0",
        "messages": [{...}, ...],
        "eval_result_list": [true, false, ...],
        "acc": 0.75,
        "final_answer_list": ["resp1", "resp2", ...]
      },
      ...
    ]
    Note: task_id can repeat (one entry per conversation history, not per task).
    Entries with null messages are skipped.
"""

import sys
sys.path.append("../")

import re
import os
import json
import glob
import tqdm
import argparse
import concurrent.futures

from openai import OpenAI
from utils.api_gen import singleturn_gen_nothink
from utils.eval_math_train import eval_math_withans
from conversation_simulator.task_config import load_problems

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="LCF: evaluate multi-turn conversation histories "
                    "(reads trail files, re-generates last turn via precontext, evaluates)"
    )
    parser.add_argument("--base-url",        type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model-name",      type=str, default=None)
    parser.add_argument("--openai-api-key",  type=str, default="EMPTY")
    parser.add_argument("--input-dir",       type=str, required=True,
                        help="Directory containing trail JSONL files from Step 1 "
                             "(e.g. eval_responses/<model>/mt_add_math_train/)")
    parser.add_argument("--datatype",       type=str, default="math_train",
                        help="Task type for loading ground-truth answers "
                             "(must be a key in task_config.TASK_DATA_PATHS, e.g. math_train)")
    parser.add_argument("--responses-num",   type=int, default=4,
                        help="Number of candidate responses to generate per conversation history")
    parser.add_argument("--max-workers",     type=int, default=16)
    parser.add_argument("--max-tokens",      type=int, default=4096)
    parser.add_argument("--output-file-name", type=str, required=True,
                        help="Output filename (written under acc_rawdata/<model_name>/)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Answer extraction (GSM8K style)
# ---------------------------------------------------------------------------

_ANS_RE = re.compile(r"#### (\-?[0-9\.,]+)")
_INVALID_ANS = "[invalid]"


def _extract_answer(completion: str) -> str:
    match = _ANS_RE.search(completion)
    if match:
        return match.group(1).strip().replace(",", "")
    return _INVALID_ANS


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class MultiturnEval:
    """
    For each conversation history in the input trail files, strip the last
    assistant turn, re-generate N candidate responses (precontext approach),
    and evaluate accuracy against the ground-truth answer.

    Output is a flat JSON list — task_id may repeat across entries.
    Entries with null full_messages are skipped.
    """

    def __init__(self, args=None):
        self.args = args or _parse_args()
        self.client = OpenAI(
            api_key=self.args.openai_api_key,
            base_url=self.args.base_url,
        )

        # Auto-detect model name if not provided
        self.model_name = self.args.model_name
        if self.model_name is None:
            try:
                models = self.client.models.list()
                self.model_name = models.data[0].id
            except Exception as e:
                raise RuntimeError(
                    f"Failed to auto-detect model name from {self.args.base_url}/models: {e}\n"
                    "Please specify --model-name explicitly."
                ) from e

        # Load all records flat (each record = one conversation history)
        # Each element: {"task_id": str, "full_messages": list | None}
        self.records = self._load_all_records()

        # Load ground-truth answers via task_config
        problems = load_problems(self.args.datatype)
        self.problems_dict = {}
        for idx, item in enumerate(problems):
            if "task_id" in item:
                tid = item["task_id"]
            elif "unique_id" in item:
                tid = item["unique_id"]
            else:
                tid = f"math/{idx}"
            self.problems_dict[tid] = item

    @staticmethod
    def _infer_model_name_from_path(input_dir: str) -> str:
        """Extract model name from input-dir path.

        Expected pattern: .../eval_responses/<model_name>/<mode>_<datatype>/
        Falls back to 'unknown_model' if the pattern doesn't match.
        """
        normed = os.path.normpath(input_dir)
        parts = normed.split(os.sep)
        if len(parts) >= 2:
            return parts[-2]
        return "unknown_model"

    def _infer_model_name(self) -> str:
        return self._infer_model_name_from_path(self.args.input_dir)

    def _load_all_records(self) -> list:
        """
        Read all *.jsonl files in --input-dir and return a flat list of records.
        Each record is a dict with at least {"task_id": str, "full_messages": list | None}.
        Records with null full_messages are filtered out here.

        Returns:
            [{"task_id": "gsm8k/0", "full_messages": [{...}, ...]}, ...]
        """
        input_dir = self.args.input_dir
        jsonl_files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl")))
        if not jsonl_files:
            raise FileNotFoundError(
                f"No .jsonl files found in {input_dir}"
            )

        records = []
        skipped = 0
        for fpath in jsonl_files:
            with open(fpath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    # Skip entries with null messages
                    if record.get("full_messages") is None:
                        skipped += 1
                        continue
                    records.append(record)

        print(f"[multiturn_eval] Loaded {len(records)} conversation histories "
              f"from {len(jsonl_files)} trail files (skipped {skipped} null entries)")
        return records

    # ------------------------------------------------------------------
    # Generation (precontext approach)
    # ------------------------------------------------------------------

    def _gen_one(self, record: dict) -> list[str] | None:
        """
        Generate `responses_num` candidates for the last turn of one
        conversation history (precontext approach).

        Given the conversation history (full_messages), strip the last
        assistant turn and re-generate N candidate responses in a single
        call to singleturn_gen_nothink (which uses n=responses_num).

        full_messages[-1] must be the assistant turn to replace.
        full_messages[-2] is the user question for the last turn.
        full_messages[:-2] is the preceding context.
        """
        full_messages = record["full_messages"]
        if full_messages is None or len(full_messages) < 2:
            return None
        assert full_messages[-1]["role"] == "assistant", (
            "Expected last message to be assistant turn"
        )

        context_messages = full_messages[:-2]
        user_question = full_messages[-2]["content"]

        try:
            resp = singleturn_gen_nothink(
                self.client,
                questions=user_question,
                responses_num=self.args.responses_num,
                messages=context_messages,
                model_name=self.model_name,
                retry=True,
                max_tokens=self.args.max_tokens,
            )
            return [r.message.content for r in resp]
        except Exception as e:
            print(f"[multiturn_eval] generation error for "
                  f"task_id={record.get('task_id')}: {e}")
            return None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _eval_one(self, task_id: str, generated_list: list[str] | None) -> dict:
        """
        Evaluate a list of generated responses against the ground-truth answer.

        Returns:
            {
                "eval_result_list": [bool, ...],
                "acc": float,
                "final_answer_list": [str, ...]
            }
        """
        problem = self.problems_dict.get(task_id)
        if problem is None:
            print(f"[multiturn_eval] WARNING: task_id {task_id!r} not found in problems_dict, skipping eval")
            return {
                "eval_result_list": [],
                "acc": 0.0,
                "final_answer_list": [],
            }

        correct_answer = _extract_answer(problem["answer"])

        if generated_list is None:
            return {
                "eval_result_list": [],
                "acc": 0.0,
                "final_answer_list": [],
            }

        eval_result_list = []
        for response in generated_list:
            if response is None:
                eval_result_list.append(False)
            else:
                result = eval_math_withans(response, correct_answer)
                eval_result_list.append(result)

        acc = (sum(float(r) for r in eval_result_list) / len(eval_result_list)
               if eval_result_list else 0.0)

        return {
            "eval_result_list": eval_result_list,
            "acc": acc,
            "final_answer_list": generated_list,
        }

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        """
        For each conversation history record (flat, not grouped by task_id):
        1. Use precontext approach to re-generate last turn with N candidates
        2. Evaluate each candidate against ground-truth
        3. Collect results into a flat list (task_id may repeat)
        """
        # Parallel generation across all records
        print(f"[multiturn_eval] Generating {self.args.responses_num} candidates "
              f"for each of {len(self.records)} conversation histories ...")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.args.max_workers
        ) as executor:
            generated_list = list(
                tqdm.tqdm(
                    executor.map(self._gen_one, self.records),
                    total=len(self.records),
                    desc="multiturn_eval gen",
                )
            )

        # Evaluate and build output list
        output_list = []
        for record, gen in zip(self.records, generated_list):
            task_id = record["task_id"]
            ev = self._eval_one(task_id, gen)
            output_list.append({
                "task_id": task_id,
                "messages": record["full_messages"],
                "eval_result_list": ev["eval_result_list"],
                "acc": ev["acc"],
                "final_answer_list": ev["final_answer_list"],
            })

        # Write output – infer model name from input-dir path
        # Expected: eval_responses/<model_name>/<mode>_<datatype>/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        model_name = self._infer_model_name()
        output_dir = os.path.join(project_root, "acc_rawdata", model_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, self.args.output_file_name)
        with open(output_path, "w") as f:
            json.dump(output_list, f)
        print(f"[multiturn_eval] Written to {output_path} "
              f"({len(output_list)} entries)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    MultiturnEval().run()
