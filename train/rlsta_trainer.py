"""
RLSTATrainer — RLSTA (Reinforcement Learning with Self-verification Through Accuracy)

This trainer extends TRL's GRPOTrainer by incorporating a **self-verification
reward signal** derived from the reference model's likelihood of the completion
under a simplified (single-turn) prompt.

============================================================================
Mechanism Overview
============================================================================

Standard GRPO trains the policy by:
    1. Generating N completions per prompt
    2. Scoring each completion with a reward function (e.g. math correctness)
    3. Computing group-relative advantages and optimizing the policy

RLSTATrainer adds an **eval_prompt likelihood reward** on top:

    For each multi-turn prompt P = [user1, assistant1, user2, ..., userK]:
        - The model generates N completions {c1, ..., cN} conditioned on P
        - A "single-turn eval_prompt" E is constructed by merging all user
          turns into a single message: E = [userK_merged]
        - The **reference model** computes log P_ref(ci | E) for each ci
        - This gives an "eval_reward" = exp(mean_logp_per_token) / N

    The final reward for each completion becomes:
        reward_total = eval_reward + correctness_reward

    This means:
        - Completions that are **correct** get a direct +1.0 reward bonus
        - All completions (correct or not) also receive a **likelihood-based
          reward** from the reference model, measuring how "natural" or
          "probable" the completion would be if the question were asked
          directly (without the multi-turn history).

Why this helps:
    - In multi-turn RL training, the conversation history can degrade over
      time (the model's own prior turns may be low-quality). This causes the
      model to "forget" how to solve problems it could solve in single-turn.
    - The eval_prompt likelihood acts as a **regularizer**: it rewards
      completions that the reference model (trained on single-turn) would
      also assign high probability to, preventing the policy from drifting
      into pathological multi-turn behaviors.
    - It combines the benefits of:
        (a) Task-specific correctness reward (sparse but accurate)
        (b) Reference model likelihood (dense, smooth signal)

============================================================================
Key Difference from Standard GRPOTrainer
============================================================================

In `_generate_and_score_completions`:

    1. After generating completions from the multi-turn prompt, it also
       tokenizes the `eval_prompt` (single-turn merged prompt).

    2. It concatenates [eval_prompt_ids, completion_ids] and computes
       per-token log probabilities using the **reference model**.

    3. The eval_reward is computed as:
           eval_reward = exp( sum(logp * mask) / sum(mask) ) / num_generations

    4. The final reward = eval_reward + correctness_reward, and advantages
       are computed from this combined reward.

============================================================================
"""

from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    prepare_multimodal_messages,
)
from trl.trainer.utils import (
    nanmax,
    nanmin,
    nanstd,
    pad,
)
from trl.extras.profiling import profiling_decorator
from accelerate.utils import gather, gather_object

import torch
from typing import Any

from trl import GRPOTrainer


class RLSTATrainer(GRPOTrainer):
    """
    RLSTA Trainer: extends GRPOTrainer with self-verification through
    reference-model likelihood under a single-turn eval prompt.

    The training data must include an `eval_prompt` field in each sample,
    which is a single-turn version of the multi-turn prompt (all user turns
    merged into one). This is used to compute a likelihood-based reward
    from the reference model, which is added to the correctness reward.
    """

    @profiling_decorator
    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        # --- Handle multimodal images (if present) ---
        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [
                [example.get("image")] if example.get("image") is not None else None
                for example in inputs
            ]
        else:
            images = None
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        if images is not None:
            prompts = [
                prepare_multimodal_messages(prompt, image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]

        # --- Generate completions from the multi-turn prompt ---
        prompt_ids_list, completion_ids_list, num_items_in_batch, sampling_per_token_logps_list, extra_fields = (
            self._generate(prompts)
        )

        # --- Tokenize single-turn eval_prompt (for self-verification) ---
        eval_prompts_ids, eval_prompts_mask = None, None
        if "eval_prompt" in inputs[0]:
            grandparent = super(GRPOTrainer, self)
            eval_prompts = [x["eval_prompt"] for x in inputs]
            eval_prompts = self.processing_class(
                text=eval_prompts, return_tensors="pt", padding=True,
                padding_side="left", add_special_tokens=False
            )
            eval_prompts = grandparent._prepare_inputs(eval_prompts)
            eval_prompts_ids = eval_prompts["input_ids"]
            eval_prompts_mask = eval_prompts["attention_mask"]

        # --- Convert token ID lists to padded tensors ---
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")

        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        # Mask truncated completions if configured
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor(
                [ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device
            )
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt + completion for full-sequence log-prob computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # Concatenate eval_prompt + completion for self-verification
        if eval_prompts_ids is not None:
            eval_prompts_completion_ids = torch.cat(
                [eval_prompts_ids, completion_ids], dim=1
            ).to(device)
            eval_prompts_completion_mask = torch.cat(
                [eval_prompts_mask, completion_mask], dim=1
            ).to(device)

        logits_to_keep = completion_ids.size(1)
        batch_size = (
            self.args.per_device_train_batch_size if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        num_images = [len(img_list) for img_list in images] if images is not None else None

        # Handle multimodal forward kwargs
        if images is not None:
            prompts_text = [
                apply_chat_template(
                    {"prompt": prompt}, self.processing_class,
                    **self.chat_template_kwargs
                )["prompt"]
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(
                images=images, text=prompts_text, padding=True, return_tensors="pt"
            )
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {
                k: v for k, v in prompt_inputs.items()
                if k not in ["input_ids", "attention_mask"]
            }
        else:
            forward_kwargs = {}

        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        # =====================================================================
        # Compute log probabilities (no gradient needed for reward computation)
        # =====================================================================
        with torch.no_grad():
            # --- Old per-token logps (for importance sampling if needed) ---
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model, prompt_completion_ids, attention_mask,
                    logits_to_keep, batch_size, num_images=num_images,
                    **forward_kwargs,
                )
            else:
                old_per_token_logps = None

            # --- Importance sampling correction for vLLM ---
            if self.use_vllm and self.vllm_importance_sampling_correction:
                importance_sampling_ratio = torch.exp(
                    old_per_token_logps - sampling_per_token_logps
                )
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )

            # --- Reference model KL penalty ---
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model, prompt_completion_ids, attention_mask,
                        logits_to_keep, batch_size=batch_size,
                        num_images=num_images, **forward_kwargs,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model, prompt_completion_ids, attention_mask,
                            logits_to_keep, batch_size=batch_size,
                            num_images=num_images, **forward_kwargs,
                        )
            else:
                ref_per_token_logps = None

            # =================================================================
            # RLSTA CORE: Self-verification via eval_prompt likelihood
            # =================================================================
            # Compute how likely each completion is under the reference model
            # when conditioned on the single-turn eval_prompt (not the full
            # multi-turn history). This gives a dense reward signal that
            # measures "would this answer make sense for the original question?"
            # =================================================================
            if eval_prompts_ids is not None:
                eval_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.ref_model,
                    eval_prompts_completion_ids,
                    eval_prompts_completion_mask,
                    logits_to_keep,
                    batch_size=batch_size,
                    num_images=num_images,
                    **forward_kwargs,
                )
                # Convert mean log-prob to a probability-like reward:
                #   eval_reward = exp(mean_logp_per_token) / num_generations
                # Dividing by num_generations normalizes the scale so it
                # doesn't dominate the correctness reward.
                eval_rewards = torch.exp(
                    (eval_per_token_logps * completion_mask).sum(-1)
                    / completion_mask.sum(-1)
                ) / self.num_generations
            else:
                eval_rewards = None

        if eval_rewards is not None:
            eval_rewards = gather(eval_rewards)

        # --- Decode completions to text ---
        prompts_text = self.processing_class.batch_decode(
            prompt_ids, skip_special_tokens=True
        )
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text, strict=True):
                bootstrap = (
                    prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        # Merge extra_fields from rollout into inputs
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # --- Compute correctness rewards ---
        rewards_per_func = self._calculate_rewards(
            inputs, prompts, completions, completion_ids_list
        )

        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        # =================================================================
        # RLSTA reward combination:
        #   For samples where at least one completion in the group is correct
        #   (reward > 0.9), incorrect completions (reward < 0.9) get masked
        #   to 0 — preventing the model from being encouraged to produce
        #   wrong answers just because they have high likelihood.
        # =================================================================
        grouped_rewards_mask1 = (
            rewards.view(-1, self.num_generations)
            .max(dim=-1, keepdim=True)
            .values > 0.9
        ).expand((-1, self.num_generations)).reshape_as(rewards)
        grouped_rewards_mask2 = (
            rewards.view(-1, self.num_generations) < 0.9
        ).reshape_as(rewards)

        # Add the eval_prompt likelihood reward to the correctness reward
        if eval_rewards is not None:
            rewards = eval_rewards + rewards

        # --- Compute group-relative advantages ---
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. "
                "Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        # Slice to keep only local process data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        # --- Logging ---
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(
            is_std_zero.float().mean().item()
        )

        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[completion_mask.bool()]
            mean_delta = (
                torch.mean(delta) if delta.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            max_delta = (
                torch.max(delta) if delta.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
            min_is = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            mean_is = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            max_is = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_is)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_is).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_is)).item()
            )

        # --- Build output dict ---
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        return output
