"""
Multi-turn refine (mt_refine) simulation with information correction.

Supports three strategies via --strategy:
  - mt-refine  (default): First turn gives all modified shards at once,
                         then each subsequent turn corrects one shard.
  - atr      (Add Then Revise): First gives shards one by one (all modified),
                         then gives all corrections one by one.
  - awr      (Add With Revise): Alternates modified shard, then its correction,
                         for each shard.

Data files: mt_refine_math_eval.jsonl / mt_refine_math_train.jsonl
"""

from utils.api_gen import multiturn_gen_think, multiturn_gen_nothink
from .base import SimulationBase


CORRECTION_PREFIX = (
    "Actually, there were multiple errors in the conditions I provided. "
    "Here are the correct conditions: "
)
CORRECTION_SUFFIX = " Please provide the final answer based on these corrections."


class MtRefineSimulation(SimulationBase):
    DEFAULT_GENTYPE = "nothink"
    OUTPUT_ROOT = "eval_responses"

    def _add_extra_args(self, parser):
        parser.add_argument(
            "--strategy",
            type=str,
            default="mt-refine",
            choices=["mt-refine", "atr", "awr"],
            help="Correction strategy: mt-refine | atr | awr",
        )

    def _get_output_dir_prefix(self) -> str:
        return "mt_refine"

    def _build_questions(self, item, task_info):
        """
        Build the list of user questions depending on strategy.
        Requires items with 'modified_shard' field.
        """
        modified_shards = [i["shard"] for i in item["modified_shard"]]
        correct_shards = [i["shard"] for i in item["shards"]]
        strategy = self.args.strategy

        corrections = [
            CORRECTION_PREFIX + s + CORRECTION_SUFFIX
            for s in correct_shards[1:]
        ]

        if strategy == "mt-refine":
            # First turn: background + all modified shards at once
            # Then: one correction per turn
            first_turn = " ".join([correct_shards[0]] + modified_shards)
            return [first_turn] + corrections

        elif strategy == "atr":
            # Add all modified shards one by one, then all corrections
            return [correct_shards[0]] + modified_shards + corrections

        elif strategy == "awr":
            # Alternate: modified shard, then its correction
            interleaved = []
            for mod_s, corr in zip(modified_shards, corrections):
                interleaved.append(mod_s)
                interleaved.append(corr)
            return [correct_shards[0]] + interleaved

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _generate(self, idx, item, task_info):
        system_prompt = task_info["system_prompt"]

        # For items with modified shards, build special correction questions
        if "modified_shard" in item:
            questions = self._build_questions(item, task_info)
        else:
            # Fallback: use regular shards
            questions = task_info["shards"]

        if self.gentype == "nothink":
            return multiturn_gen_nothink(
                self.client,
                questions=questions,
                messages=[system_prompt],
                model_name=self.model_name,
                return_messages=True,
                retry=self.retry,
                max_tokens=4096,
            )
        else:
            return multiturn_gen_think(
                self.client,
                questions=questions,
                messages=[system_prompt],
                model_name=self.model_name,
                return_messages=True,
                retry=self.retry,
                max_tokens=4096 * 2,
            )


if __name__ == "__main__":
    MtRefineSimulation().run()
