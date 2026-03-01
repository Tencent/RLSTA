"""
Self-correction simulation.

Sends the full question first, then asks the model to self-correct 3 times.
"""

from utils.api_gen import multiturn_gen_think, multiturn_gen_nothink
from .base import SimulationBase

SELF_CORRECT_PROMPT = (
    "There may be something wrong with the previous answer. "
    "Please review it and provide a corrected or improved answer?"
)


class SelfCorrectSimulation(SimulationBase):
    DEFAULT_GENTYPE = "nothink"
    OUTPUT_ROOT = "eval_responses"

    def _get_output_dir_prefix(self) -> str:
        return "self_correct"

    def _generate(self, idx, item, task_info):
        system_prompt = task_info["system_prompt"]
        full_question = " ".join(task_info["shards"])

        # First turn: full question, then 3 rounds of self-correction
        questions = [
            full_question,
            SELF_CORRECT_PROMPT,
            SELF_CORRECT_PROMPT,
            SELF_CORRECT_PROMPT,
        ]

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
            # In think mode, the original code used shards instead of
            # the self-correct pattern — preserving that behaviour.
            shards = task_info["shards"]
            return multiturn_gen_think(
                self.client,
                questions=shards,
                messages=[system_prompt],
                model_name=self.model_name,
                return_messages=True,
                retry=self.retry,
                max_tokens=4096 * 2,
            )


if __name__ == "__main__":
    SelfCorrectSimulation().run()
