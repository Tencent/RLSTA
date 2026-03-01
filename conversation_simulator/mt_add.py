"""
Multi-turn conversation simulation.

Shards are delivered one per turn to the LLM via multiturn_gen_think/nothink.
"""

from utils.api_gen import multiturn_gen_think, multiturn_gen_nothink
from .base import SimulationBase


class MultiturnSimulation(SimulationBase):
    DEFAULT_GENTYPE = "nothink"
    OUTPUT_ROOT = "eval_responses"

    def _get_output_dir_prefix(self) -> str:
        return "mt_add"

    def _generate(self, idx, item, task_info):
        shards = task_info["shards"]
        system_prompt = task_info["system_prompt"]

        if self.gentype == "nothink":
            return multiturn_gen_nothink(
                self.client,
                questions=shards,
                messages=[system_prompt],
                model_name=self.model_name,
                return_messages=True,
                retry=self.retry,
                max_tokens=4096,
            )
        else:
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
    MultiturnSimulation().run()
