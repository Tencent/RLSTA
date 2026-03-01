"""
Single-turn conversation simulation (baseline).

All shards are concatenated into one prompt and sent in a single request.
"""

import concurrent.futures
import tqdm

from utils.api_gen import singleturn_gen, singleturn_gen_nothink
from utils.data import write_jsonl
from .base import SimulationBase
from .task_config import load_problems, prepare_task


class SingleturnSimulation(SimulationBase):
    DEFAULT_GENTYPE = "nothink"
    OUTPUT_ROOT = "eval_responses"

    def _get_output_dir_prefix(self) -> str:
        return "singleturn"

    def _generate(self, idx, item, task_info):
        system_prompt = task_info["system_prompt"]
        # For single-turn, use concat_prompt if available (task classes), otherwise join shards
        question = task_info.get("concat_prompt", " ".join(task_info["shards"]))
        messages = [system_prompt] if system_prompt is not None else []
        if self.gentype == "nothink":
            resp = singleturn_gen_nothink(
                self.client,
                questions=question,
                responses_num=1,
                messages=messages,
                model_name=self.model_name,
                retry=self.retry,
                max_tokens=4096,
            )
        else:
            resp = singleturn_gen(
                self.client,
                questions=question,
                responses_num=1,
                messages=messages,
                model_name=self.model_name,
                retry=self.retry,
                max_tokens=4096 * 2,
            )
        return resp[0].message.content


if __name__ == "__main__":
    SingleturnSimulation().run()
