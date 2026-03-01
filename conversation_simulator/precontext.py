"""
Pre-context simulation.

Reads existing multi-turn conversation results, then re-generates
the last turn with multiple samples (responses_num=8) for analysis.

This script has a different workflow from other simulations:
  - It reads from an existing full-messages file (--fullmsg-file-name)
  - It does NOT use the standard task preparation pipeline
  - Output goes to a local tmp/ directory
"""

import sys
import os

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PACKAGE_DIR)
sys.path.append(_PROJECT_ROOT)

import json
import logging
import argparse
import concurrent.futures

import tqdm
import requests
from openai import OpenAI

from utils.api_gen import singleturn_gen_nothink
from utils.data import write_jsonl, stream_jsonl

logger = logging.getLogger(__name__)


class PrecontextSimulation:
    """
    Re-generate the last turn of existing multi-turn conversations
    with multiple samples for analysis.
    """

    def __init__(self):
        self.args = self._parse_args()
        self.retry = True
        self.base_url = self.args.base_url
        self.model_name = self.args.model_name
        self.max_tokens = 4096

        self.client = OpenAI(
            api_key=self.args.openai_api_key,
            base_url=self.base_url,
        )

        if self.model_name is None:
            self.model_name = (
                requests.get(url=f"{self.base_url}/models")
                .json()["data"][0]["id"]
            )

        self.fullmsgs = json.load(open(self.args.fullmsg_file_name, "r"))

    def _parse_args(self):
        parser = argparse.ArgumentParser(description="Pre-context simulation")
        parser.add_argument("--base-url", type=str, default="http://localhost:8000")
        parser.add_argument("--model-name", type=str, default=None)
        parser.add_argument("--fullmsg-file-name", type=str, default="")
        parser.add_argument("--openai-api-key", type=str, default="EMPTY")
        parser.add_argument("--gentype", type=str, default="nothink")
        parser.add_argument("--responses-num", type=int, default=8)
        parser.add_argument("--output-file-name", type=str, default="")
        parser.add_argument("--max-workers", type=int, default=16)
        return parser.parse_args()

    def _gen_for_full_messages(self, entry):
        """Generate multiple responses for one conversation's last turn."""
        full_messages = entry["full_messages"]
        if full_messages is None:
            return None
        assert full_messages[-1]["role"] == "assistant"
        try:
            resp = singleturn_gen_nothink(
                self.client,
                questions=full_messages[-2]["content"],
                responses_num=self.args.responses_num,
                messages=full_messages[:-2],
                model_name=self.model_name,
                retry=self.retry,
                max_tokens=self.max_tokens,
            )
            return [r.message.content for r in resp]
        except Exception as e:
            logger.error("Generation failed: %s", e, exc_info=True)
            return None

    def run(self):
        """Execute parallel generation and write results."""
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.args.max_workers
        ) as executor:
            final_answer_list = list(
                executor.map(self._gen_for_full_messages, self.fullmsgs)
            )

        results = [
            {"full_messages": fm, "final_answer": fa}
            for fm, fa in zip(self.fullmsgs, final_answer_list)
        ]

        if os.path.isabs(self.args.output_file_name):
            output_path = self.args.output_file_name
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            _package_dir = os.path.dirname(os.path.abspath(__file__))
            tmp_dir = os.path.join(_package_dir, "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            output_path = os.path.join(tmp_dir, self.args.output_file_name)
        json.dump(results, open(output_path, "w"))
        logger.info("Results written to %s (%d entries)", output_path, len(results))


if __name__ == "__main__":
    PrecontextSimulation().run()
