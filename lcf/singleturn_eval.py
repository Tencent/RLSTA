"""
Singleturn Evaluation for Latent Capability Filtering (LCF).

Reads all trail JSONL files from a directory produced by Step 1 (singleturn mode),
aggregates responses by task_id across trails, and evaluates each response
against the ground-truth answer.

This is the "E[Ver(m | i_full)]" side of the LCF condition:
    E_m~pi(·|i_full)[Ver(m)] > E_mn~pi(·|H)[Ver(mn)]

Input  (--input-dir): directory containing trail JSONL files (e.g.
       eval_responses/<model>/singleturn_math_train/)
       Each JSONL line: {"task_id": "math/0", "full_messages": "<response_string>"}

Output (--output-file-name, written to acc_rawdata/):
    {
      "task_id_1": {
          "eval_result_list": [bool, ...],
          "final_acc_list":   [float],
          "final_answer_list": [str, ...]
      },
      ...
    }
"""

import sys
sys.path.append("../")

import re
import os
import json
import glob
import argparse

from utils.eval_math_train import eval_math_withans
from conversation_simulator.task_config import load_problems

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="LCF: evaluate single-turn full-information baseline "
                    "(reads existing responses from trail files)"
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing trail JSONL files from Step 1 "
                             "(e.g. eval_responses/<model>/singleturn_math_train/)")
    parser.add_argument("--datatype", type=str, default="math_train",
                        help="Task type for loading ground-truth answers "
                             "(must be a key in task_config.TASK_DATA_PATHS, e.g. math_train)")
    parser.add_argument("--output-file-name", type=str, required=True,
                        help="Output filename (written under acc_rawdata/)")
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

class SingleturnEval:
    """
    Evaluate the model's single-turn full-information capability by reading
    existing responses from trail files (produced by Step 1 singleturn mode).

    Reads all *.jsonl files in the input directory, aggregates responses
    by task_id, and evaluates each response against the ground-truth answer.
    """

    def __init__(self, args=None):
        self.args = args or _parse_args()

        # Load ground-truth answers via task_config
        problems = load_problems(self.args.datatype)
        # Build index: task_id -> problem dict
        # For math_train, task_config uses unique_id or f"math/{idx}" as task_id
        self.problems_dict = {}
        for idx, item in enumerate(problems):
            # Use the same logic as prepare_task to derive task_id
            if "task_id" in item:
                tid = item["task_id"]
            elif "unique_id" in item:
                tid = item["unique_id"]
            else:
                tid = f"math/{idx}"
            self.problems_dict[tid] = item

        # Read all trail files from input directory
        self.responses_by_task = self._load_all_trails()

    @staticmethod
    def _infer_model_name_from_path(input_dir: str) -> str:
        """Extract model name from input-dir path.

        Expected pattern: .../eval_responses/<model_name>/<mode>_<datatype>/
        Falls back to 'unknown_model' if the pattern doesn't match.
        """
        # Normalise: strip trailing slashes so os.path.split works correctly
        normed = os.path.normpath(input_dir)
        parts = normed.split(os.sep)
        # The model name is the second-to-last component
        # e.g. eval_responses / hunyuan-2.0-instruct-20251111 / singleturn_math_train
        if len(parts) >= 2:
            return parts[-2]
        return "unknown_model"

    def _infer_model_name(self) -> str:
        return self._infer_model_name_from_path(self.args.input_dir)

    def _load_all_trails(self) -> dict:
        """
        Read all *.jsonl files in --input-dir and aggregate responses by task_id.

        Returns:
            {task_id: [response_str_1, response_str_2, ...]}
        """
        input_dir = self.args.input_dir
        jsonl_files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl")))
        if not jsonl_files:
            raise FileNotFoundError(
                f"No .jsonl files found in {input_dir}"
            )

        responses: dict = {}
        for fpath in jsonl_files:
            with open(fpath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    tid = record["task_id"]
                    fm = record["full_messages"]
                    # singleturn: full_messages is either a string (response text)
                    # or a list of message dicts (extract last assistant content)
                    if isinstance(fm, str):
                        response_text = fm
                    elif isinstance(fm, list):
                        # Find the last assistant message
                        response_text = None
                        for msg in reversed(fm):
                            if isinstance(msg, dict) and msg.get("role") == "assistant":
                                response_text = msg["content"]
                                break
                        if response_text is None:
                            response_text = ""
                    else:
                        response_text = str(fm)
                    responses.setdefault(tid, []).append(response_text)

        print(f"[singleturn_eval] Loaded {sum(len(v) for v in responses.values())} "
              f"responses across {len(responses)} tasks "
              f"from {len(jsonl_files)} trail files")
        return responses

    def run(self):
        singleturn_acc = {}

        for task_id, response_list in self.responses_by_task.items():
            problem = self.problems_dict.get(task_id)
            if problem is None:
                print(f"[singleturn_eval] WARNING: task_id {task_id!r} not found in "
                      f"problems_dict, skipping")
                continue

            correct_answer = _extract_answer(problem["answer"])

            eval_result_list = []
            for response in response_list:
                result = eval_math_withans(response, correct_answer)
                eval_result_list.append(result)

            avg_acc = (sum(float(r) for r in eval_result_list) / len(eval_result_list)
                       if eval_result_list else 0.0)

            singleturn_acc[task_id] = {
                "eval_result_list": eval_result_list,
                "final_acc_list": [avg_acc],
                "final_answer_list": response_list,
            }

        # Write output – infer model name from input-dir path
        # Expected: eval_responses/<model_name>/<mode>_<datatype>/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        model_name = self._infer_model_name()
        output_dir = os.path.join(project_root, "acc_rawdata", model_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, self.args.output_file_name)
        with open(output_path, "w") as f:
            json.dump(singleturn_acc, f)
        print(f"[singleturn_eval] Written to {output_path} "
              f"({len(singleturn_acc)} tasks)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SingleturnEval().run()
