"""
Latent Capability Filtering (LCF) — Filter & RL Dataset Construction.

Implements the core LCF condition (Eq. 2 from the paper):

    E_m~pi(·|i_full)[Ver(m)] > E_mn~pi(·|H)[Ver(mn)]

We retain conversation histories where the model has latent capability
to solve the problem given full information, but fails under the original
multi-turn history. This yields dataset D_M for model M.

Usage:
    python -m lcf.filter \\
        --singleturn-eval  <model_name>/<st_eval_filename> \\
        --multiturn-eval   <model_name>/<mt_eval_filename> \\
        --output-tag       <model_tag> \\
        --filter-item-numbers 200,400 \\
        [--output-prefix   lcf_gsm8k]
"""

import sys
sys.path.append("../")

import re
import os
import json
import argparse
import numpy as np

from utils.data import write_jsonl, stream_jsonl
from conversation_simulator.task_config import load_problems

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="LCF: filter multi-turn histories and build RL training set"
    )
    parser.add_argument("--singleturn-eval", type=str, required=True,
                        help="Path relative to acc_rawdata/ "
                             "for singleturn accuracy (e.g. <model_name>/st_acc_math_train.json)")
    parser.add_argument("--multiturn-eval",  type=str, required=True,
                        help="Path relative to acc_rawdata/ "
                             "for multiturn accuracy (e.g. <model_name>/mt_acc_math_train.json)")
    parser.add_argument("--datatype",         type=str, default="math_train",
                        help="Task type for loading ground-truth answers "
                             "(must be a key in task_config, e.g. math_train, math_test, code_test)")
    parser.add_argument("--output-tag",      type=str, default="model",
                        help="Model tag used as subdirectory under train_datas/")
    parser.add_argument("--output-prefix",   type=str, default="lcf_gsm8k",
                        help="Prefix for output JSONL filenames")
    parser.add_argument("--filter-item-numbers", type=str, default="200,400",
                        help="Comma-separated list of dataset sizes to produce")
    parser.add_argument("--sample-per-task", type=int, default=2,
                        help="Max conversation histories to sample per task_id")
    parser.add_argument("--add-single",      action="store_true",
                        help="Also add a single-turn anchor entry per task_id")
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

class LCFFilter:
    """
    Apply the Latent Capability Filtering condition and produce RL training data.

    Filtering condition (Eq. 2):
        singleturn_acc[task_id] > multiturn_acc   (for each conversation history)

    We exclude histories where single-turn and multi-turn performances are
    comparable (i.e., multiturn_acc >= singleturn_acc), focusing only on
    instances where the model has superior single-turn capability.
    """

    # ------------------------------------------------------------------
    # Helpers for locating accuracy files
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_acc_path(acc_dir: str, rel_path: str, model_name: str | None) -> str:
        """Resolve an accuracy-file path under *acc_dir*.

        Strategy (tried in order):
        1. ``acc_dir / rel_path``  – user already included model sub-dir
        2. ``acc_dir / model_name / rel_path`` – auto-prepend model name
        Raises ``FileNotFoundError`` if neither exists.
        """
        candidate = os.path.join(acc_dir, rel_path)
        if os.path.isfile(candidate):
            return candidate
        if model_name:
            candidate2 = os.path.join(acc_dir, model_name, rel_path)
            if os.path.isfile(candidate2):
                return candidate2
        raise FileNotFoundError(
            f"Cannot find accuracy file. Tried:\n"
            f"  1) {candidate}\n"
            f"  2) {os.path.join(acc_dir, model_name or '<no model_name>', rel_path)}\n"
            f"Hint: pass --singleturn-eval / --multiturn-eval as "
            f"'<model_name>/<filename>' or ensure --output-tag matches the model sub-directory."
        )

    @staticmethod
    def _infer_model_name(args) -> str | None:
        """Try to infer the model name from --output-tag."""
        # --output-tag is typically the model name
        if args.output_tag and args.output_tag != "model":
            return args.output_tag
        return None

    def __init__(self, args=None):
        self.args = args or _parse_args()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        acc_dir = os.path.join(project_root, "acc_rawdata")

        model_name = self._infer_model_name(self.args)

        st_path = self._resolve_acc_path(acc_dir, self.args.singleturn_eval, model_name)
        self.singleturn_acc_dict: dict = json.load(open(st_path))

        # multiturn eval may be a list (new format) or a dict (legacy format).
        # New list format: [{"task_id": ..., "messages": ..., "acc": ..., ...}, ...]
        # We convert list → dict keyed by task_id for the LCF condition check.
        mt_path = self._resolve_acc_path(acc_dir, self.args.multiturn_eval, model_name)
        raw_mt = json.load(open(mt_path))
        if isinstance(raw_mt, list):
            # Convert list format to grouped dict:
            # {task_id: [{"messages": ..., "acc": ..., ...}, ...]}
            self.multiturn_acc_list = raw_mt   # keep the flat list for direct access
            self.multiturn_acc_dict = {}
            for entry in raw_mt:
                tid = entry["task_id"]
                self.multiturn_acc_dict.setdefault(tid, []).append(entry)
        else:
            # Legacy dict format: {task_id: {"eval_result_list": ..., "final_acc_list": ...}}
            self.multiturn_acc_list = None
            self.multiturn_acc_dict = raw_mt

        # Load ground-truth problems for answer extraction and single-turn anchor
        problems = load_problems(self.args.datatype)
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

        self.filter_item_numbers = sorted(
            [int(x) for x in self.args.filter_item_numbers.split(",")],
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Step 1: Apply LCF condition
    # ------------------------------------------------------------------

    def _apply_lcf_condition(self) -> dict:
        """
        Retain conversation histories where:
            E[Ver(m | i_full)] > E[Ver(mn | H)]

        i.e., singleturn_acc > multiturn_acc for that specific history.

        Supports two multiturn eval formats:
        - New list format: multiturn_acc_dict is {task_id: [{"messages":.."acc":..}, ...]}
        - Legacy dict format: multiturn_acc_dict is {task_id: {"final_acc_list": [...], ...}}

        Returns:
            filtered_data: {task_id: [{"full_messages": ..., "acc": float}, ...]}
        """
        filtered_data: dict = {}
        accs = []

        if self.multiturn_acc_list is not None:
            # New list format: entries already grouped into multiturn_acc_dict
            for task_id, entries in self.multiturn_acc_dict.items():
                st_entry = self.singleturn_acc_dict.get(task_id)
                if st_entry is None:
                    continue
                singleturn_acc = st_entry["final_acc_list"][0]

                for entry in entries:
                    mt_acc = entry["acc"]
                    if mt_acc < singleturn_acc:
                        accs.append(mt_acc)
                        filtered_data.setdefault(task_id, []).append({
                            "full_messages": entry["messages"],
                            "acc": mt_acc,
                        })
        else:
            # Legacy dict format (no full_messages available)
            for task_id in self.multiturn_acc_dict:
                st_entry = self.singleturn_acc_dict.get(task_id)
                if st_entry is None:
                    continue
                singleturn_acc = st_entry["final_acc_list"][0]

                mt_acc_list = self.multiturn_acc_dict[task_id]["final_acc_list"]

                for mt_acc in mt_acc_list:
                    if mt_acc < singleturn_acc:
                        accs.append(mt_acc)
                        filtered_data.setdefault(task_id, []).append({
                            "full_messages": None,
                            "acc": mt_acc,
                        })

        if accs:
            print(f"[lcf_filter] Passed LCF condition: {len(accs)} histories "
                  f"across {len(filtered_data)} tasks "
                  f"(mean multiturn_acc={sum(accs)/len(accs):.4f})")
        else:
            print("[lcf_filter] Warning: no histories passed the LCF condition.")

        return filtered_data

    # ------------------------------------------------------------------
    # Step 2: Sample per task
    # ------------------------------------------------------------------

    def _get_answer(self, task_id: str) -> str:
        """Extract the ground-truth answer string for a task.

        Supports multiple data schemas:
        - math_train / math_test: ``problem["answer"]`` in GSM8K ``#### <number>`` format
        - code_test: ``problem["test"]`` (test code)
        - database: ``problem["reference_sql"]``
        - actions: ``problem["reference_answer"]``
        - summary: empty string (no exact answer)
        """
        problem = self.problems_dict.get(task_id)
        if problem is None:
            return _INVALID_ANS

        # Math-style answer (#### 42)
        if "answer" in problem:
            return _extract_answer(problem["answer"])
        # Code test cases
        if "test" in problem:
            return problem["test"]
        # Database SQL
        if "reference_sql" in problem:
            return problem["reference_sql"]
        # Actions reference answer
        if "reference_answer" in problem:
            return str(problem["reference_answer"])
        return ""

    def _sample_data(self, filtered_data: dict) -> list:
        """
        For each task_id, randomly sample up to `sample_per_task` histories.
        Attaches the ground-truth answer for RL training.
        """
        sampled = []
        for task_id, entries in filtered_data.items():
            n = min(self.args.sample_per_task, len(entries))
            chosen = np.random.choice(entries, size=n, replace=False)
            answer = self._get_answer(task_id)
            for entry in chosen:
                assert entry["full_messages"][-1]["role"] == "assistant"
                sampled.append({
                    "task_id": task_id,
                    "completion": entry["full_messages"],
                    "answer": answer,
                })
        return sampled

    # ------------------------------------------------------------------
    # Step 3: Build RL dataset (strip last assistant turn)
    # ------------------------------------------------------------------

    def _make_rl_dataset(self, sampled: list) -> list:
        """
        Convert sampled conversations into RL training format:
          - Strip the last assistant message (the model will generate it)
          - Optionally add a single-turn anchor entry per task_id

        The single-turn anchor uses the system prompt + merged shards as
        the user message, providing a high-quality supervision signal.
        """
        # Group by task_id to handle add_single correctly
        by_task: dict = {}
        for item in sampled:
            by_task.setdefault(item["task_id"], []).append(item)

        rl_data = []
        for task_id, items in by_task.items():
            for item in items:
                assert item["completion"][-1]["role"] == "assistant"
                rl_data.append({
                    "task_id": task_id,
                    "completion": item["completion"][:-1],   # drop last assistant turn
                    "answer": item["answer"],
                })

            # Single-turn anchor: system prompt + full merged query
            if self.args.add_single:
                problem = self.problems_dict[task_id]
                shards = problem.get("shards", [])
                full_query = " ".join(shards) if shards else ""
                system_msg = items[0]["completion"][0]   # reuse system prompt from history
                rl_data.append({
                    "task_id": task_id,
                    "completion": [
                        system_msg,
                        {"role": "user", "content": full_query},
                    ],
                    "answer": problem["answer"],
                })

        assert len(rl_data) >= len(sampled)
        return rl_data

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self):
        # Step 1: LCF filtering
        filtered_data = self._apply_lcf_condition()

        # Step 2: Sample
        sampled = self._sample_data(filtered_data)
        print(f"[lcf_filter] After sampling: {len(sampled)} entries")

        # Step 3: Produce datasets at each requested size
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)          # RLSTA/
        output_dir = os.path.join(project_root, "train_datas", self.args.output_tag)
        os.makedirs(output_dir, exist_ok=True)

        remaining = sampled
        for n in self.filter_item_numbers:
            if n > len(remaining):
                print(f"[lcf_filter] Skipping size={n} (only {len(remaining)} available)")
                continue
            remaining = np.random.choice(remaining, size=n, replace=False).tolist()
            rl_data = self._make_rl_dataset(remaining)
            out_path = os.path.join(
                output_dir, f"{self.args.output_prefix}_{n}.jsonl"
            )
            write_jsonl(out_path, rl_data)
            print(f"[lcf_filter] Written {len(rl_data)} entries → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    LCFFilter().run()
