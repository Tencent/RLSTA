"""
Unified evaluation entry point.

Usage (via run.sh):
    bash eval/run.sh math   --responses-dir eval_responses/hunyuan-2.0-instruct-20251111
    bash eval/run.sh code   --responses-dir eval_responses/hunyuan-2.0-instruct-20251111
    bash eval/run.sh all    --responses-dir eval_responses/hunyuan-2.0-instruct-20251111

Direct usage:
    python -m eval.run_eval --task math --responses-dir eval_responses/...
"""

import sys
import os
import glob
import json
import argparse
import concurrent.futures
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.data import stream_jsonl
from eval.evaluators import (
    eval_one_math, eval_one_math_llm,
    eval_one_code, eval_one_actions,
    eval_one_summary, eval_one_database,
)
from eval.db_extract import run_database_stage1

# ─────────────────────────── task-type detection ────────────────────────────

TASK_KEYWORDS = [
    ("database", "database"),
    ("summary",  "summary"),
    ("action",   "actions"),
    ("code",     "code"),
    ("math",     "math"),
]

def detect_task(folder_name: str) -> str:
    name = folder_name.lower()
    for keyword, task in TASK_KEYWORDS:
        if keyword in name:
            return task
    return "unknown"

# ─────────────────────────── folder runner ──────────────────────────────────

def run_folder(folder_path, output_dir, task_type,
               api_base_url=None, api_key=None, api_model=None,
               math_eval_mode="rule"):
    folder_name = os.path.basename(folder_path.rstrip("/"))
    files = glob.glob(os.path.join(folder_path, "*.jsonl"))
    if not files:
        print(f"  [skip] no jsonl files in {folder_path}")
        return

    items = []
    for f in files:
        items.extend(stream_jsonl(f))
    print(f"  [{task_type}] {folder_name}: {len(items)} items")
    os.makedirs(output_dir, exist_ok=True)

    # ── math ──────────────────────────────────────────────────────────────
    if task_type == "math":
        from utils.eval.task_math import TaskMath
        task = TaskMath()
        if math_eval_mode == "llm":
            with concurrent.futures.ThreadPoolExecutor(max_workers=64) as ex:
                futs = [ex.submit(eval_one_math_llm, it, task,
                                  api_base_url, api_key, api_model) for it in items]
                scores = [f.result() for f in tqdm(
                    concurrent.futures.as_completed(futs),
                    total=len(futs), desc=f"{folder_name}(llm)")]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=16) as ex:
                futs = [ex.submit(eval_one_math, it, task) for it in items]
                scores = [f.result() for f in tqdm(
                    concurrent.futures.as_completed(futs),
                    total=len(futs), desc=folder_name)]
        scores = [s for s in scores if s is not None]
        acc = sum(scores) / len(scores) if scores else 0.0
        mode_tag = f"_{math_eval_mode}" if math_eval_mode != "rule" else ""
        out = os.path.join(output_dir, f"{folder_name}{mode_tag}_acc_{acc:.3f}.json")
        json.dump([], open(out, "w"))
        print(f"  → acc={acc:.3f}  saved: {out}")

    # ── code ──────────────────────────────────────────────────────────────
    elif task_type == "code":
        from utils.eval.task_code import TaskCode
        task = TaskCode()
        task_scores = {}
        all_scores = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=256) as ex:
            futs = {ex.submit(eval_one_code, it, task): it["task_id"] for it in items}
            for f in tqdm(concurrent.futures.as_completed(futs),
                          total=len(futs), desc=folder_name):
                tid = futs[f]
                s = f.result()
                if s is None:
                    continue
                all_scores.append(s)
                task_scores.setdefault(tid, []).append(s)
        pass1 = sum(all_scores) / len(all_scores) if all_scores else 0.0
        passk_list = [any(v) for v in task_scores.values()]
        passk = sum(passk_list) / len(passk_list) if passk_list else 0.0
        out = os.path.join(output_dir, f"{folder_name}_pass1_{pass1:.3f}_passk_{passk:.3f}.json")
        json.dump([passk, pass1], open(out, "w"))
        print(f"  → pass@1={pass1:.3f}  pass@k={passk:.3f}  saved: {out}")

    # ── actions ───────────────────────────────────────────────────────────
    elif task_type == "actions":
        from utils.eval.task_actions import TaskActions
        task = TaskActions()
        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as ex:
            futs = [ex.submit(eval_one_actions, it, task) for it in items]
            scores = [f.result() for f in tqdm(
                concurrent.futures.as_completed(futs),
                total=len(futs), desc=folder_name)]
        scores = [s for s in scores if s is not None]
        acc = sum(scores) / len(scores) if scores else 0.0
        out = os.path.join(output_dir, f"{folder_name}_acc_{acc:.3f}.json")
        json.dump([], open(out, "w"))
        print(f"  → acc={acc:.3f}  saved: {out}")

    # ── summary ───────────────────────────────────────────────────────────
    elif task_type == "summary":
        # TaskSummary internally uses model_openai which reads from env vars
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
        if api_base_url:
            os.environ.setdefault("OPENAI_BASE_URL", api_base_url)
        from utils.eval.task_summary import TaskSummary
        task = TaskSummary()
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=64) as ex:
            futs = [ex.submit(eval_one_summary, it, task) for it in items]
            for f in tqdm(concurrent.futures.as_completed(futs),
                          total=len(futs), desc=folder_name):
                results.append(f.result())
        out = os.path.join(output_dir, f"{folder_name}_summary.json")
        json.dump(results, open(out, "w"))
        valid = [r for r in results if r and r.get("joint_score", -1) >= 0]
        avg = sum(r["joint_score"] for r in valid) / len(valid) if valid else 0.0
        print(f"  → joint_score={avg:.3f}  saved: {out}")

    # ── database ──────────────────────────────────────────────────────────
    elif task_type == "database":
        from utils.eval.task_database import TaskDatabase
        extracted = run_database_stage1(
            items, base_url=api_base_url, api_key=api_key, model=api_model)
        task = TaskDatabase()
        with concurrent.futures.ProcessPoolExecutor(max_workers=16) as ex:
            futs = [ex.submit(eval_one_database, it, task) for it in extracted]
            scores = [f.result() for f in tqdm(
                concurrent.futures.as_completed(futs),
                total=len(futs), desc=folder_name)]
        scores = [s for s in scores if s is not None]
        acc = sum(scores) / len(scores) if scores else 0.0
        out = os.path.join(output_dir, f"{folder_name}_acc_{acc:.3f}.json")
        json.dump([], open(out, "w"))
        print(f"  → acc={acc:.3f}  saved: {out}")

    else:
        print(f"  [skip] unknown task type: {folder_name}")

# ─────────────────────────── main ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified evaluation runner")
    parser.add_argument("--responses-dir", required=True,
                        help="Root dir containing per-mode subfolders, "
                             "e.g. eval_responses/hunyuan-2.0-instruct-20251111")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save result JSON files "
                             "(default: <project-dir>/eval_results)")
    parser.add_argument("--folder", default=None,
                        help="Evaluate only this subfolder name (default: all)")
    parser.add_argument("--task", default=None,
                        choices=["math", "code", "actions", "summary", "database"],
                        help="Evaluate only folders of this task type")
    parser.add_argument("--math-eval-mode", default="rule", choices=["rule", "llm"],
                        help="Math evaluation mode: rule (default) or llm")
    parser.add_argument("--api-base-url", default=None,
                        help="Base URL for LLM API (math-llm & database extract)")
    parser.add_argument("--api-key", default=None,
                        help="API key for LLM API")
    parser.add_argument("--api-model", default="gpt-4o-mini",
                        help="Model name for LLM API (default: gpt-4o-mini)")
    args = parser.parse_args()

    responses_dir = os.path.abspath(args.responses_dir)
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = args.output_dir or os.path.join(project_dir, "eval_results")
    os.makedirs(output_dir, exist_ok=True)

    if args.folder:
        folders = [os.path.join(responses_dir, args.folder)]
    else:
        folders = sorted(
            os.path.join(responses_dir, d)
            for d in os.listdir(responses_dir)
            if os.path.isdir(os.path.join(responses_dir, d))
        )

    print(f"Responses dir : {responses_dir}")
    print(f"Output dir    : {output_dir}")
    print(f"Folders found : {len(folders)}")

    for folder_path in folders:
        folder_name = os.path.basename(folder_path)
        task_type = detect_task(folder_name)
        if args.task and task_type != args.task:
            continue
        if task_type == "unknown":
            print(f"  [skip] cannot detect task type: {folder_name}")
            continue
        run_folder(
            folder_path, output_dir, task_type,
            api_base_url=args.api_base_url,
            api_key=args.api_key,
            api_model=args.api_model,
            math_eval_mode=args.math_eval_mode,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
