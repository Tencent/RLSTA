"""
Per-item evaluator functions for each task type.
Each function returns:
    True/False  – for math / code / actions / database
    dict        – for summary (6 metrics)
    None        – if the item has no valid response (full_messages is None)
"""

import sys
import os
import re
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.data import extract_python_code


def _last_content(item):
    fm = item["full_messages"]
    if isinstance(fm, str):
        return fm
    return fm[-1]["content"]


# ─────────────────────────── math ───────────────────────────────────────────

def eval_one_math(item, task):
    if item["full_messages"] is None:
        return None
    sample = task.get_sample(item["task_id"])
    score = task.evaluator_function(_last_content(item), sample).get("score", 0.0)
    return bool(score > 0.5)


def eval_one_math_llm(item, task, base_url, api_key, model):
    """LLM-based math evaluator: model extracts answer + local float verification."""
    if item["full_messages"] is None:
        return None
    from openai import OpenAI
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "API key is required for math-llm evaluation. "
            "Pass --api-key YOUR_KEY or set the OPENAI_API_KEY environment variable."
        )
    client = OpenAI(base_url=base_url, api_key=api_key)

    response = _last_content(item)
    sample = task.get_sample(item["task_id"])
    question = sample.get("question", "")
    correct_answer = sample.get("answer", "")
    if "#### " in correct_answer:
        correct_answer = correct_answer.split("#### ")[-1].strip()

    system_prompt = (
        "You are a strict evaluator. Your job is to extract the final answer "
        "from a model's response and compare it to the correct answer."
    )
    user_prompt = f"""
Original Question: {question}

Model's Full Response:
{response}

Correct Answer (Ground Truth): {correct_answer}

Task:
1. Extract the final specific answer from the "Model's Full Response".
2. Compare the extracted answer with the "Correct Answer".
3. Determine if they match (True/False).

Output format (JSON):
{{
    "extracted_answer": "the answer you found",
    "is_correct": true or false
}}
"""

    def _skip_special(s):
        for pat in [",", "\\$", "(?s).*#### ", "\\.$"]:
            s = re.sub(pat, "", s)
        return s

    def _safe_float(s):
        try:
            return float(_skip_special(s))
        except Exception:
            return None

    for _ in range(5000):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            result = json.loads(resp.choices[0].message.content)
            gpt_extracted = result.get("extracted_answer", "")
            gpt_verdict = result.get("is_correct", False)

            # local float verification
            try:
                gold = float(_skip_special(correct_answer))
                nums = re.findall(r"(-?[0-9.,]{2,})|(-?[0-9]+)", gpt_extracted.strip())
                extracted_val = _safe_float(
                    nums[-1][0] if nums[-1][0] else nums[-1][1]
                ) if nums else None
            except Exception:
                gold, extracted_val = None, None

            if gold is not None and extracted_val is not None:
                return abs(extracted_val - gold) < 1e-3
            return bool(gpt_verdict)
        except Exception as e:
            print(f"  math-llm retry: {e}")
            time.sleep(1)
    return False


# ─────────────────────────── code ───────────────────────────────────────────

def eval_one_code(item, task):
    if item["full_messages"] is None:
        return None
    codes = extract_python_code(_last_content(item))[-5:]
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        sample = task.get_sample(item["task_id"])
        result = any(
            task.evaluator_function(c, sample).get("is_correct", False)
            for c in codes
        )
    except Exception:
        result = False
    finally:
        sys.stdout = original_stdout
    return result


# ─────────────────────────── actions ────────────────────────────────────────

def eval_one_actions(item, task):
    if item["full_messages"] is None:
        return None
    try:
        sample = task.get_sample(item["task_id"])
        return task.evaluator_function(_last_content(item), sample).get("is_correct", False)
    except Exception:
        return False


# ─────────────────────────── summary ────────────────────────────────────────

def eval_one_summary(item, task, retry=20):
    if item["full_messages"] is None:
        return None
    response = _last_content(item)
    sample = task.get_sample(item["task_id"])
    for _ in range(retry):
        try:
            return task.evaluator_function(response, sample)
        except Exception as e:
            print(f"  summary retry: {e}")
            time.sleep(5)
    return {"coverage_score": -1, "citation_score": -1, "joint_score": -1,
            "citation_precision": -1, "citation_recall": -1, "trim_ratio": -1}


# ─────────────────────────── database ───────────────────────────────────────

def eval_one_database(item, task):
    """Stage-2: item already has correct_answer extracted by db_extract."""
    if item.get("full_messages") is None or item.get("correct_answer") is None:
        return None
    try:
        sample = task.get_sample(item["task_id"])
        return task.evaluator_function(item["correct_answer"], sample).get("score", 0) > 0.5
    except Exception:
        return False
