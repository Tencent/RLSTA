"""
Database stage-1: use LLM to extract the final answer from each conversation item.
The extracted answer is stored in item["correct_answer"] for stage-2 rule evaluation.
"""

import sys
import os
import json
import time
import concurrent.futures
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _last_content(item):
    fm = item["full_messages"]
    if isinstance(fm, str):
        return fm
    return fm[-1]["content"]


def _extract_db_answer(item, answer_description, base_url, api_key, model):
    from openai import OpenAI
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "API key is required for database extraction. "
            "Pass --api-key YOUR_KEY or set the OPENAI_API_KEY environment variable."
        )
    client = OpenAI(base_url=base_url, api_key=api_key)
    last = _last_content(item)
    prompt = (
        "You are reviewing a multi-turn conversation between a user and an assistant, "
        "and are given the last turn of the conversation.\n"
        "In the final response from the assistant, a final answer has been provided. "
        "Your goal is to extract verbatim what the answer is:\n"
        "- If the answer is short (less than 10 words), copy verbatim in the `answer` field.\n"
        "- If the answer is long, produce it with an ellipsis: "
        "```start [...] end``` (≥4 words each side).\n\n"
        "Rules:\n"
        "- [Exact Answer Only] only extract the exact answer.\n"
        "- [Verbatim Only] do not modify the text in any way.\n"
        f"- [Task Specific Answer] {answer_description}\n"
        "- [String output] the <answer_str> must be a string.\n\n"
        'Output JSON: {"answer": "<answer_str>"}\n\n'
        f"Conversation's last turn:\n{last}"
    )
    for _ in range(5000):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            content = json.loads(resp.choices[0].message.content)["answer"]
            if "[...]" in content and content.count("[...]") == 1:
                prefix, suffix = [s.strip() for s in content.split("[...]")]
                start = last.find(prefix)
                end = last.rfind(suffix)
                content = last[start: end + len(suffix)]
            return content
        except Exception as e:
            print(f"  db extract error: {e}")
            time.sleep(1)
    return None


def run_database_stage1(items, base_url, api_key, model, workers=64):
    """LLM-extract correct_answer for each item (stage 1 of database eval)."""
    from utils.eval.task_database import TaskDatabase
    answer_description = TaskDatabase().get_answer_description()

    def _process(idx_item):
        idx, item = idx_item
        answer = _extract_db_answer(item, answer_description, base_url, api_key, model)
        return idx, {**item, "correct_answer": answer}

    results = [None] * len(items)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process, (i, it)): i for i, it in enumerate(items)}
        for f in tqdm(concurrent.futures.as_completed(futs),
                      total=len(futs), desc="db-extract"):
            idx, res = f.result()
            results[idx] = res
    return results
