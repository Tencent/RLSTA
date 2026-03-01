import os
import re
import copy
import glob
import json
import argparse
import concurrent.futures

from tqdm import tqdm
from math_verify import parse, verify
from .eval.task_code import TaskCode
from .eval.task_math import TaskMath

def is_number(s):
    try:
        float(s)  # Try converting to float first
        return True
    except ValueError:
        return False
def eval_math_withans(response, answer):
    split_response = response.split("\n")
    if "=" not in answer:
        split_response = sum([r.split("=") for r in split_response],[])
    if is_number(answer):
        def skip_special_tokens(extracted_answer):
            regexes_to_ignore = [",", "\\$", "(?s).*#### ", "\\.$"]
            for regex in regexes_to_ignore:
                extracted_answer = re.sub(regex, "", extracted_answer)
            return extracted_answer
        def safe_float_convert(text):
            try:
                return float(skip_special_tokens(text))
            except (ValueError, TypeError):
                return None
        try:
            gold = float(answer)
            extracted_answer = response.strip()
            extracted_answer = re.findall(r"(-?[0-9.,]{2,})|(-?[0-9]+)", extracted_answer)
            extracted_answer = [skip_special_tokens(m[0] if m[0] else m[1]) for m in extracted_answer if m][-3:]
            extracted_answer =  [num for num in map(safe_float_convert, extracted_answer) if num is not None]
        except:
            return False
        return any(abs(test_answer - gold) < 1e-3 for test_answer in extracted_answer)
    else:
        gold = parse(f"${answer}$")
        for r in split_response:
            r = r.strip().strip(".")
            pred1 = parse(r)
            pred2 = parse(f"${r}$")
            if verify(gold, pred1) or verify(gold, pred2):
                return True
        return False

def process_item_math_withans(item,answer):
    final_response = {}
    final_response["task_id"] = item["task_id"]
    try:
        if "final_answer" not in item:
            if "full_messages" in item:
                final_response['final_response_eval'] = eval_math_withans(item["full_messages"], answer) 
        else:
            final_response['final_response_eval'] = [eval_math_withans(i, answer) for i in item["final_answer"]]
    except:
        final_response['final_response_eval'] = None
    return final_response
