
import os
import re
import sys
import copy
import glob
import json
import argparse
import concurrent.futures

from tqdm import tqdm
from Levenshtein import ratio
from math_verify import parse, verify
from .eval.task_code import TaskCode
from .eval.task_math import TaskMath
from .data import stream_jsonl, write_jsonl, read_problems, extract_python_code

def extract_singleturn_answer(completion):
    extracted_python_code = extract_python_code(completion)
    if len(extracted_python_code) > 0:
        longest_code = max(extracted_python_code, key=len)
        return longest_code
    return ""
def eval_code_single(py_code,task_id,code_task):
    # extracted_answer = extract_singleturn_answer(response)
    sharded_sample = code_task.get_sample(task_id)
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try :
        evaluation_return = code_task.evaluator_function(py_code, sharded_sample)
        is_correct = evaluation_return.get("is_correct", False)
    except:
        is_correct = False
    sys.stdout = original_stdout
    return is_correct

def eval_code(response,task_id,code_task):
    return any(eval_code_single(py_code,task_id,code_task) for py_code in extract_python_code(response))

def process_item_code(item):
    final_response = {}
    final_response["task_id"] = item["task_id"]
    if type(item["final_response"]) is list:
        final_response['final_response_eval'] = [eval_code(i, item['task_id'], TaskCode()) for i in item["final_response"]]
    else:
        final_response['final_response_eval'] = eval_code(item["final_response"], item['task_id'], TaskCode())
    return final_response
