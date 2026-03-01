import os
import re
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

def eval_math(response,task_id,math_task):
    sharded_sample = math_task.get_sample(task_id)
    evaluation_return = math_task.evaluator_function(response, sharded_sample)
    is_correct = evaluation_return.get("score", 0.0) > 0.5
    return is_correct


def process_item_math(item):
    final_response = {}
    final_response["task_id"] = item["task_id"]
    # final_response['final_response_eval'] = [eval_math(i, item['task_id'], TaskMath()) for i in item["final_response"]]
    if type(item["final_response"]) is list:
        final_response['final_response_eval'] = [eval_math(i, item['task_id'], TaskMath()) for i in item["final_response"]]
    else:
        final_response['final_response_eval'] = eval_math(item["final_response"], item['task_id'], TaskMath())
    return final_response
