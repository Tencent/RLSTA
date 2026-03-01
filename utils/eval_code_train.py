
import os
import re
import ast
import sys
import copy
import glob
import json
import zlib
import base64
import pickle
import argparse
import concurrent.futures

from tqdm import tqdm
from .eval.task_math import TaskMath
from math_verify import parse, verify
from .eval.task_code import TaskCode, check_correctness
from .data import stream_jsonl, write_jsonl, read_problems, extract_python_code

def load_test_cases(sample):
    public_test_cases = json.loads(sample["public_test_cases"])  # type: ignore

    if "private_test_cases" in sample:
        try:
            private_test_cases = json.loads(sample["private_test_cases"])  # type: ignore
        except:
            private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(sample["private_test_cases"].encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
    else:
        private_test_cases = []

    return json.dumps(
        {
            "inputs": [
                t["input"]
                for t in public_test_cases + private_test_cases
            ],
            "outputs": [
                t["output"]
                for t in public_test_cases + private_test_cases
            ],
            "fn_name": sample["function_name"],
        }
    )
def evaluator_function(extracted_answer, sample) -> bool:
    pred_python_code = extracted_answer.replace("```python", "").replace("```", "")

    if "def " not in pred_python_code:
        return {"is_correct": False, "pass@1": 0, "score": 0}

    # Adding imports for HE-derived samples
    # if "prompt" in sample or "question_content" in sample:
    #     # Extract imports from sample["prompt"] -- this affects full
    #     prompt_ast = ast.parse(sample["prompt"] if "prompt" in sample else sample["question_content"])
    #     imports = []
    #     for node in prompt_ast.body:
    #         if isinstance(node, (ast.Import, ast.ImportFrom)):
    #             imports.append(ast.unparse(node))

    #     # Prepend imports to pred_python_func
    #     if imports:
    #         pred_python_code = "\n".join(imports) + "\n\n" + pred_python_code

    # Force update the function name with the true function name
    old_func_name = pred_python_code.split("def ")[1].split("(")[0].strip()
    pred_python_code = pred_python_code.replace(old_func_name, sample["function_name"])

    # load tests
    testcases = load_test_cases(sample)

    output, metadata = check_correctness(sample, pred_python_code, testcases, timeout=6)
    all_test_cases_passed = all(o is True for o in output)

    score = 1 if all_test_cases_passed else 0
    return {"is_correct": all_test_cases_passed, "pass@1": 1 if all_test_cases_passed else 0, "score": score}

def extract_singleturn_answer(completion):
    extracted_python_code = extract_python_code(completion)
    if len(extracted_python_code) > 0:
        longest_code = max(extracted_python_code, key=len)
        return longest_code
    return ""
def eval_code_single(py_code,eval_sample):
    # extracted_answer = extract_singleturn_answer(response)
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try :
        evaluation_return = evaluator_function(py_code, eval_sample)
        is_correct = evaluation_return.get("is_correct", False)
    except:
        is_correct = False
    sys.stdout = original_stdout
    return is_correct

def eval_code(response,eval_sample):
    return any(eval_code_single(py_code,eval_sample) for py_code in extract_python_code(response))

# def process_item_code(item,eval_sample):
#     last_query = item['last_query']
#     final_response = {prompt: {} for prompt in last_query}
#     for prompt in last_query:
#         final_response[prompt] = {
#             "eval_result": [eval_code(response, eval_sample) for response in item['final_response'][prompt]],
#             "completion": item['final_response'][prompt],
#             "query": last_query[prompt]
#             }
#         if "full_messages" in item:
#             final_response["full_messages"] = item["full_messages"]
#     final_response["task_id"] = item["task_id"]
#     return final_response
def process_item_code_train(item,eval_sample):
    eval_sample = {
        "function_name": eval_sample["function_name"],
        "public_test_cases": eval_sample["public_test_cases"],
        
    }
    final_response = {}
    final_response["task_id"] = item["task_id"]
    try:
        if "final_answer" not in item:
            if "full_messages" in item:
                final_response['final_response_eval'] = eval_code(item["full_messages"], eval_sample) 
        else:
            final_response['final_response_eval'] = [eval_code(i, eval_sample) for i in item["final_answer"]]
    except:
        final_response['final_response_eval'] = None




    # if item['full_messages'] is None:
    #     final_response["task_id"] = item["task_id"]
    #     # final_response["answer"] = item['answer']
    #     return final_response
    # assistant_response = [i["content"] for i in item['full_messages'] if i['role'] == 'assistant']
    # eval_result = [eval_code(response, eval_sample) for response in assistant_response]
    # final_response['final_response_eval'] = [eval_code(i, eval_sample) for i in item["final_answer"]]
    # # final_response["eval_result"] = eval_result
    # final_response["task_id"] = item["task_id"]
    # # final_response["answer"] = item['answer']
    # final_response["full_messages"] = item['full_messages']
    return final_response


# def process_item_math_withans(item,answer):
#     final_response = {}
#     final_response["task_id"] = item["task_id"]
#     try:
#         if "final_answer" not in item:
#             if "full_messages" in item:
#                 final_response['final_response_eval'] = eval_math_withans(item["full_messages"], answer) 
#         else:
#             final_response['final_response_eval'] = [eval_math_withans(i, answer) for i in item["final_answer"]]
#     except:
#         final_response['final_response_eval'] = None
#     return final_response
