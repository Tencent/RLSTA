from typing import List, Dict, Any
from .task_base import Task
import json, random, re, os
class TaskMath(Task):
    def __init__(self):
        # with open("prompts/math/math_full_prompt.txt", "r") as f:
        #     self.fully_specified_prompt = f.read()
        # with open("prompts/math/math_system_prompt.txt", "r") as f:
        #     self.system_prompt = f.read()
        self.seed = 42
        random.seed(self.seed)

        self.answer_extraction_strategy = "gen"

    def get_dataset_file(self) -> str:
        return os.path.join(os.path.dirname(__file__), "../../sharded_data/lost_in_conversation_math.json")

    def get_samples(self, filter="full"):
        with open(self.get_dataset_file(), "r") as f:
            data = json.load(f)
        return data

    def get_task_name(self):
        return "math"
    
    def get_answer_description(self) -> str:
        return "The answer should be a single number (it could be decimal, or negative, or a fraction, etc.)."
    
    def generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        return self.system_prompt

    def evaluator_function(self, extracted_answer: str, sample: Dict[str, Any]) -> bool:
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
            gold = sample["answer"].split("####")[1].strip().lower()
            gold = float(skip_special_tokens(gold))
            # https://github.com/EleutherAI/lm-evaluation-harness/blob/bb098f13b05e361f01a5afe7b612779ce362b3f2/lm_eval/tasks/gsm8k/gsm8k.yaml#L42
            extracted_answer = extracted_answer.strip()
            # strict
            # extracted_answer = re.findall(r"(\-?[0-9\.\,]+)", extracted_answer)[0]
            # flexible
            extracted_answer = re.findall(r"(-?[0-9.,]{2,})|(-?[0-9]+)", extracted_answer)
            # extracted_answer = [skip_special_tokens(''.join(m)) for m in extracted_answer if any(m)]
            extracted_answer = [skip_special_tokens(m[0] if m[0] else m[1]) for m in extracted_answer if m][-3:]
            extracted_answer =  [num for num in map(safe_float_convert, extracted_answer) if num is not None]
        except:
            return {"score": 0.0, "error": f"Answer could not be extracted: {repr(extracted_answer)}"}

        score = 1.0 if any(abs(test_answer - gold) < 1e-3 for test_answer in extracted_answer) else 0.0
        return {"score": score}

    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        return self.fully_specified_prompt.replace("[[QUESTION]]", sample["question"])

    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        query = ""
        for shard in sample["shards"]:
            query += f"- {shard['shard']}\n"
        return self.fully_specified_prompt.replace("[[QUESTION]]", query)

    def extract_fully_specific_response(self, response: str, sample: Dict[str, Any]) -> str:
        # FIXME(hiro): "completion" is not the best name for the field because we ask for a full function
        return response["answer"]

    def process_original_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process GSM8K sample for annotation UI display"""
        return {
            "task_id": sample["task_id"],
            "question": sample["question"],
            "answer": sample["answer"],
        }
