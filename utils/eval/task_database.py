import os
from .eval_spider_exec import eval_exec_match
from typing import Dict, Any, List
from .task_base import Task
import json, re
# """You must generate the valid SQL query that answers the given user query in natural language.

# The schema of the database you are responding to is as follows:
# [[DATABASE_SCHEMA]]

# The user query is:
# [[USER_QUERY]]

# Rules:
# - [Single Answer] Produce a single valid SQL query based on the database schema and the user query.
# - [Only SQL] Do not include any other text or comments.
# - [Validity] The SQL query must be valid and executable.
# - [Single Line] Produce your entire response in a single line."""
class TaskDatabase(Task):
    def __init__(self):
        # with open(f"prompts/database/database_full_prompt.txt", "r") as f:
        self.fully_specified_prompt = """You must generate the valid SQL query that answers the given user query in natural language.

The schema of the database you are responding to is as follows:
[[DATABASE_SCHEMA]]

The user query is:
[[USER_QUERY]]

Rules:
- [Single Answer] Produce a single valid SQL query based on the database schema and the user query.
- [Only SQL] Do not include any other text or comments.
- [Validity] The SQL query must be valid and executable.
- [Single Line] Produce your entire response in a single line."""
        # with open(f"prompts/database/database_system_prompt.txt", "r") as f:
        self.system_prompt = """You are helping a user write SQL queries to a database. If something is not clear, you can ask the user to clarify what they need. The schema for the database being accessed is the following:
        
[[SCHEMA]]""" + """
[Important] I will provide additional conditions incrementally in turns rather than all at once. Do not provide a valid answer until you have all necessary details.
After each new condition, review your previous response to ensure it remains correct and compatible.
Update your solution accordingly while maintaining correctness under all conditions provided so far."""
        self.answer_extraction_strategy = "prefix_suffix"

    def get_dataset_file(self) -> str:
        return os.path.join(os.path.dirname(__file__), "../../sharded_data/lost_in_conversation_database.json")


    def get_samples(self):
        with open(self.get_dataset_file(), "r") as f:
            data = json.load(f)
        data = [d for d in data if d["task"] == "database"]
        return data

    def get_task_name(self) -> str:
        return "database"

    def get_answer_description(self) -> str:
        return "If the response contains a complete SQL query (and not just partial or templated SQL used as an example), then it is an answer attempt. You must only extract the SQL query, nothing before or after, as it will be executed as is."

    def generate_system_prompt(self, sample: Dict[str, Any]) -> str:
        return self.system_prompt.replace("[[SCHEMA]]", sample["schema_sql"])

    def evaluator_function(self, extracted_answer: str, sample: Dict[str, Any]) -> bool:
        # simpler and easier than using parsing, based on this paper
        # https://arxiv.org/pdf/2010.02840 (followup to spider)

        pred_sql = extracted_answer.replace("```sql", "").replace("```", "")
        pred_sql = re.sub(r"\s+", " ", pred_sql).strip()
        ref_sql = sample["reference_sql"]
        # if there's no /root/tmp/database/ folder, then throw an error
        if not os.path.exists("/root/tmp/database/"):
            raise FileNotFoundError("/root/tmp/database/ folder not found; please see data/spider/README.md for instructions")


        try:
            is_correct = eval_exec_match(f"/root/tmp/database/{sample['db_id']}/", pred_sql, ref_sql, plug_value=True, keep_distinct=False, progress_bar_for_each_datapoint=False) == 1
        except Exception as e:
            print(f"Error evaluating SQL: {e}")
            is_correct = False
        score = 1.0 if is_correct else 0.0
        return {"score": score}

    def populate_fully_specific_prompt(self, sample: Dict[str, Any]) -> str:
        return self.fully_specified_prompt.replace("[[DATABASE_SCHEMA]]", sample["schema_sql"]).replace("[[USER_QUERY]]", sample["fully_specified_question"])

    def populate_concat_prompt(self, sample: Dict[str, Any]) -> str:
        user_query = "Consider all the following:\n"

        for shard in sample["shards"]:
            user_query += f"- {shard['shard']}\n"
        return self.fully_specified_prompt.replace("[[DATABASE_SCHEMA]]", sample["schema_sql"]).replace("[[USER_QUERY]]", user_query)

    def extract_fully_specific_response(self, response: str, sample: Dict[str, Any]) -> str:
        return response["sql"]

    def process_original_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process Spider sample for annotation UI display"""
        return {
            "task_id": sample["task_id"],
            "question": sample["fully_specified_question"],
            "reference_sql": sample["reference_sql"],
            "db_id": sample["db_id"],
            "spider_difficulty": sample.get("spider_difficulty", "NA"),
            # "schema": sample["schema_sql"],
        }


if __name__ == "__main__":
    task = TaskDatabase()
    samples = task.get_samples()
    print(len(samples))
