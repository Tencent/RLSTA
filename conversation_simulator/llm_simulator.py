"""
LLM-as-user simulation.

Uses GPT-4o-mini to simulate a "lazy user" who reveals shards one at a time,
while the local LLM tries to answer the evolving question.
"""

import json
import os
import time

from openai import OpenAI
from utils.api_gen import singleturn_gen, multiturn_gen_think, multiturn_gen_nothink
from utils.prompt import CODE_SYSTEM_PROMPT_BASE
from .base import SimulationBase


# ---- GPT-4o-mini user simulator ----

_LLM_USER_PROMPT_TEMPLATE = """You are simulating a user of an interactive LLM system (like ChatGPT).
    The user is inherently lazy, and answers in short form, providing only minimal information to the system. You should not be proactive.

    Here's the conversation so far:
    {conversation_str}

    Here are the shards that have already been revealed:
    {shards_revealed_str}

    Here are all the shards that have not been revealed yet:
    {shards_not_revealed_str}

    You must generate a response to the conversation so far. Here are the rules:
    - [Providing a Shard] You can reveal the content of a shard to the system in your response if it will help the system move closer to answering the problem. You should select the shard to reveal that is most "basic" and is the current most relevant shard.
    - [One Shard at a Time] You should only reveal at most one shard at a time.
    - [Reveal Entire Shard] If you reveal a shard, you must make sure to include *all the information in the shard*. For example, if the shard is "your symptoms are that you have a headache in the mornings", your response can't just be ``yeah I have headaches'', you must say ``yup mostly headaches in the mornings``.
    - [Irrelevant Clarifications] If the system asks you a question irrelevant to the shards, asks you a generic question (``Can you give me a hint?``), you should respond with an answer that does not provide a shard. (``I don't know``, ``Is that really important?``, etc.) You should not reveal any information beyond what is available in the shards.
    - [No Repeated Shards] You should not reveal the same shard more than once. Carefully review the shards revealed already, and only reveal a shard if its `shard_id` is not on the list.
    - [Rephrase Shards] If you reveal a shard, you should rephrase it in a conversational way. Do not copy the shard verbatim.
    - [Do Not Ask Questions] Your response should always be declarative sentences, and not questions.
    - [Brevity of Response] You should favor being succint. Your answer can also have typos, improper grammar, capitalization, etc. You are simulating a real person talking to an AI, who is in a hurry.
    - [Format] Your response should be formatted as a JSON object with the following keys:
        - `response`: The response to the conversation so far.
        - `shard_id`: The shard you are revealing to the system. The shard_id can be an integer, or -1 if you did not reveal any shards.
    For example:
    {{"response": "I don't know", "shard_id": -1}}
    or:
    {{"response": "yeah I want it to [...]", "shard_id": 1}}"""


def _get_user_response(shards, shard_ids_revealed, shard_ids_not_revealed, msg):
    """
    Call GPT-4o-mini to generate the next user utterance.
    Returns (user_response_str, shard_id).
    """
    _base_url = os.environ.get("LLM_USER_BASE_URL")
    _api_key = os.environ.get("LLM_USER_API_KEY")
    if not _base_url or not _api_key:
        raise EnvironmentError(
            "LLM user simulator requires environment variables: "
            "LLM_USER_BASE_URL and LLM_USER_API_KEY. "
            "e.g. export LLM_USER_BASE_URL='https://your-api-endpoint/v1/' "
            "     export LLM_USER_API_KEY='your-key'"
        )
    llm_client = OpenAI(
        base_url=_base_url,
        api_key=_api_key,
    )
    shard_texts_revealed = [s for s in shards if s["shard_id"] in shard_ids_revealed]
    shard_texts_not_revealed = [s for s in shards if s["shard_id"] in shard_ids_not_revealed]

    conversation_str = "\n\n".join(
        f"[{item['role']}] {item['content']}" for item in msg
    )

    prompt = _LLM_USER_PROMPT_TEMPLATE.format(
        conversation_str=conversation_str,
        shards_revealed_str=json.dumps(shard_texts_revealed),
        shards_not_revealed_str=json.dumps(shard_texts_not_revealed),
    )

    max_retry = 1000
    while max_retry > 0:
        try:
            response = singleturn_gen(
                llm_client,
                questions=prompt,
                responses_num=1,
                messages=[],
                model_name="gpt-4o-mini",
                retry=True,
                max_tokens=1024,
            )[0].message.content
            response_json = json.loads(response)
            user_response = response_json["response"]
            shard_id = response_json["shard_id"]
            assert shard_id in shard_ids_not_revealed
            return user_response, shard_id
        except Exception as e:
            print(e)
            max_retry -= 1
            time.sleep(5)

    # Fallback: reveal the first unrevealed shard
    return shard_texts_not_revealed[0], list(shard_ids_not_revealed)[0]


def _multiturn_llm_user(sample, local_client, model_name, gentype="nothink", retry=True, system_prompt=None):
    """Run a full multi-turn conversation with LLM-simulated user."""
    if system_prompt is None:
        system_prompt = CODE_SYSTEM_PROMPT_BASE
    shards = sample["shards"]
    question = shards[0]["shard"]

    shard_ids_revealed = set()
    shard_ids_not_revealed = set(i["shard_id"] for i in shards)

    gen_fn = multiturn_gen_nothink if gentype == "nothink" else multiturn_gen_think
    max_tokens = 4096 if gentype == "nothink" else 4096 * 2

    msg = gen_fn(
        local_client,
        questions=[question],
        messages=[system_prompt],
        model_name=model_name,
        return_messages=True,
        retry=retry,
        max_tokens=max_tokens,
    )

    for _ in range(len(shards) - 1):
        user_response, shard_id = _get_user_response(
            shards, shard_ids_revealed, shard_ids_not_revealed, msg
        )
        shard_ids_revealed.add(shard_id)
        shard_ids_not_revealed.remove(shard_id)
        msg = gen_fn(
            local_client,
            questions=[user_response],
            messages=msg,
            model_name=model_name,
            return_messages=True,
            retry=retry,
            max_tokens=max_tokens,
        )
    return msg


class LLMSimulatorSimulation(SimulationBase):
    DEFAULT_GENTYPE = "nothink"
    OUTPUT_ROOT = "eval_responses"

    def _get_output_dir_prefix(self) -> str:
        return "llm_simulator_multiturn"

    def _generate(self, idx, item, task_info):
        return _multiturn_llm_user(
            item, self.client, self.model_name,
            gentype=self.gentype,
            retry=self.retry,
            system_prompt=task_info.get("system_prompt"),
        )


if __name__ == "__main__":
    LLMSimulatorSimulation().run()
