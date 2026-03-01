import time
import copy

MAX_RETRIES = 300

def singleturn_gen(client, questions = "", responses_num = 10, messages = [], model_name = "Qwen2.5-7B-Instruct", retry = False, max_tokens = 2048):
    if retry:
        retry_times = 0
        model_response = None
        while model_response is None or len(model_response[0].message.content) < 1:
            retry_times += 1
            if retry_times > MAX_RETRIES:
                raise RuntimeError(f"singleturn_gen: max retries ({MAX_RETRIES}) exceeded for model={model_name}")
            try:
                if model_name == "o3" or model_name == "gpt-5-mini":
                        model_response = client.chat.completions.create(
                        n = responses_num,
                        model = model_name,
                        messages = messages + [{ "role": "user", "content": questions}],
                        # max_tokens = max_tokens,
                        # extra_body={
                        #     "thinking_config": {
                        #         "thinking_budget": 4096,
                        #         "include_thoughts": False
                        #     }
                        # },
                        timeout=480
                    ).choices
                else: 
                    model_response = client.chat.completions.create(
                    n = responses_num,
                    model = model_name,
                    messages = messages + [{ "role": "user", "content": questions}],
                    max_tokens = max_tokens,
                    # extra_body={
                    #     "thinking_config": {
                    #         "thinking_budget": 4096,
                    #         "include_thoughts": False
                    #     }
                    # },
                    timeout=480
                ).choices
                # time.sleep(1)
            except Exception as e:
                if retry_times % 5 == 0:
                    print(f"Error: {e}, len messages: {len(messages)}, model: {model_name}, {retry_times}-th retry...")
                time.sleep(min(1.5 * retry_times, 5))
    else:
        if model_name == "o3" or model_name == "gpt-5-mini":
            model_response = client.chat.completions.create(
            n = responses_num,
            model = model_name,
            messages = messages + [{ "role": "user", "content": questions}],
            # max_tokens = max_tokens,
            # extra_body={
            #     "thinking_config": {
            #         "thinking_budget": 4096,
            #         "include_thoughts": False
            #     }
            # },
            timeout=180
            ).choices
        else: 
            model_response = client.chat.completions.create(
            n = responses_num,
            model = model_name,
            messages = messages + [{ "role": "user", "content": questions}],
            max_tokens = max_tokens,
            # extra_body={
            #     "thinking_config": {
            #         "thinking_budget": 4096,
            #         "include_thoughts": False
            #     }
            # },
            timeout=180
        ).choices
    return model_response

def singleturn_gen_nothink(client, questions = "", responses_num = 10, messages = [], model_name = "Qwen2.5-7B-Instruct", retry = False, max_tokens = 2048):
    if retry:
        retry_times = 0
        model_response = None
        while model_response is None or len(model_response[0].message.content) < 1:
            retry_times += 1
            if retry_times > MAX_RETRIES:
                raise RuntimeError(f"singleturn_gen_nothink: max retries ({MAX_RETRIES}) exceeded for model={model_name}")
            try:
                model_response = client.chat.completions.create(
                    n = responses_num,
                    model = model_name,
                    messages = messages + [{ "role": "user", "content": questions}],
                    max_tokens = max_tokens,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                ).choices
                time.sleep(1)
            except Exception as e:
                if retry_times % 5 == 0:
                    print(f"Error: {e}, len messages: {len(messages)}, model: {model_name}, {retry_times}-th retry...")
                time.sleep(min(1.5 * retry_times, 5))
    else:
        model_response = client.chat.completions.create(
            n = responses_num,
            model = model_name,
            messages = messages + [{ "role": "user", "content": questions}],
            max_tokens = max_tokens,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ).choices
    return model_response

def multiturn_gen(client, questions = [], messages = [], model_name = "Qwen2.5-7B-Instruct", return_messages = False, retry = False):
    for question in questions:
        model_response = singleturn_gen(client, questions = question, responses_num = 1, messages = messages, model_name = model_name, retry = retry)
        messages += [
            { "role": "user", "content": question},
            { "role": "assistant", "content": model_response[0].message.content},
        ]
    if return_messages is False:
        return model_response
    else:
        return messages
    
def multiturn_gen_think(client, questions = [], messages = [], model_name = "Qwen2.5-7B-Instruct", return_messages = False, retry = False, max_tokens = 2048):
    model_response = copy.deepcopy(messages)
    for question in questions:
        model_response = singleturn_gen(client, questions = question, responses_num = 1, messages = messages, model_name = model_name, retry = retry, max_tokens = max_tokens)
        messages += [
            { "role": "user", "content": question},
            { "role": "assistant", "content": model_response[0].message.content},
        ]
    if return_messages is False:
        return model_response
    else:
        return messages

def multiturn_gen_empty(client, questions = [], messages = [], assistant_response = "", **kwargs):
    for question in questions:
        messages += [
            { "role": "user", "content": question},
            { "role": "assistant", "content": assistant_response},
        ]
    return messages
    
def multiturn_gen_nothink(client, questions = [], messages = [], model_name = "Qwen2.5-7B-Instruct", return_messages = False, retry = False, max_tokens = 2048):
    for question in questions:
        model_response = singleturn_gen_nothink(client, questions = question, responses_num = 1, messages = messages, model_name = model_name, retry = retry, max_tokens = max_tokens)
        messages += [
            { "role": "user", "content": question},
            { "role": "assistant", "content": model_response[0].message.content},
        ]
    if return_messages is False:
        return model_response
    else:
        return messages

def apply_chat_template(messages, add_generation_prompt=False, tokenize=False):
    """
    Applies the Qwen chat template to a list of messages.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        add_generation_prompt: Whether to add a prompt for the model to generate a response
        tokenize: Whether to tokenize the result
        
    Returns:
        Formatted chat template as a string
    """
    conversation = ""
    
    # Check if we have any messages
    if not messages:
        # If messages is empty, return empty string or just the generation prompt
        if add_generation_prompt:
            return "<|im_start|>assistant\n"
        return ""
    
    # Process each message
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            conversation += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            conversation += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            conversation += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        else:
            raise ValueError(f"Unsupported role: {role}")
    
    # Add generation prompt if requested
    if add_generation_prompt:
        conversation += "<|im_start|>assistant\n"
    
    return conversation

# def singleturn_gen_precontext(client, questions = "", responses_num = 1, messages = [], extra_prompt = "", model_name = "Qwen2.5-7B-Instruct"):
#     model_response = None
#     while model_response is None or len(model_response[0].text) < 1:
#         try:
#             model_response = client.completions.create(
#                 n = responses_num,
#                 model = model_name,
#                 prompt = apply_chat_template(messages + [{ "role": "user", "content": questions}], add_generation_prompt = True) + extra_prompt,
#             ).choices
#             time.sleep(1)
#         except Exception as e:
#             time.sleep(1)
#     return model_response[0].text

def singleturn_gen_precontext(client, questions = "", responses_num = 1, messages = [], extra_prompt = "", model_name = "Qwen2.5-7B-Instruct", retry = False):
    model_response = None
    if retry:
        retry_times = 0
        model_response = None
        while model_response is None or len(model_response[0].text) < 1:
            retry_times += 1
            if retry_times > MAX_RETRIES:
                raise RuntimeError(f"singleturn_gen_precontext: max retries ({MAX_RETRIES}) exceeded for model={model_name}")
            try:
                model_response = client.completions.create(
                    n = responses_num,
                    model = model_name,
                    prompt = apply_chat_template(messages + [{ "role": "user", "content": questions}], add_generation_prompt = True) + extra_prompt,
                    max_tokens = 4096
                ).choices
                time.sleep(1)
            except Exception as e:
                if retry_times % 5 == 0:
                    print(f"Error: {e}, len messages: {len(messages)}, model: {model_name}, {retry_times}-th retry...")
                time.sleep(min(1.5 * retry_times, 5))
    else:
        model_response = client.completions.create(
            n = responses_num,
            model = model_name,
            prompt = apply_chat_template(messages + [{ "role": "user", "content": questions}], add_generation_prompt = True) + extra_prompt,
            max_tokens = 4096
        ).choices
    return [i.text for i in model_response]
