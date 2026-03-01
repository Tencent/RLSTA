import os
import re
import ast
import gzip
import json

from typing import Iterable, Dict

ROOT = os.path.dirname(os.path.abspath(__file__))
HUMAN_EVAL = os.path.join(ROOT, "..", "data", "HumanEval.jsonl.gz")


def read_problems(evalset_file: str = HUMAN_EVAL) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))

# def extract_python_code(text):
#     """
#     Extract Python code blocks from text.
#     Looks for code blocks marked with ```python and ```
    
#     Args:
#         text (str): The text containing Python code blocks
        
#     Returns:
#         list: A list of extracted Python code blocks
#     """
#     # Pattern to match Python code blocks
#     # Uses re.DOTALL to make . match newlines
#     pattern = r'```python\n(.*?)```'
    
#     # Find all matches
#     matches = re.findall(pattern, text, re.DOTALL)
    
#     return matches

def extract_python_code(text):
    """
    Extract and process Python code blocks from text.
    Looks for code blocks marked with ```python or ``` and processes them using AST.
    
    Args:
        text (str): The text containing Python code blocks
        
    Returns:
        list: A list of processed Python code blocks (functions with imports)
    """

    
    # Pattern to match Python code blocks (with or without 'python' keyword)
    # More flexible pattern that handles whitespace
    pattern = r'```(?:python)?\s*(.*?)\s*```'
    
    # Find all code block matches
    code_blocks = re.findall(pattern, text, re.DOTALL)
    
    # Special handling for "class Solution" pattern (common in LeetCode-style problems)
    if "class Solution" in text:
        solution_blocks = [c for c in code_blocks if "class Solution" in c]
        if solution_blocks:
            return [solution_blocks[-1]]  # Return the last Solution class
    
    # Process code blocks from last to first
    processed_blocks = []
    if code_blocks:
        for block in reversed(code_blocks):
            result = _extract_function_from_code(block)
            if result:
                processed_blocks.append(result)
        
        # Reverse to maintain original order
        return list(reversed(processed_blocks)) if processed_blocks else []
    
    # If no code blocks found, try to extract from raw text
    text = text.strip()
    if text.startswith("```") or text.startswith("`"):
        text = text[text.find("\n"):].strip()
    
    import_idx = text.rfind("import")
    def_idx = text.rfind("def")
    start_idx = import_idx if import_idx >= 0 else def_idx
    
    if start_idx >= 0:
        text = text[start_idx:]
        result = _extract_function_from_code(text)
        if result:
            return [result]
    
    return []


def _add_parent_info(node, parent=None):
    """Add parent information to all nodes in the AST."""
    node.parent = parent
    for child in ast.iter_child_nodes(node):
        _add_parent_info(child, node)


def _extract_function_from_code(code: str) -> str:
    """Helper method to extract function from pure Python code using AST."""
    import ast
    
    try:
        # Parse the code
        tree = ast.parse(code)
        
        # Add parent information to all nodes
        _add_parent_info(tree)
        
        # Find all import statements
        import_nodes = [node for node in ast.walk(tree) 
                       if isinstance(node, (ast.Import, ast.ImportFrom))]
        imports = [ast.unparse(node) for node in import_nodes]
        
        # Find all function definitions
        function_nodes = [node for node in ast.walk(tree) 
                         if isinstance(node, ast.FunctionDef)]
        
        if not function_nodes:
            return ""  # No functions found
        
        # Get the last function definition that is at the top level
        last_function = None
        for node in reversed(function_nodes):
            # Check if the parent is the module (top level)
            if isinstance(node.parent, ast.Module):
                last_function = node
                break
        
        if not last_function:
            return ""  # No top-level functions found
        
        # Get the source lines from the original text
        source_lines = code.splitlines()
        
        # Function spans from its first line to the last line of its body
        # If there are decorators, start from the first decorator
        start_line = (last_function.decorator_list[0].lineno - 1
                     if last_function.decorator_list
                     else last_function.lineno - 1)  # ast line numbers are 1-based
        end_line = last_function.end_lineno
        
        # Extract the complete function text
        function_text = '\n'.join(source_lines[start_line:end_line])
        
        # Prepend imports if any were found
        if imports:
            return '\n'.join(imports) + '\n\n' + function_text
        return function_text
        
    except Exception:
        return ""
