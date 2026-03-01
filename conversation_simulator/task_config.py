"""
Task configuration: data paths, system prompts, and task preparation logic.

This module centralizes all task-type-specific configuration so that
individual simulation scripts only need to describe *how* they generate,
not *where* data lives or how prompts are selected.
"""

import os
import json
import logging
from utils.data import stream_jsonl
from utils.prompt import (
    MATH_SYSTEM_PROMPT_BASE,
    CODE_SYSTEM_PROMPT_BASE,
    SUMMARY_SYSTEM_PROMPT_BASE,
    prepare_multiturn_shards,
)
from utils.eval.task_code import TaskCode
from utils.eval.task_database import TaskDatabase
from utils.eval.task_actions import TaskActions

logger = logging.getLogger(__name__)

# Root directory: conversation_simulator's parent (i.e. RLSTA/)
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PACKAGE_DIR)

# ---------------------------------------------------------------------------
# Data loading configuration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Base data files – keyed by the user-facing datatype name.
# Each value: (file_path_relative_to_PROJECT_ROOT, loader_type)
#   loader_type: "json" | "jsonl"
# ---------------------------------------------------------------------------
_BASE_DATA_FILES = {
    "code_test": ("sharded_data/lost_in_conversation_code.json",     "json"),
    "math_test": ("sharded_data/lost_in_conversation_math.json",     "json"),
    "summary":   ("sharded_data/lost_in_conversation_summary.json",  "json"),
    "database":  ("sharded_data/lost_in_conversation_database.json", "json"),
    "actions":   ("sharded_data/lost_in_conversation_actions.json",  "json"),
    # math_train has dedicated files per mode – see _MODE_OVERRIDES
    "math_train": ("sharded_data/mt_add_math_train.jsonl",          "jsonl"),
}

# ---------------------------------------------------------------------------
# Mode-specific overrides.
# When a mode needs a *different* data file for a datatype, register it here.
# Key: (mode, datatype) -> (file_path, loader_type)
# If a (mode, datatype) pair is NOT listed here, _BASE_DATA_FILES[datatype] is used.
# ---------------------------------------------------------------------------
_MODE_OVERRIDES = {
    ("mt_refine", "math_train"): ("sharded_data/mt_refine_math_train.jsonl", "jsonl"),
    ("mt_refine", "math_test"):  ("sharded_data/mt_refine_math_eval.jsonl",  "jsonl"),
    ("mt_add",    "math_train"): ("sharded_data/mt_add_math_train.jsonl",    "jsonl"),
}

# Modes that REQUIRE the modified_shard field in data (and therefore only
# work with specific data files).  If a user picks one of these modes but
# the resolved data has no modified_shard, we raise early.
_MODES_REQUIRING_MODIFIED_SHARD = {"mt_refine"}

# ---------------------------------------------------------------------------
# Flat canonical lookup table (built from above).
# Keys are canonical task identifiers used everywhere after resolution:
#   - plain datatypes: "math_test", "code_test", …
#   - prefixed keys  : "mt_refine_math_train", "mt_add_math_train", …
# ---------------------------------------------------------------------------
TASK_DATA_PATHS: dict[str, tuple[str, str]] = {}

# 1) Plain datatype keys
TASK_DATA_PATHS.update(_BASE_DATA_FILES)

# 2) Prefixed keys from overrides
for (mode, dt), file_info in _MODE_OVERRIDES.items():
    TASK_DATA_PATHS[f"{mode}_{dt}"] = file_info

# 3) For modes that have NO override for a datatype, the prefixed key still
#    points to the base file so that e.g. "singleturn_math_test" resolves OK.
_ALL_MODES = {"mt_add", "mt_refine", "singleturn", "self_correct", "llm_simulator_multiturn"}
for _mode in _ALL_MODES:
    for _dt, _file_info in _BASE_DATA_FILES.items():
        _prefixed = f"{_mode}_{_dt}"
        if _prefixed not in TASK_DATA_PATHS:
            TASK_DATA_PATHS[_prefixed] = _file_info

# System prompt mapping: task_group -> base prompt
SYSTEM_PROMPT_MAP = {
    "code":    CODE_SYSTEM_PROMPT_BASE,
    "math":    MATH_SYSTEM_PROMPT_BASE,
    "summary": SUMMARY_SYSTEM_PROMPT_BASE,
}

# Map task_type -> prompt group  (we derive this automatically)
def _infer_prompt_group(key: str) -> str | None:
    """Infer the prompt group from a canonical task key."""
    lower = key.lower()
    if "code" in lower or "database" in lower or "actions" in lower:
        return "code"
    if "math" in lower:
        return "math"
    if "summary" in lower:
        return "summary"
    return None

TASK_PROMPT_GROUP: dict[str, str] = {}
for _key in TASK_DATA_PATHS:
    _g = _infer_prompt_group(_key)
    if _g is not None:
        TASK_PROMPT_GROUP[_key] = _g

# Task classes that override system_prompt via generate_system_prompt()
TASK_CLASS_MAP = {
    "code_test": TaskCode,
    "database":  TaskDatabase,
    "actions":   TaskActions,
}


def resolve_task_type(task_type: str, mode: str = "") -> str:
    """Resolve a (possibly shortened) task_type to its canonical key in TASK_DATA_PATHS.

    Resolution order (when *mode* is given):
      1. Prefixed match: ``"{mode}_{task_type}"``  – checked FIRST so that
         mode-specific data is always preferred (e.g. mode="mt_refine",
         task_type="math_train" -> "mt_refine_math_train").
      2. Exact match on ``task_type`` alone (e.g. "math_test").

    When *mode* is empty, only the exact match is tried.
    """
    # 1. mode-prefixed key
    if mode:
        prefixed = f"{mode}_{task_type}"
        if prefixed in TASK_DATA_PATHS:
            return prefixed
    # 2. exact key
    if task_type in TASK_DATA_PATHS:
        return task_type
    raise ValueError(
        f"Unknown task type: {task_type!r} (mode={mode!r}). "
        f"Available datatypes: {sorted(_BASE_DATA_FILES.keys())}. "
        f"All canonical keys: {sorted(TASK_DATA_PATHS.keys())}"
    )


def load_problems(task_type: str, limit: int = 0):
    """Load the dataset for *task_type* and return a list of problem dicts.

    Args:
        task_type: key in TASK_DATA_PATHS
        limit: if > 0, only return the first *limit* items (useful for testing)
    """
    if task_type not in TASK_DATA_PATHS:
        raise ValueError(f"Unknown task type: {task_type}. "
                         f"Available: {list(TASK_DATA_PATHS.keys())}")
    path, loader = TASK_DATA_PATHS[task_type]
    path = os.path.join(_PROJECT_ROOT, path)
    if loader == "json":
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = list(stream_jsonl(path))
    if limit > 0:
        data = data[:limit]
    return data


def get_system_prompt(task_type: str):
    """Return the base system prompt for the given task_type."""
    group = TASK_PROMPT_GROUP.get(task_type)
    if group is None:
        return None
    if group not in SYSTEM_PROMPT_MAP:
        raise ValueError(f"No system prompt for group={group}")
    return SYSTEM_PROMPT_MAP[group]


def _extract_base_datatype(task_type: str) -> str:
    """Strip any known mode prefix from a canonical task_type to get the base datatype.

    Examples:
        "mt_refine_math_train" -> "math_train"
        "singleturn_code_test" -> "code_test"
        "math_test"            -> "math_test"
    """
    for prefix in sorted(_ALL_MODES, key=len, reverse=True):  # longest first
        tag = f"{prefix}_"
        if task_type.startswith(tag):
            return task_type[len(tag):]
    return task_type


def prepare_task(task_type: str, idx: int, item: dict):
    """
    Prepare task metadata for one problem.

    *task_type* is the canonical (possibly prefixed) key returned by
    ``resolve_task_type()``.  We strip the mode prefix to determine the
    underlying data schema and then dispatch accordingly.

    Returns a dict with:
        - task_id:       unique identifier
        - shards:        list of shard strings
        - system_prompt: system prompt (dict or str)
    Extra keys may be present for specific task types.
    """
    base = _extract_base_datatype(task_type)
    result = {}

    # ---- Code-like tasks (code_test, database, actions) ----
    if base in ("code_test", "database", "actions"):
        shards = [i["shard"] for i in item["shards"]]

        # These tasks use a task class that generates a specialised system prompt
        task_cls = TASK_CLASS_MAP[base]
        task_obj = task_cls()
        system_prompt = {"role": "system", "content": task_obj.generate_system_prompt(item)}
        # For single-turn, we can also get a concat prompt:
        result["concat_prompt"] = task_obj.populate_concat_prompt(item)

        result.update(
            task_id=item["task_id"],
            shards=shards,
            system_prompt=system_prompt,
        )

    # ---- Math test (eval set from LiC benchmark or mt_refine_math_eval, shards are dicts) ----
    elif base == "math_test":
        system_prompt = get_system_prompt(task_type)
        # Shards may be dicts (LiC / mt_refine_eval) or plain strings
        raw_shards = item["shards"]
        if raw_shards and isinstance(raw_shards[0], dict):
            shards = [s["shard"] for s in raw_shards]
        else:
            shards = raw_shards
        result.update(
            task_id=item["task_id"],
            shards=shards,
            system_prompt=system_prompt,
        )

    # ---- math_train (shards are plain strings) ----
    elif base == "math_train":
        system_prompt = get_system_prompt(task_type)
        # Shards may be plain strings (mt_add) or dicts (mt_refine)
        raw_shards = item["shards"]
        if raw_shards and isinstance(raw_shards[0], dict):
            shards = [s["shard"] for s in raw_shards]
        else:
            shards = raw_shards  # plain string list
        task_id = item.get("task_id") or item.get("unique_id", f"math/{idx}")
        result.update(
            task_id=task_id,
            shards=shards,
            system_prompt=system_prompt,
        )

    # ---- Summary ----
    elif base == "summary":
        system_prompt = get_system_prompt(task_type)
        shards = prepare_multiturn_shards(item)
        result.update(
            task_id=item["task_id"],
            shards=shards,
            system_prompt=system_prompt,
        )

    else:
        raise ValueError(
            f"Invalid task type: {task_type!r} (base={base!r}). "
            f"Available base datatypes: {sorted(_BASE_DATA_FILES.keys())}"
        )

    return result
