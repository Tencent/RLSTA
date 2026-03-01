"""
Base class for all conversation simulation experiments.

Handles:
  - CLI argument parsing
  - OpenAI client creation
  - Data loading (via task_config)
  - Output directory creation
  - Parallel execution via ThreadPoolExecutor
  - Result serialization

Subclasses only need to implement:
  - _get_output_dir_prefix() -> str          (e.g. "1new_multiturn")
  - _generate(idx, item, task_info) -> Any   (the core generation logic)
"""

import sys
import os

# Root directory of the conversation_simulator package (resolved at import time).
# All relative output paths are anchored to this directory's parent (i.e. RLSTA/).
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_PACKAGE_DIR)

sys.path.append(PROJECT_ROOT)

import json
import logging
import argparse
import concurrent.futures

import tqdm
import requests
from openai import OpenAI

from utils.data import write_jsonl
from .task_config import load_problems, prepare_task, resolve_task_type

logger = logging.getLogger(__name__)


class SimulationBase:
    """Abstract base class for simulation experiments."""

    # Subclasses may override default values
    DEFAULT_GENTYPE = "nothink"
    DEFAULT_MAX_WORKERS = 16
    DEFAULT_RETRY = True
    # Subclass sets this to "eval_responses" or "ablation_responses"
    OUTPUT_ROOT = "eval_responses"

    def __init__(self):
        self.args = self._parse_args()
        self.retry = self.DEFAULT_RETRY
        self.trail = self.args.trail
        self.base_url = self.args.base_url
        self.task_type = self.args.datatype        # user-supplied (used in output dir name)
        self.model_name = self.args.model_name
        self.gentype = self.args.gentype
        self.max_workers = getattr(self.args, "max_workers", self.DEFAULT_MAX_WORKERS)

        # Resolve task_type to its canonical key for data loading.
        # E.g. mode="mt_add", task_type="math_train" -> "mt_add_math_train"
        mode = self._get_output_dir_prefix()        # e.g. "mt_add", "mt_refine", "singleturn"
        self._resolved_task_type = resolve_task_type(self.task_type, mode=mode)

        self.client = self._init_client()
        self.problems = load_problems(self._resolved_task_type, limit=self.args.limit)
        self.output_dir = self._build_output_dir()
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # CLI argument parsing
    # ------------------------------------------------------------------
    def _parse_args(self):
        parser = argparse.ArgumentParser(
            description=f"Run {self.__class__.__name__} simulation"
        )
        parser.add_argument("--trail", type=int, required=True)
        parser.add_argument("--base-url", type=str, default="http://localhost:8000")
        parser.add_argument("--datatype", type=str, default="math_test")
        parser.add_argument("--model-name", type=str, default=None)
        parser.add_argument("--openai-api-key", type=str, default="EMPTY")
        parser.add_argument("--gentype", type=str, default=self.DEFAULT_GENTYPE)
        parser.add_argument("--max-workers", type=int, default=self.DEFAULT_MAX_WORKERS)
        parser.add_argument("--limit", type=int, default=0,
                            help="Limit number of problems to load (0 = all, useful for testing)")
        self._add_extra_args(parser)
        return parser.parse_args()

    def _add_extra_args(self, parser):
        """Override in subclass to add extra CLI args."""
        pass

    # ------------------------------------------------------------------
    # OpenAI client
    # ------------------------------------------------------------------
    def _init_client(self):
        client = OpenAI(
            api_key=self.args.openai_api_key,
            base_url=self.base_url,
        )
        if self.model_name is None:
            self.model_name = (
                requests.get(url=f"{self.base_url}/models")
                .json()["data"][0]["id"]
            )
        return client

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    def _get_output_dir_prefix(self) -> str:
        """
        Return the directory name prefix that distinguishes this simulation type.
        E.g. "1new_multiturn", "new_singleturn", "self_correct", etc.
        Subclasses MUST override this.
        """
        raise NotImplementedError

    def _build_output_dir(self) -> str:
        prefix = self._get_output_dir_prefix()
        think_prefix = "think_" if self.gentype == "think" else ""
        dir_name = f"{think_prefix}{prefix}_{self.task_type}"

        url_parts = self.base_url.split("/")
        port_str = url_parts[-2].split(":")[-1] if len(url_parts) >= 2 else ""
        if port_str.isdigit():
            model_id = port_str
        else:
            model_id = self.model_name.split("/")[-1]

        return os.path.join(PROJECT_ROOT, self.OUTPUT_ROOT, model_id, dir_name)

    # ------------------------------------------------------------------
    # Core generation (subclass implements)
    # ------------------------------------------------------------------
    def _generate(self, idx: int, item: dict, task_info: dict):
        """
        Execute the actual LLM generation for one problem.

        Args:
            idx:       problem index
            item:      raw problem dict
            task_info: dict returned by task_config.prepare_task()

        Returns:
            The generation result (messages list, string, etc.)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Single-problem processing
    # ------------------------------------------------------------------
    def _process_one_problem(self, args_tuple):
        idx, item = args_tuple
        try:
            task_info = prepare_task(self._resolved_task_type, idx, item)
        except Exception as e:
            logger.error("Task preparation failed for idx=%d: %s", idx, e, exc_info=True)
            return dict(task_id=f"unknown/{idx}", full_messages=None)

        try:
            result = self._generate(idx, item, task_info)
        except Exception as e:
            logger.error("Generation failed for task_id=%s: %s",
                         task_info.get("task_id"), e, exc_info=True)
            result = None

        return dict(
            task_id=task_info["task_id"],
            full_messages=result,
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self):
        """Execute the simulation with parallel workers and write results."""
        tasks_iter = list(enumerate(self.problems))
        samples = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for sample in tqdm.tqdm(
                executor.map(self._process_one_problem, tasks_iter),
                total=len(tasks_iter),
            ):
                samples.append(sample)

        output_path = f"{self.output_dir}/{self.trail}.jsonl"
        write_jsonl(output_path, samples)
        logger.info("Results written to %s (%d samples)", output_path, len(samples))
