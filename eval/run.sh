#!/bin/bash
# Unified entry script for evaluation.
#
# Usage:
#   bash eval/run.sh <task> [options]
#
# Tasks:
#   math      - Math evaluation (GSM8K-style numeric answer)
#   code      - Code evaluation (execute Python, pass@1 + pass@k)
#   actions   - Actions evaluation
#   summary   - Summary evaluation (LLM-based, 6 metrics)
#   database  - Database evaluation (2-stage: LLM extract → rule match)
#   all       - Evaluate all task types
#
# Options:
#   --responses-dir DIR       Root dir of model responses (required)
#   --output-dir DIR          Where to save results (default: <project-dir>/eval_results)
#   --folder NAME             Evaluate only this subfolder
#   --math-eval-mode MODE     rule (default) or llm
#   --api-base-url URL        LLM API base URL (for math-llm, summary & database)
#   --api-key KEY             LLM API key (required for math-llm, summary & database)
#   --api-model MODEL         LLM model name (default: gpt-4o-mini)
#
# Examples:
#   bash eval/run.sh all      --responses-dir eval_responses/hunyuan-2.0-instruct-20251111
#   bash eval/run.sh math     --responses-dir eval_responses/hunyuan-2.0-instruct-20251111
#   bash eval/run.sh math     --responses-dir eval_responses/hunyuan-2.0-instruct-20251111 \
#                             --math-eval-mode llm \
#                             --api-base-url YOUR_BASE_URL \
#                             --api-key YOUR_KEY \
#                             --api-model YOUR_MODEL
#   bash eval/run.sh database --responses-dir eval_responses/hunyuan-2.0-instruct-20251111 \
#                             --api-base-url YOUR_BASE_URL \
#                             --api-key YOUR_KEY \
#                             --api-model YOUR_MODEL
#   bash eval/run.sh math     --responses-dir eval_responses/hunyuan-2.0-instruct-20251111 \
#                             --folder nothink_multiturn_math_test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

TASK="${1:?Please specify task: math|code|actions|summary|database|all}"
shift  # remaining args forwarded to Python

case "$TASK" in
  math|code|actions|summary|database)
    python -m eval.run_eval --task "$TASK" "$@"
    ;;
  all)
    python -m eval.run_eval "$@"
    ;;
  *)
    echo "Unknown task: $TASK"
    echo "Available: math | code | actions | summary | database | all"
    exit 1
    ;;
esac
