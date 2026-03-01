#!/bin/bash
# Latent Capability Filtering (LCF) pipeline.
#
# Three steps:
#   1. singleturn_eval - Read singleturn trail files and evaluate accuracy.
#                        Compute E[Ver(m | i_full)]
#   2. multiturn_eval  - Read multiturn trail files, re-generate last turn,
#                        and evaluate accuracy. Compute E[Ver(mn | H)]
#   3. filter          - Apply LCF condition and build RL training dataset
#
# Usage:
#   bash run.sh <step> [extra_args...]
#
# Steps:
#   singleturn_eval - Evaluate single-turn responses (no API call needed)
#   multiturn_eval  - Re-generate + evaluate multi-turn histories (needs API)
#   filter          - Apply LCF condition and produce RL training data
#
# Examples:
#   # Step 2a: evaluate single-turn responses (reads existing trail files)
#   bash lcf/run.sh singleturn_eval \
#       --input-dir eval_responses/<model>/singleturn_math_train \
#       --datatype math_train \
#       --output-file-name st_acc_math_train.json
#
#   # Step 2b: evaluate multi-turn histories (re-generates last turn via API)
#   bash lcf/run.sh multiturn_eval \
#       --input-dir eval_responses/<model>/mt_add_math_train \
#       --base-url http://localhost:8000/v1 \
#       --openai-api-key EMPTY \
#       --datatype math_train \
#       --responses-num 4 \
#       --output-file-name mt_acc_math_train.json
#
#   # Step 3: filter and build RL dataset
#   bash lcf/run.sh filter \
#       --singleturn-eval st_acc_math_train.json \
#       --multiturn-eval  mt_acc_math_train.json \
#       --fullmsg-file-name eval_responses/<model>/mt_add_math_train/1.jsonl \
#       --output-tag Qwen2.5-7B-Instruct \
#       --filter-item-numbers 200,400 \
#       --output-prefix lcf_gsm8k
set -euo pipefail

# Ensure we run from the directory containing the lcf package (i.e. RLSTA/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

STEP="${1:?Please specify step: singleturn_eval|multiturn_eval|filter}"
shift  # remaining args forwarded to Python
case "$STEP" in
  singleturn_eval)
    python -m lcf.singleturn_eval "$@"
    ;;
  multiturn_eval)
    python -m lcf.multiturn_eval "$@"
    ;;
  filter)
    python -m lcf.filter "$@"
    ;;
  *)
    echo "Unknown step: $STEP"
    echo "Available: singleturn_eval | multiturn_eval | filter"
    exit 1
    ;;
esac
