#!/bin/bash
# Unified entry script for conversation simulation experiments.
#
# Usage:
#   bash run.sh <mode> [extra_args...]
#
# Modes:
#   mt_add       - Multi-turn conversation simulation (add shards one per turn)
#   mt_refine    - Multi-turn refine / correction (use --strategy mt-refine|atr|awr)
#   singleturn   - Single-turn baseline
#   self_correct - Self-correction simulation
#   llm_sim      - LLM-as-user simulation
#   precontext   - Pre-context re-generation
#
# The --trails flag accepts a comma-separated list (e.g. --trails 1,2,3,4,5).
# Each trail is launched in parallel via background processes and waited on.
#
# Examples:
#   bash run.sh mt_add      --trails 1,2,3,4,5 --base-url http://localhost:8000/v1 --datatype math_test --gentype think
#   bash run.sh singleturn  --trails 1,2,3 --base-url http://localhost:8000/v1 --datatype code_test --gentype nothink
#   bash run.sh mt_refine   --trails 1,2 --base-url http://localhost:8000/v1 --datatype math_train --strategy mt-refine
#   bash run.sh precontext  --base-url http://localhost:8000/v1 --fullmsg-file-name /path/to/file.json --output-file-name out.json
set -euo pipefail

MODE="${1:?Please specify mode: mt_add|mt_refine|singleturn|self_correct|llm_sim|precontext}"
shift  # remaining args forwarded to Python

# --- Determine the Python module for the selected mode ---
case "$MODE" in
  mt_add)       MODULE="conversation_simulator.mt_add" ;;
  mt_refine)    MODULE="conversation_simulator.mt_refine" ;;
  singleturn)   MODULE="conversation_simulator.singleturn" ;;
  self_correct) MODULE="conversation_simulator.self_correct" ;;
  llm_sim)      MODULE="conversation_simulator.llm_simulator" ;;
  precontext)   MODULE="conversation_simulator.precontext" ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Available: mt_add | mt_refine | singleturn | self_correct | llm_sim | precontext"
    exit 1
    ;;
esac

# --- Parse --trails from arguments; collect the rest into OTHER_ARGS ---
TRAILS=""
OTHER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --trails)
      TRAILS="$2"
      shift 2
      ;;
    *)
      OTHER_ARGS+=("$1")
      shift
      ;;
  esac
done

# --- Execute ---
if [[ -z "$TRAILS" ]]; then
  # No --trails provided: run once without --trail (for modes like precontext)
  python -m "$MODULE" "${OTHER_ARGS[@]}"
else
  # Split comma-separated trails and launch each in parallel
  IFS=',' read -ra TRAIL_LIST <<< "$TRAILS"
  for t in "${TRAIL_LIST[@]}"; do
    echo "[run.sh] Launching $MODE --trail $t in background ..."
    python -m "$MODULE" --trail "$t" "${OTHER_ARGS[@]}" &
  done
  echo "[run.sh] Waiting for all ${#TRAIL_LIST[@]} trail(s) to finish ..."
  wait
  echo "[run.sh] All trails completed."
fi
