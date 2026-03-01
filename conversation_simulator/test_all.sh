#!/bin/bash
# ===========================================================================
# Comprehensive test script for conversation_simulator
#
# Usage:
#   cd /root/h20_sh_new/RLSTA
#   bash conversation_simulator/test_all.sh <BASE_URL> <MODEL_NAME> [OPENAI_API_KEY]
#
# Example:
#   bash conversation_simulator/test_all.sh http://localhost:8000/v1 Qwen/Qwen3-8B
#   bash conversation_simulator/test_all.sh http://localhost:8000/v1 Qwen/Qwen3-8B EMPTY
#
# Each test only loads 5 items (--limit 5) for quick validation.
# All tests use gentype=nothink for speed.
#
# For llm_sim mode, you must also set:
#   export LLM_USER_BASE_URL='https://your-api-endpoint/v1/'
#   export LLM_USER_API_KEY='your-key'
# ===========================================================================
set -uo pipefail

BASE_URL="${1:?Usage: bash test_all.sh <BASE_URL> <MODEL_NAME> [OPENAI_API_KEY]}"
MODEL_NAME="${2:?Usage: bash test_all.sh <BASE_URL> <MODEL_NAME> [OPENAI_API_KEY]}"
API_KEY="${3:-EMPTY}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

LIMIT=5
TRAIL=999
MAX_WORKERS=10
GENTYPE="nothink"

PASS=0
FAIL=0
SKIP=0
FAILED_CASES=()

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Track output files for cleanup
OUTPUT_FILES=()

# ------------------------------------------------------------------
# Verify that the output JSONL file exists and every record has
# a non-null full_messages field (i.e. conversation history saved).
# ------------------------------------------------------------------
verify_output() {
    local output_file="$1"
    local test_name="$2"

    if [[ ! -f "$output_file" ]]; then
        echo -e "${RED}   ⚠ VERIFY FAIL: output file not found: ${output_file}${NC}"
        FAIL=$((FAIL + 1))
        FAILED_CASES+=("${test_name} [output file missing]")
        return 1
    fi

    local total_lines
    total_lines=$(wc -l < "$output_file")
    if [[ "$total_lines" -eq 0 ]]; then
        echo -e "${RED}   ⚠ VERIFY FAIL: output file is empty: ${output_file}${NC}"
        FAIL=$((FAIL + 1))
        FAILED_CASES+=("${test_name} [output file empty]")
        return 1
    fi

    # Check that no record has full_messages: null
    local null_count
    null_count=$(python3 -c "
import json, sys
count = 0
with open('$output_file') as f:
    for line in f:
        obj = json.loads(line.strip())
        if obj.get('full_messages') is None:
            count += 1
print(count)
" 2>/dev/null || echo "-1")

    if [[ "$null_count" == "-1" ]]; then
        echo -e "${RED}   ⚠ VERIFY FAIL: could not parse output file: ${output_file}${NC}"
        FAIL=$((FAIL + 1))
        FAILED_CASES+=("${test_name} [output parse error]")
        return 1
    elif [[ "$null_count" -gt 0 ]]; then
        echo -e "${RED}   ⚠ VERIFY FAIL: ${null_count}/${total_lines} records have null full_messages${NC}"
        FAIL=$((FAIL + 1))
        FAILED_CASES+=("${test_name} [${null_count} null full_messages]")
        return 1
    fi

    echo -e "${GREEN}   ✔ VERIFY OK: ${total_lines} records, all have full_messages${NC}"
    return 0
}

run_test() {
    local test_name="$1"
    shift
    local cmd="$*"

    echo ""
    echo -e "${CYAN}======================================${NC}"
    echo -e "${CYAN}TEST: ${test_name}${NC}"
    echo -e "${CYAN}CMD:  ${cmd}${NC}"
    echo -e "${CYAN}======================================${NC}"

    if eval "$cmd"; then
        echo -e "${GREEN}✅ PASSED: ${test_name}${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}❌ FAILED: ${test_name} (exit code $?)${NC}"
        FAIL=$((FAIL + 1))
        FAILED_CASES+=("$test_name")
    fi
}

skip_test() {
    local test_name="$1"
    local reason="$2"
    echo ""
    echo -e "${YELLOW}⏭️  SKIPPED: ${test_name} — ${reason}${NC}"
    SKIP=$((SKIP + 1))
}

# Run a standard test and then verify its output file
run_and_verify() {
    local test_name="$1"
    local output_file="$2"
    shift 2
    local cmd="$*"

    OUTPUT_FILES+=("$output_file")

    echo ""
    echo -e "${CYAN}======================================${NC}"
    echo -e "${CYAN}TEST: ${test_name}${NC}"
    echo -e "${CYAN}OUTPUT: ${output_file}${NC}"
    echo -e "${CYAN}======================================${NC}"

    # Check if output file already exists
    if [[ -f "$output_file" ]]; then
        echo -e "${YELLOW}⏭️  SKIPPED: ${test_name} — output file already exists${NC}"
        SKIP=$((SKIP + 1))
        # Verify the existing output file
        verify_output "$output_file" "$test_name"
        return 0
    fi

    echo -e "${CYAN}CMD:  ${cmd}${NC}"

    if eval "$cmd"; then
        echo -e "${GREEN}✅ PASSED: ${test_name}${NC}"
        PASS=$((PASS + 1))
        # Verify output
        verify_output "$output_file" "$test_name"
    else
        echo -e "${RED}❌ FAILED: ${test_name} (exit code $?)${NC}"
        FAIL=$((FAIL + 1))
        FAILED_CASES+=("$test_name")
    fi
}

# Helper: build the expected output path for a given mode and task
# Mirrors the logic in base.py _build_output_dir
get_output_path() {
    local prefix="$1"    # e.g. "1new_multiturn"
    local task="$2"      # e.g. "math_test"
    local gentype="$3"   # "think" or "nothink"

    local nothink_prefix=""
    if [[ "$gentype" == "nothink" ]]; then
        nothink_prefix="nothink_"
    fi

    local dir_name="${nothink_prefix}${prefix}_${task}"

    # Determine model_id (same logic as base.py)
    local port_str
    port_str=$(echo "$BASE_URL" | grep -oP ':\K[0-9]+(?=/)')
    local model_id
    if [[ -n "$port_str" ]]; then
        model_id="$port_str"
    else
        model_id="${MODEL_NAME##*/}"
    fi

    echo "eval_responses/${model_id}/${dir_name}/${TRAIL}.jsonl"
}

# Common args shared by all standard (non-precontext) tests
COMMON="--trail $TRAIL --base-url $BASE_URL --model-name $MODEL_NAME --openai-api-key $API_KEY --limit $LIMIT --max-workers $MAX_WORKERS --gentype $GENTYPE"

echo ""
echo "============================================"
echo " Conversation Simulator — Full Test Suite"
echo "============================================"
echo " BASE_URL:    ${BASE_URL}"
echo " MODEL_NAME:  ${MODEL_NAME}"
echo " API_KEY:     ${API_KEY}"
echo " GENTYPE:     ${GENTYPE}"
echo " LIMIT:       ${LIMIT} items per test"
echo " TRAIL:       ${TRAIL}"
echo " MAX_WORKERS: ${MAX_WORKERS}"
echo " PROJECT_DIR: ${PROJECT_DIR}"
echo "============================================"

# ===================================================================
# 1. MULTITURN (mt_add) — all task_types, gentype=nothink
#    Supports: math_test, code_test, summary, database, actions,
#              mt_add_math_train
# ===================================================================

for TASK in math_test code_test summary database actions mt_add_math_train; do
    OUT=$(get_output_path "multiturn" "$TASK" "$GENTYPE")
    run_and_verify "multiturn / ${TASK} / ${GENTYPE}" "$OUT" \
        python -m conversation_simulator.multiturn \
        $COMMON --datatype "$TASK"
done

# ===================================================================
# 2. SINGLETURN — all task_types, gentype=nothink
#    Supports the same task_types as multiturn
# ===================================================================

for TASK in math_test code_test summary database actions mt_add_math_train; do
    OUT=$(get_output_path "singleturn" "$TASK" "$GENTYPE")
    run_and_verify "singleturn / ${TASK} / ${GENTYPE}" "$OUT" \
        python -m conversation_simulator.singleturn \
        $COMMON --datatype "$TASK"
done

# ===================================================================
# 3. SELF_CORRECT — all task_types, gentype=nothink
# ===================================================================

for TASK in math_test code_test summary database actions mt_add_math_train; do
    OUT=$(get_output_path "self_correct" "$TASK" "$GENTYPE")
    run_and_verify "self_correct / ${TASK} / ${GENTYPE}" "$OUT" \
        python -m conversation_simulator.self_correct \
        $COMMON --datatype "$TASK"
done

# ===================================================================
# 4. MT_REFINE — mt_refine_math_eval/train × mt-refine/atr/awr
#    These task_types have 'modified_shard' field required by mt_refine
# ===================================================================

for TASK in mt_refine_math_eval mt_refine_math_train; do
    for STRAT in mt-refine atr awr; do
        OUT=$(get_output_path "mt_refine_${STRAT}" "$TASK" "$GENTYPE")
        run_and_verify "mt_refine / ${TASK} / ${STRAT} / ${GENTYPE}" "$OUT" \
            python -m conversation_simulator.mt_refine \
            $COMMON --datatype "$TASK" --strategy "$STRAT"
    done
done

# ===================================================================
# 5. LLM_SIM — only code_test (needs shard_id dict-structured shards)
#    Requires LLM_USER_BASE_URL and LLM_USER_API_KEY env vars
# ===================================================================

if [[ -n "${LLM_USER_BASE_URL:-}" && -n "${LLM_USER_API_KEY:-}" ]]; then
    OUT=$(get_output_path "llm_simulator_multiturn" "code_test" "$GENTYPE")
    run_and_verify "llm_sim / code_test / ${GENTYPE}" "$OUT" \
        python -m conversation_simulator.llm_simulator \
        $COMMON --datatype code_test
else
    skip_test "llm_sim / code_test / ${GENTYPE}" \
        "LLM_USER_BASE_URL and LLM_USER_API_KEY not set"
fi

# ===================================================================
# 6. PRECONTEXT — uses real multiturn results from eval_responses as input
#    Reads JSONL results from prior tests, converts to JSON array format
#    expected by precontext.py, then re-generates last turn with multiple
#    samples for analysis.
# ===================================================================

EVAL_RESPONSES_DIR="eval_responses/hunyuan-2.0-instruct-20251111"
PRECONTEXT_INPUT_FILE="/tmp/test_precontext_fullmsgs_$$.json"
PRECONTEXT_OUTPUT_FILE="/tmp/test_precontext_output_$$.json"

# Try to find a suitable multiturn JSONL from prior eval_responses
PRECONTEXT_SOURCE=""
for candidate in \
    "${EVAL_RESPONSES_DIR}/nothink_multiturn_math_test/999.jsonl" \
    "${EVAL_RESPONSES_DIR}/nothink_multiturn_code_test/999.jsonl" \
    "${EVAL_RESPONSES_DIR}/multiturn_math_test/999.jsonl" \
    "${EVAL_RESPONSES_DIR}/multiturn_code_test/999.jsonl"; do
    if [[ -f "$candidate" ]]; then
        PRECONTEXT_SOURCE="$candidate"
        break
    fi
done

if [[ -n "$PRECONTEXT_SOURCE" ]]; then
    echo -e "${CYAN}   Using precontext source: ${PRECONTEXT_SOURCE}${NC}"

    # Convert JSONL → JSON array (take first $LIMIT items, keep only full_messages)
    python3 -c "
import json, sys

items = []
with open('$PRECONTEXT_SOURCE') as f:
    for i, line in enumerate(f):
        if i >= $LIMIT:
            break
        obj = json.loads(line.strip())
        fm = obj.get('full_messages')
        if fm is not None and len(fm) >= 3 and fm[-1]['role'] == 'assistant':
            items.append({'full_messages': fm})

if not items:
    print('ERROR: no valid full_messages found in source file', file=sys.stderr)
    sys.exit(1)

with open('$PRECONTEXT_INPUT_FILE', 'w') as out:
    json.dump(items, out)
print(f'Prepared {len(items)} conversations for precontext test')
"

    if [[ $? -eq 0 ]]; then
        run_test "precontext / real multiturn data / responses_num=2" \
            python -m conversation_simulator.precontext \
            --base-url "$BASE_URL" --model-name "$MODEL_NAME" --openai-api-key "$API_KEY" \
            --fullmsg-file-name "$PRECONTEXT_INPUT_FILE" \
            --output-file-name "$PRECONTEXT_OUTPUT_FILE" \
            --responses-num 2 --max-workers $MAX_WORKERS

        # Verify precontext output exists and is valid JSON with expected structure
        if [[ -f "$PRECONTEXT_OUTPUT_FILE" ]]; then
            PRECONTEXT_VERIFY=$(python3 -c "
import json, sys
data = json.load(open('$PRECONTEXT_OUTPUT_FILE'))
total = len(data)
valid = sum(1 for d in data if d.get('final_answer') is not None)
print(f'{total} {valid}')
" 2>/dev/null || echo "0 0")
            PRECONTEXT_TOTAL=$(echo "$PRECONTEXT_VERIFY" | cut -d' ' -f1)
            PRECONTEXT_VALID=$(echo "$PRECONTEXT_VERIFY" | cut -d' ' -f2)

            if [[ "$PRECONTEXT_TOTAL" -gt 0 && "$PRECONTEXT_VALID" -gt 0 ]]; then
                echo -e "${GREEN}   ✔ VERIFY OK: precontext output has ${PRECONTEXT_TOTAL} items, ${PRECONTEXT_VALID} with valid final_answer${NC}"
            else
                echo -e "${RED}   ⚠ VERIFY FAIL: precontext output has ${PRECONTEXT_TOTAL} items but ${PRECONTEXT_VALID} valid (expected > 0)${NC}"
                FAIL=$((FAIL + 1))
                FAILED_CASES+=("precontext [${PRECONTEXT_VALID}/${PRECONTEXT_TOTAL} valid final_answer]")
            fi
        else
            echo -e "${RED}   ⚠ VERIFY FAIL: precontext output file not found${NC}"
            FAIL=$((FAIL + 1))
            FAILED_CASES+=("precontext [output file missing]")
        fi
    else
        echo -e "${RED}❌ FAILED: precontext — could not prepare input from ${PRECONTEXT_SOURCE}${NC}"
        FAIL=$((FAIL + 1))
        FAILED_CASES+=("precontext [input preparation failed]")
    fi
else
    skip_test "precontext / real multiturn data" \
        "No multiturn results found in ${EVAL_RESPONSES_DIR}"
fi

# ===================================================================
# Summary
# ===================================================================
echo ""
echo "============================================"
echo -e " ${CYAN}TEST SUMMARY${NC}"
echo "============================================"
TOTAL=$((PASS + FAIL))
echo -e " ${GREEN}PASSED:  ${PASS}${NC}"
echo -e " ${RED}FAILED:  ${FAIL}${NC}"
echo -e " ${YELLOW}SKIPPED: ${SKIP}${NC}"
echo " TOTAL:   ${TOTAL}"
echo "============================================"

if [[ ${FAIL} -gt 0 ]]; then
    echo ""
    echo -e "${RED}Failed test cases:${NC}"
    for tc in "${FAILED_CASES[@]}"; do
        echo -e "  ${RED}• ${tc}${NC}"
    done
    echo ""
    exit 1
else
    echo ""
    echo -e "${GREEN}All tests passed! 🎉${NC}"
    echo ""
    exit 0
fi
