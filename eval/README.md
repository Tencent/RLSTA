# Evaluation

This directory contains the unified evaluation pipeline for all task types.

## Quick Start

```bash
# Evaluate all tasks
bash eval/run.sh all --responses-dir eval_responses/<model-name>

# Evaluate a specific task
bash eval/run.sh math     --responses-dir eval_responses/<model-name>
bash eval/run.sh code     --responses-dir eval_responses/<model-name>
bash eval/run.sh actions  --responses-dir eval_responses/<model-name>
bash eval/run.sh summary  --responses-dir eval_responses/<model-name> --api-base-url <URL> --api-key <KEY>
bash eval/run.sh database --responses-dir eval_responses/<model-name> --api-base-url <URL> --api-key <KEY>
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--responses-dir DIR` | *(required)* | Root directory of model responses |
| `--output-dir DIR` | `<project-dir>/eval_results` | Where to save result JSON files |
| `--folder NAME` | *(all folders)* | Evaluate only one specific subfolder |
| `--math-eval-mode MODE` | `rule` | `rule` or `llm` for math evaluation |
| `--api-base-url URL` | — | LLM API base URL (required for `summary`, `database`, `math --math-eval-mode llm`) |
| `--api-key KEY` | — | LLM API key (same tasks as above) |
| `--api-model MODEL` | `gpt-4o-mini` | LLM model name |

## Tasks

| Task | Eval Method | Needs API |
|------|-------------|-----------|
| `math` | Rule-based numeric match (default) or LLM judge | Only in `llm` mode |
| `code` | Execute Python, compute pass@1 & pass@k | No |
| `actions` | Exact / fuzzy action match | No |
| `summary` | LLM judge across 6 metrics | **Yes** |
| `database` | Stage 1: LLM extracts SQL → Stage 2: execute & compare | **Yes** |

## File Structure

```
eval/
├── run.sh          # Entry script (bash)
├── run_eval.py     # Main orchestration logic
├── evaluators.py   # Per-task eval functions (eval_one_*)
└── db_extract.py   # Database Stage 1: LLM-based SQL extraction
```

## Database Task: Data Preparation

The `database` task uses [test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval)
to execute and compare SQL queries against real SQLite databases.

**Required databases (19 Spider db_ids):**
`battle_death`, `car_1`, `concert_singer`, `course_teach`, `cre_Doc_Template_Mgt`,
`dog_kennels`, `employee_hire_evaluation`, `flight_2`, `museum_visit`, `network_1`,
`orchestra`, `pets_1`, `poker_player`, `singer`, `student_transcripts_tracking`,
`tvshow`, `voter_1`, `world_1`, `wta_1`

### Setup Steps

**Step 1**: Download the Spider database files from the Google Drive link in the
[test-suite-sql-eval README](https://github.com/taoyds/test-suite-sql-eval?tab=readme-ov-file).

After downloading and decompressing, the path
`test-suite-sql-eval/database/atis/atis.sqlite` should be valid.

**Step 2**: Create the directory expected by the evaluator and symlink the databases:

```bash
mkdir -p /root/tmp/database
ln -s "$(pwd)/test-suite-sql-eval/database/"* /root/tmp/database/
```

Or copy instead of symlink:

```bash
mkdir -p /root/tmp/database
cp -r test-suite-sql-eval/database/* /root/tmp/database/
```

**Step 3**: Verify the setup:

```bash
ls /root/tmp/database/concert_singer/
# Expected: concert_singer.sqlite
```

**Step 4**: Run database evaluation:

```bash
bash eval/run.sh database \
    --responses-dir eval_responses/<model-name> \
    --api-base-url <YOUR_BASE_URL> \
    --api-key <YOUR_KEY> \
    --api-model <YOUR_MODEL>
```

> **Note**: The evaluator looks for `.sqlite` files under `/root/tmp/database/{db_id}/`.
> The directory must exist before running, otherwise a `FileNotFoundError` will be raised.
