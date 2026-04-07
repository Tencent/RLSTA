# RLSTA — Reinforcement Learning with Single-Turn Anchors

A generalizable training framework designed to break *Contextual Inertia* in multi-turn LLM interactions. RLSTA leverages the model's superior single-turn capabilities as internal reward signals, applicable to diverse interaction scenarios including both incremental information addition (**MT-Add**) and error correction (**MT-Refine**).

The project covers the full end-to-end pipeline: **Data Generation → Evaluation → Latent Capability Filtering → Training**.

---

## Project Structure

```
RLSTA/
├── sharded_data/              # Seed data
├── conversation_simulator/    # Step 1: Single-turn / multi-turn conversation generator
├── eval/                      # Step 2: Response evaluation (accuracy scoring)
├── lcf/                       # Step 3: Latent Capability Filtering — data filtering
├── train_datas/               # Step 3 output: filtered training data (gitignored)
├── train/                     # Step 4: GRPO / RLSTA reinforcement learning training
├── utils/                     # Shared utilities (API calls, eval functions, prompt templates)
├── acc_rawdata/               # Evaluation intermediates (raw accuracy data, gitignored)
└── eval_responses/            # Model inference response cache (gitignored)
```

---

## Pipeline Overview

```mermaid
flowchart LR
    A[sharded_data<br/>Seed Problems] --> B[conversation_simulator<br/>Generate Conversations]
    B --> C[eval<br/>Evaluate Accuracy]
    C --> D[lcf<br/>Latent Capability Filtering]
    D --> E[train_datas<br/>Training Data]
    E --> F[train<br/>GRPO / RLSTA Training]
```

| Stage | Module | Description |
|-------|--------|-------------|
| **1. Generation** | `conversation_simulator/` | Supports singleturn, mt_add, mt_refine modes; calls LLM API to generate conversations |
| **2. Evaluation** | `eval/` | Scores generated responses for accuracy on math tasks |
| **3. Filtering** | `lcf/` | **Latent Capability Filtering**: isolates instances where the model possesses the latent capability to solve a problem in single-turn (given merged full information), yet fails under the original multi-turn history — retaining cases where E[Ver(m\|i_full)] > E[Ver(m_n\|H)], thereby using the model's single-turn ability as a stable anchor |
| **4. Training** | `train/` | Supports standard GRPO and RLSTA (GRPO + reference-model likelihood reward) training modes |

---

## Quick Start

```bash
# 1. Generate conversation data
bash conversation_simulator/run.sh mt_add \
    --trails 1,2,3,4 \
    --base-url "$BASE_URL" --openai-api-key "$API_KEY" \
    --model-name hunyuan-2.0-instruct-20251111 \
    --datatype math_train

# 2. Evaluate
bash lcf/run.sh singleturn_eval --input-dir eval_responses/... --output-tag ...
bash lcf/run.sh multiturn_eval --input-dir eval_responses/... --output-tag ...

# 3. Filter
bash lcf/run.sh filter \
    --singleturn-eval st_acc_math_train.json \
    --multiturn-eval mt_acc_math_train.json \
    --output-tag hunyuan-2.0-instruct-20251111 \
    --datatype math_train \
    --filter-item-numbers 2,4 \
    --output-prefix lcf_math \
    --sample-per-task 2

# 4. Train
bash train/run_grpo.sh lcf_math_200 hunyuan-2.0-instruct-20251111 48 0,1,2,3,4,5,6,7
```

---

## Module Documentation

- [`conversation_simulator/`](conversation_simulator/) — Conversation generator
- [`eval/`](eval/README.md) — Evaluation module
- [`lcf/`](lcf/README.md) — Latent Capability Filtering module
- [`train/`](train/README.md) — Training module

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{chen2026breaking,
  title     = {Breaking Contextual Inertia: Reinforcement Learning with Single-Turn Anchors for Stable Multi-Turn Interaction},
  author    = {Chen, Xingwu and Zhang, Zhanqiu and Guo, Yiwen and Zou, Difan},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026}
}
```
