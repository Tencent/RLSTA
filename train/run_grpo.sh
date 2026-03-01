#!/bin/bash
# =============================================================
# GRPO Training Launch Script
# =============================================================
#
# This script launches Group Relative Policy Optimization (GRPO)
# training using DeepSpeed ZeRO-3 via HuggingFace Accelerate.
#
# Usage:
#   bash train/run_grpo.sh <data_name> <model_name> [epochs] [gpu_ids] [train_mode]
#
# Example:
#   bash train/run_grpo.sh lcf_math_200 hunyuan-2.0-instruct-20251111 48 0,1,2,3,4,5,6,7 base
#   bash train/run_grpo.sh lcf_math_200 hunyuan-2.0-instruct-20251111 48 0,1,2,3,4,5,6,7 rlsta
#
# Arguments:
#   $1 - data_name   : Name of the training data file (without .jsonl extension)
#                       Looked up at: train_datas/<model_name>/<data_name>.jsonl
#   $2 - model_name  : Name of the base model directory under /root/models/
#   $3 - epochs      : Number of training epochs (default: 48)
#   $4 - gpu_ids     : Comma-separated GPU device IDs (default: 0,1,2,3,4,5,6,7)
#   $5 - train_mode  : Training mode: "base" or "rlsta" (default: base)
# =============================================================

# Navigate to this script's directory so relative paths work
cd "$(dirname "$0")"

# --------------- Configurable Parameters ---------------
data_name=${1:-multiturn_gsm8k_all_100}
model_name=${2:-Qwen2.5-7B-Instruct}
epochs=${3:-48}
gpuids=${4:-0,1,2,3,4,5,6,7}

train_mode=${5:-base}                     # Training mode: base = GRPOTrainer, rlsta = RLSTATrainer

base_model="/root/models/$model_name"     # Path to the pre-trained model weights
output_dir="/root/pipeline_ckpts/${model_name}-${data_name}-${train_mode}"
data_path="../train_datas/${model_name}/${data_name}.jsonl"
test_data_path="../grpo/rl_train_data/eval_dataset.jsonl"

# --------------- Hyperparameters ---------------
gpu_count=8
max_prompt_length=8192
max_completion_length=1024
micro_batch_size=2
gradient_accumulation_steps=2
lr=3e-7
weight_decay=1e-2

# --------------- Setup ---------------
rm -rf "${output_dir}"
mkdir -p "${output_dir}"
mkdir -p logs/${model_name}

# --------------- DeepSpeed JSON Config (ZeRO-2 + CPU offload) ---------------
# This config is consumed by DeepSpeed at runtime for optimizer/scheduler settings.
# Note: The accelerate yaml (accelerate_ds_zero3.yaml) defines ZeRO stage 3 for
# model parallelism, while this JSON handles optimizer-level ZeRO-2 offload.
cat > ds_config.json << EOL
{
    "train_micro_batch_size_per_gpu": ${micro_batch_size},
    "gradient_accumulation_steps": ${gradient_accumulation_steps},
    "bf16": {
        "enabled": true
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": ${lr},
            "weight_decay": ${weight_decay},
            "betas": [0.9, 0.95],
            "eps": 1e-8
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": ${lr},
            "warmup_num_steps": "auto",
            "warmup_type": "linear",
            "total_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_scatter": true,
        "reduce_bucket_size": "auto",
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "train_batch_size": "auto",
    "wall_clock_breakdown": false,
    "zero_allow_untested_optimizer": true,
    "fp16": {
        "enabled": false
    }
}
EOL

# --------------- Launch Training ---------------
export TOKENIZERS_PARALLELISM=true

CUDA_VISIBLE_DEVICES=$gpuids accelerate launch \
  --config_file accelerate_ds_zero3.yaml \
  train_grpo.py \
  --model_name_or_path $base_model \
  --data_path $data_path \
  --test_data_path $test_data_path \
  --output_dir $output_dir \
  --per_device_train_batch_size $micro_batch_size \
  --gradient_accumulation_steps $gradient_accumulation_steps \
  --learning_rate $lr \
  --weight_decay $weight_decay \
  --num_train_epochs $epochs \
  --gradient_checkpointing \
  --max_prompt_length $max_prompt_length \
  --max_completion_length $max_completion_length \
  --train_mode $train_mode \
  --num_generations 8 \
  --bf16 | tee logs/${model_name}/${data_name}_${train_mode}.log
