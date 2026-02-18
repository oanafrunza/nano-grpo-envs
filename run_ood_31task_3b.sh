#!/usr/bin/env bash
set -euo pipefail

# OOD 31-Task Test for 3B: Run best 3B configurations on expanded task suite
# to test if masking/zeroing helps with OOD generalization
# This uses the same 31 tasks as the 7B validation experiments

if [ -z "${BASH_VERSION:-}" ]; then
  echo "[WARN] This launcher requires bash. Re-executing with bash..."
  exec bash "$0" "$@"
fi

OUTDIR_BASE="exp_output/ood_31task_3b"
mkdir -p "$OUTDIR_BASE"
MASTER_LOG="${OUTDIR_BASE}/batch_ood_31task_3b_launch.log"
echo "==== OOD 31-task 3B test launch started $(date) ====" | tee -a "$MASTER_LOG"

# vLLM server defaults
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_PORT=${VLLM_PORT:-8000}

# GPU detection
autodetect_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local idxs
    idxs=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    if [ -n "$idxs" ]; then echo "$idxs"; return 0; fi
  fi
  if command -v python >/dev/null 2>&1; then
    local idxs
    idxs=$(python - <<'PY'
try:
  import torch
  n = torch.cuda.device_count()
  print(",".join(str(i) for i in range(n)) if n > 0 else "")
except Exception:
  print("")
PY
    )
    if [ -n "$idxs" ]; then echo "$idxs"; return 0; fi
  fi
  echo "0"
}

GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-"$(autodetect_gpus)"}}
BASE_ENV="NCCL_P2P_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 CUDA_VISIBLE_DEVICES=${GPUS} PYTORCH_ALLOC_CONF=max_split_size_mb:64 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 TORCHDYNAMO_DISABLE=1"
echo "[INFO] Using GPUs: ${GPUS}" | tee -a "$MASTER_LOG"

# Find Python
if [ -f ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "[ERROR] No python found!" | tee -a "$MASTER_LOG"
  exit 1
fi
echo "[INFO] Using Python: $PYTHON" | tee -a "$MASTER_LOG"

# Check vLLM server health
check_vllm() {
  if command -v curl >/dev/null 2>&1; then
    if curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
      echo "[INFO] vLLM server at ${VLLM_HOST}:${VLLM_PORT} is healthy" | tee -a "$MASTER_LOG"
      return 0
    else
      echo "[WARN] vLLM server at ${VLLM_HOST}:${VLLM_PORT} not responding" | tee -a "$MASTER_LOG"
      return 1
    fi
  fi
  return 0
}
check_vllm

# Training parameters
MODEL="Qwen/Qwen2.5-3B-Instruct"
NUM_TRAIN_ITERS=1000
EVAL_EVERY=100
TRAIN_SIZE=5000
EVAL_SIZE=500
NUM_CHAINS=4
MAX_COMPLETION_LENGTH=512
COMMON_ARGS="--format_full_threshold 0.9"

# Expanded 31-task suite matching the 7B validation experiments
# These are the same 31 tasks used in validation/source_reasoning_gym_30.jsonl
SPECS_31='[
  {"name":"simple_equations","weight":1},
  {"name":"polynomial_multiplication","weight":1},
  {"name":"complex_arithmetic","weight":1},
  {"name":"simple_integration","weight":1},
  {"name":"binary_matrix","weight":1},
  {"name":"graph_color","weight":1},
  {"name":"group_anagrams","weight":1},
  {"name":"advanced_geometry","weight":1},
  {"name":"circuit_logic","weight":1},
  {"name":"codeio","weight":1},
  {"name":"course_schedule","weight":1},
  {"name":"decimal_arithmetic","weight":1},
  {"name":"figlet_font","weight":1},
  {"name":"fraction_simplification","weight":1},
  {"name":"gcd","weight":1},
  {"name":"knights_knaves","weight":1},
  {"name":"largest_island","weight":1},
  {"name":"mini_sudoku","weight":1},
  {"name":"modulo_grid","weight":1},
  {"name":"n_queens","weight":1},
  {"name":"palindrome_partitioning","weight":1},
  {"name":"prime_factorization","weight":1},
  {"name":"rectangle_count","weight":1},
  {"name":"rotate_matrix","weight":1},
  {"name":"rush_hour","weight":1},
  {"name":"sentence_reordering","weight":1},
  {"name":"shortest_path","weight":1},
  {"name":"spiral_matrix","weight":1},
  {"name":"string_synthesis","weight":1},
  {"name":"time_intervals","weight":1},
  {"name":"word_ladder","weight":1}
]'
export SPECS_JSON="$SPECS_31"

# Parse task names and weights
readarray -t TRAIN_NAMES < <(python - <<'PY'
import os, json
specs=json.loads(os.environ.get('SPECS_JSON','[]'))
for s in specs:
  print(s['name'])
PY
)
readarray -t TRAIN_WEIGHTS < <(python - <<'PY'
import os, json
specs=json.loads(os.environ.get('SPECS_JSON','[]'))
for s in specs:
  print(s['weight'])
PY
)
EVAL_NAMES=("${TRAIN_NAMES[@]}")
EVAL_COUNT=${#EVAL_NAMES[@]}
if [ "$EVAL_COUNT" -gt 0 ]; then
  EW_UNIFORM=$(awk -v n="$EVAL_COUNT" 'BEGIN {printf "%.6f", 1.0/n}')
else
  EW_UNIFORM=1.0
fi
EVAL_WEIGHTS=()
for _ in "${EVAL_NAMES[@]}"; do EVAL_WEIGHTS+=("$EW_UNIFORM"); done

if [ ${#TRAIN_NAMES[@]} -eq 0 ]; then
  echo "[ERROR] No TRAIN_NAMES parsed from SPECS. Check SPECS JSON." | tee -a "$MASTER_LOG"
  exit 1
fi

echo "[INFO] Running on ${#TRAIN_NAMES[@]} tasks" | tee -a "$MASTER_LOG"

# Build CLI args for tasks
TRAIN_NAMES_STR=$(IFS=,; echo "${TRAIN_NAMES[*]}")
TRAIN_WEIGHTS_STR=$(IFS=,; echo "${TRAIN_WEIGHTS[*]}")
EVAL_NAMES_STR=$(IFS=,; echo "${EVAL_NAMES[*]}")
EVAL_WEIGHTS_STR=$(IFS=,; echo "${EVAL_WEIGHTS[*]}")

# Experiment definitions
# Best 3B configurations from science2_3b_suite:
# 1. 3B Baseline seed0 (27.12% on 10 tasks, no masking/zeroing)
# 2. 3B Continuous seed0 (25.28% on 10 tasks, masking every 10, weight 0.7, warmup 400)
# 3. 3B Phase-Adapt seed1 (22.20% on 10 tasks, masking + late zeroing + phase adaptation)

declare -A EXPERIMENTS

EXPERIMENTS["baseline_3b_seed0"]="main.py \
  --model_name ${MODEL} \
  --use_vllm --vllm_host ${VLLM_HOST} --vllm_port ${VLLM_PORT} \
  --train-names ${TRAIN_NAMES_STR} \
  --train-weights ${TRAIN_WEIGHTS_STR} \
  --train-size ${TRAIN_SIZE} \
  --eval-names ${EVAL_NAMES_STR} \
  --eval-weights ${EVAL_WEIGHTS_STR} \
  --eval-size ${EVAL_SIZE} \
  --num_train_iters ${NUM_TRAIN_ITERS} \
  --eval_every ${EVAL_EVERY} \
  --num_chains ${NUM_CHAINS} \
  --max_completion_length ${MAX_COMPLETION_LENGTH} \
  --seed 0 \
  --project_name nano-grpo-ood-31task-3b \
  --run_name baseline_3b_seed0_31task \
  --output_dir ${OUTDIR_BASE}/baseline_3b_seed0_31task \
  ${COMMON_ARGS}"

EXPERIMENTS["continuous_3b_seed0"]="main.py \
  --model_name ${MODEL} \
  --use_vllm --vllm_host ${VLLM_HOST} --vllm_port ${VLLM_PORT} \
  --train-names ${TRAIN_NAMES_STR} \
  --train-weights ${TRAIN_WEIGHTS_STR} \
  --train-size ${TRAIN_SIZE} \
  --eval-names ${EVAL_NAMES_STR} \
  --eval-weights ${EVAL_WEIGHTS_STR} \
  --eval-size ${EVAL_SIZE} \
  --num_train_iters ${NUM_TRAIN_ITERS} \
  --eval_every ${EVAL_EVERY} \
  --num_chains ${NUM_CHAINS} \
  --max_completion_length ${MAX_COMPLETION_LENGTH} \
  --reward_mask_strategy every_n \
  --reward_mask_every_n 10 \
  --reward_mask_weight 0.7 \
  --mask_format \
  --mask_warmup_steps 400 \
  --seed 0 \
  --project_name nano-grpo-ood-31task-3b \
  --run_name continuous_3b_seed0_31task \
  --output_dir ${OUTDIR_BASE}/continuous_3b_seed0_31task \
  ${COMMON_ARGS}"

EXPERIMENTS["phase_adapt_3b_seed1"]="main_phase_adapt.py \
  --model_name ${MODEL} \
  --use_vllm --vllm_host ${VLLM_HOST} --vllm_port ${VLLM_PORT} \
  --train-names ${TRAIN_NAMES_STR} \
  --train-weights ${TRAIN_WEIGHTS_STR} \
  --train-size ${TRAIN_SIZE} \
  --eval-names ${EVAL_NAMES_STR} \
  --eval-weights ${EVAL_WEIGHTS_STR} \
  --eval-size ${EVAL_SIZE} \
  --num_train_iters ${NUM_TRAIN_ITERS} \
  --eval_every ${EVAL_EVERY} \
  --num_chains ${NUM_CHAINS} \
  --max_completion_length ${MAX_COMPLETION_LENGTH} \
  --reward_mask_strategy every_n \
  --reward_mask_every_n 10 \
  --reward_mask_weight 0.7 \
  --mask_format \
  --mask_warmup_steps 400 \
  --full_correct_zero_strategy late_schedule \
  --zero_warmup_steps 600 \
  --enable_phase_adaptive \
  --gradient_checkpointing \
  --seed 1 \
  --project_name nano-grpo-ood-31task-3b \
  --run_name phase_adapt_3b_seed1_31task \
  --output_dir ${OUTDIR_BASE}/phase_adapt_3b_seed1_31task \
  ${COMMON_ARGS}"

# Run experiments sequentially
for exp_name in baseline_3b_seed0 continuous_3b_seed0 phase_adapt_3b_seed1; do
  COMPLETION_MARKER="${OUTDIR_BASE}/${exp_name}_31task.completed"
  RUN_LOG="${OUTDIR_BASE}/${exp_name}_31task_run.log"
  
  if [ -f "$COMPLETION_MARKER" ]; then
    echo "[INFO] Experiment ${exp_name} already completed, skipping" | tee -a "$MASTER_LOG"
    continue
  fi
  
  echo "[INFO] Starting experiment: ${exp_name}" | tee -a "$MASTER_LOG"
  echo "[INFO] Started at: $(date)" | tee -a "$MASTER_LOG"
  START_TIME=$(date +%s)
  
  CMD="${EXPERIMENTS[$exp_name]}"
  echo "[CMD] env ${BASE_ENV} $PYTHON $CMD" | tee -a "$MASTER_LOG"
  
  if env ${BASE_ENV} $PYTHON $CMD > "$RUN_LOG" 2>&1; then
    touch "$COMPLETION_MARKER"
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINS=$(( (ELAPSED % 3600) / 60 ))
    echo "[INFO] Experiment ${exp_name} completed successfully" | tee -a "$MASTER_LOG"
    echo "[INFO] Completed at: $(date) (took ${HOURS}h ${MINS}m)" | tee -a "$MASTER_LOG"
  else
    EXIT_CODE=$?
    echo "[ERROR] Experiment ${exp_name} failed with exit code ${EXIT_CODE}" | tee -a "$MASTER_LOG"
    echo "[ERROR] Check log: $RUN_LOG" | tee -a "$MASTER_LOG"
  fi
done

echo "==== OOD 31-task 3B test finished $(date) ====" | tee -a "$MASTER_LOG"
echo "[INFO] All experiments complete. Check ${OUTDIR_BASE} for results." | tee -a "$MASTER_LOG"
