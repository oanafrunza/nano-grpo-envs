#!/usr/bin/env bash
set -euo pipefail

# Ensure we're running under bash even if invoked by sh
if [ -z "${BASH_VERSION:-}" ]; then
  echo "[WARN] This launcher requires bash. Re-executing with bash..."
  exec bash "$0" "$@"
fi

# Science2 3B Suite: baseline, continuous-best, phase-adapt-best
OUTDIR_BASE="exp_output/science2_3b_suite"
mkdir -p "$OUTDIR_BASE"
MASTER_LOG="${OUTDIR_BASE}/batch_science2_3b_launch.log"
echo "==== Science2 3B suite launch started $(date) ====" | tee -a "$MASTER_LOG"

# vLLM server defaults; you will start it separately
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_PORT=${VLLM_PORT:-8000}

# Python runner: prefer project venv
VENV="/mnt/home/oana/projects/nano-grpo-envs/.venv"
if [ -x "$VENV/bin/python" ]; then
  PY_BIN="$VENV/bin/python"
else
  PY_BIN="python"
fi

# GPU/config env
autodetect_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local idxs
    idxs=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    if [ -n "$idxs" ]; then echo "$idxs"; return 0; fi
  fi
  if command -v "$PY_BIN" >/dev/null 2>&1; then
    local idxs
    idxs=$($PY_BIN - <<'PY'
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
BASE_ENV="NCCL_P2P_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 CUDA_VISIBLE_DEVICES=${GPUS} PYTORCH_ALLOC_CONF=max_split_size_mb:64 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64"
echo "[INFO] Using training GPUs: ${GPUS}" | tee -a "$MASTER_LOG"

# Trainer knobs
NUM_TRAIN_ITERS=${NUM_TRAIN_ITERS:-1000}
EVAL_EVERY=${EVAL_EVERY:-100}
SAVE_EVERY=${SAVE_EVERY:-500}
TRAIN_SIZE=${TRAIN_SIZE:-5000}
EVAL_SIZE=${EVAL_SIZE:-500}
NUM_CHAINS=${NUM_CHAINS:-4}
SAVE_ONLY_LAST=${SAVE_ONLY_LAST:-1}
DISABLE_CHECKPOINT_SAVING=${DISABLE_CHECKPOINT_SAVING:-0}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}

# Masking/zeroing defaults
LIGHT_EVERY_N=${LIGHT_EVERY_N:-20}
LIGHT_WEIGHT=${LIGHT_WEIGHT:-0.7}
LIGHT_MASK_WARMUP_STEPS=${LIGHT_MASK_WARMUP_STEPS:-400}
ZERO_WARMUP_STEPS_LATE=${ZERO_WARMUP_STEPS_LATE:-600}
FULLZERO_MAX_FRAC=${FULLZERO_MAX_FRAC:-0.15}

# Seeds: two by default
SEEDS=(${SEEDS:-0 1})
COMMON_ARGS="--format_full_threshold 0.9"

# Composite 10-task sweep
SPECS='[{"name":"polynomial_equations","weight":1},{"name":"palindrome_generation","weight":1},{"name":"leg_counting","weight":1},{"name":"family_relationships","weight":1},{"name":"bf","weight":1},{"name":"sokoban","weight":1},{"name":"simple_geometry","weight":1},{"name":"maze","weight":1},{"name":"number_sequence","weight":1},{"name":"propositional_logic","weight":1}]'
export SPECS_JSON="$SPECS"

readarray -t TRAIN_NAMES < <($PY_BIN - <<'PY'
import os, json
specs=json.loads(os.environ.get('SPECS_JSON','[]'))
for s in specs:
  print(s['name'])
PY
)
readarray -t TRAIN_WEIGHTS < <($PY_BIN - <<'PY'
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
if [ ${#TRAIN_WEIGHTS[@]} -ne ${#TRAIN_NAMES[@]} ]; then
  echo "[ERROR] TRAIN_WEIGHTS count (${#TRAIN_WEIGHTS[@]}) != TRAIN_NAMES count (${#TRAIN_NAMES[@]})." | tee -a "$MASTER_LOG"
  exit 1
fi

# Experiments list (name | runner | ARGS)
EXPERIMENTS=$(cat <<EXPS
baseline_len512                | main.py             | --reward_mask_strategy none --full_correct_zero_strategy none --max_completion_length 512
continuous_best_len512        | main.py             | --reward_mask_strategy every_n --reward_mask_every_n ${LIGHT_EVERY_N} --reward_mask_weight ${LIGHT_WEIGHT} --mask_format --mask_warmup_steps ${LIGHT_MASK_WARMUP_STEPS} --full_correct_zero_strategy none --max_completion_length 512
phase_adapt_best_len512       | main_phase_adapt.py | --enable_phase_adaptive --reward_mask_strategy every_n --reward_mask_every_n ${LIGHT_EVERY_N} --reward_mask_weight ${LIGHT_WEIGHT} --mask_format --mask_warmup_steps ${LIGHT_MASK_WARMUP_STEPS} --full_correct_zero_strategy every_n --full_correct_zero_every_n 20 --zero_warmup_steps ${ZERO_WARMUP_STEPS_LATE} --full_correct_zero_max_frac ${FULLZERO_MAX_FRAC} --max_completion_length 512 --post_phase_chain_scale 0.5 --diversity_weight 0.05
EXPS
)

# vLLM reachability check (non-fatal)
if command -v curl >/dev/null 2>&1; then
  if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null; then
    echo "[INFO] vLLM server reachable at ${VLLM_HOST}:${VLLM_PORT}" | tee -a "$MASTER_LOG"
  else
    echo "[WARN] vLLM server not reachable at ${VLLM_HOST}:${VLLM_PORT}. Ensure server is running before trainers." | tee -a "$MASTER_LOG"
  fi
else
  echo "[WARN] curl not found; skipping vLLM health check." | tee -a "$MASTER_LOG"
fi

for SEED in "${SEEDS[@]}"; do
  echo "[INFO] Launching 3B suite experiments for seed=${SEED}" | tee -a "$MASTER_LOG"
  while IFS='|' read -r NAME RUNNER ARGS; do
    NAME=$(echo "$NAME" | xargs)
    [ -z "$NAME" ] && continue
    RUNNER=$(echo "$RUNNER" | xargs)
    ARGS=$(echo "$ARGS" | xargs)

    OUTDIR="${OUTDIR_BASE}/${NAME}_seed${SEED}"
    COMPLETED_MARKER="${OUTDIR}.completed"
    if [ -f "$COMPLETED_MARKER" ]; then
      echo "[SKIP] $NAME (seed=${SEED}) already completed (outdir: $OUTDIR)" | tee -a "$MASTER_LOG"
      continue
    fi
    mkdir -p "$OUTDIR"
    echo "[LAUNCH] $NAME (seed=${SEED}) -> $OUTDIR" | tee -a "$MASTER_LOG"

    EFFECTIVE_COMMON="$COMMON_ARGS"
    if [[ "$ARGS" == *"--format_full_threshold "* ]]; then
      EFFECTIVE_COMMON="${EFFECTIVE_COMMON//--format_full_threshold 0.9/}"
    fi

    PER_EXP_ENV="TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}"

    EXTRA_SAVE_ARGS=""
    if [ "$DISABLE_CHECKPOINT_SAVING" = "1" ]; then
      EXTRA_SAVE_ARGS="--disable_checkpoint_saving"
    elif [ "$SAVE_ONLY_LAST" = "1" ]; then
      EXTRA_SAVE_ARGS="--save_only_last"
    fi

    # Choose W&B project
    case "$RUNNER" in
      main_phase_adapt.py) PROJECT=${WANDB_PROJECT_PHASE:-nano-grpo-science2-phase} ;;
      main.py)             PROJECT=${WANDB_PROJECT_CONT:-nano-grpo-science2-continuous} ;;
      *)                   PROJECT=${WANDB_PROJECT_CONT:-nano-grpo-science2-continuous} ;;
    esac

    CMD="${BASE_ENV} ${PER_EXP_ENV} ${PY_BIN} ${RUNNER} --use_vllm \
      --vllm_host ${VLLM_HOST} --vllm_port ${VLLM_PORT} \
      --output_dir ${OUTDIR} \
      --model_name Qwen/Qwen2.5-3B-Instruct \
      ${EFFECTIVE_COMMON} ${ARGS} ${EXTRA_SAVE_ARGS} \
      --num_train_iters ${NUM_TRAIN_ITERS} --eval_every ${EVAL_EVERY} --save_every ${SAVE_EVERY} \
      --num_chains ${NUM_CHAINS} \
      --seed ${SEED} \
      --train-names ${TRAIN_NAMES[@]} \
      --train-weights ${TRAIN_WEIGHTS[@]} \
      --train-size ${TRAIN_SIZE} \
      --eval-names ${EVAL_NAMES[@]} \
      --eval-weights ${EVAL_WEIGHTS[@]} \
      --eval-size ${EVAL_SIZE} \
      --use_wandb --wandb_project ${PROJECT} --wandb_run ${NAME}_seed${SEED}"

    if [ "$RUNNER" = "main_phase_adapt.py" ]; then
      CMD+=" --problem_overrides_file configs/phase_adapt_problem_overrides.json"
      if [ "$GRADIENT_CHECKPOINTING" = "1" ]; then
        CMD+=" --gradient_checkpointing"
      fi
    fi

    echo "[DEBUG] train-names: ${TRAIN_NAMES[*]}" | tee -a "$MASTER_LOG"
    echo "[DEBUG] train-weights: ${TRAIN_WEIGHTS[*]}" | tee -a "$MASTER_LOG"
    echo "[DEBUG] eval-names: ${EVAL_NAMES[*]}" | tee -a "$MASTER_LOG"
    echo "[DEBUG] eval-weights: ${EVAL_WEIGHTS[*]}" | tee -a "$MASTER_LOG"
    echo "Command: $CMD" >> "$MASTER_LOG"
    echo "=== START ${NAME} seed=${SEED} $(date) ===" | tee -a "${OUTDIR}_run.log" "$MASTER_LOG"

    set +e
    eval "$CMD" 2>&1 | tee -a "${OUTDIR}_run.log"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
      touch "$COMPLETED_MARKER"
    else
      echo "[ERROR] ${NAME} (seed=${SEED}) exited with code $EXIT_CODE" | tee -a "${OUTDIR}_run.log" "$MASTER_LOG"
    fi
    echo "=== END ${NAME} seed=${SEED} $(date) ===" | tee -a "${OUTDIR}_run.log" "$MASTER_LOG"
    sleep 3
  done <<< "$EXPERIMENTS"

done

echo "==== Science2 3B suite launch finished $(date) ====" | tee -a "$MASTER_LOG"
echo "3B suite done. Inspect logs and summaries under $OUTDIR_BASE" | tee -a "$MASTER_LOG"
