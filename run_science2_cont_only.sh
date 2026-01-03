#!/usr/bin/env bash
set -euo pipefail

# Science2 Continuous Only: run just the continuous add-on experiments
CONT_OUTDIR_BASE="exp_output/science2_cont_suite"
mkdir -p "$CONT_OUTDIR_BASE"
CONT_MASTER_LOG="${CONT_OUTDIR_BASE}/batch_science2_cont_launch.log"
echo "==== Science2 continuous-only launch started $(date) ====" | tee -a "$CONT_MASTER_LOG"

# vLLM server defaults; override via env if needed
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_PORT=${VLLM_PORT:-8000}

# GPU and comms env. Override GPUS to select devices.
GPUS=${GPUS:-"1,2,3"}
BASE_ENV="NCCL_P2P_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 CUDA_VISIBLE_DEVICES=${GPUS} PYTORCH_ALLOC_CONF=max_split_size_mb:64"

# Larger defaults (override via env for custom runs)
NUM_TRAIN_ITERS=${NUM_TRAIN_ITERS:-1000}
EVAL_EVERY=${EVAL_EVERY:-100}
SAVE_EVERY=${SAVE_EVERY:-500}
TRAIN_SIZE=${TRAIN_SIZE:-5000}
EVAL_SIZE=${EVAL_SIZE:-500}
NUM_CHAINS=${NUM_CHAINS:-4}
# Project for continuous experiments
WANDB_PROJECT_CONT=${WANDB_PROJECT_CONT:-nano-grpo-science2-continuous}

# Checkpoint saving controls (default: only save final to conserve disk)
SAVE_ONLY_LAST=${SAVE_ONLY_LAST:-1}
DISABLE_CHECKPOINT_SAVING=${DISABLE_CHECKPOINT_SAVING:-0}

# Warmup defaults (use new flags implemented in main.py)
MASK_WARMUP_STEPS=${MASK_WARMUP_STEPS:-200}
ZERO_WARMUP_STEPS=${ZERO_WARMUP_STEPS:-200}

# Light softmask defaults (gentler variant)
LIGHT_EVERY_N=${LIGHT_EVERY_N:-30}
LIGHT_WEIGHT=${LIGHT_WEIGHT:-0.5}
LIGHT_MASK_WARMUP_STEPS=${LIGHT_MASK_WARMUP_STEPS:-400}

# Single-seed for continuous experiments (set SEED_CONT to override)
SEED_CONT=${SEED_CONT:-1}

# Default to graded formatting unless experiment overrides
COMMON_ARGS="--format_full_threshold 0.9"

# Composite SPECS (same set as science suite)
SPECS='[{"name":"polynomial_equations","weight":1},{"name":"palindrome_generation","weight":1},{"name":"leg_counting","weight":1},{"name":"family_relationships","weight":1},{"name":"bf","weight":1},{"name":"sokoban","weight":1},{"name":"simple_geometry","weight":1},{"name":"maze","weight":1},{"name":"number_sequence","weight":1},{"name":"propositional_logic","weight":1}]'
export SPECS_JSON="$SPECS"

# Parse names/weights from SPECS
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

# Eval: reuse names, uniform weights
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
  echo "[ERROR] No TRAIN_NAMES parsed from SPECS. Check SPECS JSON." | tee -a "$CONT_MASTER_LOG"
  exit 1
fi
if [ ${#TRAIN_WEIGHTS[@]} -ne ${#TRAIN_NAMES[@]} ]; then
  echo "[ERROR] TRAIN_WEIGHTS count (${#TRAIN_WEIGHTS[@]}) != TRAIN_NAMES count (${#TRAIN_NAMES[@]})." | tee -a "$CONT_MASTER_LOG"
  exit 1
fi

# vLLM health check
if command -v curl >/dev/null 2>&1; then
  if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null || \
     curl -s "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null || \
     curl -s "http://${VLLM_HOST}:${VLLM_PORT}/" >/dev/null; then
    echo "[INFO] vLLM server reachable at ${VLLM_HOST}:${VLLM_PORT}" | tee -a "$CONT_MASTER_LOG"
  else
    echo "[ERROR] vLLM server at ${VLLM_HOST}:${VLLM_PORT} not reachable. Start server, then re-run." | tee -a "$CONT_MASTER_LOG"
    exit 1
  fi
else
  echo "[WARN] curl not found; skipping vLLM health check." | tee -a "$CONT_MASTER_LOG"
fi

# Continuous experiments (name | ARGS)
CONT_EXPERIMENTS=$(cat <<'EXPS'
cont_softmask_light               | --reward_mask_strategy every_n --reward_mask_every_n ${LIGHT_EVERY_N} --reward_mask_weight ${LIGHT_WEIGHT} --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy none --mask_warmup_steps ${LIGHT_MASK_WARMUP_STEPS}
cont_softmask_prob_p             | --reward_mask_strategy prob_p --reward_mask_prob ${SOFTMASK_PROB:-0.2} --reward_mask_weight ${SOFTMASK_WEIGHT:-0.5} --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy none --mask_warmup_steps ${MASK_WARMUP_STEPS}
cont_softmask_roundrobin         | --reward_mask_strategy round_robin_k --reward_mask_round_robin_k ${ROUNDROBIN_K:-2} --reward_mask_weight ${ROUNDROBIN_WEIGHT:-0.5} --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy none --mask_warmup_steps ${MASK_WARMUP_STEPS}
cont_fullzero_everyN_t09         | --reward_mask_strategy none --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy every_n --full_correct_zero_every_n ${FULLZERO_EVERY_N:-20} --format_full_threshold 0.9 --zero_warmup_steps ${ZERO_WARMUP_STEPS}
cont_fullzero_prob_t09           | --reward_mask_strategy none --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy prob_p --full_correct_zero_prob ${FULLZERO_PROB:-0.2} --format_full_threshold 0.9 --zero_warmup_steps ${ZERO_WARMUP_STEPS}
cont_grpo_baseline               | --loss_type grpo --reward_mask_strategy none --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy none
EXPS
)

for SEED in "$SEED_CONT"; do
  echo "[INFO] Launching continuous experiments for seed=${SEED}" | tee -a "$CONT_MASTER_LOG"
  while IFS='|' read -r NAME ARGS; do
    NAME=$(echo "$NAME" | xargs)
    [ -z "$NAME" ] && continue
    ARGS=$(echo "$ARGS" | xargs)

    OUTDIR="${CONT_OUTDIR_BASE}/${NAME}_seed${SEED}"
    COMPLETED_MARKER="${OUTDIR}.completed"

    if [ -f "$COMPLETED_MARKER" ]; then
      echo "[SKIP] $NAME (seed=${SEED}) already completed (outdir: $OUTDIR)" | tee -a "$CONT_MASTER_LOG"
      continue
    fi

    mkdir -p "$OUTDIR"
    echo "[LAUNCH] $NAME (seed=${SEED}) -> $OUTDIR" | tee -a "$CONT_MASTER_LOG"

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

    PROJECT="$WANDB_PROJECT_CONT"

    CMD="${BASE_ENV} ${PER_EXP_ENV} uv run main.py --use_vllm --use_liger \
      --vllm_host ${VLLM_HOST} --vllm_port ${VLLM_PORT} \
      --output_dir ${OUTDIR} \
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

    echo "[DEBUG] train-names: ${TRAIN_NAMES[*]}" | tee -a "$CONT_MASTER_LOG"
    echo "[DEBUG] train-weights: ${TRAIN_WEIGHTS[*]}" | tee -a "$CONT_MASTER_LOG"
    echo "[DEBUG] eval-names: ${EVAL_NAMES[*]}" | tee -a "$CONT_MASTER_LOG"
    echo "[DEBUG] eval-weights: ${EVAL_WEIGHTS[*]}" | tee -a "$CONT_MASTER_LOG"
    printf 'Command: %s\n' "$CMD" >> "$CONT_MASTER_LOG"
    echo "=== START ${NAME} seed=${SEED} $(date) ===" | tee -a "${OUTDIR}_run.log" "$CONT_MASTER_LOG"

    set +e
    eval "$CMD" 2>&1 | tee -a "${OUTDIR}_run.log"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    if [ $EXIT_CODE -eq 0 ]; then
      touch "$COMPLETED_MARKER"
    else
      echo "[ERROR] ${NAME} (seed=${SEED}) exited with code $EXIT_CODE" | tee -a "${OUTDIR}_run.log" "$CONT_MASTER_LOG"
    fi
    echo "=== END ${NAME} seed=${SEED} $(date) ===" | tee -a "${OUTDIR}_run.log" "$CONT_MASTER_LOG"
    sleep 3
  done <<< "$CONT_EXPERIMENTS"
 done

echo "==== Science2 continuous-only launch finished $(date) ====" | tee -a "$CONT_MASTER_LOG"
echo "Continuous set done. Inspect logs under $CONT_OUTDIR_BASE" | tee -a "$CONT_MASTER_LOG"
