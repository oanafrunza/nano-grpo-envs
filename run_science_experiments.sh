#!/usr/bin/env bash
set -euo pipefail

# Science-grade masking/zeroing experiment suite for nano-grpo-envs
# - Graded formatting by default (format_full_threshold=0.9)
# - Explicit variants toggle binary format or format masking when needed
# - Reuses composite reasoning_envs like run_specs_experiments.sh

OUTDIR_BASE="exp_output/science_suite"
mkdir -p "$OUTDIR_BASE"
MASTER_LOG="${OUTDIR_BASE}/batch_science_launch.log"
echo "==== Science suite launch started $(date) ====" >> "$MASTER_LOG"

# vLLM server defaults; override via env if needed
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_PORT=${VLLM_PORT:-8000}

# GPU and comms env (mirrors run_specs_experiments.sh). Override GPUS to select devices.
GPUS=${GPUS:-"1,2,3"}
BASE_ENV="NCCL_P2P_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 CUDA_VISIBLE_DEVICES=${GPUS} PYTORCH_ALLOC_CONF=max_split_size_mb:64"

# Small, fast defaults with env overrides for smoke tests
NUM_TRAIN_ITERS=${NUM_TRAIN_ITERS:-200}
EVAL_EVERY=${EVAL_EVERY:-50}
SAVE_EVERY=${SAVE_EVERY:-200}
TRAIN_SIZE=${TRAIN_SIZE:-1000}
EVAL_SIZE=${EVAL_SIZE:-100}
NUM_CHAINS=${NUM_CHAINS:-4}
WANDB_PROJECT=${WANDB_PROJECT:-nano-grpo-science}

# Default to graded formatting (omit binary), keep threshold for full-correct gating
COMMON_ARGS="--format_full_threshold 0.9"

# Baseline composite SPECS (same as run_specs_experiments.sh)
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
  echo "[ERROR] No TRAIN_NAMES parsed from SPECS. Check SPECS JSON." | tee -a "$MASTER_LOG"
  exit 1
fi
if [ ${#TRAIN_WEIGHTS[@]} -ne ${#TRAIN_NAMES[@]} ]; then
  echo "[ERROR] TRAIN_WEIGHTS count (${#TRAIN_WEIGHTS[@]}) != TRAIN_NAMES count (${#TRAIN_NAMES[@]})." | tee -a "$MASTER_LOG"
  exit 1
fi

# vLLM health check (supports OpenAI-style and custom endpoints)
if command -v curl >/dev/null 2>&1; then
  if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null || \
     curl -s "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null || \
     curl -s "http://${VLLM_HOST}:${VLLM_PORT}/" >/dev/null; then
    echo "[INFO] vLLM server reachable at ${VLLM_HOST}:${VLLM_PORT}" | tee -a "$MASTER_LOG"
  else
    echo "[ERROR] vLLM server at ${VLLM_HOST}:${VLLM_PORT} not reachable. Start server, then re-run." | tee -a "$MASTER_LOG"
    exit 1
  fi
else
  echo "[WARN] curl not found; skipping vLLM health check." | tee -a "$MASTER_LOG"
fi

# Define experiments: name | ARGS (use cat heredoc to avoid set -e issues)
#baseline_graded          | --reward_mask_strategy none --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy none
EXPERIMENTS=$(cat <<'EXPS'
baseline_binary          | --reward_mask_strategy none --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy none --format_binary_threshold 0.9
softmask_every10_wt05    | --reward_mask_strategy every_n --reward_mask_every_n 10 --reward_mask_weight 0.5 --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy none
fullzero_every10         | --reward_mask_strategy none --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy every_n --full_correct_zero_every_n 10
roundrobin_zero_k4       | --reward_mask_strategy none --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy round_robin_k --full_correct_zero_round_robin_k 4
hybrid_soft_rrk4         | --reward_mask_strategy every_n --reward_mask_every_n 10 --reward_mask_weight 0.5 --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy round_robin_k --full_correct_zero_round_robin_k 4
softmask_every10_wt05_maskfmt | --reward_mask_strategy every_n --reward_mask_every_n 10 --reward_mask_weight 0.5 --mask_format --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy none
baseline_nomask          | --reward_mask_strategy none --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy none
baseline_graded_noliger          | --reward_mask_strategy none --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy none
EXPS
)

# Iterate and launch
while IFS='|' read -r NAME ARGS; do
  NAME=$(echo "$NAME" | xargs)
  [ -z "$NAME" ] && continue
  ARGS=$(echo "$ARGS" | xargs)

  OUTDIR="${OUTDIR_BASE}/${NAME}"
  COMPLETED_MARKER="${OUTDIR}.completed"

  if [ -f "$COMPLETED_MARKER" ]; then
    echo "[SKIP] $NAME already completed (outdir: $OUTDIR)" | tee -a "$MASTER_LOG"
    continue
  fi

  mkdir -p "$OUTDIR"
  echo "[LAUNCH] $NAME -> $OUTDIR" | tee -a "$MASTER_LOG"

  # Effective common args: drop default full threshold if overridden; handle optional binary
  EFFECTIVE_COMMON="$COMMON_ARGS"
  if [[ "$ARGS" == *"--format_full_threshold "* ]]; then
    EFFECTIVE_COMMON="${EFFECTIVE_COMMON//--format_full_threshold 0.9/}"
  fi

  # Sanitize: avoid stray bare --format_binary_threshold
  SANITIZED_ARGS=()
  prev=""
  for t in $ARGS; do
    if [ "$prev" = "--format_binary_threshold" ] && [[ "$t" =~ ^-- ]]; then
      # previous flag had no value; drop it and treat this token as new flag
      prev=""
    fi
    SANITIZED_ARGS+=("$t")
    prev="$t"
  done
  ARGS="${SANITIZED_ARGS[*]}"
  # Defensive: strip any stray double quotes that could break the eval
  ARGS=${ARGS//\"/}

  # Per-experiment environment overrides
  PER_EXP_ENV=""
  if [ "$NAME" = "baseline_graded_noliger" ]; then
    PER_EXP_ENV="TORCHDYNAMO_DISABLE=1"
  fi

  CMD="${BASE_ENV} ${PER_EXP_ENV} uv run main.py --use_vllm --use_liger \
    --vllm_host ${VLLM_HOST} --vllm_port ${VLLM_PORT} \
    --output_dir ${OUTDIR} \
    ${EFFECTIVE_COMMON} ${ARGS} \
    --num_train_iters ${NUM_TRAIN_ITERS} --eval_every ${EVAL_EVERY} --save_every ${SAVE_EVERY} \
    --num_chains ${NUM_CHAINS} \
    --train-names ${TRAIN_NAMES[@]} \
    --train-weights ${TRAIN_WEIGHTS[@]} \
    --train-size ${TRAIN_SIZE} \
    --eval-names ${EVAL_NAMES[@]} \
    --eval-weights ${EVAL_WEIGHTS[@]} \
    --eval-size ${EVAL_SIZE} \
    --use_wandb --wandb_project ${WANDB_PROJECT} --wandb_run ${NAME}"

  echo "[DEBUG] train-names: ${TRAIN_NAMES[*]}" | tee -a "$MASTER_LOG"
  echo "[DEBUG] train-weights: ${TRAIN_WEIGHTS[*]}" | tee -a "$MASTER_LOG"
  echo "[DEBUG] eval-names: ${EVAL_NAMES[*]}" | tee -a "$MASTER_LOG"
  echo "[DEBUG] eval-weights: ${EVAL_WEIGHTS[*]}" | tee -a "$MASTER_LOG"
  echo "Command: $CMD" >> "$MASTER_LOG"
  echo "=== START $NAME $(date) ===" | tee -a "${OUTDIR}_run.log" "$MASTER_LOG"

  set +e
  eval "$CMD" 2>&1 | tee -a "${OUTDIR}_run.log"
  EXIT_CODE=${PIPESTATUS[0]}
  set -e

  if [ $EXIT_CODE -eq 0 ]; then
    touch "$COMPLETED_MARKER"
  else
    echo "[ERROR] $NAME exited with code $EXIT_CODE" | tee -a "${OUTDIR}_run.log" "$MASTER_LOG"
  fi
  echo "=== END $NAME $(date) ===" | tee -a "${OUTDIR}_run.log" "$MASTER_LOG"
  sleep 3
done <<< "$EXPERIMENTS"

echo "==== Science suite launch finished $(date) ====" | tee -a "$MASTER_LOG"
echo "All done. Inspect per-experiment logs: *_run.log and directories in $OUTDIR_BASE" | tee -a "$MASTER_LOG"
