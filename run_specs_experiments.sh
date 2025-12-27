#!/usr/bin/env bash
set -euo pipefail
set -x

# Batch launcher for a single composite spec list using nano GRPO main.py
# Uses reasoning_envs with train/eval names+weights. Total training examples: 2000, eval: 50.
# Edit GPUS and model flags as needed.

GPUS="1,2,3,4"                                  # Set desired GPUs
BASE_ENV="NCCL_P2P_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 CUDA_VISIBLE_DEVICES=${GPUS} PYTORCH_ALLOC_CONF=max_split_size_mb:64"
BASE_MODEL_FLAGS="--use_vllm --use_liger"   # Mirror batch script defaults
# vLLM server settings (adjust if you run on a different port)
VLLM_HOST="127.0.0.1"
VLLM_PORT=8000

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Specs JSON provided by user
SPECS='[{"name":"polynomial_equations","weight":1},{"name":"palindrome_generation","weight":1},{"name":"leg_counting","weight":1},{"name":"family_relationships","weight":1},{"name":"bf","weight":1},{"name":"sokoban","weight":1},{"name":"simple_geometry","weight":1},{"name":"maze","weight":1},{"name":"number_sequence","weight":1},{"name":"propositional_logic","weight":1}]'

# Derive names and weights arrays from SPECS using python (read from env)
export SPECS_JSON="$SPECS"
readarray -t TRAIN_NAMES < <(python - <<'PY'
import os,json
specs=json.loads(os.environ.get('SPECS_JSON','[]'))
for s in specs:
  print(s['name'])
PY
)
readarray -t TRAIN_WEIGHTS < <(python - <<'PY'
import os,json
specs=json.loads(os.environ.get('SPECS_JSON','[]'))
for s in specs:
  print(float(s.get('weight',1)))
PY
)

# Use same names for eval; uniform weights
EVAL_NAMES=("${TRAIN_NAMES[@]}")
EVAL_WEIGHTS=()
for _ in "${EVAL_NAMES[@]}"; do EVAL_WEIGHTS+=("$(python - <<'PY'
print(1/1)
PY
)"); done

# Normalize eval weights to uniform 1/len (bash awk for portability)
EVAL_COUNT=${#EVAL_NAMES[@]}
if [ "$EVAL_COUNT" -gt 0 ]; then
  EW_UNIFORM=$(awk -v n="$EVAL_COUNT" 'BEGIN {printf "%.6f", 1.0/n}')
else
  EW_UNIFORM=1.0
fi
EVAL_WEIGHTS=()
for _ in "${EVAL_NAMES[@]}"; do EVAL_WEIGHTS+=("$EW_UNIFORM"); done

# Safety checks
if [ ${#TRAIN_NAMES[@]} -eq 0 ]; then
  echo "[ERROR] No TRAIN_NAMES parsed from SPECS. Check SPECS JSON." >&2
  exit 2
fi
if [ ${#TRAIN_WEIGHTS[@]} -ne ${#TRAIN_NAMES[@]} ]; then
  echo "[ERROR] TRAIN_WEIGHTS count (${#TRAIN_WEIGHTS[@]}) != TRAIN_NAMES count (${#TRAIN_NAMES[@]})." >&2
  exit 2
fi

# Common formatting thresholds and other shared args (mirror batch script)
# Default to graded formatting: only set full threshold, omit binary threshold
COMMON_ARGS="--format_full_threshold 0.9"

# Experiment definitions: name | args (copied from run_batch_experiments.sh)
EXPERIMENTS=$(cat <<'EOF'
baseline_default          | --reward_mask_strategy none --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy none
baseline_fullzero_every10 | --reward_mask_strategy none --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy every_n --full_correct_zero_every_n 10
fullzero_every10_fw13     | --reward_mask_strategy none --correctness_weight 1.0 --format_weight 1.3 --full_correct_zero_strategy every_n --full_correct_zero_every_n 10
softmask_every10_wt05     | --reward_mask_strategy every_n --reward_mask_every_n 10 --reward_mask_weight 0.5 --correctness_weight 1.0 --format_weight 1.2 --full_correct_zero_strategy none
roundrobin_zero_k4        | --reward_mask_strategy none --correctness_weight 1.0 --format_weight 1.2 --full_correct_zero_strategy round_robin_k --full_correct_zero_round_robin_k 4
cosine_zero_period200     | --reward_mask_strategy none --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy cosine --full_correct_zero_period 200 --full_correct_zero_max_frac 0.3
prob_zero_p01             | --reward_mask_strategy none --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy prob_p --full_correct_zero_prob 0.1
hybrid_soft_roundrobin    | --reward_mask_strategy every_n --reward_mask_every_n 10 --reward_mask_weight 0.5 --correctness_weight 1.0 --format_weight 1.2 --full_correct_zero_strategy round_robin_k --full_correct_zero_round_robin_k 4
graded_format_softmask    | --reward_mask_strategy every_n --reward_mask_every_n 10 --reward_mask_weight 0.5 --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy none --format_full_threshold 0.9
EOF
)

OUTDIR_BASE="exp_specs_runs"
mkdir -p "$OUTDIR_BASE"
MASTER_LOG="${OUTDIR_BASE}/batch_specs_launch.log"
echo "==== Batch specs launch started $(date) ====" >> "$MASTER_LOG"

# Preflight: verify single vLLM server is healthy (support both vLLM OpenAI and custom servers)
if command -v curl >/dev/null 2>&1; then
  HEALTH_OK=0
  # Try /health first (vLLM OpenAI server)
  if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/health/" | grep -q "OK"; then
    HEALTH_OK=1
  else
    # Fallback: try OpenAI models list
    if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" | grep -q "id"; then
      HEALTH_OK=1
    else
      # Fallback: try root or /generate ping for custom servers
      if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/" >/dev/null 2>&1 || curl -s "http://${VLLM_HOST}:${VLLM_PORT}/generate" >/dev/null 2>&1; then
        HEALTH_OK=1
      fi
    fi
  fi
  if [ "$HEALTH_OK" -eq 1 ]; then
    echo "[INFO] vLLM server reachable at ${VLLM_HOST}:${VLLM_PORT}" | tee -a "$MASTER_LOG"
  else
    echo "[ERROR] vLLM server at ${VLLM_HOST}:${VLLM_PORT} not reachable. Start server, then re-run." | tee -a "$MASTER_LOG"
    exit 3
  fi
else
  echo "[WARN] curl not found; skipping vLLM health check." | tee -a "$MASTER_LOG"
fi

while IFS='|' read -r NAME ARGS; do
  NAME=$(echo "$NAME" | xargs)
  ARGS=$(echo "$ARGS" | xargs)
  [ -z "$NAME" ] && continue
  OUTDIR="${OUTDIR_BASE}/${NAME}"
  if [ -d "$OUTDIR" ] && [ -f "$OUTDIR/.completed" ]; then
    echo "[SKIP] $NAME already completed (outdir: $OUTDIR)" | tee -a "$MASTER_LOG"
    continue
  fi
  echo "[LAUNCH] $NAME -> $OUTDIR" | tee -a "$MASTER_LOG"
  EFFECTIVE_COMMON="$COMMON_ARGS"
  if [[ "$ARGS" == *"--format_binary_threshold ''"* ]] || [[ "$ARGS" == *"--no_format_binary"* ]]; then
    EFFECTIVE_COMMON="${EFFECTIVE_COMMON//--format_binary_threshold 1.0/}"
    ARGS="${ARGS//--format_binary_threshold ''/}"
    ARGS="${ARGS//--no_format_binary/}"
  fi
  # If ARGS provides a custom full threshold, drop the default from COMMON
  if [[ "$ARGS" == *"--format_full_threshold "* ]]; then
    EFFECTIVE_COMMON="${EFFECTIVE_COMMON//--format_full_threshold 1.0/}"
  fi
  # Sanitize ARGS: remove lone --format_binary_threshold with no value
  SANITIZED_ARGS=()
  read -ra TOKS <<< "$ARGS"
  i=0
  while [ $i -lt ${#TOKS[@]} ]; do
    t="${TOKS[$i]}"
    if [ "$t" = "--format_binary_threshold" ]; then
      if [ $((i+1)) -lt ${#TOKS[@]} ] && [[ ! "${TOKS[$((i+1))]}" =~ ^- ]]; then
        SANITIZED_ARGS+=("$t" "${TOKS[$((i+1))]}")
        i=$((i+2))
        continue
      else
        i=$((i+1))
        continue
      fi
    fi
    SANITIZED_ARGS+=("$t")
    i=$((i+1))
  done
  ARGS="${SANITIZED_ARGS[*]}"
  CMD="$BASE_ENV uv run python main.py $BASE_MODEL_FLAGS --output_dir $OUTDIR \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --learning_rate 5e-6 --gradient_accumulation_steps 4 \
    --num_train_iters 1000 --eval_every 50 --save_every 250 \
    --temperature 0.9 --num_chains 4 \
    --max_prompt_length 4096 --max_completion_length 4096 \
    --train-names ${TRAIN_NAMES[@]} \
    --train-weights ${TRAIN_WEIGHTS[@]} \
    --train-size 5000 \
    --eval-names ${EVAL_NAMES[@]} \
    --eval-weights ${EVAL_WEIGHTS[@]} \
    --eval-size 500 \
    --use_wandb --wandb_project nano-grpo --wandb_run ${NAME} \
    $EFFECTIVE_COMMON $ARGS \
    --vllm_host $VLLM_HOST --vllm_port $VLLM_PORT"
  echo "[DEBUG] train-names: ${TRAIN_NAMES[*]}" | tee -a "$MASTER_LOG"
  echo "[DEBUG] train-weights: ${TRAIN_WEIGHTS[*]}" | tee -a "$MASTER_LOG"
  echo "[DEBUG] eval-names: ${EVAL_NAMES[*]}" | tee -a "$MASTER_LOG"
  echo "[DEBUG] eval-weights: ${EVAL_WEIGHTS[*]}" | tee -a "$MASTER_LOG"
  echo "Command: $CMD" >> "$MASTER_LOG"
  echo "=== START $NAME $(date) ===" | tee -a "${OUTDIR}_run.log" "$MASTER_LOG"
  # Keep loop going even if a run fails
  eval "$CMD" 2>&1 | tee -a "${OUTDIR}_run.log" || true
  EXIT_CODE=${PIPESTATUS[0]}
  if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] $NAME exited with code $EXIT_CODE" | tee -a "${OUTDIR}_run.log" "$MASTER_LOG"
  else
    echo "=== END $NAME $(date) ===" | tee -a "${OUTDIR}_run.log" "$MASTER_LOG"
    touch "$OUTDIR/.completed"
  fi
  # Small pause to ensure communicator shutdown before next run
  sleep 2
done <<< "$EXPERIMENTS"

echo "==== Batch specs launch finished $(date) ====" | tee -a "$MASTER_LOG"
echo "All done. Inspect per-experiment logs: *_run.log and directories in $OUTDIR_BASE" | tee -a "$MASTER_LOG"