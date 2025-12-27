#!/usr/bin/env bash
set -euo pipefail
set -o pipefail
set -x

# Batch launcher for reward masking / zeroing experiments.
# Customize GPU IDs and shared env vars here.

GPUS="5,6"               # Change to desired GPU devices
BASE_ENV="NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=${GPUS}"
BASE_MODEL_FLAGS="--use_vllm --use_liger"  # Remove flags if not desired

# Ensure we run from the repo directory containing main.py
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Common arguments (tweak as needed)
# Default to graded formatting: only set full threshold, omit binary threshold
COMMON_ARGS="--format_full_threshold 0.9"

# Experiment definitions: name | args
# You can comment/uncomment lines to select experiments.
EXPERIMENTS=$(cat <<'EOF'
baseline_fullzero_every10 | --reward_mask_strategy none --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy every_n --full_correct_zero_every_n 10
fullzero_every10_fw13     | --reward_mask_strategy none --correctness_weight 1.0 --format_weight 1.3 --full_correct_zero_strategy every_n --full_correct_zero_every_n 10
softmask_every10_wt05     | --reward_mask_strategy every_n --reward_mask_every_n 10 --reward_mask_weight 0.5 --correctness_weight 1.0 --format_weight 1.2 --full_correct_zero_strategy none
roundrobin_zero_k4        | --reward_mask_strategy none --correctness_weight 1.0 --format_weight 1.2 --full_correct_zero_strategy round_robin_k --full_correct_zero_round_robin_k 4
cosine_zero_period200     | --reward_mask_strategy none --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy cosine --full_correct_zero_period 200 --full_correct_zero_max_frac 0.3
prob_zero_p01             | --reward_mask_strategy none --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy prob_p --full_correct_zero_prob 0.1
hybrid_soft_roundrobin    | --reward_mask_strategy every_n --reward_mask_every_n 10 --reward_mask_weight 0.5 --correctness_weight 1.0 --format_weight 1.2 --full_correct_zero_strategy round_robin_k --full_correct_zero_round_robin_k 4
graded_format_softmask    | --reward_mask_strategy every_n --reward_mask_every_n 10 --reward_mask_weight 0.5 --correctness_weight 1.0 --format_weight 1.0 --full_correct_zero_strategy none --format_binary_threshold '' --format_full_threshold 0.9
EOF
)

LOG_DIR="exp_output"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/batch_launch.log"
echo "==== Batch launch started $(date) ====" >> "$MASTER_LOG"

while IFS='|' read -r NAME ARGS; do
  NAME=$(echo "$NAME" | xargs)  # trim
  ARGS=$(echo "$ARGS" | xargs)
  [ -z "$NAME" ] && continue
  OUTDIR="${LOG_DIR}/${NAME}"
  if [ -d "$OUTDIR" ]; then
    echo "[SKIP] $NAME already exists (outdir: $OUTDIR)" | tee -a "$MASTER_LOG"
    continue
  fi
  echo "[LAUNCH] $NAME -> $OUTDIR" | tee -a "$MASTER_LOG"
  # Allow per-experiment override to disable binary formatting cleanly
  EFFECTIVE_COMMON="$COMMON_ARGS"
  if [[ "$ARGS" == *"--format_binary_threshold ''"* ]] || [[ "$ARGS" == *"--no_format_binary"* ]]; then
    EFFECTIVE_COMMON="${EFFECTIVE_COMMON//--format_binary_threshold 1.0/}"
    ARGS="${ARGS//--format_binary_threshold ''/}"
    ARGS="${ARGS//--no_format_binary/}"
  fi
  CMD="$BASE_ENV uv run main.py $BASE_MODEL_FLAGS --output_dir $OUTDIR $EFFECTIVE_COMMON $ARGS"
  echo "Command: $CMD" >> "$MASTER_LOG"
  # Run and stream to terminal and per-experiment log
  echo "=== START $NAME $(date) ===" | tee -a "${OUTDIR}_run.log" "$MASTER_LOG"
  eval "$CMD" 2>&1 | tee -a "${OUTDIR}_run.log"
  EXIT_CODE=${PIPESTATUS[0]}
  if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] $NAME exited with code $EXIT_CODE" | tee -a "${OUTDIR}_run.log" "$MASTER_LOG"
  else
    echo "=== END $NAME $(date) ===" | tee -a "${OUTDIR}_run.log" "$MASTER_LOG"
  fi
done <<< "$EXPERIMENTS"

echo "==== Batch launch finished $(date) ====" | tee -a "$MASTER_LOG"
echo "Collecting completed runs for comparison..." | tee -a "$MASTER_LOG"

# Build list of runs with run_log.json present
RUNS=()
while IFS='|' read -r NAME ARGS; do
  NAME=$(echo "$NAME" | xargs)
  [ -z "$NAME" ] && continue
  OUTDIR="${LOG_DIR}/${NAME}"
  if [ -f "${OUTDIR}/run_log.json" ]; then
    RUNS+=("$OUTDIR")
  fi
done <<< "$EXPERIMENTS"

if [ ${#RUNS[@]} -gt 0 ]; then
  echo "Comparing runs: ${RUNS[*]}" | tee -a "$MASTER_LOG"
  python -u compare_multiple_runs.py --runs "${RUNS[@]}" --out_dir "$LOG_DIR" 2>&1 | tee -a "$MASTER_LOG"
  echo "Wrote summary CSV and (if available) plot in $LOG_DIR" | tee -a "$MASTER_LOG"
else
  echo "No completed runs with run_log.json found yet." | tee -a "$MASTER_LOG"
fi

echo "All done. Inspect per-experiment logs: *_run.log and directories in $LOG_DIR" | tee -a "$MASTER_LOG"