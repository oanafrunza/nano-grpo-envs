#!/usr/bin/env bash
set -euo pipefail

# Ensure we're running under bash even if invoked by sh
if [ -z "${BASH_VERSION:-}" ]; then
  echo "[WARN] This launcher requires bash. Re-executing with bash..."
  exec bash "$0" "$@"
fi

# Science2 Phase-Adaptive: runs new trainer with phase detection & schedules
OUTDIR_BASE="exp_output/science2_phase_adapt_suite"
mkdir -p "$OUTDIR_BASE"
MASTER_LOG="${OUTDIR_BASE}/batch_science2_phase_adapt_launch.log"
echo "==== Science2 phase-adapt suite launch started $(date) ====" | tee -a "$MASTER_LOG"

VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_PORT=${VLLM_PORT:-8000}
GPUS=${GPUS:-"1,2,3,4"}
BASE_ENV="NCCL_P2P_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 CUDA_VISIBLE_DEVICES=${GPUS} PYTORCH_ALLOC_CONF=max_split_size_mb:64 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64"

NUM_TRAIN_ITERS=${NUM_TRAIN_ITERS:-1000}
EVAL_EVERY=${EVAL_EVERY:-100}
SAVE_EVERY=${SAVE_EVERY:-500}
TRAIN_SIZE=${TRAIN_SIZE:-5000}
EVAL_SIZE=${EVAL_SIZE:-500}
NUM_CHAINS=${NUM_CHAINS:-4}
WANDB_PROJECT=${WANDB_PROJECT:-nano-grpo-science2-phase}
SAVE_ONLY_LAST=${SAVE_ONLY_LAST:-1}
DISABLE_CHECKPOINT_SAVING=${DISABLE_CHECKPOINT_SAVING:-0}
MASK_WARMUP_STEPS=${MASK_WARMUP_STEPS:-200}
ZERO_WARMUP_STEPS=${ZERO_WARMUP_STEPS:-200}
MAX_LEN_BASE=${MAX_LEN_BASE:-1024}
ADAPT_POST_LEN_FACTOR=${ADAPT_POST_LEN_FACTOR:-1.0}
POST_PHASE_CHAIN_SCALE=${POST_PHASE_CHAIN_SCALE:-0.5}
MAX_COMPLETION_LENGTH_CAP=${MAX_COMPLETION_LENGTH_CAP:-${MAX_LEN_BASE}}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}

LIGHT_EVERY_N=${LIGHT_EVERY_N:-20}
LIGHT_WEIGHT=${LIGHT_WEIGHT:-0.7}
LIGHT_MASK_WARMUP_STEPS=${LIGHT_MASK_WARMUP_STEPS:-400}

SEEDS=(${SEEDS:-1 2})
COMMON_ARGS="--format_full_threshold 0.9"

SPECS='[{"name":"polynomial_equations","weight":1},{"name":"palindrome_generation","weight":1},{"name":"leg_counting","weight":1},{"name":"family_relationships","weight":1},{"name":"bf","weight":1},{"name":"sokoban","weight":1},{"name":"simple_geometry","weight":1},{"name":"maze","weight":1},{"name":"number_sequence","weight":1},{"name":"propositional_logic","weight":1}]'
export SPECS_JSON="$SPECS"

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
if [ ${#TRAIN_WEIGHTS[@]} -ne ${#TRAIN_NAMES[@]} ]; then
  echo "[ERROR] TRAIN_WEIGHTS count (${#TRAIN_WEIGHTS[@]}) != TRAIN_NAMES count (${#TRAIN_NAMES[@]})." | tee -a "$MASTER_LOG"
  exit 1
fi

# Experiments (name | ARGS)
EXPERIMENTS=$(cat <<'EXPS'
phase_adapt_softmask          | --enable_phase_adaptive --reward_mask_strategy prob_p --reward_mask_prob ${ADAPT_SOFT_PROB:-0.3} --reward_mask_weight ${ADAPT_SOFT_WEIGHT:-0.3} --mask_warmup_steps ${MASK_WARMUP_STEPS} --zero_warmup_steps ${ZERO_WARMUP_STEPS} --max_completion_length ${MAX_LEN_BASE} --max_completion_length_cap ${MAX_COMPLETION_LENGTH_CAP} --post_phase_max_len_factor ${ADAPT_POST_LEN_FACTOR} --post_phase_chain_scale ${POST_PHASE_CHAIN_SCALE} --diversity_weight ${ADAPT_DIVERSITY_WEIGHT:-0.05}
phase_adapt_diversity         | --enable_phase_adaptive --reward_mask_strategy round_robin_k --reward_mask_round_robin_k ${ADAPT_ROUND_ROBIN_K:-2} --reward_mask_weight ${ADAPT_ROUND_ROBIN_WEIGHT:-0.5} --mask_warmup_steps ${MASK_WARMUP_STEPS} --zero_warmup_steps ${ZERO_WARMUP_STEPS} --max_completion_length ${MAX_LEN_BASE} --max_completion_length_cap ${MAX_COMPLETION_LENGTH_CAP} --post_phase_max_len_factor ${ADAPT_POST_LEN_FACTOR} --post_phase_chain_scale ${POST_PHASE_CHAIN_SCALE} --diversity_weight ${ADAPT_DIVERSITY_WEIGHT:-0.05}
phase_split_masking           | --enable_phase_adaptive --reward_mask_strategy every_n --reward_mask_every_n ${LIGHT_EVERY_N} --reward_mask_weight ${LIGHT_WEIGHT} --mask_format --mask_warmup_steps ${LIGHT_MASK_WARMUP_STEPS} --zero_warmup_steps ${ZERO_WARMUP_STEPS} --max_completion_length ${MAX_LEN_BASE} --max_completion_length_cap ${MAX_COMPLETION_LENGTH_CAP} --post_phase_max_len_factor ${ADAPT_POST_LEN_FACTOR} --post_phase_chain_scale ${POST_PHASE_CHAIN_SCALE} --diversity_weight ${ADAPT_DIVERSITY_WEIGHT:-0.05}
EXPS
)

# Additional apples-to-apples suite at fixed 512 length
EXPERIMENTS_LEN512=$(cat <<'EXPS512'
phase_adapt_softmask_len512   | --enable_phase_adaptive --reward_mask_strategy prob_p --reward_mask_prob ${ADAPT_SOFT_PROB:-0.3} --reward_mask_weight ${ADAPT_SOFT_WEIGHT:-0.3} --mask_warmup_steps ${MASK_WARMUP_STEPS} --zero_warmup_steps ${ZERO_WARMUP_STEPS} --max_completion_length 512 --max_completion_length_cap 512 --post_phase_max_len_factor 1.0 --post_phase_chain_scale ${POST_PHASE_CHAIN_SCALE} --diversity_weight ${ADAPT_DIVERSITY_WEIGHT:-0.05}
phase_adapt_diversity_len512  | --enable_phase_adaptive --reward_mask_strategy round_robin_k --reward_mask_round_robin_k ${ADAPT_ROUND_ROBIN_K:-2} --reward_mask_weight ${ADAPT_ROUND_ROBIN_WEIGHT:-0.5} --mask_warmup_steps ${MASK_WARMUP_STEPS} --zero_warmup_steps ${ZERO_WARMUP_STEPS} --max_completion_length 512 --max_completion_length_cap 512 --post_phase_max_len_factor 1.0 --post_phase_chain_scale ${POST_PHASE_CHAIN_SCALE} --diversity_weight ${ADAPT_DIVERSITY_WEIGHT:-0.05}
phase_split_masking_len512    | --enable_phase_adaptive --reward_mask_strategy every_n --reward_mask_every_n ${LIGHT_EVERY_N} --reward_mask_weight ${LIGHT_WEIGHT} --mask_format --mask_warmup_steps ${LIGHT_MASK_WARMUP_STEPS} --zero_warmup_steps ${ZERO_WARMUP_STEPS} --max_completion_length 512 --max_completion_length_cap 512 --post_phase_max_len_factor 1.0 --post_phase_chain_scale ${POST_PHASE_CHAIN_SCALE} --diversity_weight ${ADAPT_DIVERSITY_WEIGHT:-0.05}
EXPS512
)

for SEED in "${SEEDS[@]}"; do
  echo "[INFO] Launching phase-adapt experiments for seed=${SEED}" | tee -a "$MASTER_LOG"
  while IFS='|' read -r NAME ARGS; do
    NAME=$(echo "$NAME" | xargs)
    [ -z "$NAME" ] && continue
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

    CMD="${BASE_ENV} ${PER_EXP_ENV} uv run main_phase_adapt.py --use_vllm --use_liger \
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
      --problem_overrides_file configs/phase_adapt_problem_overrides.json \
      --use_wandb --wandb_project ${WANDB_PROJECT} --wandb_run ${NAME}_seed${SEED}"

    if [ "$GRADIENT_CHECKPOINTING" = "1" ]; then
      CMD+=" --gradient_checkpointing"
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

# Queue the 512-length apples-to-apples suite after current runs
for SEED in "${SEEDS[@]}"; do
  echo "[INFO] Launching phase-adapt LEN512 experiments for seed=${SEED}" | tee -a "$MASTER_LOG"
  while IFS='|' read -r NAME ARGS; do
    NAME=$(echo "$NAME" | xargs)
    [ -z "$NAME" ] && continue
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

    CMD="${BASE_ENV} ${PER_EXP_ENV} uv run main_phase_adapt.py --use_vllm --use_liger \
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
      --problem_overrides_file configs/phase_adapt_problem_overrides.json \
      --use_wandb --wandb_project ${WANDB_PROJECT} --wandb_run ${NAME}_seed${SEED}"

    if [ "$GRADIENT_CHECKPOINTING" = "1" ]; then
      CMD+=" --gradient_checkpointing"
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
  done <<< "$EXPERIMENTS_LEN512"
done

echo "==== Science2 phase-adapt suite launch finished $(date) ====" | tee -a "$MASTER_LOG"
echo "Phase-adapt suite done. Inspect logs and summaries under $OUTDIR_BASE" | tee -a "$MASTER_LOG"
