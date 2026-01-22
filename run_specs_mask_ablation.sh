#!/usr/bin/env bash
set -euo pipefail
set -x

# Launch a small set of mask-only ablation experiments using a JSON config.
# Each experiment line in JSON should have: {"name": "...", "args": "..."}
# This script mirrors run_specs_experiments.sh but reads experiments from a JSON file.

EXPCONFIG=${1:-configs/mask_ablation_experiments.json}
GPUS=${GPUS:-"1,2,3,4"}
BASE_ENV="NCCL_P2P_DISABLE=1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 CUDA_VISIBLE_DEVICES=${GPUS} PYTORCH_ALLOC_CONF=max_split_size_mb:64"
BASE_MODEL_FLAGS="--use_vllm --use_liger"
VLLM_HOST=${VLLM_HOST:-"127.0.0.1"}
VLLM_PORT=${VLLM_PORT:-8000}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Composite training/eval suite (same as existing specs script)
SPECS='[{"name":"polynomial_equations","weight":1},{"name":"palindrome_generation","weight":1},{"name":"leg_counting","weight":1},{"name":"family_relationships","weight":1},{"name":"bf","weight":1},{"name":"sokoban","weight":1},{"name":"simple_geometry","weight":1},{"name":"maze","weight":1},{"name":"number_sequence","weight":1},{"name":"propositional_logic","weight":1}]'

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
EVAL_NAMES=("${TRAIN_NAMES[@]}")
# Uniform eval weights
EVAL_COUNT=${#EVAL_NAMES[@]}
EW_UNIFORM=$(awk -v n="$EVAL_COUNT" 'BEGIN {printf "%.6f", n>0?1.0/n:1.0}')
EVAL_WEIGHTS=()
for _ in "${EVAL_NAMES[@]}"; do EVAL_WEIGHTS+=("$EW_UNIFORM"); done

COMMON_ARGS="--format_full_threshold 0.9"

OUTDIR_BASE="exp_specs_runs"
mkdir -p "$OUTDIR_BASE"
MASTER_LOG="${OUTDIR_BASE}/batch_specs_launch.log"
echo "==== Mask ablations launch started $(date) ====" >> "$MASTER_LOG"

# vLLM health check
if command -v curl >/dev/null 2>&1; then
  HEALTH_OK=0
  if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/health/" | grep -q "OK"; then HEALTH_OK=1; else
    if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" | grep -q "id"; then HEALTH_OK=1; else
      if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/" >/dev/null 2>&1 || curl -s "http://${VLLM_HOST}:${VLLM_PORT}/generate" >/dev/null 2>&1; then HEALTH_OK=1; fi
    fi
  fi
  if [ "$HEALTH_OK" -ne 1 ]; then echo "[ERROR] vLLM server at ${VLLM_HOST}:${VLLM_PORT} not reachable." | tee -a "$MASTER_LOG"; exit 3; fi
  echo "[INFO] vLLM server reachable at ${VLLM_HOST}:${VLLM_PORT}" | tee -a "$MASTER_LOG"
fi

# Read experiments from JSON and emit lines NAME|ARGS
readarray -t EXP_LINES < <(python - <<PY
import json,sys
path=sys.argv[1]
with open(path) as f:
  exps=json.load(f)
for e in exps:
  name=e.get('name','').strip()
  args=e.get('args','').strip()
  if name and args:
    print(f"{name}|{args}")
PY
"$EXPCONFIG")

for LINE in "${EXP_LINES[@]}"; do
  NAME="$(echo "$LINE" | cut -d'|' -f1 | xargs)"
  ARGS="$(echo "$LINE" | cut -d'|' -f2- | xargs)"
  [ -z "$NAME" ] && continue
  OUTDIR="${OUTDIR_BASE}/${NAME}"
  if [ -d "$OUTDIR" ] && [ -f "$OUTDIR/.completed" ]; then
    echo "[SKIP] $NAME already completed (outdir: $OUTDIR)" | tee -a "$MASTER_LOG"
    continue
  fi
  echo "[LAUNCH] $NAME -> $OUTDIR" | tee -a "$MASTER_LOG"
  EFFECTIVE_COMMON="$COMMON_ARGS"
  if [[ "$ARGS" == *"--format_full_threshold "* ]]; then
    EFFECTIVE_COMMON="${EFFECTIVE_COMMON//--format_full_threshold 0.9/}"
  fi
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
  echo "Command: $CMD" >> "$MASTER_LOG"
  echo "=== START $NAME $(date) ===" | tee -a "${OUTDIR}_run.log" "$MASTER_LOG"
  eval "$CMD" 2>&1 | tee -a "${OUTDIR}_run.log" || true
  EXIT_CODE=${PIPESTATUS[0]}
  if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] $NAME exited with code $EXIT_CODE" | tee -a "${OUTDIR}_run.log" "$MASTER_LOG"
  else
    echo "=== END $NAME $(date) ===" | tee -a "${OUTDIR}_run.log" "$MASTER_LOG"
    touch "$OUTDIR/.completed"
  fi
  sleep 2
done

echo "==== Mask ablations launch finished $(date) ====" | tee -a "$MASTER_LOG"
