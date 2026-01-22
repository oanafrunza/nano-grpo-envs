#!/usr/bin/env bash
set -euo pipefail

# Train using the same composite dataset mechanism as current experiments,
# with the curated Stretch task set (+10 tasks).

NAMES=(
  simple_integration \
  time_intervals \
  group_anagrams string_synthesis \
  binary_matrix \
  word_ladder rush_hour \
  advanced_geometry \
  course_schedule graph_color \
  circuit_logic
)

# Uniform weights across tasks
WEIGHTS=()
for _ in "${NAMES[@]}"; do WEIGHTS+=("0.1"); done

SIZE=${SIZE:-1500}
EVAL_SIZE=${EVAL_SIZE:-400}
SEED=${SEED:-42}
CHAINS=${CHAINS:-8}
STEPS=${STEPS:-500}
TEMP=${TEMP:-0.9}
OUTPUT_DIR=${OUTPUT_DIR:-exp_output/transfer_stretch}

mkdir -p "$OUTPUT_DIR"

uv run main.py \
  --train-names "${NAMES[@]}" \
  --train-weights "${WEIGHTS[@]}" \
  --train-size "$SIZE" \
  --eval-names "${NAMES[@]}" \
  --eval-weights "${WEIGHTS[@]}" \
  --eval-size "$EVAL_SIZE" \
  --num_train_iters "$STEPS" \
  --num_chains "$CHAINS" \
  --temperature "$TEMP" \
  --output_dir "$OUTPUT_DIR"
