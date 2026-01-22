#!/usr/bin/env bash
set -euo pipefail

# Train using the same composite dataset mechanism as current experiments,
# with the curated Core task set (20 tasks).

NAMES=(
  simple_equations polynomial_multiplication \
  fraction_simplification decimal_arithmetic complex_arithmetic \
  gcd prime_factorization rotate_matrix spiral_matrix palindrome_partitioning \
  family_relationships sentence_reordering \
  codeio \
  n_queens mini_sudoku \
  rectangle_count \
  shortest_path largest_island \
  modulo_grid \
  knights_knaves
)

# Uniform weights across tasks
WEIGHTS=()
for _ in "${NAMES[@]}"; do WEIGHTS+=("0.05"); done

SIZE=${SIZE:-2000}
EVAL_SIZE=${EVAL_SIZE:-500}
SEED=${SEED:-42}
CHAINS=${CHAINS:-8}
STEPS=${STEPS:-500}
TEMP=${TEMP:-0.9}
OUTPUT_DIR=${OUTPUT_DIR:-exp_output/transfer_core}

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
