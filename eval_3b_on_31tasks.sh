#!/usr/bin/env bash
set -euo pipefail

# Evaluate the 3 best 3B trained models on the 31-task OOD suite
# This runs inference only - no training

cd "$(dirname "$0")"

# Output directory
OUTDIR="validation/results_3b_31task"
mkdir -p "$OUTDIR"

echo "==== Evaluating 3B models on 31-task suite ===="
echo "Started: $(date)"

# GPU configuration - use Slurm-allocated GPUs or autodetect
# Don't override CUDA_VISIBLE_DEVICES if Slurm already set it
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "Using Slurm-allocated GPUs: ${CUDA_VISIBLE_DEVICES}"
else
  echo "No CUDA_VISIBLE_DEVICES set, will use available GPUs"
fi

# The 31-task dataset
DATASET="validation/source_reasoning_gym_30.jsonl"

# Check if dataset exists
if [ ! -f "$DATASET" ]; then
  echo "ERROR: Dataset not found at $DATASET"
  echo "Please ensure validation/source_reasoning_gym_30.jsonl exists"
  exit 1
fi

# The 3 best trained 3B models from science2_3b_suite
# 1. Baseline seed0 (27.12% on 10 tasks)
# 2. Continuous seed0 (25.28% on 10 tasks)  
# 3. Phase-Adapt seed1 (22.20% on 10 tasks)

MODELS=(
  "baseline_3b_seed0:exp_output/science2_3b_suite/baseline_len512_seed0"
  "continuous_3b_seed0:exp_output/science2_3b_suite/continuous_best_len512_seed0"
  "phase_adapt_3b_seed1:exp_output/science2_3b_suite/phase_adapt_best_len512_seed1"
)

# Create config file for evaluation
CONFIG_FILE="$OUTDIR/eval_config.json"
echo "{" > "$CONFIG_FILE"
echo '  "models": [' >> "$CONFIG_FILE"
first=true
for entry in "${MODELS[@]}"; do
  name="${entry%%:*}"
  path="${entry##*:}"
  
  if [ ! -d "$path" ]; then
    echo "WARNING: Checkpoint directory not found: $path"
    continue
  fi
  
  # Always use checkpoint-final if it exists
  if [ -d "$path/checkpoint_final" ]; then
    ckpt_path="$path/checkpoint_final"
  else
    ckpt_path="$path"
  fi
  
  # Verify the checkpoint directory has model files
  if [ ! -f "$ckpt_path/config.json" ]; then
    echo "WARNING: No config.json found in $ckpt_path"
    continue
  fi
  
  if [ "$first" = false ]; then
    echo "," >> "$CONFIG_FILE"
  fi
  first=false
  
  cat >> "$CONFIG_FILE" << EOF
    {
      "name": "$name",
      "checkpoint": "$ckpt_path"
    }
EOF
done
echo "" >> "$CONFIG_FILE"
echo "  ]," >> "$CONFIG_FILE"
echo '  "evaluation": {' >> "$CONFIG_FILE"
echo '    "max_new_tokens": 512,' >> "$CONFIG_FILE"
echo '    "temperature": 0.0,' >> "$CONFIG_FILE"
echo '    "format_checker": "regex",' >> "$CONFIG_FILE"
echo '    "regex_pattern": "<answer>.*</answer>"' >> "$CONFIG_FILE"
echo "  }" >> "$CONFIG_FILE"
echo "}" >> "$CONFIG_FILE"

echo "Config file created: $CONFIG_FILE"
cat "$CONFIG_FILE"

# Run evaluation
BATCH_SIZE=${BATCH_SIZE:-16}
RESULTS_CSV="$OUTDIR/results.csv"

echo ""
echo "Running evaluation..."
echo "  Dataset: $DATASET"
echo "  Batch size: $BATCH_SIZE"
echo "  Output: $RESULTS_CSV"
echo ""

python validation/evaluate_models.py \
  --config "$CONFIG_FILE" \
  --dataset "$DATASET" \
  --out "$RESULTS_CSV" \
  --backend hf \
  --use-chat-template \
  --batch-size "$BATCH_SIZE"

if [ $? -ne 0 ]; then
  echo "ERROR: Evaluation failed"
  exit 1
fi

echo ""
echo "Analyzing results..."

# Generate summary statistics
python validation/analyze_results.py \
  --results "$RESULTS_CSV" \
  --out-dir "$OUTDIR"

echo ""
echo "==== Evaluation complete ===="
echo "Finished: $(date)"
echo ""
echo "Results saved to:"
echo "  - $RESULTS_CSV (raw predictions)"
echo "  - $OUTDIR/summary_per_model.csv (aggregated by model)"
echo "  - $OUTDIR/summary_per_task.csv (aggregated by task)"
echo "  - $OUTDIR/summary_per_split.csv (core vs stretch tasks)"
echo ""
echo "Quick summary:"
python -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('$OUTDIR/summary_per_model.csv')
    print(df.to_string(index=False))
except Exception as e:
    print(f'Could not load summary: {e}', file=sys.stderr)
"
