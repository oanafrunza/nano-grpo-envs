#!/bin/bash
#SBATCH --job-name=3b_sweep
#SBATCH --output=/mnt/home/oana/projects/nano-grpo-envs/logs/3b_sweep_%A_%a.out
#SBATCH --error=/mnt/home/oana/projects/nano-grpo-envs/logs/3b_sweep_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --gpus=6
#SBATCH --mem=160G
#SBATCH --array=0-47%1  # Run only 1 job at a time (6 GPUs max)

# ===== 3B Hyperparameter Sweep Execution Script =====
# This script runs one config at a time using SLURM array jobs
# Each job trains a single model with specific hyperparameters

set -e

# Activate virtual environment
source .venv/bin/activate

cd /mnt/home/oana/projects/nano-grpo-envs/experiments_3b/phase2_hyperparameter_sweep

# Load sweep manifest
MANIFEST="sweep_manifest.json"
if [ ! -f "$MANIFEST" ]; then
    echo "Error: sweep_manifest.json not found!"
    echo "Run: python generate_sweep_configs.py"
    exit 1
fi

# Get total number of configs
TOTAL_CONFIGS=$(jq '.total_configs' $MANIFEST)
echo "Sweep manifest loaded: $TOTAL_CONFIGS configs"

# Get config for this array task
CONFIG_ID=${SLURM_ARRAY_TASK_ID}
CONFIG_FILE=$(jq -r ".configs[$CONFIG_ID].file" $MANIFEST)
METHOD=$(jq -r ".configs[$CONFIG_ID].method" $MANIFEST)

if [ "$CONFIG_FILE" == "null" ] || [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found for task $CONFIG_ID"
    exit 1
fi

echo "=========================================="
echo "Starting 3B Sweep Job"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Config ID: $CONFIG_ID"
echo "Method: $METHOD"
echo "Config File: $CONFIG_FILE"
echo "=========================================="

# Parse config parameters
EVERY_N=$(jq -r ".configs[$CONFIG_ID].params.every_n" $MANIFEST)
WEIGHT=$(jq -r ".configs[$CONFIG_ID].params.weight" $MANIFEST)
SEED=$(jq -r ".configs[$CONFIG_ID].params.seed" $MANIFEST)

if [ "$METHOD" == "continuous" ]; then
    WARMUP=$(jq -r ".configs[$CONFIG_ID].params.warmup" $MANIFEST)
    echo "Continuous config: every_n=$EVERY_N, weight=$WEIGHT, warmup=$WARMUP, seed=$SEED"
elif [ "$METHOD" == "phase_adapt" ]; then
    MASK_WARMUP=$(jq -r ".configs[$CONFIG_ID].params.mask_warmup" $MANIFEST)
    ZERO_WARMUP=$(jq -r ".configs[$CONFIG_ID].params.zero_warmup" $MANIFEST)
    echo "Phase-Adapt config: every_n=$EVERY_N, weight=$WEIGHT, mask_warmup=$MASK_WARMUP, zero_warmup=$ZERO_WARMUP, seed=$SEED"
fi

# Navigate back to main project directory
cd /mnt/home/oana/projects/nano-grpo-envs

# ===== Start vLLM Server =====
echo "Starting vLLM server in background..."
VLLM_PORT=8000
VLLM_MODEL="Qwen/Qwen2.5-3B-Instruct"

# Allocate GPUs: GPU 0,1 for vLLM (tensor parallel), GPU 2,3,4,5 for training
echo "Allocating GPU 0,1 for vLLM (tensor parallel), GPU 2,3,4,5 for training"

# Start vLLM server on GPU 0,1 with tensor parallelism (using your preferred method)
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 uv run vllm_server.py \
    --model $VLLM_MODEL \
    --port $VLLM_PORT \
    > logs/vllm_server_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1 &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for vLLM server to be ready
echo "Waiting for vLLM server to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null 2>&1; then
        echo "✓ vLLM server is ready after ${i} seconds"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "✗ vLLM server failed to start after 60 seconds"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# Trap to ensure vLLM server is killed on exit
trap "echo 'Stopping vLLM server...'; kill $VLLM_PID 2>/dev/null" EXIT

# Select training script based on method
if [ "$METHOD" == "continuous" ]; then
    TRAIN_SCRIPT="main.py"
elif [ "$METHOD" == "phase_adapt" ]; then
    TRAIN_SCRIPT="main_phase_adapt.py"
else
    echo "Error: Unknown method $METHOD"
    exit 1
fi

echo "Training script: $TRAIN_SCRIPT"
echo "Starting training..."

# Load config parameters
MODEL=$(jq -r '.model' $CONFIG_FILE)
DATASET=$(jq -r '.dataset_name' $CONFIG_FILE)
OUTPUT_DIR=$(jq -r '.output_dir' $CONFIG_FILE)
LR=$(jq -r '.learning_rate' $CONFIG_FILE)
BATCH_SIZE=$(jq -r '.per_device_train_batch_size' $CONFIG_FILE)
GRAD_ACCUM=$(jq -r '.gradient_accumulation_steps' $CONFIG_FILE)
EPOCHS=$(jq -r '.num_train_epochs' $CONFIG_FILE)
MAX_TOKENS=$(jq -r '.max_new_tokens' $CONFIG_FILE)
TEMP=$(jq -r '.temperature' $CONFIG_FILE)
NUM_GENS=$(jq -r '.num_sample_generations' $CONFIG_FILE)
BETA=$(jq -r '.beta' $CONFIG_FILE)

# Build base command
CMD="python $TRAIN_SCRIPT \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET \
    --output_dir $OUTPUT_DIR \
    --learning_rate $LR \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_train_epochs $EPOCHS \
    --max_new_tokens $MAX_TOKENS \
    --temperature $TEMP \
    --num_sample_generations $NUM_GENS \
    --beta $BETA \
    --seed $SEED \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --metric_for_best_model eval_pass@1 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --report_to wandb \
    --run_name 3b_${METHOD}_n${EVERY_N}_w${WEIGHT}_seed${SEED}"

# Add method-specific parameters
if [ "$METHOD" == "continuous" ]; then
    CMD="$CMD \
        --reward_mask_every_n $EVERY_N \
        --reward_mask_weight $WEIGHT \
        --mask_warmup_steps $WARMUP"
elif [ "$METHOD" == "phase_adapt" ]; then
    CMD="$CMD \
        --reward_mask_every_n $EVERY_N \
        --reward_mask_weight $WEIGHT \
        --mask_warmup_steps $MASK_WARMUP \
        --zero_warmup_steps $ZERO_WARMUP \
        --full_correct_zero_strategy late \
        --gradient_checkpointing"
fi

# Execute training on GPU 2,3,4,5
echo "Command: $CMD"
CUDA_VISIBLE_DEVICES=2,3,4,5 eval $CMD

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "✓ Training completed successfully"
else
    echo "✗ Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

# Check if checkpoint was saved
if [ -d "$OUTPUT_DIR/checkpoint_final" ]; then
    echo "✓ Checkpoint saved: $OUTPUT_DIR/checkpoint_final"
else
    echo "⚠ Warning: checkpoint_final not found"
fi

# Check if summary.json was created
if [ -f "$OUTPUT_DIR/summary.json" ]; then
    echo "✓ Summary saved: $OUTPUT_DIR/summary.json"
    echo "Performance:"
    jq '.overall_results' "$OUTPUT_DIR/summary.json"
else
    echo "⚠ Warning: summary.json not found"
fi

echo "=========================================="
echo "3B Sweep Job Completed"
echo "Config ID: $CONFIG_ID"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
