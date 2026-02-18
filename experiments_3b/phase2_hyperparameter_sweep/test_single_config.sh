#!/bin/bash
#SBATCH --job-name=3b_test
#SBATCH --output=/mnt/home/oana/projects/nano-grpo-envs/logs/3b_test_%j.out
#SBATCH --error=/mnt/home/oana/projects/nano-grpo-envs/logs/3b_test_%j.err
#SBATCH --time=4:00:00
#SBATCH --gpus=2
#SBATCH --mem=60G

# ===== Quick 2-GPU Test Script =====
# Test vLLM + training setup with reduced resources

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

# Test with first config (continuous, n=10, w=0.7, warmup=200, seed=0)
CONFIG_ID=0
CONFIG_FILE=$(jq -r ".configs[$CONFIG_ID].file" $MANIFEST)
METHOD=$(jq -r ".configs[$CONFIG_ID].method" $MANIFEST)

echo "=========================================="
echo "3B Test Job (2 GPUs)"
echo "Config ID: $CONFIG_ID"
echo "Method: $METHOD"
echo "Config File: $CONFIG_FILE"
echo "=========================================="

# Parse config parameters
EVERY_N=$(jq -r ".configs[$CONFIG_ID].params.every_n" $MANIFEST)
WEIGHT=$(jq -r ".configs[$CONFIG_ID].params.weight" $MANIFEST)
SEED=$(jq -r ".configs[$CONFIG_ID].params.seed" $MANIFEST)
WARMUP=$(jq -r ".configs[$CONFIG_ID].params.warmup" $MANIFEST)

echo "Config: every_n=$EVERY_N, weight=$WEIGHT, warmup=$WARMUP, seed=$SEED"

# Navigate back to main project directory
cd /mnt/home/oana/projects/nano-grpo-envs

# ===== Start vLLM Server =====
echo "Starting vLLM server in background..."
VLLM_PORT=8000
VLLM_MODEL="Qwen/Qwen2.5-3B-Instruct"

# Allocate GPUs: GPU 0 for vLLM, GPU 1 for training
echo "Allocating GPU 0 for vLLM, GPU 1 for training"

# Start vLLM server on GPU 0 (no tensor parallelism with single GPU)
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 uv run vllm_server.py \
    --model $VLLM_MODEL \
    --port $VLLM_PORT \
    > logs/vllm_test_${SLURM_JOB_ID}.log 2>&1 &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for vLLM server to be ready
echo "Waiting for vLLM server to be ready..."
for i in {1..120}; do
    if curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null 2>&1; then
        echo "✓ vLLM server is ready after ${i} seconds"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "✗ vLLM server failed to start after 120 seconds"
        echo "Check logs: logs/vllm_test_${SLURM_JOB_ID}.log"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# Trap to ensure vLLM server is killed on exit
trap "echo 'Stopping vLLM server...'; kill $VLLM_PID 2>/dev/null" EXIT

# Select training script
TRAIN_SCRIPT="main.py"  # Using continuous for test

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
    --run_name 3b_test_n${EVERY_N}_w${WEIGHT}_seed${SEED} \
    --reward_mask_every_n $EVERY_N \
    --reward_mask_weight $WEIGHT \
    --mask_warmup_steps $WARMUP"

# Execute training on GPU 1
echo "Command: $CMD"
CUDA_VISIBLE_DEVICES=1 eval $CMD

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "✓ Training completed successfully"
else
    echo "✗ Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo "=========================================="
echo "3B Test Job Completed"
echo "Check vLLM logs: logs/vllm_test_${SLURM_JOB_ID}.log"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
