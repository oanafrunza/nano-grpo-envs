#!/bin/bash
#SBATCH --job-name=3b_sweep
#SBATCH --output=/mnt/home/oana/projects/nano-grpo-envs/logs/3b_sweep_%j.out
#SBATCH --error=/mnt/home/oana/projects/nano-grpo-envs/logs/3b_sweep_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=hpc-high
#SBATCH --gpus=6
#SBATCH --mem=160G

# ===== 3B Hyperparameter Sweep - Sequential Loop =====
# Allocates 6 GPUs once, runs all configs sequentially
# Estimated: 16 configs × 3 hours = ~48 hours (2 days)

set -e

# Activate virtual environment
source .venv/bin/activate

cd /mnt/home/oana/projects/nano-grpo-envs/experiments_3b/phase2_hyperparameter_sweep

# Use reduced manifest (16 configs instead of 48)
MANIFEST="sweep_manifest_reduced.json"

# Check if manifest exists
if [ ! -f "$MANIFEST" ]; then
    echo "Error: $MANIFEST not found!"
    echo "Run: python generate_sweep_configs_reduced.py"
    exit 1
fi

# Get total number of configs
TOTAL_CONFIGS=$(jq '.total_configs' $MANIFEST)
echo "=========================================="
echo "Starting 3B Sweep - Sequential Execution"
echo "Total configs: $TOTAL_CONFIGS"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: 6 (2 vLLM + 4 training)"
echo "Estimated runtime: ${TOTAL_CONFIGS} × 3 hours = ~$((TOTAL_CONFIGS * 3)) hours"
echo "=========================================="

# Navigate to main project directory
cd /mnt/home/oana/projects/nano-grpo-envs

# ===== Start vLLM Server (shared across all training runs) =====
echo "Starting vLLM server in background..."
VLLM_PORT=8000
VLLM_MODEL="Qwen/Qwen2.5-3B-Instruct"

echo "Allocating GPU 0,1 for vLLM (tensor parallel), GPU 2,3,4,5 for training"

# Start vLLM server once - will serve all training runs
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 uv run vllm_server.py \
    --model $VLLM_MODEL \
    --port $VLLM_PORT \
    > logs/vllm_sweep_${SLURM_JOB_ID}.log 2>&1 &

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
        echo "Check logs: logs/vllm_sweep_${SLURM_JOB_ID}.log"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# Trap to ensure vLLM server is killed on exit
trap "echo 'Stopping vLLM server...'; kill $VLLM_PID 2>/dev/null" EXIT

# ===== Loop through all configs =====
SUCCESS_COUNT=0
FAIL_COUNT=0

for CONFIG_ID in $(seq 0 $((TOTAL_CONFIGS - 1))); do
    echo ""
    echo "=========================================="
    echo "Config $((CONFIG_ID + 1))/${TOTAL_CONFIGS}"
    echo "=========================================="
    
    # Load config
    CONFIG_FILE=$(jq -r ".configs[$CONFIG_ID].file" experiments_3b/phase2_hyperparameter_sweep/$MANIFEST)
    METHOD=$(jq -r ".configs[$CONFIG_ID].method" experiments_3b/phase2_hyperparameter_sweep/$MANIFEST)
    
    if [ "$CONFIG_FILE" == "null" ] || [ ! -f "$CONFIG_FILE" ]; then
        echo "✗ Config file not found for ID $CONFIG_ID"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    
    echo "Config file: $CONFIG_FILE"
    echo "Method: $METHOD"
    
    # Parse parameters
    EVERY_N=$(jq -r ".configs[$CONFIG_ID].params.every_n" experiments_3b/phase2_hyperparameter_sweep/$MANIFEST)
    WEIGHT=$(jq -r ".configs[$CONFIG_ID].params.weight" experiments_3b/phase2_hyperparameter_sweep/$MANIFEST)
    SEED=$(jq -r ".configs[$CONFIG_ID].params.seed" experiments_3b/phase2_hyperparameter_sweep/$MANIFEST)
    
    if [ "$METHOD" == "continuous" ]; then
        WARMUP=$(jq -r ".configs[$CONFIG_ID].params.warmup" experiments_3b/phase2_hyperparameter_sweep/$MANIFEST)
        echo "Parameters: every_n=$EVERY_N, weight=$WEIGHT, warmup=$WARMUP, seed=$SEED"
    elif [ "$METHOD" == "phase_adapt" ]; then
        MASK_WARMUP=$(jq -r ".configs[$CONFIG_ID].params.mask_warmup" experiments_3b/phase2_hyperparameter_sweep/$MANIFEST)
        ZERO_WARMUP=$(jq -r ".configs[$CONFIG_ID].params.zero_warmup" experiments_3b/phase2_hyperparameter_sweep/$MANIFEST)
        echo "Parameters: every_n=$EVERY_N, weight=$WEIGHT, mask_warmup=$MASK_WARMUP, zero_warmup=$ZERO_WARMUP, seed=$SEED"
    fi
    
    # Select training script
    if [ "$METHOD" == "continuous" ]; then
        TRAIN_SCRIPT="main.py"
    elif [ "$METHOD" == "phase_adapt" ]; then
        TRAIN_SCRIPT="main_phase_adapt.py"
    else
        echo "✗ Unknown method: $METHOD"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    
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
    
    echo "Starting training with $TRAIN_SCRIPT..."
    START_TIME=$(date +%s)
    
    # Build command
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
    CUDA_VISIBLE_DEVICES=2,3,4,5 eval $CMD
    
    TRAIN_EXIT=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ $TRAIN_EXIT -eq 0 ]; then
        echo "✓ Training completed successfully in ${DURATION}s ($((DURATION / 60)) minutes)"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        
        # Check outputs
        if [ -f "$OUTPUT_DIR/summary.json" ]; then
            PERFORMANCE=$(jq -r '.overall_results.["pass@1"]' "$OUTPUT_DIR/summary.json" 2>/dev/null || echo "N/A")
            echo "  Performance: ${PERFORMANCE}%"
        fi
    else
        echo "✗ Training failed with exit code $TRAIN_EXIT after ${DURATION}s"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
    echo "Progress: ${SUCCESS_COUNT} succeeded, ${FAIL_COUNT} failed, $((CONFIG_ID + 1))/${TOTAL_CONFIGS} completed"
done

# Final summary
echo ""
echo "=========================================="
echo "Sweep Completed!"
echo "=========================================="
echo "Total configs: $TOTAL_CONFIGS"
echo "Succeeded: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "vLLM logs: logs/vllm_sweep_${SLURM_JOB_ID}.log"
echo "=========================================="

if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi
