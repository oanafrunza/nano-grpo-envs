#!/bin/bash
#SBATCH --job-name=llama8b_training
#SBATCH --output=/mnt/home/oana/projects/nano-grpo-envs/logs/llama8b_training_%j.out
#SBATCH --error=/mnt/home/oana/projects/nano-grpo-envs/logs/llama8b_training_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=h100
#SBATCH --gpus=3
#SBATCH --mem=180G
#SBATCH --dependency=singleton

# Llama-3.1-8B-Instruct Training Jobs
# NOTE: Must run on same node as vLLM server!
# Submit with: sbatch --nodelist=<vllm-node> sbatch_training.sh

set -e

cd /mnt/home/oana/projects/nano-grpo-envs
source .venv/bin/activate

echo "=========================================="
echo "Llama-3.1-8B Training Started at $(date)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Export environment variables
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export PYTORCH_ALLOC_CONF=max_split_size_mb:64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
export TORCHDYNAMO_DISABLE=1

# Wait for vLLM server to be ready
echo ""
echo "Checking vLLM server connectivity..."
VLLM_HOST=127.0.0.1
VLLM_PORT=8002

for i in {1..60}; do
    if curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "✓ vLLM server is reachable"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "✗ vLLM server not reachable at ${VLLM_HOST}:${VLLM_PORT}"
        echo "Make sure vLLM server job is running: squeue -u oana"
        exit 1
    fi
    echo "Waiting for vLLM server... (${i}/60)"
    sleep 5
done

# Training configuration
OUTDIR_BASE="exp_output/llama_8b"
mkdir -p "$OUTDIR_BASE"

TRAIN_NAMES="polynomial_equations palindrome_generation leg_counting family_relationships bf sokoban simple_geometry maze number_sequence propositional_logic"
TRAIN_WEIGHTS="1 1 1 1 1 1 1 1 1 1"
EVAL_NAMES="polynomial_equations palindrome_generation leg_counting family_relationships bf sokoban simple_geometry maze number_sequence propositional_logic"
EVAL_WEIGHTS="0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1"

TRAIN_SIZE=5000
EVAL_SIZE=500
NUM_TRAIN_ITERS=1000
EVAL_EVERY=100
SAVE_EVERY=500
NUM_CHAINS=4

# Common arguments
COMMON_ARGS="--use_vllm --vllm_host ${VLLM_HOST} --vllm_port ${VLLM_PORT} \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --save_only_last \
    --num_train_iters ${NUM_TRAIN_ITERS} \
    --eval_every ${EVAL_EVERY} \
    --save_every ${SAVE_EVERY} \
    --num_chains ${NUM_CHAINS} \
    --train-names ${TRAIN_NAMES} \
    --train-weights ${TRAIN_WEIGHTS} \
    --train-size ${TRAIN_SIZE} \
    --eval-names ${EVAL_NAMES} \
    --eval-weights ${EVAL_WEIGHTS} \
    --eval-size ${EVAL_SIZE} \
    --use_wandb \
    --wandb_project llama-8b-cross-validation"

# ============================================================
# Experiment 1: Baseline (Seed 0)
# ============================================================
EXP_NAME="baseline_seed0"
OUTDIR="${OUTDIR_BASE}/${EXP_NAME}"
COMPLETED_MARKER="${OUTDIR}.completed"

if [ -f "$COMPLETED_MARKER" ]; then
    echo ""
    echo "=========================================="
    echo "Skipping ${EXP_NAME} (already completed)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Experiment 1/3: Baseline (Seed 0)"
    echo "Started at $(date)"
    echo "=========================================="
    
    uv run main.py ${COMMON_ARGS} \
        --output_dir ${OUTDIR} \
        --seed 0 \
        --wandb_run ${EXP_NAME} \
        --reward_mask_strategy none \
        --full_correct_zero_strategy none \
        --max_completion_length 512
    
    if [ $? -eq 0 ]; then
        touch "$COMPLETED_MARKER"
        echo "✓ Experiment 1/3 Completed at $(date)"
    else
        echo "✗ Experiment 1/3 Failed at $(date)"
    fi
fi

# ============================================================
# Experiment 2: Continuous Full-Zero (Seed 0)
# ============================================================
EXP_NAME="continuous_fullzero_seed0"
OUTDIR="${OUTDIR_BASE}/${EXP_NAME}"
COMPLETED_MARKER="${OUTDIR}.completed"

if [ -f "$COMPLETED_MARKER" ]; then
    echo ""
    echo "=========================================="
    echo "Skipping ${EXP_NAME} (already completed)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Experiment 2/3: Continuous Full-Zero (Seed 0)"
    echo "Started at $(date)"
    echo "=========================================="
    
    uv run main.py ${COMMON_ARGS} \
        --output_dir ${OUTDIR} \
        --seed 0 \
        --wandb_run ${EXP_NAME} \
        --reward_mask_strategy none \
        --full_correct_zero_strategy every_n \
        --full_correct_zero_every_n 20 \
        --format_full_threshold 0.9 \
        --zero_warmup_steps 200
    
    if [ $? -eq 0 ]; then
        touch "$COMPLETED_MARKER"
        echo "✓ Experiment 2/3 Completed at $(date)"
    else
        echo "✗ Experiment 2/3 Failed at $(date)"
    fi
fi

# ============================================================
# Experiment 3: Phase-Adapt (Seed 0)
# ============================================================
EXP_NAME="phase_adapt_seed0"
OUTDIR="${OUTDIR_BASE}/${EXP_NAME}"
COMPLETED_MARKER="${OUTDIR}.completed"

if [ -f "$COMPLETED_MARKER" ]; then
    echo ""
    echo "=========================================="
    echo "Skipping ${EXP_NAME} (already completed)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Experiment 3/3: Phase-Adapt (Seed 0)"
    echo "Started at $(date)"
    echo "=========================================="
    
    uv run main_phase_adapt.py ${COMMON_ARGS} \
        --output_dir ${OUTDIR} \
        --seed 0 \
        --wandb_run ${EXP_NAME} \
        --enable_phase_adaptive \
        --reward_mask_strategy every_n \
        --reward_mask_every_n 20 \
        --reward_mask_weight 0.7 \
        --mask_format \
        --mask_warmup_steps 400 \
        --zero_warmup_steps 200 \
        --format_full_threshold 0.9 \
        --max_completion_length 512 \
        --max_completion_length_cap 512 \
        --post_phase_max_len_factor 1.0 \
        --post_phase_chain_scale 1.0 \
        --diversity_weight 0.05 \
        --phase_target_nll 2.0 \
        --phase_window 5 \
        --phase_patience 3 \
        --problem_overrides_file experiments_3b/configs/phase_adapt_problem_overrides.json \
        --gradient_checkpointing
    
    if [ $? -eq 0 ]; then
        touch "$COMPLETED_MARKER"
        echo "✓ Experiment 3/3 Completed at $(date)"
    else
        echo "✗ Experiment 3/3 Failed at $(date)"
    fi
fi

echo ""
echo "=========================================="
echo "All Llama-3.1-8B Training Completed"
echo "Finished at $(date)"
echo "=========================================="
