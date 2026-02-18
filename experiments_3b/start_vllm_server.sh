#!/bin/bash
#SBATCH --job-name=vllm_3b
#SBATCH --output=/mnt/home/oana/projects/nano-grpo-envs/logs/vllm_server_%j.out
#SBATCH --error=/mnt/home/oana/projects/nano-grpo-envs/logs/vllm_server_%j.err
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --mem=40G

# ===== Standalone vLLM Server Script =====
# Alternative approach: Start vLLM server as a separate SLURM job
# Then reference this job when submitting training jobs
#
# Usage:
#   sbatch start_vllm_server.sh
#   # Note the job ID (e.g., 12345)
#   # Then submit training with: sbatch --dependency=after:12345 run_sweep.sh

set -e

source .venv/bin/activate

cd /mnt/home/oana/projects/nano-grpo-envs

VLLM_PORT=8000
MODEL="Qwen/Qwen2.5-3B-Instruct"

echo "=========================================="
echo "Starting vLLM Server"
echo "Model: $MODEL"
echo "Port: $VLLM_PORT"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port $VLLM_PORT \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --trust-remote-code

echo "vLLM server stopped"
