#!/bin/bash
#SBATCH --job-name=vllm_llama3b
#SBATCH --output=/mnt/home/oana/projects/nano-grpo-envs/logs/vllm_llama3b_%j.out
#SBATCH --error=/mnt/home/oana/projects/nano-grpo-envs/logs/vllm_llama3b_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=h100
#SBATCH --gpus=1
#SBATCH --mem=40G

# Persistent vLLM Server for Llama-3.2-3B-Instruct
# This server stays running and can serve multiple training jobs

set -e

cd /mnt/home/oana/projects/nano-grpo-envs
source .venv/bin/activate

echo "=========================================="
echo "Starting vLLM Server for Llama-3.2-3B at $(date)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Export environment variables for networking
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

# Start vLLM server
echo "Starting vLLM server on port 8001..."
uv run vllm_server.py \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --port 8001

echo ""
echo "=========================================="
echo "vLLM Server stopped at $(date)"
echo "=========================================="
