#!/bin/bash
#SBATCH --job-name=eval_3b_31task
#SBATCH --output=/mnt/home/oana/projects/nano-grpo-envs/logs/slurm_eval_3b_31tasks_%j.out
#SBATCH --error=/mnt/home/oana/projects/nano-grpo-envs/logs/slurm_eval_3b_31tasks_%j.err
#SBATCH --time=6:00:00
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Slurm batch script to evaluate 3B models on 31-task suite
# This runs inference only - no training

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Change to project directory
cd /mnt/home/oana/projects/nano-grpo-envs

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
  echo "Virtual environment activated: .venv"
else
  echo "WARNING: Virtual environment not found at .venv/bin/activate"
  echo "Continuing with system Python..."
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the evaluation script
./eval_3b_on_31tasks.sh

echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="
