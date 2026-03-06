# Llama Cross-Validation Experiments

## Quick Start

### Option 1: Use the Master Script (Recommended)

```bash
cd /mnt/home/oana/projects/nano-grpo-envs/llama

# Launch both vLLM servers
./launch_llama_experiments.sh launch-vllm all

# Wait ~2-3 minutes for servers to start, then check status
./launch_llama_experiments.sh status

# Once servers are healthy, launch training
./launch_llama_experiments.sh launch-train all
```

### Option 2: Manual Launch

```bash
cd /mnt/home/oana/projects/nano-grpo-envs

# 1. Launch vLLM servers
sbatch llama/llama_3b/sbatch_vllm_server.sh
sbatch llama/llama_8b/sbatch_vllm_server.sh

# 2. Check which nodes they're on
squeue -u $USER | grep vllm_llama

# 3. Launch training on same nodes
# Replace <NODE_3B> and <NODE_8B> with actual node names
sbatch --nodelist=<NODE_3B> llama/llama_3b/sbatch_training.sh
sbatch --nodelist=<NODE_8B> llama/llama_8b/sbatch_training.sh
```

## What Will Run

### Llama-3.2-3B (3 experiments, ~24 hours total)
- `baseline_seed0` - No masking/zeroing
- `continuous_fullzero_seed0` - Zero incorrect every 20 steps
- `phase_adapt_seed0` - Mask→Zero transition

### Llama-3.1-8B (3 experiments, ~24 hours total)
- `baseline_seed0` - No masking/zeroing
- `continuous_fullzero_seed0` - Zero incorrect every 20 steps
- `phase_adapt_seed0` - Mask→Zero transition

## Monitoring

```bash
# Check all jobs
squeue -u $USER | grep llama

# Check vLLM server logs
tail -f logs/vllm_llama3b_<JOBID>.out
tail -f logs/vllm_llama8b_<JOBID>.out

# Check training logs
tail -f logs/llama3b_training_<JOBID>.out
tail -f logs/llama8b_training_<JOBID>.out

# Check experiment progress
ls -la exp_output/llama_3b/*.completed
ls -la exp_output/llama_8b/*.completed
```

## Results Location

After completion:
- Llama-3B checkpoints: `exp_output/llama_3b/`
- Llama-8B checkpoints: `exp_output/llama_8b/`

## Configuration Details

All experiments use:
- 10 in-domain tasks (same as Qwen experiments)
- 1000 training iterations
- Evaluation every 100 steps
- Same hyperparameters as Qwen best configs
