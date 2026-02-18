# vLLM Server Setup Options for SLURM

## Current Approach (What You're Doing)
Starting vLLM manually in tmux on the same node, then running training separately.

**Pros**: Simple, flexible, you can monitor both processes
**Cons**: Manual coordination, vLLM not managed by SLURM, harder to track resources

---

## Option 1: Integrated vLLM in Training Job (RECOMMENDED)

The sweep script now automatically starts vLLM in the background before training:

```bash
sbatch run_sweep.sh
```

**How it works**:
1. SLURM allocates 2 GPUs to the job
2. Script starts vLLM server in background on first GPU
3. Waits for vLLM to be ready (up to 60 seconds)
4. Starts training on second GPU
5. Automatically kills vLLM when job ends

**Pros**: 
- Everything managed by SLURM
- Automatic cleanup
- vLLM logs saved to `logs/vllm_server_{job_id}_{array_id}.log`
- No manual coordination needed

**Cons**: 
- Uses 2 GPUs per job (might limit parallelism)
- If training fails, vLLM resources are wasted

---

## Option 2: Separate vLLM SLURM Job

Start vLLM as a long-running SLURM job, then submit training jobs that use it:

```bash
# Start vLLM server (runs for 24 hours)
sbatch experiments_3b/start_vllm_server.sh
# Note the job ID (e.g., 12345)

# Check vLLM is running
squeue -u $USER

# Submit training jobs (will use the same node)
sbatch --nodelist=<node_from_vllm_job> run_sweep.sh
```

**Pros**:
- One vLLM server serves all training jobs
- More efficient GPU usage
- vLLM can run for days

**Cons**:
- Manual node coordination
- Need to ensure training jobs land on same node
- vLLM keeps running after training finishes (manual cleanup)

---

## Option 3: tmux + SLURM Hybrid (Your Current Way)

Keep doing what you're doing, but document it:

```bash
# 1. Get allocation with multiple GPUs
salloc --gpus=4 --time=24:00:00

# 2. In first tmux session, start vLLM
tmux new -s vllm
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --port 8000
# Ctrl+B, D to detach

# 3. In second tmux session, run training
tmux new -s training
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1,2,3
cd experiments_3b/phase2_hyperparameter_sweep
python generate_sweep_configs.py
# Run configs manually or with local script
```

**Pros**:
- Full control
- Can monitor both easily
- Familiar workflow

**Cons**:
- Can't use SLURM array jobs efficiently
- Manual resource management
- Interactive allocation ties up GPUs

---

## Recommendation

**For the sweep (48 jobs)**: Use Option 1 (integrated vLLM)
- Simplest for batch jobs
- Everything tracked by SLURM
- Automatic cleanup

**For development/debugging**: Use Option 3 (tmux)
- When you want to iterate quickly
- When you need to inspect both processes

---

## Current Script Configuration

The updated `run_sweep.sh` now includes:
- Automatic vLLM startup in background
- Health check with 60-second timeout
- Trap to kill vLLM on exit
- Separate logs per array task

You can still disable this by commenting out the vLLM section and starting it manually if needed.

---

## Testing the New Setup

```bash
# Test with a single config first
cd experiments_3b/phase2_hyperparameter_sweep
python generate_sweep_configs.py

# Run just array task 0
sbatch --array=0 run_sweep.sh

# Check logs
tail -f /mnt/home/oana/projects/nano-grpo-envs/logs/3b_sweep_*_0.out
tail -f /mnt/home/oana/projects/nano-grpo-envs/logs/vllm_server_*_0.log
```
