# Nano GRPO with Reasoning Gym

A barebones, fast implementation of GRPO for training language models on reasoning tasks. This is a modernization of my previous [DeepSeekRL-Extended](https://github.com/brendanhogan/DeepSeekRL-Extended) repository - taking the simple but slow GRPO implementation and making it production-ready with modern optimizations.

## Why This Exists

The original DeepSeekRL-Extended was a great learning tool but became bloated and slow. This version focuses on:
- **Speed**: vLLM for high-throughput generation, Liger kernels for optimized loss computation
- **Scalability**: Accelerate support for large models and multi-GPU training  
- **Cleanliness**: Simple, readable codebase without unnecessary complexity
- **Reasoning Focus**: Built on reasoning_gym for structured reasoning tasks

The goal is to create a fast, clean foundation for experimenting with agent training on reasoning exercises. 

I have a lot of little ideas I want to test out, and so having something barebones like this, but also fast enough to train to do big enough experiments is great. 

## Features

- **Modern Optimizations**: vLLM server, Liger fused kernels, Accelerate multi-GPU
- **Composite Datasets**: Mix multiple reasoning tasks with custom weights via reasoning_gym
- **Clean Logging**: JSON logs + optional Weights & Biases integration
- **Extensible**: Easy to adapt to new datasets and evaluation metrics

## Quick Start

### 1. Basic Training (Local Model)

```bash
# Train with default settings
uv run main.py

# Custom dataset mixture
uv run main.py \
  --train-names leg_counting logic_grid \
  --train-weights 0.6 0.4 \
  --train-size 1000 \
  --num_train_iters 500
```

### 2. Using vLLM Server (Faster Generation)

First, start the vLLM server in a separate terminal:

```bash
# Start vLLM server (multi-GPU example)
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 uv run vllm_server.py --model Qwen/Qwen2.5-7B-Instruct --port 8000
```

Then run training with vLLM:

```bash
# Use vLLM for generation
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=2,3 uv run main.py --use_vllm --use_liger

# Custom vLLM server location
uv run main.py --use_vllm --vllm_host 192.168.1.100 --vllm_port 8001
```

### 3. Using Liger Kernels (Faster Loss Computation)

```bash
# Enable Liger kernels for faster GRPO loss (local training only)
uv run main.py --use_liger

# Combine with vLLM (uses standard model for vLLM, Liger for local training)
uv run main.py --use_vllm --use_liger
```

## Key Arguments

### Dataset Configuration
- `--train-names`: List of reasoning task names (e.g., `leg_counting figlet_font`)
- `--train-weights`: Weights for each task (e.g., `0.7 0.3`)
- `--train-size`: Total training dataset size
- `--eval-names`, `--eval-weights`, `--eval-size`: Separate eval dataset

### Training
- `--num_train_iters`: Number of training steps
- `--learning_rate`: Learning rate (default: 5e-6)
- `--num_chains`: Parallel generations per prompt (default: 8)
- `--temperature`: Sampling temperature (default: 0.9)

### Logging & Saving
- `--eval_every`: Run evaluation every N steps (default: 20)
- `--save_every`: Save checkpoint every N steps (default: 50)
- `--use_wandb`: Enable Weights & Biases logging
- `--output_dir`: Where to save logs and checkpoints

### vLLM Options
- `--use_vllm`: Enable vLLM server mode
- `--vllm_host`: vLLM server host (default: localhost)
- `--vllm_port`: vLLM server port (default: 8000)

### Liger Kernel Options
- `--use_liger`: Enable Liger kernels for faster GRPO loss computation
- `--beta`: KL penalty weight for Liger loss (default: 0.0)
- `--epsilon_low`, `--epsilon_high`: Clipping parameters for Liger loss
- `--loss_type`: Liger loss type (default: "dr_grpo")

## Output Files

- `output/run_log.json`: Complete training log with all examples
- `output/checkpoint_step_N/`: Model checkpoints
- Weights & Biases dashboard (if enabled)

## Code Structure

The codebase is intentionally simple and focused:

- **`main.py`**: Contains pretty much everything - the core GRPO training loop, generation, scoring, and logging
- **`llms.py`**: Just model setup - loads standard or Liger kernel models with proper configuration
- **`reasoning_envs.py`**: Dataset setup and loading - handles reasoning_gym composite datasets
- **`utils.py`**: Utility functions for prompt formatting, reward checking, etc.
- **`vllm_server.py` & `vllm_client.py`**: Taken directly from TRL for vLLM integration

To adapt to new use cases, you mainly need to change the dataset loading in `reasoning_envs.py` and the evaluation logic in `main.py`.

## How It Works

1. **Dataset Building**: Creates composite datasets from reasoning_gym tasks
2. **Generation**: Generates multiple completions per prompt (local model or vLLM server)
3. **Scoring**: Evaluates correctness + format compliance using dataset's `score_answer` method
4. **Training**: Uses GRPO loss (standard or Liger fused) to improve model based on rewards
5. **Logging**: Tracks everything in structured JSON format

## Future Directions

This is designed as a foundation for rapid experimentation. I'm interested in testing ideas around:
- Soft reward structures for complex reasoning
- Multi-agent reasoning competitions
- New RL algos 
- New loss/reward functions for GRPO - overall and per token level 
- A bunch of dumb ideas 

## Example Output

```json
{
  "generations": [
    {
      "text": "<think>Let me count the legs...</think>\n<answer>94</answer>",
      "extracted_answer": "94",
      "correct": 1,
      "format_reward": 0.4,
      "total_reward": 1.4
    }
  ],
  "loss": 0.023,
  "eval_metrics": {
    "accuracy": 85.0,
    "avg_format_reward": 0.35
  }
}
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{nano_grpo_reasoning_gym,
  title={Nano GRPO with Reasoning Gym: A Fast Implementation of GRPO for Reasoning Tasks},
  author={Brendan Hogan},
  year={2025},
  url={https://github.com/brendanhogan/nano-grpo-reasoning-gym},
}
```