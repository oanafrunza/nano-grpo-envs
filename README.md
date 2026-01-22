## Masking Strategy Wins (Phase-Adapt Split-Masking @ L512)

This highlights where masking clearly helps beyond overall pass@1, using the top variants from each suite:
- Phase-Adapt: phase_split_masking L512
- Continuous: cont_softmask_prob_p L512
- Science2 Baseline: baseline_nomask L512

Per-task summary:

| Task | Format winner | Pass@1 winner | Notes |
|------|----------------|----------------|-------|
| propositional_logic | Phase-Adapt | Continuous | Phase has highest format (~0.40); cont leads pass (~50.36). |
| polynomial_equations | Phase-Adapt | Baseline | Phase has highest format (~0.399); baseline leads pass (~≥67). |
| simple_geometry | Phase-Adapt | Baseline | Phase has highest format (~0.40); baseline leads pass (~≥80). |
| leg_counting | Phase-Adapt | Baseline | Phase has highest format (~0.40); baseline leads pass (~≥44). |
| number_sequence | Phase-Adapt | Baseline | Phase has highest format (~0.40); baseline leads pass (~≥66). |
| maze | Phase-Adapt | Phase-Adapt | Masking improves both pass (~14.9 vs cont ~12.0, baseline ~10.7) and format (~0.40). |
| sokoban | Phase-Adapt | Phase-Adapt | Masking wins on pass (~0.54 vs cont ~0.0, baseline ~0.18) and format (~0.399). |
| bf | — | — | Pass near-zero across all; no meaningful winner; cont slightly higher format. |

Why this matters:
- If strict formatting is required (e.g., tag compliance, auto-graders), split-masking improves format consistency across many tasks while keeping accuracy competitive.
- For hard puzzle-like tasks (maze, sokoban), split-masking delivers the best combination of accuracy and formatting.
- For pure accuracy on math/logic (simple_geometry, polynomial_equations, number_sequence), baseline/continuous still lead on pass@1.

See also:
- Aggregated metrics and plots: [exp_output/visualizations](exp_output/visualizations)
- Top-variant overall: [top_variants_overall.png](exp_output/visualizations/top_variants_overall.png)
- Per-task comparison: [per_task_top3_comparison.png](exp_output/visualizations/per_task_top3_comparison.png)

**Phase-Adapt Split-Masking: Details**
- Core: Uses a windowed moving-average of completion NLL to detect the training “consolidation” phase. Once the signal stabilizes (lower variance and slope change), it triggers a single switch to consolidation.
- After switch: Scales chain sampling/weighting while keeping sequence length fixed (L512). This increases trajectory signal without confounding length, improving gradient stability and formatting consistency.
- Masking mechanics: Alternates which token spans are reward-masked to separate signals.
  - Format-emphasis step: mask reasoning/answer spans; train structure, tags, and required formatting.
  - Correctness-emphasis step: mask formatting spans; train reasoning and final correctness.
  - Alternation prevents reward collapse and reduces interference between format and correctness.
- Why it helps: Reduces gradient interference, boosts formatting robustness across structured tasks, and improves puzzle-like tasks (maze, sokoban) on both format and pass@1.
- Trade-offs: For math/logic tasks where raw accuracy dominates (e.g., simple geometry, number sequences), no-mask or continuous softmask baselines typically lead on pass@1.
- What to monitor: Moving-average NLL plateau and the phase-switch event; chain scaling applied (with length unchanged); rising formatting metrics; per-task pass@1/format deltas.

**Quick Highlights**
- Pass@1: ~31.5 at L512; competitive with continuous softmask, below no-mask baseline.
- Format robustness: Top formatting on structured tasks (propositional_logic, polynomial_equations, simple_geometry, leg_counting, number_sequence).
- Puzzle tasks: Wins both pass@1 and format on maze and sokoban.
- Mechanism: Single consolidation switch via moving-average completion NLL; scales chains (not length); alternates masked spans to separate format vs correctness signals.
- When to use: Prefer for strict formatting requirements or puzzle-heavy curricula; choose baseline/continuous for raw accuracy on math/logic.
- See plots: [top_variants_overall.png](exp_output/visualizations/top_variants_overall.png), [per_task_top3_comparison.png](exp_output/visualizations/per_task_top3_comparison.png), [pass_at1_comparison.png](exp_output/visualizations/pass_at1_comparison.png)
## Reward Masking & Zeroing

This repo includes disciplined mechanisms to occasionally withhold rewards to encourage exploration and robustness. You can combine or run them independently.

### Implementations
- **Correctness Masking (component-level):** Applies a keep/drop mask only to the correctness reward component. Formatting is left untouched.
  - `--reward_mask_strategy {none,every_n,prob_p,cosine,round_robin_k}`
  - `--reward_mask_weight <float>`: masked-correct rewards are scaled by this weight (0.0 = drop, 1.0 = keep, 0.5 = half).
  - Tunables by strategy:
    - `every_n`: `--reward_mask_every_n <int>`
    - `prob_p`: `--reward_mask_prob <float>`
    - `cosine`: `--reward_mask_period <int>`, `--reward_mask_max_frac <float>`
    - `round_robin_k`: `--reward_mask_round_robin_k <int>`

- **Full-Correct Zeroing (combined reward-level):** Occasionally set the entire combined reward (correctness + formatting) to zero, but only when the sample is fully correct.
  - `--full_correct_zero_strategy {none,every_n,prob_p,cosine,round_robin_k}`
  - Tunables by strategy:
    - `every_n`: `--full_correct_zero_every_n <int>`
    - `prob_p`: `--full_correct_zero_prob <float>`
    - `cosine`: `--full_correct_zero_period <int>`, `--full_correct_zero_max_frac <float>`
    - `round_robin_k`: `--full_correct_zero_round_robin_k <int>`
  - Definition of “fully-correct”:
    - Correctness must be 1.
    - Formatting must be above `--format_full_threshold` (default `0.9`).

- **Formatting Reward Mode:**
  - Graded (default): formatting score in `[0,1]`.
  - Binary: set `--format_binary_threshold` (e.g., `1.0`) to convert formatting to `0/1` before combining.

- **Reward Weights:** Combine components explicitly.
  - `--correctness_weight <float>`
  - `--format_weight <float>`

### Baseline Recipes
- Strict formatting + every 10 zeroing of fully-correct:
```bash
python -u projects/nano-grpo-envs/main.py \
  --output_dir exp_output/baseline_strict_fullzero_every10 \
  --train-size 2000 --eval-size 60 \
  --reward_mask_strategy none \
  --correctness_weight 1.0 --format_weight 1.0 \
  --format_binary_threshold 1.0 \
  --full_correct_zero_strategy every_n \
  --full_correct_zero_every_n 10 \
  --format_full_threshold 1.0
```

- Soft correctness-only masking (no hard zeroing):
```bash
python -u projects/nano-grpo-envs/main.py \
  --output_dir exp_output/softmask_every10_wt05 \
  --train-size 2000 --eval-size 60 \
  --reward_mask_strategy every_n \
  --reward_mask_every_n 10 \
  --reward_mask_weight 0.5 \
  --correctness_weight 1.0 --format_weight 1.2 \
  --format_binary_threshold 1.0 \
  --full_correct_zero_strategy none \
  --format_full_threshold 1.0
```

- Round-robin zeroing (distributes zeroing across steps):
```bash
python -u projects/nano-grpo-envs/main.py \
  --output_dir exp_output/fullzero_roundrobin_k4_strict \
  --train-size 2000 --eval-size 60 \
  --reward_mask_strategy none \
  --correctness_weight 1.0 --format_weight 1.2 \
  --format_binary_threshold 1.0 \
  --full_correct_zero_strategy round_robin_k \
  --full_correct_zero_round_robin_k 4 \
  --format_full_threshold 1.0
```

### Visualize & Compare
- Visualize a run (plots saved to the run directory):
```bash
python -u projects/nano-grpo-envs/visualize_results.py --output_dir exp_output/your_run_dir
```

- Compare two runs:
```bash
python -u projects/nano-grpo-envs/compare_runs.py \
  --run_a exp_output/runA \
  --run_b exp_output/runB
```

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

### Reward Masking (Withhold Correct Rewards)
- `--reward_mask_strategy`: `none | every_n | prob_p | cosine | round_robin_k`
- `--reward_mask_every_n`: For `every_n`, mask every N-th correct example
- `--reward_mask_prob`: For `prob_p`, probability to mask a correct reward
- `--reward_mask_period`: For `cosine`, period (steps) of cosine schedule
- `--reward_mask_max_frac`: For `cosine`, max fraction of correct rewards masked
- `--reward_mask_round_robin_k`: For `round_robin_k`, number of buckets rotated
- `--reward_mask_weight`: Scale for masked rewards (0.0 = drop, 1.0 = keep)

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
  "num_masked_correct": 3,
  "reward_mask_strategy": "prob_p",
  "reward_mask_weight": 0.0,
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