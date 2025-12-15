# Reward Masking Strategies in Nano-GRPO

This guide explains each reward masking strategy available via `--reward_mask_strategy` and the related flags. Masking selectively reduces the correctness reward for some correct samples to encourage exploration and stabilize training.

## Strategies

- None
  - Keeps all rewards. Use when you want pure learning signal without withholding.
  - Flag: `--reward_mask_strategy none`

- Every-N
  - Masks the correctness reward for every N-th correct sample in a batch (stable order).
  - Flags: `--reward_mask_strategy every_n --reward_mask_every_n <N>` (default N=10)
  - When combined with `--reward_mask_weight w`, masked rewards are scaled by w (0.0 = full drop, 0.5 = half reward).

- Probabilistic (prob_p)
  - Each correct sample is masked with probability p.
  - Flags: `--reward_mask_strategy prob_p --reward_mask_prob <p>` (default p=0.1)
  - Use small p (0.05–0.2) to gently reduce exploit peaks without overly weakening signal.

- Cosine Schedule
  - Time-based schedule: masks a fraction of correct rewards following a cosine curve.
  - Flags: `--reward_mask_strategy cosine --reward_mask_period <T> --reward_mask_max_frac <f>`
  - Example: period 200, max_frac 0.5 → up to half of correct rewards masked near peaks, tapering elsewhere.

- Round-Robin-k
  - Partitions chains into k buckets; on each step, one bucket of correct samples is masked.
  - Flags: `--reward_mask_strategy round_robin_k --reward_mask_round_robin_k <k>` (default k=4)
  - Smooths variance across steps; useful when gradients spike with many perfect solutions.

## Mask Strength: `--reward_mask_weight`

- Applies only to masked correct samples: effective reward = `reward_keep_mask + (1 - reward_keep_mask) * reward_mask_weight`.
- `0.0` → full drop, `0.5` → half reward, `1.0` → no effect. Typical: `0.3–0.75`.

## Formatting Rewards & Thresholds

- Two components combine into the total reward: correctness and formatting.
- Weights: `--correctness_weight` (default 1.0), `--format_weight` (default 1.0).
- Binary formatting (optional): `--format_binary_threshold <t>`; sets format_reward to 1 if `format >= t`, else 0.
- Full-correct threshold: `--format_full_threshold <t>`; used by full-correct zeroing below.

## Full-Correct Zeroing

Occasionally zeros the combined reward when a sample is fully correct (correctness==1) and `format >= format_full_threshold`.

- Strategies: `--full_correct_zero_strategy {none,every_n,prob_p,round_robin_k,cosine}`
- Key flags:
  - `--full_correct_zero_every_n <N>`
  - `--full_correct_zero_prob <p>`
  - `--full_correct_zero_round_robin_k <k>`
  - `--full_correct_zero_period <T> --full_correct_zero_max_frac <f>`
- Use cases: prevent overfitting to very easy tasks; add exploration pressure when many completions are perfect.

## Practical Recipes

- Gentle exploration without hard drops
  - `--reward_mask_strategy prob_p --reward_mask_prob 0.1 --reward_mask_weight 0.5`
  - Keeps learning stable while reducing exploit peaks.

- Periodic resets for stability
  - `--reward_mask_strategy every_n --reward_mask_every_n 10 --reward_mask_weight 0.5`
  - Optionally add: `--full_correct_zero_strategy every_n --full_correct_zero_every_n 10`.

- Variance smoothing across chains
  - `--reward_mask_strategy round_robin_k --reward_mask_round_robin_k 4 --reward_mask_weight 0.5`
  - Useful for multi-chain setups where some chains consistently overperform.

- Scheduling-based exploration
  - `--reward_mask_strategy cosine --reward_mask_period 200 --reward_mask_max_frac 0.3`
  - Aligns exploration pressure to training phases.

## Tuning Tips

- Start with small masking and increase gradually. Watch `train/avg_total_reward`, `train/avg_correctness`, and `eval/pass_at_k`.
- If formatting falls while correctness rises, try `--format_weight 1.2–1.4` or enable binary formatting (`--format_binary_threshold 0.9`).
- If pass@k stalls, lower masking strength (e.g., reduce `prob_p` or raise `reward_mask_weight`).
- Combine strategies carefully: overlapping masking and full-correct zeroing can overly suppress rewards.

## CLI Examples

- Soft masking every 10 steps:
  - `--reward_mask_strategy every_n --reward_mask_every_n 10 --reward_mask_weight 0.5`

- Hybrid soft + round-robin:
  - `--reward_mask_strategy round_robin_k --reward_mask_round_robin_k 4 --reward_mask_weight 0.5`

- Cosine full-correct zeroing:
  - `--full_correct_zero_strategy cosine --full_correct_zero_period 200 --full_correct_zero_max_frac 0.3`

- Graded formatting (remove binary, lower full threshold):
  - `--format_full_threshold 0.9` (omit `--format_binary_threshold`)

Refer to `run_specs_experiments.sh` for ready-to-run experiment combinations and to `ResearchPlan.md` for interpretation guidance.
