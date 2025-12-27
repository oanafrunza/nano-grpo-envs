# Reward Masking Strategies in Nano-GRPO

This guide explains the reward masking and full-correct zeroing mechanisms, why you might use them, and how the latest behavior (global, stateful schedules) works. The goal is to give clear intuition without losing scientific rigor.

## Correctness Masking Strategies

- None
  - Keeps all rewards. Use when you want pure learning signal without withholding or you’re establishing a baseline.
  - Flag: `--reward_mask_strategy none`

- Every-N
  - Intuition: Introduce periodic “breathers” that soften rewards at a regular cadence to avoid overfitting to easy patterns.
  - Behavior (updated): Global and stateful across steps. The scheduler keeps a running counter of correct samples seen (`global_correct_seen`) and applies masking to every N‑th correct sample encountered over time, not just per-batch.
  - Flags: `--reward_mask_strategy every_n --reward_mask_every_n <N>` (default N=10)
  - Strength: `--reward_mask_weight w` scales masked rewards in `[0,1]` (0.0 = full drop; 0.5 = half reward).
  - Good for: Regularizing high-accuracy phases without fully turning off signal.

- Probabilistic (prob_p)
  - Intuition: Gentle, stochastic exploration pressure; reduces reward occasionally to keep gradients diverse.
  - Behavior: Each correct sample is independently masked with probability `p`.
  - Flags: `--reward_mask_strategy prob_p --reward_mask_prob <p>` (default p=0.1)
  - Use small p (0.05–0.2) to gently reduce exploit peaks without overly weakening signal.

- Cosine Schedule
  - Intuition: Phase your masking. Early training might have more masking to encourage exploration; later training tapers off.
  - Behavior: Time-based schedule; masks a fraction of correct rewards following a cosine curve over a period.
  - Flags: `--reward_mask_strategy cosine --reward_mask_period <T> --reward_mask_max_frac <f>`
  - Example: period 200, max_frac 0.5 → up to half of correct rewards masked near peaks, tapering elsewhere.
  - Good for: Curriculum-like dynamics without hand-tuning per phase.

- Round-Robin-k
  - Intuition: Smooth variance across parallel chains by rotating which subset gets softened rewards.
  - What is a "chain" here: One parallel completion sampled for the same prompt at a step. Controlled by `--num_chains` (science2 uses 4).
  - Behavior (implementation): Partitions chain indices by `i % k` into `k` buckets; at step `s`, the active bucket is `s % k`. Only correct samples whose chain index `i` falls in the active bucket are masked that step.
  - Effect with current settings: With `--num_chains 4` and `--reward_mask_round_robin_k 4`, exactly one chain index is eligible per step.
    - Step 0 → bucket 0 → masks chain index {0}
    - Step 1 → bucket 1 → masks chain index {1}
    - Step 2 → bucket 2 → masks chain index {2}
    - Step 3 → bucket 3 → masks chain index {3}
    - Step 4 → wraps to bucket 0 → masks {0}, etc.
  - Flags (masking): `--reward_mask_strategy round_robin_k --reward_mask_round_robin_k <k>` (default k=4). Mask strength still governed by `--reward_mask_weight`.
  - Flags (zeroing variant): `--full_correct_zero_strategy round_robin_k --full_correct_zero_round_robin_k <k>` applies the same bucket rotation but zeros the total reward for fully‑correct samples (correctness=1 and `format >= format_full_threshold`).
  - Flags: `--reward_mask_strategy round_robin_k --reward_mask_round_robin_k <k>` (default k=4)
  - Smooths variance across steps; useful when gradients spike with many perfect solutions.

## Mask Strength: `--reward_mask_weight`

- Applies only to masked correct samples. Effective correctness component is scaled by `reward_mask_weight`.
- `0.0` → full drop, `0.5` → half reward, `1.0` → no effect. Typical: `0.3–0.75`.

## Formatting Rewards, Thresholds, and Format Masking

- Two components combine into the total reward: correctness and formatting.
- Weights: `--correctness_weight` (default 1.0), `--format_weight` (default 1.0).
- Binary formatting (optional): `--format_binary_threshold <t>`; sets format_reward to 1 if `format >= t`, else 0.
- Full-correct threshold: `--format_full_threshold <t>`; used by full-correct zeroing below.
 - Optional format masking: `--mask_format` applies the same effective mask to the formatting component (use when you want to soften both correctness and format together).

Intuition: Formatting rewards nudge the model toward clean, parseable outputs (e.g., `<answer>...</answer>`). Graded formatting provides a softer, continuous signal; binary formatting makes formatting pass-or-fail. If your outputs rarely meet strict formatting, consider graded mode or lower thresholds.

## Full-Correct Zeroing

Occasionally zeros the combined reward for fully correct samples to inject exploration and prevent collapse onto trivial, high-reward patterns.

Definition (strict):
- “Fully correct” means both correctness == 1 and formatting >= `format_full_threshold` (graded) or passes binary formatting when enabled.
- If this condition is not met, zeroing does not apply.

- Strategies: `--full_correct_zero_strategy {none,every_n,prob_p,round_robin_k,cosine}`
- Key flags:
  - `--full_correct_zero_every_n <N>`
  - `--full_correct_zero_prob <p>`
  - `--full_correct_zero_round_robin_k <k>`
  - `--full_correct_zero_period <T> --full_correct_zero_max_frac <f>`
- Use cases: prevent overfitting to very easy tasks; add exploration pressure when many completions are perfect.

Important practical note: In many real runs, formatted scores are modest (e.g., 0.0–0.4). With `--format_full_threshold 0.9`, very few samples qualify as “fully correct,” so `num_full_correct_zeroed` can remain 0 even if the zeroing strategy is enabled. To observe effects, lower the full threshold (e.g., 0.5) or improve formatting adherence.

## Practical Recipes

- Gentle exploration without hard drops
  - `--reward_mask_strategy prob_p --reward_mask_prob 0.1 --reward_mask_weight 0.5`
  - Keeps learning stable while reducing exploit peaks.

- Periodic resets for stability
  - `--reward_mask_strategy every_n --reward_mask_every_n 10 --reward_mask_weight 0.5`
  - Optionally add: `--full_correct_zero_strategy every_n --full_correct_zero_every_n 10`
  - If you never see `num_full_correct_zeroed > 0`, try `--format_full_threshold 0.5`.

- Variance smoothing across chains
  - `--reward_mask_strategy round_robin_k --reward_mask_round_robin_k 4 --reward_mask_weight 0.5`
  - Useful for multi-chain setups where some chains consistently overperform.

- Scheduling-based exploration
  - `--reward_mask_strategy cosine --reward_mask_period 200 --reward_mask_max_frac 0.3`
  - Aligns exploration pressure to training phases.

- Mask both correctness and format (advanced)
  - `--reward_mask_strategy every_n --reward_mask_every_n 10 --reward_mask_weight 0.5 --mask_format`
  - Use when formatting adherence is too strong and you want to soften both signals together.

## Tuning Tips

- Start with small masking and increase gradually. Watch `train/avg_total_reward`, `train/avg_correctness`, and `eval/pass_at_k`.
- If formatting falls while correctness rises, try `--format_weight 1.2–1.4` or enable binary formatting (`--format_binary_threshold 0.9`).
- If pass@k stalls, lower masking strength (e.g., reduce `prob_p` or raise `reward_mask_weight`).
- Combine strategies carefully: overlapping masking and full-correct zeroing can overly suppress rewards.
 - To see full-correct zeroing in metrics, ensure your formatting clears the threshold; otherwise, lower `--format_full_threshold`.

## CLI Examples

- Soft masking every 10 steps:
  - `--reward_mask_strategy every_n --reward_mask_every_n 10 --reward_mask_weight 0.5`

- Hybrid soft + round-robin:
  - `--reward_mask_strategy round_robin_k --reward_mask_round_robin_k 4 --reward_mask_weight 0.5`

- Cosine full-correct zeroing:
  - `--full_correct_zero_strategy cosine --full_correct_zero_period 200 --full_correct_zero_max_frac 0.3`

- Graded formatting (remove binary, lower full threshold):
  - `--format_full_threshold 0.9` (omit `--format_binary_threshold`) or lower to 0.5 to observe full-correct zeroing.

- Mask format together with correctness:
  - `--mask_format` (paired with any masking strategy)

Refer to `run_science_experiments.sh` and `run_specs_experiments.sh` for ready-to-run combinations, and to `ResearchPlan.md` for interpretation guidance.
