> Reference: See [MaskingStrategies.md](projects/nano-grpo-envs/MaskingStrategies.md) for detailed explanations, tuning tips, and CLI examples.

### Suggested Experiment Plan
- Step 1: Baseline strict full-zero every 10; record overall and per-task delta.
- Step 2: Gentler cadence (every 20) if formatting-sensitive tasks degrade.
- Step 3: Replace zeroing with soft correctness-only masking (weight=0.5).
- Step 4: Try round-robin or cosine schedules to smooth exploration.
- Step 5: Tune `--format_weight` (e.g., 1.2–1.4) to stabilize structure-heavy tasks.
- Step 6: Review `run_log.json` per-task formatting vs correctness to refine thresholds.

### Results Interpretation Guidance
- **Overall vs Per-Task**: A global pass@k jump with a single task regression (e.g., `leg_counting`) usually means the intervention helps simpler tasks but destabilizes structure-heavy ones; treat that task separately.
- **Formatting Drop**: Small decreases (<0.05) in avg format reward are acceptable during exploratory phases; larger drops suggest increasing `--format_weight` or relaxing binary formatting.
- **Zeroing Cadence**: More frequent zeroing (every 10) can outperform slower cadences if it drives exploration without eroding structured tasks; monitor any steep per-task declines.
- **Correctness vs Format Tradeoff**: If correctness rises while format declines, test: (a) soft masking (`reward_mask_weight > 0`), (b) raise `format_weight`, (c) round-robin zeroing for smoother variance.
- **Binary Formatting Impact**: Switching to binary can sharpen signal but may punish borderline formatting; consider graded mode for tasks sensitive to intermediate structuring.
- **Task Sensitivity**: Tasks with sequential reasoning (counting, multi-step logic) often need consistent reward shaping—avoid aggressive zeroing or use higher `format_weight` (1.2–1.4).

### Next Experiment Matrix
| Goal | Config | Why |
|------|--------|-----|
| Confirm stability | Baseline full-zero every 10 (done) | Establish strongest current uplift |
| Reduce leg_counting drop | Full-zero every 10 + `--format_weight 1.3` | Extra emphasis on structure |
| Softer exploration | `reward_mask_strategy every_n`, weight=0.5 | Partial reward keeps gradient signal |
| Smooth variance | `full_correct_zero_strategy round_robin_k` | Spreads zero events, less shock |
| Adaptive pacing | `full_correct_zero_strategy cosine` | Phase-based exploration intensity |
| Probabilistic test | `full_correct_zero_strategy prob_p` (p=0.1) | Randomization vs deterministic cadence |
| Hybrid | Soft correctness mask + round-robin zeroing | Combine gentle shaping + occasional full resets |
| Remove binary format | Drop `--format_binary_threshold`, keep `format_full_threshold 0.9` | See if graded signal stabilizes difficult tasks |

### Recommended Sequence
1. Increase `format_weight` with current best (every 10) to see if leg_counting recovers.
2. Run soft correctness masking (no full zero) to benchmark exploration without total withholding.
3. Test round-robin zeroing; compare per-task variance.
4. Evaluate cosine schedule for gradual exploration phases.
5. If leg_counting still weak, revert to graded formatting (remove binary) and re-run best two configs.
6. Select top 2 configs and run extended training (longer `--num_train_iters`) for stability confirmation.

### Batch Script (`run_batch_experiments.sh`)
Use the provided script to launch all planned experiments sequentially.

Run:
```bash
chmod +x run_batch_experiments.sh
./run_batch_experiments.sh
```

Edit GPU / flags at top of script:
```bash
GPUS="5,6"
BASE_MODEL_FLAGS="--use_vllm --use_liger"
COMMON_ARGS="--train-size 2000 --eval-size 60 --format_full_threshold 1.0 --format_binary_threshold 1.0"
```

To disable an experiment, comment out its line in the here-doc block. Logs:
- Master: `exp_output/batch_launch.log`
- Per experiment: `exp_output/<name>_run.log`

Post-run comparison examples:
```bash
python -u compare_runs.py --run_a exp_output/baseline_fullzero_every10 --run_b exp_output/fullzero_every10_fw13
python -u compare_runs.py --run_a exp_output/baseline_fullzero_every10 --run_b exp_output/softmask_every10_wt05
```

Success criteria suggestions:
- Full-zero variants: overall pass@k +3 pts vs original baseline, no task >15 pt drop.
- Soft mask: maintain ≥ baseline overall, improve format reward ≥0.02.
- Round-robin/cosine: reduce variance (std of per-task pass@k) by ≥10% vs every_n.
- Graded formatting fallback: recover any >15 pt task regression while keeping ≥50% of improvement in other tasks.

Overview

Highest overall pass: baseline_strict_fullzero_every10 (56.33%).
Highest leg_counting pass: fullzero_every10_fw13 (48.0).
Highest family_relationships pass: hybrid_soft_roundrobin (65.88%).
Highest avg format: softmask_every10_wt05 (0.328), but low overall pass (47.0).
Most damaging to formatting: roundrobin_zero_k4 (family format 0.009).
Baselines

baseline_fullzero_every10: 53.67% pass; family format very low (0.094); leg_counting weak (41).
baseline_strict_fullzero_every10: +2.67 pts overall (56.33); leg_counting improves (47); slight format lift (0.283).
Key Effects

Raising format weight (fullzero_every10_fw13): Format ↑ (0.312 vs 0.283) and leg_counting ↑ (48) but overall pass ↓ (52.33) and coin_flip ↓.
Slower zeroing (fullzero_every20): Overall pass ↓ (51.0); leg_counting collapses (35); format ↑ (0.320) shows exploration benefit offset by accuracy loss.
Soft mask (softmask_every10_wt05): Best format (0.328) and family format (0.348) but overall pass drops (47.0) indicating under-rewarding correctness.
Round-robin zeroing: Harsh family format collapse → schedule misalignment.
Probabilistic zeroing (prob_zero_p01): High coin_flip (62.61) but family pass crashes (42.35) and overall weakest pass among zeroing variants (48.67) → randomness hurting structured reasoning.
Cosine: Middle ground but leg_counting suffers (35) without format gain.
Per-Task Trade-offs

Coin flip benefits from deterministic frequent zeroing (every 10) or stochastic zeroing—but family reasoning degrades under stochastic.
Family relationships require stronger formatting incentive (format_weight boost or soft masking) to rise (seen in softmask_every10_wt05 and hybrid_soft_roundrobin).
Leg counting responds to higher format emphasis (48.0 with fw=1.3) but is fragile under exploration schedules with lower structured reward stability (falls sharply in cosine / every20 / softmask).
Interpretation

Frequent deterministic full zeroing (every 10) drives raw accuracy exploration without overly destabilizing structured tasks.
Increasing format weight improves structure-sensitive tasks at the cost of some accuracy—suggests multi-objective tension, not inherent masking failure.
Soft partial masking weight (0.5) might be too low; correctness signal weakened and overall pass falls.
Round-robin and cosine add variance without targeted gains; current parameterization not tuned.
Recommended Next Experiments

Accuracy + Format Balance:
Full zero every 10 + format_weight 1.2 + graded formatting (remove binary) to lift family format without sacrificing coin flip.
Command: add --format_binary_threshold '' (or toggle off) and --format_weight 1.2.
Tune Soft Masking:
Increase --reward_mask_weight from 0.5 → 0.75 to restore correctness while keeping exploration.
Hybrid Stability:
Full zero every 10 AND soft correctness mask (weight 0.5) only on even steps (custom conditional) to smooth spikes.
Leg Counting Focus:
Run fullzero_every10 with --format_weight 1.3 and graded format (no binary) plus extended --num_train_iters to test if pass sustains/improves.
Eliminate Underperformers:
Drop roundrobin and prob_p until tuned; they harm family tasks without compensating gains.
Success Criteria

Balanced config: ≥55% overall pass, family pass ≥60%, leg counting ≥45%, avg format ≥0.30.
Soft mask tuned: recover ≥52% overall while keeping format ≥0.31.
Graded formatting: improve family format ≥0.20 without losing >1 pt overall pass compared to strict baseline.


Next: want a run for a lot of tasks




##################################################
Correctness Component Masking:

What: Applies a mask only to the correctness reward (format reward unaffected).
Why: Encourages exploration by selectively reducing the gradient signal from correct answers.
Strategies:
none: No masking; all correct rewards kept.
every_n: Zero or scale every N-th correct sample deterministically.
prob_p: Randomly mask correct samples with probability p.
cosine: Time-varying masking fraction following a cosine wave (period + max fraction).
round_robin_k: Cycles through k buckets; one bucket is masked each cycle.
Control: --reward_mask_strategy, plus strategy-specific flags; masked rewards scaled by --reward_mask_weight (0.0 drop, 0.5 halve, 1.0 keep).
Full-Correct Zeroing:

What: Sets the entire combined reward (correctness + formatting) to zero for samples that are fully correct and well formatted.
Why: Forces continued exploration even on polished outputs; prevents early overfitting.
Trigger: correctness == 1 AND format ≥ --format_full_threshold.
Strategies: same set (none, every_n, prob_p, cosine, round_robin_k) but applied only if fully-correct.
Control: --full_correct_zero_strategy + its params (e.g. --full_correct_zero_every_n, --full_correct_zero_prob, etc.).
Mask Weight Scaling:

What: Instead of hard dropping, scales masked correctness rewards by reward_mask_weight.
Why: Maintains gradient signal while still reducing incentive strength.
Effect: Effective reward = original * (keep_mask + (1 - keep_mask) * weight).
Binary vs Graded Formatting:

Binary: Formatting becomes 1 or 0 (strict schema adherence); enables sharper fully-correct gating.
Graded: Formatting floats in [0,1]; zeroing and masking gates apply only when above threshold; smoother learning.
Control: Presence/absence of --format_binary_threshold. Omit for graded mode.
Multi-Reward Weighting:

What: Combines components with explicit weights: reward = correctness_weight * correctness + format_weight * format.
Why: Rebalance emphasis for structure-sensitive tasks or promote clean reasoning traces.
Control: --correctness_weight, --format_weight.
Behavior Quick Reference

every_n (correctness mask): Predictable cadence; good for controlled exploration; risk of sync artifacts if N too small.
prob_p (correctness mask): Stochastic exploration; can destabilize structured tasks like multi-step reasoning.
cosine (correctness mask): Phased exploration intensity; useful for warm-up / cool-down cycles.
round_robin_k (correctness mask): Distributes masking load; smoother than every_n if tasks show periodic sensitivity.
full_correct zero every_n: Strong push to explore beyond mastered cases; works best with moderate formatting emphasis.
soft masking (weight > 0): Middle ground between full reward and complete removal; stabilizes training when hard drops hurt accuracy.


srun --nodelist=slurm-h100-206-067 --pty bash

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1,2,3 uv run main.py --use_vllm --use_liger   --output_dir exp_output/fullzero_every10_fw12_graded   --reward_mask_strategy none   --correctness_weight 1.0 --format_weight 1.2   --full_correct_zero_strategy every_n --full_correct_zero_every_n 10   --format_full_threshold 0.9

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0 uv run vllm_server.py --model Qwen/Qwen2.5-7B-Instruct --port 8000

srun --nodelist=slurm-h100-206-113 --pty bash