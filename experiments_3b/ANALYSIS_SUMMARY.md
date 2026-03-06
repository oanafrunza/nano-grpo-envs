# 3B Analysis Complete - Ready for OOD Evaluation

## Summary of Analysis

### What We Did

1. **Analyzed 3B 7B-Replication Experiments**
   - Compared 4 models using exact 7B configs on 3B architecture
   - Generated comprehensive plots and metrics
   - Results: `experiments_3b/7b_replication_suite/analysis_results/`

2. **Compared All 3B Configurations**
   - Previous 3B configs (6 models) vs 7B-replication configs (4 models)
   - Total: 10 experiment runs analyzed
   - Results: `experiments_3b/3b_config_comparison/`

### Key Findings

#### In-Domain Performance (10 tasks)

**Ranked by Performance:**
1. **Baseline (Previous)**: 26.20% ± 1.30%
2. **Continuous Masking (Previous)**: 25.18% ± 0.14%
3. **Continuous Zeroing (7B-Rep)**: 23.90% ± 2.29%
4. **Phase-Adapt 7B Config (7B-Rep)**: 22.48% ± 0.06%
5. **Phase-Adapt Old Config (Previous)**: 21.36% ± 1.19%

#### Strategy Insights

**Masking vs Zeroing (3B behaves differently than 7B!):**
- For 3B: **Masking-only > Zeroing-only** by 1.28%
- For 7B: **Zeroing-only > Masking-only** (from 7B results)

**Early vs Late Zeroing:**
- Zero@200 > Zero@600 by 1.12% (earlier is better)

**Seed Variance:**
- Most stable: Continuous Masking (0.14% variance)
- Least stable: Continuous Zeroing (2.29% variance)

### Top 3 Configs for OOD Evaluation

Selected best seed per strategy (same approach as 7B):

1. **Baseline - seed 0**
   - `exp_output/science2_3b_suite/baseline_len512_seed0`
   - In-domain: **27.12%** (best overall)
   - Config: No masking/zeroing

2. **Continuous Zero - seed 0**
   - `exp_output/3b_7b_replication/continuous_fullzero_seed0`
   - In-domain: **25.52%** (best continuous)
   - Config: Zero-only, every_n=20

3. **Phase-Adapt (7B config) - seed 0**
   - `exp_output/3b_7b_replication/phase_adapt_exact7b_seed0`
   - In-domain: **22.52%** (best phase-adapt)
   - Config: Split masking + zero@200

## Next Steps: OOD Evaluation

### Run Evaluation

```bash
cd /mnt/home/oana/projects/nano-grpo-envs
sbatch experiments_3b/run_3b_ood_eval.sh
```

**Job Details:**
- GPUs: 2 x H100
- Time: 12 hours
- Memory: 200GB
- Tasks: 31 OOD tasks from reasoning_gym

### After Completion

Once evaluation finishes, analyze results:

```bash
cd /mnt/home/oana/projects/nano-grpo-envs
source .venv/bin/activate
python experiments_3b/analyze_3b_ood_results.py
```

This will:
- Calculate OOD metrics (overall, core, stretch)
- Compare 3B vs 7B OOD performance
- Test if pattern holds: Does continuous strategy help 3B on OOD?
- Generate comparison plots

### Expected Questions to Answer

1. **Does 3B scale similarly to 7B on OOD tasks?**
   - 7B Baseline: 16.23% OOD
   - 3B Baseline: ? (to be measured)

2. **Does continuous zeroing strategy help 3B OOD?**
   - 7B: Continuous Zero improved by +4.1% over baseline
   - 3B: ? (to be measured)

3. **Does phase-adapt help 3B OOD?**
   - 7B: Phase-Adapt improved by +10.7% over baseline
   - 3B: ? (to be measured)

4. **Is the pattern consistent across model sizes?**
   - If yes: Strategy generalizes
   - If no: Need size-specific tuning

## Files Generated

### Analysis Scripts
- `experiments_3b/7b_replication_suite/analyze_3b_results.py` - In-domain analysis
- `experiments_3b/compare_3b_configs.py` - Config comparison
- `experiments_3b/evaluate_3b_ood.py` - OOD setup
- `experiments_3b/analyze_3b_ood_results.py` - OOD analysis

### Results
- `experiments_3b/7b_replication_suite/analysis_results/` - 7B-replication analysis
- `experiments_3b/3b_config_comparison/` - Config comparison results
- `experiments_3b/ood_analysis/` - OOD results (after evaluation)

### Evaluation Config
- `validation/results_3b_ood_best3/eval_config.json` - Best 3 model configs for OOD eval
- `experiments_3b/run_3b_ood_eval.sh` - SLURM submission script

## Key Insights So Far

1. **3B behaves differently than 7B in-domain**
   - 3B prefers masking-only
   - 7B prefers zeroing-only

2. **Baseline is strongest for 3B in-domain**
   - No masking/zeroing: 26.20%
   - But 7B baseline was weakest OOD: 16.23%

3. **Need OOD evaluation to make final conclusions**
   - In-domain performance ≠ OOD performance
   - 7B baseline did well in-domain but worst on OOD stretch tasks

## 7B Reference (for comparison)

From `experiments_7b/paper_ready/`:

| Model | In-Domain | OOD Overall | OOD Core | OOD Stretch |
|-------|-----------|-------------|----------|-------------|
| Baseline | 38.36% | 16.23% | 20.65% | 8.18% |
| Continuous | - | 16.90% | 20.40% | 10.55% |
| Phase-Adapt | - | 17.97% | 20.85% | 12.73% |

**Key 7B Pattern:**
- Phase-Adapt best overall (+10.7% vs baseline)
- Biggest wins on stretch tasks (+55.6%)
- Both strategies improve generalization

**Question:** Will 3B show similar pattern?
