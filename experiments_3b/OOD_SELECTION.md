# 3B OOD Evaluation - Model Selection

## Selection Approach: Same as 7B

**Method**: Best single seed per strategy (3 models total)

## Selected Models for OOD Evaluation

| Strategy | Model | Checkpoint | In-Domain Pass@1 | Config Details |
|----------|-------|------------|------------------|----------------|
| **Baseline** | seed 0 | `science2_3b_suite/baseline_len512_seed0` | **27.12%** | No masking/zeroing |
| **Continuous** | seed 0 | `3b_7b_replication/continuous_fullzero_seed0` | **25.52%** | Zero-only, every_n=20 |
| **Phase-Adapt** | seed 0 | `3b_7b_replication/phase_adapt_exact7b_seed0` | **22.52%** | Split mask + zero@200 |

## Why These Configs?

### Baseline
- **Best performing overall** on in-domain (27.12%)
- Provides clean baseline for comparison
- Same as 7B approach (used best baseline seed)

### Continuous Zero (not Continuous Mask!)
- **Best continuous strategy**: 25.52% vs 25.18% for masking-only
- Uses 7B-winning strategy (zeroing) rather than 3B-winning strategy (masking)
- **Critical test**: Does 7B's best strategy work for 3B on OOD?

### Phase-Adapt 7B Config
- **Best phase-adapt variant**: 22.52%
- Uses 7B configuration (zero@200 vs zero@600)
- Tests if 7B's winning config translates to 3B

## Comparison with Available Alternatives

| Strategy | Config | Seed 0 | Seed 1 | Selected |
|----------|--------|--------|--------|----------|
| Baseline | No mask/zero | 27.12% | 25.28% | ✅ Seed 0 |
| Continuous | Masking-only | 25.28% | 25.08% | ❌ Not selected |
| Continuous | Zero-only | **25.52%** | 22.28% | ✅ Seed 0 |
| Phase-Adapt | 7B config | **22.52%** | 22.44% | ✅ Seed 0 |
| Phase-Adapt | Old config | 20.52% | 22.20% | ❌ Not selected |

## Key Decision: Zero vs Mask for Continuous

**In-domain results (3B):**
- Continuous Masking: 25.18% (better)
- Continuous Zeroing: 23.90% (worse)

**But we selected Zeroing because:**
1. **7B pattern**: Zeroing won on OOD (+29% stretch improvement)
2. **Hypothesis test**: Will 7B's OOD winner also win for 3B on OOD?
3. **Seed 0 zeroing**: 25.52% (actually better than any masking seed!)

## Expected Outcomes

### If pattern holds (3B behaves like 7B):
- Continuous Zero improves over Baseline on OOD
- Phase-Adapt improves even more (especially stretch tasks)
- 3B OOD ~60% of 7B OOD (scaled by model size)

### If pattern breaks (3B differs from 7B):
- Continuous Zero might not help 3B on OOD
- Would suggest model-size-specific strategies needed
- Masking-only might have been better choice for 3B

## Evaluation Command

```bash
cd /mnt/home/oana/projects/nano-grpo-envs
sbatch experiments_3b/run_3b_ood_eval.sh
```

## Analysis Command (after evaluation)

```bash
cd /mnt/home/oana/projects/nano-grpo-envs
python experiments_3b/analyze_3b_ood_results.py
```

This will directly compare:
- 3B Baseline vs 7B Baseline
- 3B Continuous vs 7B Continuous  
- 3B Phase-Adapt vs 7B Phase-Adapt
