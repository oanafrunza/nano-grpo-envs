# 3B Experiments: Comprehensive Understanding Summary

**Date:** March 5, 2024  
**Status:** OOD Evaluation Running (Job 33491, ~6h/12h elapsed)  
**Goal:** Test if 7B generalization patterns hold for 3B models

---

## Quick Reference

### What We Know (In-Domain Results)

| Configuration | In-Domain Pass@1 | Variance | Rank |
|---------------|------------------|----------|------|
| Baseline (seed0) | 27.12% | ±1.30% | 🥇 1st |
| Continuous Mask | 25.18% | ±0.14% | 🥈 2nd |
| Continuous Zero | 23.90% | ±2.29% | 🥉 3rd |
| Phase-Adapt 7B | 22.48% | ±0.06% | 4th |

### What We're Testing (OOD Evaluation)

**Selected Models:**
1. Baseline seed0 (27.12% in-domain)
2. Continuous Zero seed0 (25.52% in-domain)
3. Phase-Adapt 7B seed0 (22.52% in-domain)

**Key Question:** Will rankings reverse on OOD like they did for 7B?

---

## The Pattern We're Testing

### 7B Pattern (Established)

```
In-Domain:  Baseline best (38.36%)
OOD:        Phase-Adapt best (17.97%)
Effect:     +10.7% overall, +55.6% on stretch tasks
```

**Key insight:** Strategies hurt in-domain but help OOD, especially on hard tasks.

### 3B Observations So Far

```
In-Domain:  Baseline best (26.20%)
            Strategies HURT performance (-1.0% to -3.7%)
OOD:        ??? ← RUNNING NOW
```

**Key question:** Will the same OOD reversal occur?

---

## Three Possible Outcomes

### Outcome 1: Pattern Holds ✅ (Best for Paper)

**Expected Results:**
```
                      Overall    Stretch    vs Baseline
Baseline:             ~11%      ~5.6%      reference
Continuous Zero:      ~12%      ~7%        +4%, +27% stretch
Phase-Adapt:          ~13%      ~8.7%      +11%, +55% stretch
```

**Implications:**
- ✨ **Strong generalization claim** across model sizes
- ✨ Fundamental principle of GRPO reward shaping
- ✨ Strategies provide robust OOD improvements regardless of size
- 📝 Paper: "Adaptive Reward Masking for Robust LLM Alignment Across Model Scales"

### Outcome 2: Pattern Scales ⚖️ (Good for Paper)

**Expected Results:**
```
                      Overall    Stretch    vs Baseline
Baseline:             ~11%      ~5.6%      reference
Continuous Zero:      ~11.5%    ~6%        +2%, +9% stretch
Phase-Adapt:          ~12%      ~6.5%      +5%, +18% stretch
```

**Implications:**
- ✓ Same ranking but weaker effects
- ✓ Strategies help but need size-specific tuning
- ✓ Characterizes capacity-dependent scaling
- 📝 Paper: "Scale-Aware Reward Shaping for LLM Alignment"

### Outcome 3: Pattern Breaks ❌ (Interesting Science)

**Expected Results:**
```
                      Overall    Stretch    vs Baseline
Baseline:             ~11%      ~5.6%      best (unchanged)
Phase-Adapt:          ~10%      ~5%        -9%
Continuous Zero:      ~10.5%    ~5.2%      -5%
```

**Implications:**
- ⚠️ In-domain rankings preserved on OOD
- ⚠️ Strategies require minimum model capacity (≥7B)
- ⚠️ Need different approaches for smaller models
- 📝 Paper: "Capacity Requirements for Adaptive Reward Masking"

---

## Why Each Metric Matters

### 1. Overall OOD Pass@1
- **Tests:** General robustness across all 31 tasks
- **7B Improvement:** +10.7% for Phase-Adapt
- **Threshold:** +5% = good, +10% = strong generalization

### 2. Stretch Task Performance  
- **Tests:** Hardest OOD tasks (our key discriminator)
- **7B Improvement:** +55.6% for Phase-Adapt
- **Threshold:** +30% = good, +50% = pattern holds strongly

### 3. Core Task Performance
- **Tests:** Easier OOD tasks
- **7B:** Strategies ~same as baseline (20.4-20.8%)
- **Expected:** Minimal differences (strategies don't hurt core)

### 4. Continuous Zero vs Baseline
- **Tests:** If zeroing catches up on OOD (like 7B)
- **7B:** +4.1% overall despite worse in-domain
- **Key:** Does 3B show same masking→zeroing OOD preference?

---

## Key Scientific Questions

### 1. Does OOD Reversal Generalize?

**7B Pattern:**
```
In-Domain: Baseline > Strategies
OOD:       Strategies > Baseline
```

**Question:** Is this size-invariant or capacity-dependent?

**Why it matters:** If invariant → fundamental RL principle. If dependent → need size-specific methods.

### 2. Masking vs Zeroing Strategy

**Current Evidence:**
- 7B OOD: Zeroing wins
- 3B In-Domain: Masking wins

**Question:** Does zeroing always win OOD, or only for large models?

**Why it matters:** Zeroing is simpler but harsher. Need to know when to use which.

### 3. Phase-Adapt Stability

**Observed:** Phase-Adapt has lowest variance (0.06% vs 2.29%)

**Question:** Does stability translate to consistent OOD gains across sizes?

**Why it matters:** Stability valuable for production, but not if it sacrifices performance.

### 4. Scaling Properties

**Observed:** 3B = 68% of 7B in-domain performance

**Question:** Does this ratio hold for OOD? For improvements?

**Why it matters:** Helps predict larger model behavior, guides model selection.

---

## What We've Learned Already

### 1. In-Domain Consistency ✓

**Both 3B and 7B:** Baseline is best in-domain

This suggests strategies don't improve on training-similar tasks, regardless of size.

### 2. Strategy Trade-offs

**3B observations:**
- Continuous: -1.0% in-domain, stable (±0.14%)
- Phase-Adapt: -3.7% in-domain, very stable (±0.06%)

Strategies trade performance for stability in-domain. Question: Do they get OOD robustness too?

### 3. Training Dynamics

**Continuous Zero:** High variance (±2.29%) → unstable training

**Phase-Adapt:** Low variance (±0.06%) → adaptive schedule works well

Adaptive mechanisms provide training stability across sizes.

### 4. Size Effects

**3B vs 7B in-domain:** 26.20% vs 38.36% (68% ratio)

Consistent with general LLM scaling. Key question: Does OOD ratio differ?

---

## Practical Implications by Outcome

### If Pattern Holds (Outcome 1)

**For Practitioners:**
- Use Phase-Adapt or Continuous Zero for OOD robustness
- Accept minor in-domain degradation
- Works across model sizes → no retuning needed

**For Deployment:**
- Can use smaller models (3B) with strategies for diverse production
- Cost-effective OOD robustness
- No need for expensive 7B+ models if task diversity is primary concern

### If Pattern Scales (Outcome 2)

**For Practitioners:**
- Strategies help but need size-specific tuning
- Larger models benefit more
- Cost-benefit analysis needed per deployment

**For Deployment:**
- May need 7B for strong OOD robustness
- 3B with strategies as middle ground
- Consider task distribution: narrow → baseline, diverse → strategies

### If Pattern Breaks (Outcome 3)

**For Practitioners:**
- Use strategies only for models ≥7B
- Smaller models: stick with baseline or find alternative methods
- Research needed for small-model OOD robustness

**For Deployment:**
- Need larger models for OOD robustness
- 3B models for in-domain, narrow tasks only
- May need ensemble or other techniques for 3B OOD

---

## Next Steps (Waiting for Results)

### Immediate (When Job Completes)

1. **Check results:** `cat validation/results_3b_ood_best3/results.csv`
2. **Run analysis:** `python experiments_3b/analyze_3b_ood_results.py`
3. **Review plots:** Check `experiments_3b/3b_ood_analysis/` directory
4. **Compare to 7B:** Load 7B OOD results and compare patterns

### Analysis Checklist

- [ ] Overall OOD pass@1 rankings
- [ ] Core vs Stretch task breakdown
- [ ] Per-task performance (which tasks do strategies help?)
- [ ] Relative improvements vs baseline
- [ ] Comparison to 7B pattern
- [ ] Statistical significance tests

### Decision Points

**If Pattern Holds:**
- ✅ Proceed to paper writing
- Consider testing Continuous Mask OOD (since it was better in-domain)
- Focus on "why" the pattern holds (mechanistic analysis)

**If Pattern Scales:**
- 🔬 Characterize scaling properties more precisely
- Test intermediate size (5B?) to see scaling curve
- Investigate what hyperparameters need size-tuning

**If Pattern Breaks:**
- 🔍 Deep dive into why (gradient analysis, attention patterns)
- Test what's different about 3B (capacity? architecture?)
- Explore alternative strategies for smaller models

---

## Documents Created

1. **[ANALYSIS_SUMMARY.md](ANALYSIS_SUMMARY.md)** - Overview of 3B experiments
2. **[OOD_SELECTION.md](OOD_SELECTION.md)** - Rationale for model selection
3. **[DEEP_DIVE_ANALYSIS.md](DEEP_DIVE_ANALYSIS.md)** - Detailed patterns & hypotheses
4. **[PATTERN_ANALYSIS.md](PATTERN_ANALYSIS.md)** - Expected outcomes by scenario
5. **[WHY_THIS_MATTERS.md](WHY_THIS_MATTERS.md)** - Scientific significance
6. **[THIS FILE]** - Comprehensive understanding summary

### Visualizations

1. **pattern_hypothesis_visualization.png** - Full comparison of 3 outcomes
2. **stretch_task_hypothesis.png** - Focus on key discriminator metric

---

## Current Status

```bash
# Job Status
Job ID: 33491
Status: Running (~6h/12h elapsed)
Resources: 2x H100 GPUs, 200GB RAM
Command: evaluate_models.py on 31 OOD tasks

# Results Location
Config: validation/results_3b_ood_best3/eval_config.json
Output: validation/results_3b_ood_best3/results.csv
Logs: logs/slurm_eval_3b_ood_33491.{out,err}

# Check Status
squeue -u oana | grep eval_3b
tail -f logs/slurm_eval_3b_ood_33491.out
```

---

## Summary of Understanding

### What We Know
- ✅ 3B in-domain results: Baseline best, strategies hurt performance
- ✅ 7B OOD pattern: Phase-Adapt best, especially on stretch tasks
- ✅ Masking vs Zeroing: Different preferences for 3B in-domain
- ✅ Training stability: Phase-Adapt most stable across sizes

### What We're Testing
- ❓ Do 3B rankings reverse on OOD like 7B?
- ❓ Do strategies improve stretch tasks for 3B?
- ❓ Does zeroing catch up on OOD for 3B?
- ❓ Is the pattern size-invariant or capacity-dependent?

### Why It Matters
- 🎯 Tests if we found a fundamental principle vs size-specific trick
- 🎯 Determines paper narrative (strong generalization vs characterization)
- 🎯 Guides practical deployment (when to use which strategy)
- 🎯 Opens theoretical questions (why does OOD reversal happen?)

### Expected Timeline
- ⏳ Job completes in ~6 hours
- 📊 Analysis ready within 1 hour of completion
- ✍️ Paper narrative clear based on results
- 🚀 Next steps determined by outcome

---

## The Bottom Line

We're not just benchmarking 3B models on OOD tasks.

We're testing whether **reward masking/zeroing strategies generalize across model scales**, which would establish a **fundamental principle** of RL fine-tuning for LLMs.

The key discriminator is **stretch task performance**:
- If Phase-Adapt improves stretch tasks significantly → Pattern holds
- If improvements scale but remain positive → Pattern scales  
- If Baseline stays best → Pattern breaks

Any outcome is scientifically valuable, but Outcome 1 (pattern holds) gives the strongest paper.

**Now we wait for the data to tell us which story to tell.**
