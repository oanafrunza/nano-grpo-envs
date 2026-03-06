# Pattern Analysis: 3B vs 7B Models

## Executive Summary

We are testing whether the **7B generalization pattern** holds for **3B models**. This is critical for the paper's strength—if the pattern holds across model sizes, it suggests a fundamental principle of GRPO training with masking/zeroing strategies.

---

## The 7B Pattern (Established)

### Key Finding: **In-Domain vs OOD Reversal**

**In-Domain Performance (10 tasks):**
- Baseline: **38.36%** ← Best

**OOD Performance (31 tasks):**
```
                    Overall    Core(21)   Stretch(10)   Stretch Improvement
Baseline:           16.23%     20.65%     8.18%         baseline
Continuous Zero:    16.90%     20.40%     10.55%        +29.0%
Phase-Adapt:        17.97%     20.85%     12.73%        +55.6%
```

**Pattern**: Baseline wins in-domain → Phase-Adapt wins OOD (especially on stretch tasks)

---

## Current 3B Results (In-Domain Only)

### Performance Rankings

```
                        In-Domain     Variance    Status
Baseline:               26.20%        ±1.30%      ✓ Best
Continuous Mask:        25.18%        ±0.14%      ✓ Strong
Continuous Zero:        23.90%        ±2.29%      ⚠ Unstable
Phase-Adapt (7B):       22.48%        ±0.06%      ✓ Most stable
Phase-Adapt (Old):      21.36%        ±1.19%      ✗ Worst
```

### Critical Observations

1. **Baseline is best** (consistent with 7B)
2. **Strategies hurt in-domain performance** (not tested for 7B)
3. **Masking > Zeroing** for 3B (opposite of 7B OOD)
4. **Phase-Adapt has lowest variance** (0.06% vs 2.29% for Continuous)

---

## What We're Testing (OOD Evaluation Running)

### Models Selected

1. **Baseline** (seed0): 27.12% in-domain
2. **Continuous Zero** (seed0): 25.52% in-domain  
3. **Phase-Adapt 7B** (seed0): 22.52% in-domain

*Note: Selected Continuous Zero (not Mask) to test 7B zeroing pattern*

### Three Possible Outcomes

#### Outcome 1: Pattern Holds (Best for Paper)
```
3B OOD Results:
                      Overall    Core      Stretch    vs Baseline
Baseline:             ~11%      ~14%      ~5.5%      baseline
Continuous Zero:      ~12%      ~14%      ~7%        +4%, +27% stretch
Phase-Adapt:          ~13%      ~14%      ~8.5%      +11%, +55% stretch
```

**Implications:**
- ✅ Strong generalization claim across model sizes
- ✅ Strategies consistently improve OOD, especially stretch tasks
- ✅ Pattern is size-invariant (fundamental principle)
- 📝 Paper narrative: "Our approach generalizes across 3B-7B models"

#### Outcome 2: Pattern Scales (Good for Paper)
```
3B OOD Results:
                      Overall    Core      Stretch    vs Baseline
Baseline:             ~11%      ~14%      ~5.5%      baseline
Continuous Zero:      ~11.5%    ~14%      ~6%        +2%, +9% stretch
Phase-Adapt:          ~12%      ~14%      ~6.5%      +5%, +18% stretch
```

**Implications:**
- ✓ Same ranking but smaller effects
- ✓ Size-dependent but consistent direction
- ✓ Strategies still help, just need tuning
- 📝 Paper narrative: "Benefits scale with model capacity"

#### Outcome 3: Pattern Breaks (Interesting but Weaker)
```
3B OOD Results:
                      Overall    Core      Stretch    vs Baseline
Baseline:             ~11%      ~14%      ~5.5%      best (unchanged)
Phase-Adapt:          ~10%      ~13%      ~5%        -9%
Continuous Zero:      ~10.5%    ~13.5%    ~5.2%      -5%
```

**Implications:**
- ⚠ In-domain rankings match OOD rankings
- ⚠ Strategies don't help smaller models
- ⚠ May require minimum model capacity
- 📝 Paper narrative: "Effective for models ≥7B parameters"

---

## Key Questions to Answer

### 1. Does the ranking reverse?
- **7B**: Baseline best in-domain → Phase-Adapt best OOD
- **3B**: Baseline best in-domain → ??? OOD

### 2. Where do improvements come from?
- **7B**: Phase-Adapt gains +55.6% on stretch tasks
- **3B**: Will we see similar stretch task improvements?

### 3. Does zeroing beat masking OOD?
- **7B OOD**: Continuous Zero (16.90%) used
- **3B In-Domain**: Continuous Mask (25.18%) > Zero (23.90%)
- **3B OOD**: Will Zero catch up like in 7B?

### 4. Is Phase-Adapt stability model-size invariant?
- **Both 3B & 7B**: Phase-Adapt shows lowest variance
- **Question**: Does this translate to consistent OOD gains?

---

## Scaling Analysis

### Performance Ratios
```
3B / 7B In-Domain: 26.20% / 38.36% = 68.3%
Expected 3B OOD: 16.23% × 0.683 = 11.1% (baseline)
```

If pattern holds with same relative improvements:
- 3B Continuous Zero: 11.1% × 1.041 = **11.6%**
- 3B Phase-Adapt: 11.1% × 1.107 = **12.3%**

### Stretch Task Analysis
```
7B Stretch Baseline: 8.18%
7B Stretch Phase-Adapt: 12.73% (+55.6%)

Expected 3B Stretch Baseline: 8.18% × 0.683 = 5.6%
Expected 3B Stretch Phase-Adapt: 5.6% × 1.556 = 8.7%
```

---

## Paper Implications by Outcome

### Strong Paper (Outcome 1)
**Title Angle**: "Adaptive Reward Masking for Robust LLM Alignment Across Model Scales"
- Pattern holds across 3B-7B (tested)
- Likely holds for larger models (extrapolated)
- Fundamental principle, not size-specific trick

### Good Paper (Outcome 2)
**Title Angle**: "Scale-Adaptive Reward Strategies for LLM Alignment"
- Consistent direction, variable magnitude
- Size-specific tuning beneficial
- Framework for scaling strategies

### Interesting Science (Outcome 3)
**Title Angle**: "Capacity Requirements for Adaptive Reward Masking in LLM Alignment"
- Identifies minimum viable model size
- Characterizes capacity-dependent effects
- Still valuable for practitioners

---

## What Happens Next

1. **Job completes** → Check results in `validation/results_3b_ood_best3/`
2. **Run analysis** → `python experiments_3b/analyze_3b_ood_results.py`
3. **Compare patterns** → Generate 3B vs 7B comparison plots
4. **Interpret results** → Map to one of three outcomes above
5. **Decide next steps**:
   - If Outcome 1/2: Proceed to writing, maybe test Continuous Mask OOD
   - If Outcome 3: Need to understand why (analyze gradients, attention, etc.)

---

## Current Status

- ⏳ OOD evaluation running (submitted via SLURM)
- ✅ In-domain analysis complete
- ✅ Hypotheses formulated
- ✅ Comparison framework ready
- ⏳ Waiting for results to validate pattern

**Check status**: `squeue -u oana | grep eval_3b`
**Check logs**: `tail -f logs/slurm_eval_3b_ood_*.out`
