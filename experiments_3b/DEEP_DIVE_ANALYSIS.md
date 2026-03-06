# Deep Dive: 3B Experiments Analysis

## The Central Question

**Does the 7B pattern hold for 3B?** If yes → Strong generalization claim for paper

## 7B Pattern (What We're Testing)

### 7B Results Summary

| Metric | Baseline | Continuous | Phase-Adapt | Winner |
|--------|----------|------------|-------------|---------|
| **In-Domain (10 tasks)** | 38.36% | Not tested | Not tested | Baseline |
| **OOD Overall (31 tasks)** | 16.23% | 16.90% (+4.1%) | **17.97% (+10.7%)** | Phase-Adapt |
| **OOD Core (21 tasks)** | 20.65% | 20.40% (-1.2%) | **20.85% (+1.0%)** | Phase-Adapt |
| **OOD Stretch (10 tasks)** | 8.18% | 10.55% (+29.0%) | **12.73% (+55.6%)** | Phase-Adapt |

### Key 7B Observations

1. **In-domain vs OOD divergence**: Baseline strong in-domain but weakest on OOD stretch
2. **Strategy effectiveness scales with difficulty**: 
   - Core tasks: Minimal improvement (+1-2%)
   - Stretch tasks: Massive improvement (+29-56%)
3. **Phase-Adapt > Continuous > Baseline** on OOD
4. **Zeroing-only** strategy won for continuous approach

## 3B Results So Far (In-Domain Only)

### 3B In-Domain Performance (10 tasks)

| Strategy | Seed 0 | Seed 1 | Mean ± Std | Config |
|----------|--------|--------|------------|--------|
| **Baseline** | 27.12% | 25.28% | 26.20% ± 1.30% | No mask/zero |
| **Continuous Mask** | 25.28% | 25.08% | 25.18% ± 0.14% | Masking-only, every_n=20 |
| **Continuous Zero** | 25.52% | 22.28% | 23.90% ± 2.29% | Zero-only, every_n=20 |
| **Phase-Adapt (7B)** | 22.52% | 22.44% | 22.48% ± 0.06% | Split mask + zero@200 |
| **Phase-Adapt (Old)** | 20.52% | 22.20% | 21.36% ± 1.19% | Split mask + zero@600 |

### 3B Key Observations (In-Domain)

1. **Baseline is strongest** (26.20%) - opposite of 7B OOD trend
2. **Masking > Zeroing** for 3B (25.18% vs 23.90%) - **opposite of 7B!**
3. **High seed variance** for Continuous Zero (2.29%) vs very low for Phase-Adapt (0.06%)
4. **Earlier zeroing helps**: Phase-Adapt 7B config (zero@200) > Old config (zero@600)

## Critical Differences: 3B vs 7B

### Configuration Effectiveness

| Strategy | 3B Rank | 7B OOD Rank | Matches? |
|----------|---------|-------------|----------|
| Baseline | **#1** (26.20%) | #3 (16.23%) | ❌ Different |
| Continuous | #2 (25.18% mask) | #2 (16.90% zero) | ⚠️ Different strategy |
| Phase-Adapt | #4-5 (21-22%) | **#1** (17.97%) | ❌ Opposite! |

### Strategy Preferences

| Aspect | 3B (In-Domain) | 7B (OOD) | Match? |
|--------|----------------|----------|---------|
| **Best overall** | Baseline | Phase-Adapt | ❌ |
| **Continuous type** | Masking wins | Zeroing wins | ❌ |
| **Helps vs baseline?** | Hurts (-1.0%) | Helps (+10.7%) | ❌ |
| **Seed stability** | High variance | Not tested | ❓ |

## Hypothesis: Why 3B Differs In-Domain

### Theory 1: In-Domain vs OOD Fundamental Difference
- **In-domain**: Model sees similar tasks during training
  - Baseline memorizes well → high performance
  - Strategies might interfere with memorization
  
- **OOD**: Model faces novel task types
  - Baseline struggles to generalize → low performance
  - Strategies force better generalization → high performance

### Theory 2: Model Size Matters
- **3B**: Smaller capacity
  - Needs simpler strategies (masking better than zeroing?)
  - Less room for complex adaptive mechanisms
  
- **7B**: Larger capacity
  - Can handle complex strategies (zeroing + adaptation)
  - More parameters → better generalization potential

### Theory 3: Training Dynamics
- **Continuous Zero high variance (2.29%)**: Unstable training for 3B
- **Phase-Adapt low variance (0.06%)**: Stable training mechanism
- Maybe 3B needs stability more than 7B?

## What We're Testing Now (OOD Evaluation)

### Selected Models for OOD
1. **Baseline seed0** (27.12% in-domain) - Best baseline
2. **Continuous Zero seed0** (25.52% in-domain) - Best continuous, uses 7B strategy
3. **Phase-Adapt 7B seed0** (22.52% in-domain) - Best phase-adapt

### Critical Test Questions

#### Q1: Does baseline performance collapse on OOD for 3B?
**7B**: Baseline 38.36% → 16.23% (57.7% drop)
- Core: 20.65% (46.2% of in-domain)
- Stretch: 8.18% (21.3% of in-domain)

**3B Prediction**: Baseline 27.12% → ???
- If pattern holds: ~10-12% OOD overall (similar 60% drop)
- If 3B differs: ~15-18% OOD overall (smaller drop)

#### Q2: Does Continuous Zero improve 3B OOD?
**7B**: +4.1% overall, +29% stretch improvement

**3B Scenarios**:
- ✅ **Pattern holds**: Continuous improves by 5-10% on stretch tasks
- ❌ **Pattern breaks**: Continuous doesn't help or hurts performance
- 🤔 **Masking better**: Should've tested masking instead (can test later)

#### Q3: Does Phase-Adapt win on 3B OOD?
**7B**: +10.7% overall, +55.6% stretch improvement

**3B Scenarios**:
- ✅ **Pattern holds**: Phase-Adapt > Continuous > Baseline on OOD
- ❌ **Pattern breaks**: Phase-Adapt ≤ Baseline on OOD
- 📊 **Scaled effect**: Phase-Adapt helps but less than 7B (+5-8% vs +10.7%)

## Expected Outcomes & Paper Implications

### Scenario A: Pattern Fully Holds ⭐⭐⭐
**Results**:
- 3B Baseline: ~10-12% OOD
- 3B Continuous: ~11-13% OOD (+5-10% stretch)
- 3B Phase-Adapt: ~12-15% OOD (+20-30% stretch)

**Paper Claim**: 
> "Our approach generalizes across model sizes (3B-7B). Despite 3B models having different in-domain preferences (masking vs zeroing), the same strategies improve OOD generalization, with consistent patterns: Phase-Adapt > Continuous > Baseline on novel tasks."

**Strength**: 🔥🔥🔥 Very strong generalization claim

### Scenario B: Pattern Partially Holds ⭐⭐
**Results**:
- 3B strategies help but effects are smaller/different
- Maybe Continuous > Phase-Adapt for 3B
- Or Phase-Adapt only helps on some task types

**Paper Claim**:
> "Our approach shows effectiveness across model sizes, though optimal strategies may vary. 3B models benefit from [X] while 7B models benefit more from [Y], suggesting size-dependent tuning is valuable."

**Strength**: 🔥🔥 Good contribution, shows nuance

### Scenario C: Pattern Breaks ⭐
**Results**:
- 3B strategies don't improve OOD or hurt performance
- In-domain rankings match OOD rankings (baseline best)

**Paper Claim**:
> "We find that generalization strategies show size-dependent effects. While effective for 7B models, 3B models may require alternative approaches. This suggests [theory about capacity/training dynamics]."

**Strength**: 🔥 Honest negative result, good science but weaker paper

## Per-Task Pattern Analysis

### 7B Task-Specific Patterns

**Tasks where Phase-Adapt helps most (>20% improvement)**:
- Stretch tasks: Almost all (8/10 tasks)
- Core difficult tasks: Maze, Sokoban, BF

**Tasks where strategies barely help (<5% improvement)**:
- Easy core tasks: Propositional Logic, Number Sequence
- Well-structured tasks: Polynomial Equations

### What to Watch in 3B Results

1. **Stretch task improvement**: Key indicator
   - 7B: +55.6% for Phase-Adapt
   - 3B: Should see similar relative improvement if pattern holds

2. **Task difficulty correlation**:
   - Do strategies help more on harder tasks?
   - Is this consistent across model sizes?

3. **Task type patterns**:
   - Logic/Math vs Spatial/Sequential
   - Structured vs Unstructured

## Variance Analysis Deep Dive

### Seed Variance Comparison

| Strategy | 3B Variance | Interpretation |
|----------|-------------|----------------|
| Baseline | 1.30% | Moderate - expected for baseline |
| Continuous Mask | 0.14% | Very low - stable strategy |
| **Continuous Zero** | **2.29%** | High - unstable for 3B! |
| Phase-Adapt 7B | 0.06% | Extremely low - very stable |
| Phase-Adapt Old | 1.19% | Moderate - less stable |

### Implications

**Continuous Zero instability (2.29% variance)**:
- Seed 0: 25.52% (good)
- Seed 1: 22.28% (poor)
- **Why?** Zeroing might be too aggressive for 3B
- **Alternative**: Masking provides smoother training

**Phase-Adapt stability (0.06% variance)**:
- Seed 0: 22.52%
- Seed 1: 22.44%
- **Why?** Adaptive mechanism self-corrects
- **Benefit**: Reproducible results

## Timeline of 3B Experiments

### Phase 1: Original 3B Experiments
- **Baseline**: 26.20% ± 1.30%
- **Continuous Masking**: 25.18% ± 0.14%
- **Phase-Adapt (zero@600)**: 21.36% ± 1.19%
- **Finding**: Baseline wins, masking better than phase-adapt

### Phase 2: 7B Replication on 3B
- **Continuous Zero**: 23.90% ± 2.29%
- **Phase-Adapt (zero@200)**: 22.48% ± 0.06%
- **Finding**: 7B configs don't improve 3B in-domain performance
- **But**: Earlier zeroing helps (+1.1%), more stable

### Phase 3: OOD Evaluation (Running Now)
- **Testing**: Will 7B OOD pattern emerge for 3B?
- **Models**: Best seed of each strategy
- **Dataset**: Same 31 tasks as 7B

## Statistical Significance Notes

### What Would Convince Us?

**Strong evidence pattern holds**:
- Phase-Adapt beats Baseline by >5% on OOD overall
- Phase-Adapt beats Continuous by >3% on OOD stretch
- Improvements consistent across majority of stretch tasks (>7/10)

**Weak evidence / inconclusive**:
- Improvements 1-3% (within noise)
- Mixed results across task types
- High variability across seeds

**Evidence pattern breaks**:
- Phase-Adapt ≤ Baseline on OOD
- No improvement on stretch tasks
- Different ranking than 7B

## Next Steps After OOD Results

### If Pattern Holds
1. ✅ Write paper with strong generalization claim
2. Test on even smaller model (1B?) to confirm trend
3. Analyze which specific mechanisms transfer across sizes

### If Pattern Partially Holds
1. Deep dive into which aspects transfer
2. Characterize size-dependent effects
3. Develop size-specific tuning guidelines

### If Pattern Breaks
1. Analyze why 3B differs
2. Test alternative strategies (maybe masking-only continuous on OOD?)
3. Investigate capacity vs strategy complexity tradeoff

## Key Metrics to Extract from Results

### Primary Metrics
- [ ] Overall Pass@1 per model
- [ ] Core vs Stretch breakdown
- [ ] Relative improvements (% over baseline)
- [ ] Per-task Pass@1

### Secondary Metrics
- [ ] Format correctness rates
- [ ] Error type distribution
- [ ] Task difficulty correlation
- [ ] Strategy × task type interactions

### Comparison Metrics
- [ ] 3B vs 7B absolute performance
- [ ] 3B vs 7B relative improvements
- [ ] 3B vs 7B scaling factors

## Questions to Answer

1. **Generalization**: Do patterns generalize 7B → 3B?
2. **Scaling**: How does performance scale with model size?
3. **Strategy**: Are optimal strategies size-dependent?
4. **Stability**: Does variance matter more for smaller models?
5. **Mechanisms**: Which strategy components are most important?

---

**Status**: ⏳ Waiting for OOD evaluation results
**ETA**: Check job status with `squeue -u oana` or check logs in `logs/slurm_eval_3b_ood_*.out`
