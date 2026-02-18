# Science2 Results: Phase-Adapt vs Continuous vs Baselines

This folder contains aggregated metrics and visualizations comparing the three main experiment sets:

- Phase-Adapt (softmask, diversity bonus, split-masking; lengths 512 and 1024)
- Continuous (fixed 512-length runs)
- Science2 Baselines (baseline_nomask, softmask_every10_wt05, fullzero_every10_nothresh)

## Quick Links
- Metrics CSV: [metrics_comparison.csv](metrics_comparison.csv)
- All-variants pass@1 plot: [pass_at1_comparison.png](pass_at1_comparison.png)
- Top variants overall (one per suite): [top_variants_overall.png](top_variants_overall.png)
- Per-task deltas vs baseline_nomask: [per_task_deltas_vs_baseline_nomask.csv](per_task_deltas_vs_baseline_nomask.csv)
- Per-task comparison (top 3: phase_split_masking L512, cont_softmask_prob_p L512, baseline_nomask L512): [per_task_top3_comparison.png](per_task_top3_comparison.png)

## Headline Results (avg pass@1)
- Phase-Adapt (best): phase_split_masking L512 ≈ 31.54
- Continuous (best): cont_softmask_prob_p L512 ≈ 33.96
- Science2 Baseline (best): baseline_nomask L512 ≈ 37.48

## Key Takeaways
- Overall ranking: baseline_nomask ≥ cont_softmask_prob_p > phase_split_masking (512) > phase_diversity/softmask.
- Best phase-adapt masking: split-masking at 512 achieves the highest pass@1 among phase-adapt variants while preserving formatting (~0.39).
- Length effects: diversity improves at 1024 vs 512; split-masking performs best at 512; softmask is roughly flat across lengths.
- Task-level notes:
  - simple_geometry: baseline_nomask and continuous fullzero/softmask lead; phase-adapt split-masking is competitive at 512.
  - family_relationships: diversity can spike on some runs, but not enough to surpass top baselines overall.
  - bf/sokoban: near-zero across all methods; masking strategy doesn’t materially change outcomes.

## Approach Summary

| Suite | Variant | Length | Phase Logic | Masking Strategy | Notable Settings |
|-------|---------|--------|-------------|------------------|------------------|
| Phase-Adapt | phase_split_masking | 512 | Phase-adaptive training with consolidation detection via completion-NLL moving average; scales `num_chains` after phase switch (keeps length fixed) | Split-masking that alternates reward emphasis across token spans (format vs correctness) to stabilize learning | Diversity bonus off; `<think>/<answer>` format; pass@1 ≈ 31.54, format ≈ 0.391 |
| Continuous | cont_softmask_prob_p | 512 | Single-phase, fixed parameters; no phase detection | Probabilistic softmask (p) that attenuates reward on format-sensitive tokens while keeping signal | Fixed length; strong overall balance; pass@1 ≈ 33.96, format ≈ 0.389 |
| Science2 Baseline | baseline_nomask | 512 | Standard DR-GRPO without scheduling | No masking; all tokens contribute uniformly to reward | Simplest setup; consistently highest pass@1; pass@1 ≈ 37.48, format ≈ 0.377 |

## How to Recreate Figures
Figures were generated from the summaries under:
- Phase-Adapt: ../science2_phase_adapt_suite/*/summary.json
- Continuous: ../science2_cont_suite/*/summary.json
- Science2 Baselines: ../science2_suite/*/summary.json

The aggregation scripts produced the CSVs and plots in this folder. If you want code snippets to regenerate or extend these charts, ping me and I’ll drop a small `analyze_results.py` here.
