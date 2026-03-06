"""
Comprehensive 3B Model Overview: All Checkpoints In-Domain and OOD Performance
With sanity checks for model integrity and evaluation correctness
"""

import json
import pandas as pd
import os
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("/mnt/home/oana/projects/nano-grpo-envs")

print('='*100)
print('COMPREHENSIVE 3B MODEL OVERVIEW')
print('='*100)
print()

# ============================================================================
# PART 1: COLLECT ALL IN-DOMAIN RESULTS
# ============================================================================

print('PART 1: IN-DOMAIN PERFORMANCE (10 reasoning_gym tasks)')
print('='*100)
print()

experiments = []

# Previous 3B experiments (science2_3b_suite)
previous_configs = [
    ('baseline_len512_seed0', 'Baseline', 0, 'Previous'),
    ('baseline_len512_seed1', 'Baseline', 1, 'Previous'),
    ('continuous_best_len512_seed0', 'Continuous (Mask)', 0, 'Previous'),
    ('continuous_best_len512_seed1', 'Continuous (Mask)', 1, 'Previous'),
    ('phase_adapt_best_len512_seed0', 'Phase-Adapt (Old)', 0, 'Previous'),
    ('phase_adapt_best_len512_seed1', 'Phase-Adapt (Old)', 1, 'Previous'),
]

for exp_name, strategy, seed, source in previous_configs:
    summary_path = PROJECT_ROOT / f"exp_output/science2_3b_suite/{exp_name}/summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
            experiments.append({
                'experiment': exp_name,
                'strategy': strategy,
                'seed': seed,
                'source': source,
                'pass@1': summary['pass_at_k'],  # Already a percentage
                'format_reward': summary['avg_format_reward'] * 100,
                'n_problems': len(summary.get('per_problem_type', {})),
                'checkpoint_exists': (PROJECT_ROOT / f"exp_output/science2_3b_suite/{exp_name}/checkpoint_final").exists()
            })
    else:
        print(f"⚠️  WARNING: Missing summary for {exp_name}")

# 7B-Replication experiments (3b_7b_replication)
replication_configs = [
    ('continuous_fullzero_seed0', 'Continuous (Zero)', 0, '7B-Replication'),
    ('continuous_fullzero_seed1', 'Continuous (Zero)', 1, '7B-Replication'),
    ('phase_adapt_exact7b_seed0', 'Phase-Adapt (7B)', 0, '7B-Replication'),
    ('phase_adapt_exact7b_seed1', 'Phase-Adapt (7B)', 1, '7B-Replication'),
]

for exp_name, strategy, seed, source in replication_configs:
    summary_path = PROJECT_ROOT / f"exp_output/3b_7b_replication/{exp_name}/summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
            experiments.append({
                'experiment': exp_name,
                'strategy': strategy,
                'seed': seed,
                'source': source,
                'pass@1': summary['pass_at_k'],  # Already a percentage
                'format_reward': summary['avg_format_reward'] * 100,
                'n_problems': len(summary.get('per_problem_type', {})),
                'checkpoint_exists': (PROJECT_ROOT / f"exp_output/3b_7b_replication/{exp_name}/checkpoint_final").exists()
            })
    else:
        print(f"⚠️  WARNING: Missing summary for {exp_name}")

df_indomain = pd.DataFrame(experiments)

# Display in-domain results
print(f"{'Strategy':<25} {'Seed':<6} {'Source':<18} {'Pass@1':<10} {'Format':<10} {'Checkpoint':<12}")
print('-'*100)
for _, row in df_indomain.iterrows():
    checkpoint_status = '✓' if row['checkpoint_exists'] else '✗ MISSING'
    print(f"{row['strategy']:<25} {row['seed']:<6} {row['source']:<18} {row['pass@1']:>7.2f}%  {row['format_reward']:>7.2f}%  {checkpoint_status:<12}")

print()
print('Summary by Strategy (averaged across seeds):')
print('-'*100)
strategy_summary = df_indomain.groupby('strategy').agg({
    'pass@1': ['mean', 'std'],
    'format_reward': 'mean',
    'checkpoint_exists': 'sum'
}).round(2)
print(strategy_summary.to_string())

# ============================================================================
# PART 2: OOD EVALUATION RESULTS
# ============================================================================

print()
print('='*100)
print('PART 2: OOD PERFORMANCE (31 reasoning_gym tasks)')
print('='*100)
print()

# Check which OOD evaluations exist
ood_dirs = [
    ('results_3b_31task', 'Previous 3B (Feb 17)'),
    ('results_3b_ood_best3', 'New 3B Best 3 (Mar 5)'),
    ('results_3b_ood_top2', 'Top 2 configs'),
]

ood_results = []

for dir_name, description in ood_dirs:
    ood_path = PROJECT_ROOT / f"validation/{dir_name}"
    if ood_path.exists():
        # Load summary files
        model_summary = ood_path / "summary_per_model.csv"
        split_summary = ood_path / "summary_per_split.csv"
        
        if model_summary.exists() and split_summary.exists():
            df_model = pd.read_csv(model_summary)
            df_split = pd.read_csv(split_summary)
            
            print(f"\n{description}:")
            print(f"Location: {dir_name}")
            print('-'*100)
            
            # Display per-model results
            print(f"{'Model':<30} {'Overall':<12} {'Core (21)':<12} {'Stretch (10)':<15} {'Format OK':<12}")
            print('-'*100)
            
            for _, model_row in df_model.iterrows():
                model_name = model_row['model']
                overall = model_row['pass1_mean'] * 100
                format_ok = model_row['format_ok_mean'] * 100
                
                # Get split results
                core_row = df_split[(df_split['model'] == model_name) & (df_split['split'] == 'core')]
                stretch_row = df_split[(df_split['model'] == model_name) & (df_split['split'] == 'stretch')]
                
                core = core_row['pass1_mean'].values[0] * 100 if len(core_row) > 0 else 0
                stretch = stretch_row['pass1_mean'].values[0] * 100 if len(stretch_row) > 0 else 0
                
                print(f"{model_name:<30} {overall:>9.2f}%  {core:>9.2f}%  {stretch:>12.2f}%  {format_ok:>9.2f}%")
                
                # Store for later comparison
                ood_results.append({
                    'evaluation': description,
                    'model': model_name,
                    'overall': overall,
                    'core': core,
                    'stretch': stretch,
                    'format_ok': format_ok
                })

# ============================================================================
# PART 3: SANITY CHECKS
# ============================================================================

print()
print('='*100)
print('PART 3: SANITY CHECKS & VALIDATION')
print('='*100)
print()

print('CHECK 1: Checkpoint Integrity')
print('-'*100)
missing_checkpoints = df_indomain[~df_indomain['checkpoint_exists']]
if len(missing_checkpoints) > 0:
    print(f"⚠️  WARNING: {len(missing_checkpoints)} checkpoints missing!")
    for _, row in missing_checkpoints.iterrows():
        print(f"   - {row['experiment']}")
else:
    print("✓ All checkpoints exist")

print()
print('CHECK 2: Format Reward (should be high for valid models)')
print('-'*100)
low_format = df_indomain[df_indomain['format_reward'] < 70]
if len(low_format) > 0:
    print(f"⚠️  WARNING: {len(low_format)} models have low format reward (<70%):")
    for _, row in low_format.iterrows():
        print(f"   - {row['experiment']}: {row['format_reward']:.1f}%")
else:
    print("✓ All models have format reward >= 70%")

print()
print('CHECK 3: OOD Format Reward (should be high)')
print('-'*100)
if ood_results:
    df_ood = pd.DataFrame(ood_results)
    low_format_ood = df_ood[df_ood['format_ok'] < 70]
    if len(low_format_ood) > 0:
        print(f"⚠️  WARNING: {len(low_format_ood)} OOD evaluations have low format reward (<70%):")
        for _, row in low_format_ood.iterrows():
            print(f"   - {row['evaluation']} | {row['model']}: {row['format_ok']:.1f}%")
    else:
        print("✓ All OOD evaluations have format reward >= 70%")

print()
print('CHECK 4: Performance Consistency (in-domain vs baseline)')
print('-'*100)
baseline_mean = df_indomain[df_indomain['strategy'] == 'Baseline']['pass@1'].mean()
print(f"Baseline mean: {baseline_mean:.2f}%")
for strategy in df_indomain['strategy'].unique():
    if strategy != 'Baseline':
        strat_mean = df_indomain[df_indomain['strategy'] == strategy]['pass@1'].mean()
        diff = strat_mean - baseline_mean
        print(f"{strategy:<25}: {strat_mean:>6.2f}% ({diff:+.2f}% vs baseline)")

print()
print('CHECK 5: Seed Variance (should be reasonable)')
print('-'*100)
for strategy in df_indomain['strategy'].unique():
    strat_data = df_indomain[df_indomain['strategy'] == strategy]
    if len(strat_data) >= 2:
        variance = strat_data['pass@1'].std()
        print(f"{strategy:<25}: σ = {variance:.2f}%", end='')
        if variance > 2.0:
            print(" ⚠️  HIGH VARIANCE")
        else:
            print(" ✓")

print()
print('CHECK 6: OOD Model Matching (verify correct checkpoints used)')
print('-'*100)

# Check which checkpoints were used in latest OOD evaluation
ood_config_path = PROJECT_ROOT / "validation/results_3b_ood_best3/eval_config.json"
if ood_config_path.exists():
    with open(ood_config_path) as f:
        ood_config = json.load(f)
        print("Models evaluated in results_3b_ood_best3:")
        for model in ood_config['models']:
            print(f"  {model['name']:<30} -> {model['checkpoint']}")
            # Check if checkpoint exists
            checkpoint_path = PROJECT_ROOT / model['checkpoint']
            if not checkpoint_path.exists():
                print(f"    ⚠️  WARNING: Checkpoint does not exist!")
            else:
                # Check if it's a valid model directory
                config_file = checkpoint_path / "config.json"
                if config_file.exists():
                    print(f"    ✓ Valid checkpoint")
                else:
                    print(f"    ⚠️  WARNING: Missing config.json")

# ============================================================================
# PART 4: COMPARISON SUMMARY
# ============================================================================

print()
print('='*100)
print('PART 4: KEY FINDINGS SUMMARY')
print('='*100)
print()

print('IN-DOMAIN RANKINGS (10 tasks):')
print('-'*100)
indomain_rankings = df_indomain.groupby('strategy')['pass@1'].mean().sort_values(ascending=False)
for i, (strategy, score) in enumerate(indomain_rankings.items(), 1):
    print(f"{i}. {strategy:<25}: {score:.2f}%")

print()
print('OOD RANKINGS (31 tasks - Latest Evaluation):')
print('-'*100)
if ood_results:
    latest_ood = [r for r in ood_results if 'Mar 5' in r['evaluation']]
    if latest_ood:
        latest_df = pd.DataFrame(latest_ood).sort_values('overall', ascending=False)
        for i, (_, row) in enumerate(latest_df.iterrows(), 1):
            print(f"{i}. {row['model']:<30}: {row['overall']:.2f}% overall ({row['stretch']:.2f}% stretch)")

print()
print('STRETCH TASK PATTERN (Key Finding):')
print('-'*100)
if latest_ood:
    baseline_stretch = latest_df[latest_df['model'].str.contains('baseline', case=False)]['stretch'].values
    if len(baseline_stretch) > 0:
        baseline_stretch = baseline_stretch[0]
        print(f"Baseline stretch:     {baseline_stretch:.2f}%")
        for _, row in latest_df.iterrows():
            if 'baseline' not in row['model'].lower():
                improvement = ((row['stretch'] - baseline_stretch) / baseline_stretch) * 100
                print(f"{row['model']:<30}: {row['stretch']:.2f}% ({improvement:+.1f}%)")

print()
print('='*100)
print('VALIDATION COMPLETE')
print('='*100)
