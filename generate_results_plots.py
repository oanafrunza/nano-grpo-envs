#!/usr/bin/env python3
"""
Generate publication-quality plots for 3B and 7B experiment results.
Creates plots suitable for PowerPoint presentations.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Color scheme
COLORS = {
    'baseline': '#2ecc71',
    'continuous': '#3498db',
    'phase_adapt': '#e74c3c',
    '3b': '#9b59b6',
    '7b': '#f39c12'
}

def load_summary(path: str) -> Dict:
    """Load a summary.json file."""
    with open(path, 'r') as f:
        return json.load(f)

def aggregate_seeds(exp_paths: List[str]) -> Dict:
    """Aggregate results across multiple seeds."""
    summaries = [load_summary(p) for p in exp_paths]
    
    # Calculate mean and std for overall metrics
    pass_at_k_values = [s['pass_at_k'] for s in summaries]
    
    result = {
        'pass_at_k_mean': np.mean(pass_at_k_values),
        'pass_at_k_std': np.std(pass_at_k_values),
        'num_seeds': len(summaries),
        'per_problem_type': {}
    }
    
    # Aggregate per-problem-type results
    all_problems = set()
    for s in summaries:
        all_problems.update(s['per_problem_type'].keys())
    
    for problem in all_problems:
        pass_rates = []
        for s in summaries:
            if problem in s['per_problem_type']:
                pass_rates.append(s['per_problem_type'][problem]['pass_at_1'])
        
        if pass_rates:
            result['per_problem_type'][problem] = {
                'pass_at_1_mean': np.mean(pass_rates),
                'pass_at_1_std': np.std(pass_rates) if len(pass_rates) > 1 else 0,
                'num_seeds': len(pass_rates)
            }
    
    return result

def plot_overall_comparison_3b(results: Dict[str, Dict], output_dir: Path):
    """Plot overall performance comparison for 3B model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = list(results.keys())
    means = [results[m]['pass_at_k_mean'] for m in methods]
    stds = [results[m]['pass_at_k_std'] for m in methods]
    
    x = np.arange(len(methods))
    colors = [COLORS.get(m.split('_')[0], '#95a5a6') for m in methods]
    
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color=colors, 
                   edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                f'{mean:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel('Pass@1 (%)', fontweight='bold')
    ax.set_title('Overall Performance: Qwen2.5-3B-Instruct (10-Task Suite)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in methods])
    ax.set_ylim(0, max(means) * 1.3)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '3b_overall_comparison.png', bbox_inches='tight')
    print(f"Saved: {output_dir / '3b_overall_comparison.png'}")
    plt.close()

def plot_per_problem_comparison_3b(results: Dict[str, Dict], output_dir: Path):
    """Plot per-problem performance comparison for 3B model."""
    # Get all problems
    all_problems = set()
    for method_results in results.values():
        all_problems.update(method_results['per_problem_type'].keys())
    
    problems = sorted(list(all_problems))
    methods = list(results.keys())
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(problems))
    width = 0.25
    
    for i, method in enumerate(methods):
        means = []
        stds = []
        for problem in problems:
            if problem in results[method]['per_problem_type']:
                means.append(results[method]['per_problem_type'][problem]['pass_at_1_mean'])
                stds.append(results[method]['per_problem_type'][problem]['pass_at_1_std'])
            else:
                means.append(0)
                stds.append(0)
        
        offset = (i - 1) * width
        color = COLORS.get(method.split('_')[0], '#95a5a6')
        ax.bar(x + offset, means, width, label=method.replace('_', ' ').title(),
               alpha=0.8, color=color, edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('Task', fontweight='bold')
    ax.set_ylabel('Pass@1 (%)', fontweight='bold')
    ax.set_title('Per-Task Performance: Qwen2.5-3B-Instruct', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', '\n') for p in problems], rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '3b_per_task_comparison.png', bbox_inches='tight')
    print(f"Saved: {output_dir / '3b_per_task_comparison.png'}")
    plt.close()

def plot_3b_vs_7b_comparison(results_3b: Dict[str, Dict], results_7b: Dict[str, Dict], 
                              output_dir: Path):
    """Plot comparison between 3B and 7B models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Overall comparison
    models = ['3B Baseline', '3B Continuous', '3B Phase-Adapt', 
              '7B Baseline', '7B Continuous', '7B Phase-Adapt']
    
    means_3b = [results_3b[k]['pass_at_k_mean'] for k in ['baseline', 'continuous', 'phase_adapt']]
    means_7b = [results_7b[k]['pass_at_k_mean'] for k in ['baseline', 'continuous', 'phase_adapt']]
    means = means_3b + means_7b
    
    stds_3b = [results_3b[k]['pass_at_k_std'] for k in ['baseline', 'continuous', 'phase_adapt']]
    stds_7b = [results_7b[k]['pass_at_k_std'] for k in ['baseline', 'continuous', 'phase_adapt']]
    stds = stds_3b + stds_7b
    
    x = np.arange(len(models))
    colors_list = [COLORS['3b']] * 3 + [COLORS['7b']] * 3
    
    bars = ax1.bar(x, means, yerr=stds, capsize=5, alpha=0.8, color=colors_list,
                   edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                f'{mean:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Model & Method', fontweight='bold')
    ax1.set_ylabel('Pass@1 (%)', fontweight='bold')
    ax1.set_title('3B vs 7B: Overall Performance', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=9)
    ax1.set_ylim(0, max(means) * 1.3)
    ax1.grid(axis='y', alpha=0.3)
    
    # Right plot: Best method per model
    best_methods = ['3B\nBaseline', '7B\nBaseline']
    best_means = [results_3b['baseline']['pass_at_k_mean'], 
                  results_7b['baseline']['pass_at_k_mean']]
    best_stds = [results_3b['baseline']['pass_at_k_std'],
                 results_7b['baseline']['pass_at_k_std']]
    
    x2 = np.arange(len(best_methods))
    colors_best = [COLORS['3b'], COLORS['7b']]
    
    bars2 = ax2.bar(x2, best_means, yerr=best_stds, capsize=5, alpha=0.8, 
                    color=colors_best, edgecolor='black', linewidth=1.2)
    
    # Add value labels and improvement percentage
    for i, (bar, mean, std) in enumerate(zip(bars2, best_means, best_stds)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{mean:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement arrow
    if best_means[1] > best_means[0]:
        improvement = ((best_means[1] - best_means[0]) / best_means[0]) * 100
        mid_x = (x2[0] + x2[1]) / 2
        mid_y = max(best_means) * 0.6
        ax2.annotate('', xy=(x2[1], mid_y), xytext=(x2[0], mid_y),
                    arrowprops=dict(arrowstyle='->', lw=2, color='green'))
        ax2.text(mid_x, mid_y + 2, f'+{improvement:.1f}%',
                ha='center', fontsize=11, fontweight='bold', color='green')
    
    ax2.set_xlabel('Model (Best Method)', fontweight='bold')
    ax2.set_ylabel('Pass@1 (%)', fontweight='bold')
    ax2.set_title('Best Method Comparison', fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(best_methods)
    ax2.set_ylim(0, max(best_means) * 1.35)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '3b_vs_7b_comparison.png', bbox_inches='tight')
    print(f"Saved: {output_dir / '3b_vs_7b_comparison.png'}")
    plt.close()

def plot_task_difficulty_heatmap(results_3b: Dict[str, Dict], results_7b: Dict[str, Dict],
                                   output_dir: Path):
    """Create a heatmap showing task difficulty across models and methods."""
    # Gather all problems
    all_problems = set()
    for method_results in results_3b.values():
        all_problems.update(method_results['per_problem_type'].keys())
    
    problems = sorted(list(all_problems))
    
    # Create data matrix
    configs = ['3B-Baseline', '3B-Continuous', '3B-Phase',
               '7B-Baseline', '7B-Continuous', '7B-Phase']
    
    data = []
    for problem in problems:
        row = []
        for model_results, prefix in [(results_3b, '3B'), (results_7b, '7B')]:
            for method in ['baseline', 'continuous', 'phase_adapt']:
                if problem in model_results[method]['per_problem_type']:
                    row.append(model_results[method]['per_problem_type'][problem]['pass_at_1_mean'])
                else:
                    row.append(0)
        data.append(row)
    
    df = pd.DataFrame(data, index=[p.replace('_', ' ').title() for p in problems], 
                     columns=configs)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn', center=30,
                cbar_kws={'label': 'Pass@1 (%)'}, ax=ax, linewidths=0.5)
    
    ax.set_title('Task Difficulty Heatmap: All Configurations', fontweight='bold', pad=20)
    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Task', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'task_difficulty_heatmap.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'task_difficulty_heatmap.png'}")
    plt.close()

def create_summary_table(results_3b: Dict[str, Dict], results_7b: Dict[str, Dict],
                         output_dir: Path):
    """Create a summary statistics table."""
    summary_data = []
    
    for model_name, results in [('3B', results_3b), ('7B', results_7b)]:
        for method in ['baseline', 'continuous', 'phase_adapt']:
            method_data = results[method]
            
            # Get task scores
            task_scores = [v['pass_at_1_mean'] 
                          for v in method_data['per_problem_type'].values()]
            
            summary_data.append({
                'Model': model_name,
                'Method': method.replace('_', ' ').title(),
                'Overall (%)': f"{method_data['pass_at_k_mean']:.2f} ± {method_data['pass_at_k_std']:.2f}",
                'Best Task (%)': f"{max(task_scores):.2f}",
                'Worst Task (%)': f"{min(task_scores):.2f}",
                'Solved Tasks': sum(1 for s in task_scores if s > 0),
                'Tasks > 50%': sum(1 for s in task_scores if s > 50)
            })
    
    df = pd.DataFrame(summary_data)
    
    # Save as CSV
    csv_path = output_dir / 'summary_statistics.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Create a pretty table plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center',
                    colColours=['#e8e8e8'] * len(df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code by model
    for i in range(len(df)):
        if df.iloc[i]['Model'] == '3B':
            table[(i+1, 0)].set_facecolor(COLORS['3b'])
            table[(i+1, 0)].set_alpha(0.3)
        else:
            table[(i+1, 0)].set_facecolor(COLORS['7b'])
            table[(i+1, 0)].set_alpha(0.3)
    
    plt.title('Performance Summary: All Configurations', fontweight='bold', pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_table.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'summary_table.png'}")
    plt.close()

def main():
    base_dir = Path('/mnt/home/oana/projects/nano-grpo-envs/exp_output')
    output_dir = base_dir / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    print("Loading 3B results...")
    # Load 3B results
    results_3b = {
        'baseline': aggregate_seeds([
            str(base_dir / 'science2_3b_suite/baseline_len512_seed0/summary.json'),
            str(base_dir / 'science2_3b_suite/baseline_len512_seed1/summary.json')
        ]),
        'continuous': aggregate_seeds([
            str(base_dir / 'science2_3b_suite/continuous_best_len512_seed0/summary.json'),
            str(base_dir / 'science2_3b_suite/continuous_best_len512_seed1/summary.json')
        ]),
        'phase_adapt': aggregate_seeds([
            str(base_dir / 'science2_3b_suite/phase_adapt_best_len512_seed0/summary.json'),
            str(base_dir / 'science2_3b_suite/phase_adapt_best_len512_seed1/summary.json')
        ])
    }
    
    print("Loading 7B results...")
    # Load 7B results
    results_7b = {
        'baseline': aggregate_seeds([
            str(base_dir / 'science2_suite/baseline_nomask_seed1/summary.json'),
            str(base_dir / 'science2_suite/baseline_nomask_seed2/summary.json')
        ]),
        'continuous': aggregate_seeds([
            str(base_dir / 'science2_suite/softmask_every10_wt05_seed1/summary.json'),
            str(base_dir / 'science2_suite/softmask_every10_wt05_seed2/summary.json')
        ]),
        'phase_adapt': aggregate_seeds([
            str(base_dir / 'science2_next_suite/phase_adapt_masking_latezero_len512_seed1/summary.json')
        ])
    }
    
    print("\nGenerating plots...")
    print("-" * 50)
    
    # Generate all plots
    plot_overall_comparison_3b(results_3b, output_dir)
    plot_per_problem_comparison_3b(results_3b, output_dir)
    plot_3b_vs_7b_comparison(results_3b, results_7b, output_dir)
    plot_task_difficulty_heatmap(results_3b, results_7b, output_dir)
    create_summary_table(results_3b, results_7b, output_dir)
    
    print("-" * 50)
    print(f"\n✓ All plots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - 3b_overall_comparison.png")
    print("  - 3b_per_task_comparison.png")
    print("  - 3b_vs_7b_comparison.png")
    print("  - task_difficulty_heatmap.png")
    print("  - summary_table.png")
    print("  - summary_statistics.csv")

if __name__ == '__main__':
    main()
