"""
Compare 3B Configurations: Previous vs 7B-Replication
Identifies best performing configs for OOD evaluation
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Paths
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = Path("/mnt/home/oana/projects/nano-grpo-envs")
OUTPUT_DIR = BASE_DIR / "3b_config_comparison"
OUTPUT_DIR.mkdir(exist_ok=True)

# Experiment directories
PREVIOUS_3B = PROJECT_ROOT / "exp_output/science2_3b_suite"
REPLICATION_3B = PROJECT_ROOT / "exp_output/3b_7b_replication"

def load_all_experiments():
    """Load all 3B experiment results."""
    experiments = []
    
    # Previous 3B experiments (original configs)
    previous_configs = [
        ('baseline_len512_seed0', 'Baseline', 'No masking/zeroing', 'Previous', 0),
        ('baseline_len512_seed1', 'Baseline', 'No masking/zeroing', 'Previous', 1),
        ('continuous_best_len512_seed0', 'Continuous (Mask)', 'Masking-only (every_n=20)', 'Previous', 0),
        ('continuous_best_len512_seed1', 'Continuous (Mask)', 'Masking-only (every_n=20)', 'Previous', 1),
        ('phase_adapt_best_len512_seed0', 'Phase-Adapt (Old)', 'Split mask + zero@600', 'Previous', 0),
        ('phase_adapt_best_len512_seed1', 'Phase-Adapt (Old)', 'Split mask + zero@600', 'Previous', 1),
    ]
    
    for exp_name, strategy, config, source, seed in previous_configs:
        summary_path = PREVIOUS_3B / exp_name / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
                experiments.append({
                    'experiment': exp_name,
                    'strategy': strategy,
                    'config': config,
                    'source': source,
                    'seed': seed,
                    'pass@1': summary['pass_at_k'],
                    'format_reward': summary['avg_format_reward'],
                    'per_problem_type': summary['per_problem_type']
                })
    
    # 7B-Replication experiments (7B configs on 3B)
    replication_configs = [
        ('continuous_fullzero_seed0', 'Continuous (Zero)', 'Zero-only (every_n=20)', '7B-Replication', 0),
        ('continuous_fullzero_seed1', 'Continuous (Zero)', 'Zero-only (every_n=20)', '7B-Replication', 1),
        ('phase_adapt_exact7b_seed0', 'Phase-Adapt (7B)', 'Split mask + zero@200', '7B-Replication', 0),
        ('phase_adapt_exact7b_seed1', 'Phase-Adapt (7B)', 'Split mask + zero@200', '7B-Replication', 1),
    ]
    
    for exp_name, strategy, config, source, seed in replication_configs:
        summary_path = REPLICATION_3B / exp_name / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
                experiments.append({
                    'experiment': exp_name,
                    'strategy': strategy,
                    'config': config,
                    'source': source,
                    'seed': seed,
                    'pass@1': summary['pass_at_k'],
                    'format_reward': summary['avg_format_reward'],
                    'per_problem_type': summary['per_problem_type']
                })
    
    return pd.DataFrame(experiments)


def create_comparison_table(df):
    """Create comprehensive comparison table."""
    # Group by strategy and source
    summary = []
    
    for source in ['Previous', '7B-Replication']:
        source_data = df[df['source'] == source]
        for strategy in source_data['strategy'].unique():
            strategy_data = source_data[source_data['strategy'] == strategy]
            
            mean_pass = strategy_data['pass@1'].mean()
            std_pass = strategy_data['pass@1'].std()
            
            summary.append({
                'Source': source,
                'Strategy': strategy,
                'Config': strategy_data.iloc[0]['config'],
                'Pass@1 Mean': mean_pass,
                'Pass@1 Std': std_pass,
                'Seed 0': strategy_data[strategy_data['seed']==0]['pass@1'].values[0] if len(strategy_data[strategy_data['seed']==0]) > 0 else None,
                'Seed 1': strategy_data[strategy_data['seed']==1]['pass@1'].values[0] if len(strategy_data[strategy_data['seed']==1]) > 0 else None,
            })
    
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('Pass@1 Mean', ascending=False)
    
    # Save CSV
    summary_df.to_csv(OUTPUT_DIR / "all_3b_configs_ranked.csv", index=False)
    
    print("\n" + "="*100)
    print("3B CONFIGURATION COMPARISON - Previous vs 7B-Replication")
    print("="*100)
    print(f"{'Rank':<6}{'Source':<18}{'Strategy':<25}{'Config':<30}{'Mean±Std':<15}{'Seed0':<8}{'Seed1':<8}")
    print("-"*100)
    
    for idx, row in summary_df.iterrows():
        rank = summary_df.index.get_loc(idx) + 1
        mean_std = f"{row['Pass@1 Mean']:.2f}±{row['Pass@1 Std']:.2f}"
        seed0 = f"{row['Seed 0']:.2f}" if row['Seed 0'] is not None else "N/A"
        seed1 = f"{row['Seed 1']:.2f}" if row['Seed 1'] is not None else "N/A"
        print(f"{rank:<6}{row['Source']:<18}{row['Strategy']:<25}{row['Config']:<30}{mean_std:<15}{seed0:<8}{seed1:<8}")
    
    print("="*100 + "\n")
    
    return summary_df


def plot_comparison(df):
    """Create comparison visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Overall comparison grouped by source
    ax1 = axes[0, 0]
    strategies = df['strategy'].unique()
    sources = ['Previous', '7B-Replication']
    
    x = np.arange(len(strategies))
    width = 0.35
    
    colors = {'Previous': '#1f77b4', '7B-Replication': '#ff7f0e'}
    
    for i, source in enumerate(sources):
        source_data = df[df['source'] == source]
        means = []
        for strategy in strategies:
            strat_data = source_data[source_data['strategy'] == strategy]
            if len(strat_data) > 0:
                means.append(strat_data['pass@1'].mean())
            else:
                means.append(0)
        
        offset = (i - 0.5) * width
        bars = ax1.bar(x + offset, means, width, label=source, color=colors[source], alpha=0.8)
        
        # Add value labels
        for j, (bar, val) in enumerate(zip(bars, means)):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, val + 0.3, f"{val:.1f}", 
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax1.set_ylabel('Pass@1 (%)', fontweight='bold')
    ax1.set_title('Mean Performance by Strategy & Source', fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace(' ', '\n') for s in strategies], fontsize=9)
    ax1.legend(frameon=True)
    ax1.set_ylim(0, 30)
    
    # Plot 2: Best configs with error bars
    ax2 = axes[0, 1]
    summary_df = df.groupby(['source', 'strategy']).agg({
        'pass@1': ['mean', 'std']
    }).reset_index()
    summary_df.columns = ['source', 'strategy', 'mean', 'std']
    summary_df = summary_df.sort_values('mean', ascending=False).head(6)
    
    x_pos = np.arange(len(summary_df))
    colors_list = [colors[src] for src in summary_df['source']]
    
    ax2.barh(x_pos, summary_df['mean'], xerr=summary_df['std'], 
             color=colors_list, alpha=0.7, capsize=4)
    
    for i, (idx, row) in enumerate(summary_df.iterrows()):
        ax2.text(row['mean'] + row['std'] + 0.3, i, f"{row['mean']:.2f}±{row['std']:.2f}", 
                va='center', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Pass@1 (%)', fontweight='bold')
    ax2.set_title('Top 6 Configs Ranked (with seed variance)', fontweight='bold', pad=15)
    ax2.set_yticks(x_pos)
    labels = [f"{row['strategy']}\n({row['source']})" for _, row in summary_df.iterrows()]
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlim(0, 30)
    
    # Plot 3: Seed variance comparison
    ax3 = axes[1, 0]
    variance_data = df.groupby(['source', 'strategy'])['pass@1'].std().reset_index()
    variance_data.columns = ['source', 'strategy', 'std']
    
    x_pos = np.arange(len(variance_data))
    colors_list = [colors[src] for src in variance_data['source']]
    
    bars = ax3.bar(x_pos, variance_data['std'], color=colors_list, alpha=0.7)
    
    for bar, val in zip(bars, variance_data['std']):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.05, f"{val:.2f}", 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax3.set_ylabel('Std Dev (Pass@1)', fontweight='bold')
    ax3.set_title('Seed Variance by Config', fontweight='bold', pad=15)
    ax3.set_xticks(x_pos)
    labels = [f"{row['strategy']}\n({row['source']})" for _, row in variance_data.iterrows()]
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax3.legend()
    
    # Plot 4: Individual seeds scatter
    ax4 = axes[1, 1]
    
    for source in sources:
        source_data = df[df['source'] == source]
        for strategy in source_data['strategy'].unique():
            strat_data = source_data[source_data['strategy'] == strategy]
            seeds = strat_data['seed'].values
            values = strat_data['pass@1'].values
            
            marker = 'o' if source == 'Previous' else 's'
            ax4.scatter(seeds, values, label=f"{strategy} ({source})", 
                       s=100, alpha=0.7, marker=marker)
            
            # Connect seeds with a line
            if len(values) == 2:
                ax4.plot(seeds, values, alpha=0.3, linewidth=1)
    
    ax4.set_xlabel('Seed', fontweight='bold')
    ax4.set_ylabel('Pass@1 (%)', fontweight='bold')
    ax4.set_title('Individual Seed Results', fontweight='bold', pad=15)
    ax4.set_xticks([0, 1])
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax4.set_ylim(15, 30)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "3b_config_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / '3b_config_comparison.png'}")
    plt.close()


def identify_best_configs(df):
    """Identify top 2 configs for OOD evaluation."""
    print("\n" + "="*100)
    print("BEST CONFIGS FOR OOD EVALUATION")
    print("="*100)
    
    # Calculate mean performance
    summary = df.groupby(['source', 'strategy', 'config']).agg({
        'pass@1': ['mean', 'std'],
        'experiment': 'first'
    }).reset_index()
    summary.columns = ['source', 'strategy', 'config', 'mean', 'std', 'first_exp']
    summary = summary.sort_values('mean', ascending=False)
    
    # Get top 2
    top2 = summary.head(2)
    
    print("\n🏆 TOP 2 CONFIGURATIONS:\n")
    for idx, row in top2.iterrows():
        rank = top2.index.get_loc(idx) + 1
        print(f"Rank {rank}: {row['strategy']} ({row['source']})")
        print(f"  Config: {row['config']}")
        print(f"  Performance: {row['mean']:.2f}% ± {row['std']:.2f}%")
        print(f"  Source: {row['source']}")
        
        # Get both seeds
        config_runs = df[(df['source'] == row['source']) & (df['strategy'] == row['strategy'])]
        print(f"  Models to evaluate:")
        for _, run in config_runs.iterrows():
            checkpoint_path = PREVIOUS_3B / run['experiment'] if row['source'] == 'Previous' else REPLICATION_3B / run['experiment']
            print(f"    - {run['experiment']} (seed {run['seed']}): {checkpoint_path}")
        print()
    
    # Save recommendation
    with open(OUTPUT_DIR / "ood_evaluation_plan.txt", 'w') as f:
        f.write("="*100 + "\n")
        f.write("3B OOD EVALUATION PLAN - Top 2 Configs\n")
        f.write("="*100 + "\n\n")
        
        f.write("Based on in-domain (10 tasks) performance, evaluate these configs on 31 OOD tasks:\n\n")
        
        for idx, row in top2.iterrows():
            rank = top2.index.get_loc(idx) + 1
            f.write(f"\n{rank}. {row['strategy']} ({row['source']})\n")
            f.write(f"   Config: {row['config']}\n")
            f.write(f"   In-domain: {row['mean']:.2f}% ± {row['std']:.2f}%\n")
            
            config_runs = df[(df['source'] == row['source']) & (df['strategy'] == row['strategy'])]
            f.write(f"   Models:\n")
            for _, run in config_runs.iterrows():
                if row['source'] == 'Previous':
                    checkpoint_path = f"exp_output/science2_3b_suite/{run['experiment']}"
                else:
                    checkpoint_path = f"exp_output/3b_7b_replication/{run['experiment']}"
                f.write(f"     - {checkpoint_path} (seed {run['seed']})\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("EVALUATION COMMAND:\n")
        f.write("="*100 + "\n\n")
        f.write("cd /mnt/home/oana/projects/nano-grpo-envs\n")
        f.write("python experiments_3b/evaluate_3b_ood.py\n")
        f.write("\nThis will:\n")
        f.write("- Evaluate all models from top 2 configs on 31 OOD tasks\n")
        f.write("- Compare with 7B OOD results (16.23-17.97% range)\n")
        f.write("- Generate plots similar to 7B analysis\n")
    
    print(f"✓ Saved: {OUTPUT_DIR / 'ood_evaluation_plan.txt'}")
    
    return top2


def compare_strategies(df):
    """Compare masking vs zeroing strategies."""
    print("\n" + "="*100)
    print("STRATEGY ANALYSIS: Masking vs Zeroing")
    print("="*100 + "\n")
    
    # Continuous strategies
    continuous_mask = df[df['strategy'] == 'Continuous (Mask)']['pass@1'].mean()
    continuous_zero = df[df['strategy'] == 'Continuous (Zero)']['pass@1'].mean()
    
    print(f"CONTINUOUS STRATEGY:")
    print(f"  Masking-only (Previous):  {continuous_mask:.2f}%")
    print(f"  Zeroing-only (7B-Rep):    {continuous_zero:.2f}%")
    print(f"  → Difference:             {continuous_zero - continuous_mask:+.2f}% ({'Zeroing wins' if continuous_zero > continuous_mask else 'Masking wins'})")
    print()
    
    # Phase-adapt strategies
    phase_old = df[df['strategy'] == 'Phase-Adapt (Old)']['pass@1'].mean()
    phase_7b = df[df['strategy'] == 'Phase-Adapt (7B)']['pass@1'].mean()
    
    print(f"PHASE-ADAPT STRATEGY:")
    print(f"  Split + zero@600 (Previous):  {phase_old:.2f}%")
    print(f"  Split + zero@200 (7B-Rep):    {phase_7b:.2f}%")
    print(f"  → Difference:                 {phase_7b - phase_old:+.2f}% ({'Earlier zeroing wins' if phase_7b > phase_old else 'Later zeroing wins'})")
    print()
    
    # Baseline
    baseline = df[df['strategy'] == 'Baseline']['pass@1'].mean()
    print(f"BASELINE (no masking/zeroing): {baseline:.2f}%")
    print()
    
    print("KEY FINDINGS:")
    if continuous_zero > continuous_mask:
        print(f"  ✓ Zeroing-only outperforms masking-only by {continuous_zero - continuous_mask:.2f}%")
    else:
        print(f"  ✗ Masking-only outperforms zeroing-only by {continuous_mask - continuous_zero:.2f}%")
    
    if phase_7b > phase_old:
        print(f"  ✓ Earlier zeroing (200) better than later (600) by {phase_7b - phase_old:.2f}%")
    else:
        print(f"  ✗ Later zeroing (600) better than earlier (200) by {phase_old - phase_7b:.2f}%")
    
    print("="*100 + "\n")


def main():
    """Main comparison pipeline."""
    print("\n" + "="*100)
    print("3B CONFIG COMPARISON: Previous vs 7B-Replication")
    print("="*100 + "\n")
    
    # Load all experiments
    print("Loading all 3B experiments...")
    df = load_all_experiments()
    print(f"✓ Loaded {len(df)} experiment runs\n")
    
    # Create comparison table
    summary_df = create_comparison_table(df)
    
    # Generate plots
    print("\nGenerating comparison visualizations...")
    plot_comparison(df)
    
    # Strategy analysis
    compare_strategies(df)
    
    # Identify best configs
    top2 = identify_best_configs(df)
    
    print("\n" + "="*100)
    print(f"Comparison complete! Results saved to: {OUTPUT_DIR}")
    print("="*100)
    print("\nNext step: Run OOD evaluation on top 2 configs")
    print("Command: python experiments_3b/evaluate_3b_ood.py")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
