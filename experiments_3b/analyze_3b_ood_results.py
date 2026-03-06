"""
Analyze 3B OOD Results and Compare with 7B
Shows if the pattern of improvement holds for 3B models
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

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
BASE_DIR = Path("/mnt/home/oana/projects/nano-grpo-envs")
RESULTS_3B = BASE_DIR / "validation/results_3b_ood_best3/results.csv"
OUTPUT_DIR = BASE_DIR / "experiments_3b/ood_analysis"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 7B Reference (from experiments_7b/paper_ready/README.md)
RESULTS_7B = {
    'Baseline': {'overall': 16.23, 'core': 20.65, 'stretch': 8.18},
    'Continuous': {'overall': 16.90, 'core': 20.40, 'stretch': 10.55},
    'Phase-Adapt': {'overall': 17.97, 'core': 20.85, 'stretch': 12.73},
}

# Task splits (from reasoning_gym)
CORE_TASKS = [
    'number_sequence', 'palindrome_generation', 'propositional_logic',
    'leg_counting', 'simple_geometry', 'maze', 'family_relationships',
    'bf', 'sokoban', 'polynomial_equations', 'wordle', 'sudoku',
    'set', 'twenty_questions', 'ascii_math', 'dyck_language',
    'graph_coloring', 'graph_connectivity', 'othello', 'polynomial_roots',
    'tsp'
]

STRETCH_TASKS = [
    'boolean_circuits', 'lambda_calculus', 'hanoi', 'rubiks_cube',
    'rubiks_cube_2x2', 'sorting', 'zebra_puzzles', 'chemistry_equations',
    'physics_problems', 'probability_theory'
]


def load_ood_results():
    """Load 3B OOD evaluation results."""
    if not RESULTS_3B.exists():
        print(f"❌ Results file not found: {RESULTS_3B}")
        print("\nRun evaluation first:")
        print("  sbatch experiments_3b/run_3b_ood_eval.sh")
        return None
    
    df = pd.read_csv(RESULTS_3B)
    print(f"✓ Loaded {len(df)} results from {RESULTS_3B}")
    return df


def calculate_metrics(df):
    """Calculate pass@1 metrics by model and task split."""
    results = []
    
    # Group by model
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        # Overall
        overall_pass = (model_data['pass1'].sum() / len(model_data)) * 100
        
        # Core tasks
        core_data = model_data[model_data['id'].str.contains('|'.join(CORE_TASKS), case=False, na=False)]
        core_pass = (core_data['pass1'].sum() / len(core_data)) * 100 if len(core_data) > 0 else 0
        
        # Stretch tasks
        stretch_data = model_data[model_data['id'].str.contains('|'.join(STRETCH_TASKS), case=False, na=False)]
        stretch_pass = (stretch_data['pass1'].sum() / len(stretch_data)) * 100 if len(stretch_data) > 0 else 0
        
        # Determine strategy from model name
        if 'baseline' in model.lower():
            strategy = 'Baseline'
        elif 'continuous' in model.lower() or 'zero' in model.lower():
            strategy = 'Continuous'
        elif 'phase' in model.lower() or 'adapt' in model.lower():
            strategy = 'Phase-Adapt'
        else:
            strategy = 'Other'
        
        results.append({
            'model': model,
            'strategy': strategy,
            'overall': overall_pass,
            'core': core_pass,
            'stretch': stretch_pass,
        })
    
    return pd.DataFrame(results)


def create_comparison_table(df_3b_metrics):
    """Create comparison table: 3B vs 7B."""
    print("\n" + "="*100)
    print("3B vs 7B OOD PERFORMANCE COMPARISON (31 Tasks)")
    print("="*100)
    print(f"{'Model':<35}{'Size':<8}{'Overall':<12}{'Core (21)':<12}{'Stretch (10)':<12}")
    print("-"*100)
    
    # 7B results
    for strategy, metrics in RESULTS_7B.items():
        print(f"{strategy:<35}{'7B':<8}{metrics['overall']:<12.2f}{metrics['core']:<12.2f}{metrics['stretch']:<12.2f}")
    
    print("-"*100)
    
    # 3B results (average by strategy)
    for strategy in df_3b_metrics['strategy'].unique():
        strategy_data = df_3b_metrics[df_3b_metrics['strategy'] == strategy]
        overall_mean = strategy_data['overall'].mean()
        core_mean = strategy_data['core'].mean()
        stretch_mean = strategy_data['stretch'].mean()
        
        print(f"{strategy:<35}{'3B':<8}{overall_mean:<12.2f}{core_mean:<12.2f}{stretch_mean:<12.2f}")
    
    print("="*100 + "\n")
    
    # Save CSV
    comparison_data = []
    
    for strategy, metrics in RESULTS_7B.items():
        comparison_data.append({
            'Model': strategy, 'Size': '7B',
            'Overall': metrics['overall'],
            'Core': metrics['core'],
            'Stretch': metrics['stretch']
        })
    
    for strategy in df_3b_metrics['strategy'].unique():
        strategy_data = df_3b_metrics[df_3b_metrics['strategy'] == strategy]
        comparison_data.append({
            'Model': strategy, 'Size': '3B',
            'Overall': strategy_data['overall'].mean(),
            'Core': strategy_data['core'].mean(),
            'Stretch': strategy_data['stretch'].mean()
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(OUTPUT_DIR / "3b_vs_7b_comparison.csv", index=False)
    print(f"✓ Saved: {OUTPUT_DIR / '3b_vs_7b_comparison.csv'}")
    
    return comparison_df


def plot_3b_vs_7b(df_3b_metrics):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Overall comparison
    ax1 = axes[0, 0]
    
    strategies = ['Baseline', 'Continuous', 'Phase-Adapt']
    x = np.arange(len(strategies))
    width = 0.35
    
    # 7B data
    values_7b = [RESULTS_7B[s]['overall'] for s in strategies]
    ax1.bar(x - width/2, values_7b, width, label='7B', color='#1f77b4', alpha=0.8)
    
    # 3B data
    values_3b = []
    for s in strategies:
        strategy_data = df_3b_metrics[df_3b_metrics['strategy'] == s]
        if len(strategy_data) > 0:
            values_3b.append(strategy_data['overall'].mean())
        else:
            values_3b.append(0)
    
    ax1.bar(x + width/2, values_3b, width, label='3B', color='#ff7f0e', alpha=0.8)
    
    # Add value labels
    for i, (v7, v3) in enumerate(zip(values_7b, values_3b)):
        ax1.text(i - width/2, v7 + 0.3, f"{v7:.1f}", ha='center', va='bottom', fontweight='bold')
        ax1.text(i + width/2, v3 + 0.3, f"{v3:.1f}", ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel('Pass@1 (%)', fontweight='bold')
    ax1.set_title('Overall OOD Performance (31 tasks)', fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontweight='bold')
    ax1.legend(frameon=True)
    ax1.set_ylim(0, 20)
    
    # Plot 2: Core vs Stretch
    ax2 = axes[0, 1]
    
    x = np.arange(len(strategies) * 2)
    width = 0.35
    
    # Prepare data
    core_7b = [RESULTS_7B[s]['core'] for s in strategies]
    core_3b = [df_3b_metrics[df_3b_metrics['strategy']==s]['core'].mean() if len(df_3b_metrics[df_3b_metrics['strategy']==s]) > 0 else 0 for s in strategies]
    stretch_7b = [RESULTS_7B[s]['stretch'] for s in strategies]
    stretch_3b = [df_3b_metrics[df_3b_metrics['strategy']==s]['stretch'].mean() if len(df_3b_metrics[df_3b_metrics['strategy']==s]) > 0 else 0 for s in strategies]
    
    all_7b = core_7b + stretch_7b
    all_3b = core_3b + stretch_3b
    
    ax2.bar(x - width/2, all_7b, width, label='7B', color='#1f77b4', alpha=0.8)
    ax2.bar(x + width/2, all_3b, width, label='3B', color='#ff7f0e', alpha=0.8)
    
    ax2.set_ylabel('Pass@1 (%)', fontweight='bold')
    ax2.set_title('Core vs Stretch Task Performance', fontweight='bold', pad=15)
    ax2.set_xticks(x)
    labels = [f'{s}\nCore' for s in strategies] + [f'{s}\nStretch' for s in strategies]
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.legend(frameon=True)
    ax2.set_ylim(0, 25)
    
    # Plot 3: Improvement analysis
    ax3 = axes[1, 0]
    
    # Calculate improvements
    improvements_7b = []
    improvements_3b = []
    
    baseline_7b = RESULTS_7B['Baseline']['overall']
    continuous_7b = RESULTS_7B['Continuous']['overall']
    improvements_7b.append(((continuous_7b - baseline_7b) / baseline_7b) * 100)
    
    baseline_3b = df_3b_metrics[df_3b_metrics['strategy']=='Baseline']['overall'].mean()
    continuous_3b = df_3b_metrics[df_3b_metrics['strategy']=='Continuous']['overall'].mean()
    if baseline_3b > 0:
        improvements_3b.append(((continuous_3b - baseline_3b) / baseline_3b) * 100)
    else:
        improvements_3b.append(0)
    
    x = np.arange(1)
    width = 0.35
    
    ax3.bar(x - width/2, improvements_7b, width, label='7B', color='#1f77b4', alpha=0.8)
    ax3.bar(x + width/2, improvements_3b, width, label='3B', color='#ff7f0e', alpha=0.8)
    
    for i, (v7, v3) in enumerate(zip(improvements_7b, improvements_3b)):
        ax3.text(i - width/2, v7 + 0.2 if v7 > 0 else v7 - 0.2, f"{v7:.1f}%", 
                ha='center', va='bottom' if v7 > 0 else 'top', fontweight='bold')
        ax3.text(i + width/2, v3 + 0.2 if v3 > 0 else v3 - 0.2, f"{v3:.1f}%", 
                ha='center', va='bottom' if v3 > 0 else 'top', fontweight='bold')
    
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('Improvement over Baseline (%)', fontweight='bold')
    ax3.set_title('Continuous Strategy: % Improvement', fontweight='bold', pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Continuous\nvs Baseline'])
    ax3.legend(frameon=True)
    
    # Plot 4: Model size effect
    ax4 = axes[1, 1]
    
    # Compare each strategy across model sizes
    for strategy in strategies:
        val_7b = RESULTS_7B[strategy]['overall']
        val_3b = df_3b_metrics[df_3b_metrics['strategy']==strategy]['overall'].mean()
        
        ax4.plot([0, 1], [val_3b, val_7b], 'o-', linewidth=2, markersize=8, label=strategy, alpha=0.7)
        
        # Add value labels
        ax4.text(0, val_3b, f"{val_3b:.1f}", ha='right', va='center', fontsize=9, fontweight='bold')
        ax4.text(1, val_7b, f"{val_7b:.1f}", ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax4.set_ylabel('Pass@1 (%)', fontweight='bold')
    ax4.set_title('Scaling: 3B → 7B', fontweight='bold', pad=15)
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(['3B', '7B'], fontweight='bold', fontsize=12)
    ax4.legend(frameon=True)
    ax4.set_ylim(0, 20)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "3b_vs_7b_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / '3b_vs_7b_comparison.png'}")
    plt.close()


def generate_insights(df_3b_metrics):
    """Generate key insights."""
    insights = []
    insights.append("="*100)
    insights.append("KEY INSIGHTS - 3B OOD vs 7B OOD")
    insights.append("="*100)
    insights.append("")
    
    # Get metrics
    baseline_3b = df_3b_metrics[df_3b_metrics['strategy']=='Baseline']['overall'].mean()
    continuous_3b = df_3b_metrics[df_3b_metrics['strategy']=='Continuous']['overall'].mean()
    
    baseline_7b = RESULTS_7B['Baseline']['overall']
    continuous_7b = RESULTS_7B['Continuous']['overall']
    
    # 1. Absolute performance
    insights.append("1. ABSOLUTE OOD PERFORMANCE:")
    insights.append(f"   3B Baseline:    {baseline_3b:.2f}% OOD")
    insights.append(f"   7B Baseline:    {baseline_7b:.2f}% OOD")
    insights.append(f"   → 7B is {baseline_7b - baseline_3b:+.2f}% better ({((baseline_7b - baseline_3b) / baseline_3b * 100):+.1f}%)")
    insights.append("")
    
    # 2. Strategy effectiveness
    insights.append("2. STRATEGY EFFECTIVENESS:")
    
    if baseline_3b > 0:
        imp_3b = ((continuous_3b - baseline_3b) / baseline_3b) * 100
        insights.append(f"   3B Continuous vs Baseline: {continuous_3b:.2f}% vs {baseline_3b:.2f}% = {imp_3b:+.1f}%")
    
    imp_7b = ((continuous_7b - baseline_7b) / baseline_7b) * 100
    insights.append(f"   7B Continuous vs Baseline: {continuous_7b:.2f}% vs {baseline_7b:.2f}% = {imp_7b:+.1f}%")
    
    if baseline_3b > 0:
        if imp_3b > 0 and imp_7b > 0:
            insights.append(f"   → Both sizes benefit from continuous strategy")
        elif imp_3b < 0 and imp_7b > 0:
            insights.append(f"   → Strategy works for 7B but NOT for 3B!")
        elif imp_3b > 0 and imp_7b < 0:
            insights.append(f"   → Strategy works for 3B but NOT for 7B!")
        else:
            insights.append(f"   → Strategy doesn't help either size")
    
    insights.append("")
    
    # 3. Core vs Stretch
    baseline_3b_stretch = df_3b_metrics[df_3b_metrics['strategy']=='Baseline']['stretch'].mean()
    continuous_3b_stretch = df_3b_metrics[df_3b_metrics['strategy']=='Continuous']['stretch'].mean()
    
    baseline_7b_stretch = RESULTS_7B['Baseline']['stretch']
    continuous_7b_stretch = RESULTS_7B['Continuous']['stretch']
    
    insights.append("3. STRETCH TASK PERFORMANCE:")
    insights.append(f"   3B: {baseline_3b_stretch:.2f}% → {continuous_3b_stretch:.2f}% ({continuous_3b_stretch - baseline_3b_stretch:+.2f}%)")
    insights.append(f"   7B: {baseline_7b_stretch:.2f}% → {continuous_7b_stretch:.2f}% ({continuous_7b_stretch - baseline_7b_stretch:+.2f}%)")
    insights.append("")
    
    # 4. Pattern consistency
    insights.append("4. PATTERN CONSISTENCY:")
    if imp_3b > 0 and imp_7b > 0:
        insights.append("   ✓ Pattern HOLDS: Continuous strategy improves OOD for both 3B and 7B")
    elif imp_3b < 0 and imp_7b > 0:
        insights.append("   ✗ Pattern BREAKS: Different optimal strategies for 3B vs 7B")
    elif imp_3b > imp_7b:
        insights.append("   ~ Pattern VARIES: 3B benefits MORE from continuous strategy than 7B")
    else:
        insights.append("   ~ Pattern VARIES: 7B benefits MORE from continuous strategy than 3B")
    
    insights.append("")
    insights.append("="*100)
    
    # Print and save
    insights_text = "\n".join(insights)
    print(insights_text)
    
    with open(OUTPUT_DIR / "key_insights_3b_ood.txt", 'w') as f:
        f.write(insights_text)
    print(f"\n✓ Saved: {OUTPUT_DIR / 'key_insights_3b_ood.txt'}")


def main():
    """Main analysis pipeline."""
    print("\n" + "="*100)
    print("ANALYZING 3B OOD RESULTS vs 7B")
    print("="*100 + "\n")
    
    # Load results
    df = load_ood_results()
    if df is None:
        return
    
    # Calculate metrics
    print("\nCalculating metrics...")
    df_metrics = calculate_metrics(df)
    print(f"✓ Calculated metrics for {len(df_metrics)} models\n")
    
    # Create comparison table
    comparison_df = create_comparison_table(df_metrics)
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_3b_vs_7b(df_metrics)
    
    # Generate insights
    print("\n")
    generate_insights(df_metrics)
    
    print("\n" + "="*100)
    print(f"Analysis complete! Results saved to: {OUTPUT_DIR}")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
