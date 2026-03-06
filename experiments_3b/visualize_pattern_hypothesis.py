"""
Visualize the pattern hypothesis: comparing expected 3B OOD results
if the 7B pattern holds vs if it breaks.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# 7B OOD Data (Actual)
data_7b = {
    'Baseline': {'Overall': 16.23, 'Core': 20.65, 'Stretch': 8.18},
    'Continuous\nZero': {'Overall': 16.90, 'Core': 20.40, 'Stretch': 10.55},
    'Phase-Adapt': {'Overall': 17.97, 'Core': 20.85, 'Stretch': 12.73}
}

# 3B In-Domain (Actual)
data_3b_indomain = {
    'Baseline': 26.20,
    'Continuous\nZero': 23.90,
    'Phase-Adapt': 22.48
}

# 3B OOD Predictions (scaled from 7B, assuming pattern holds)
scaling_factor = 0.683  # 3B/7B in-domain ratio
data_3b_ood_expected = {
    'Baseline': {'Overall': 11.1, 'Core': 14.1, 'Stretch': 5.6},
    'Continuous\nZero': {'Overall': 11.6, 'Core': 14.0, 'Stretch': 7.2},
    'Phase-Adapt': {'Overall': 12.3, 'Core': 14.2, 'Stretch': 8.7}
}

# 3B OOD if pattern breaks (in-domain rankings preserved)
data_3b_ood_broken = {
    'Baseline': {'Overall': 11.1, 'Core': 14.1, 'Stretch': 5.6},
    'Continuous\nZero': {'Overall': 10.2, 'Core': 13.5, 'Stretch': 4.9},
    'Phase-Adapt': {'Overall': 9.6, 'Core': 12.8, 'Stretch': 4.2}
}

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# === Row 1: 7B Results (Actual) ===
ax1 = fig.add_subplot(gs[0, :])
configs = list(data_7b.keys())
overall = [data_7b[c]['Overall'] for c in configs]
core = [data_7b[c]['Core'] for c in configs]
stretch = [data_7b[c]['Stretch'] for c in configs]

x = np.arange(len(configs))
width = 0.25

bars1 = ax1.bar(x - width, overall, width, label='Overall', color='#2ecc71', alpha=0.8)
bars2 = ax1.bar(x, core, width, label='Core (21 tasks)', color='#3498db', alpha=0.8)
bars3 = ax1.bar(x + width, stretch, width, label='Stretch (10 tasks)', color='#e74c3c', alpha=0.8)

ax1.set_ylabel('Pass@1 (%)', fontsize=12, fontweight='bold')
ax1.set_title('7B OOD Results (ACTUAL) - Pattern: Phase-Adapt wins on stretch tasks', 
              fontsize=14, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(configs)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Add improvement annotations
baseline_stretch = data_7b['Baseline']['Stretch']
phaseadapt_stretch = data_7b['Phase-Adapt']['Stretch']
improvement = ((phaseadapt_stretch - baseline_stretch) / baseline_stretch) * 100
ax1.annotate(f'+55.6% stretch\nimprovement',
            xy=(2 + width, phaseadapt_stretch), xytext=(2.4, 15),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, fontweight='bold', color='red')

# === Row 2: 3B In-Domain vs Expected OOD (if pattern holds) ===
ax2 = fig.add_subplot(gs[1, 0])
configs = list(data_3b_indomain.keys())
indomain_vals = [data_3b_indomain[c] for c in configs]
x = np.arange(len(configs))
bars = ax2.bar(x, indomain_vals, color=['#2ecc71', '#e67e22', '#9b59b6'], alpha=0.8)
ax2.set_ylabel('Pass@1 (%)', fontsize=11, fontweight='bold')
ax2.set_title('3B In-Domain (ACTUAL)\nBaseline wins', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(configs, rotation=15, ha='right')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 30)

for bar, val in zip(bars, indomain_vals):
    ax2.text(bar.get_x() + bar.get_width()/2., val,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Expected OOD if pattern holds
ax3 = fig.add_subplot(gs[1, 1])
configs = list(data_3b_ood_expected.keys())
overall = [data_3b_ood_expected[c]['Overall'] for c in configs]
core = [data_3b_ood_expected[c]['Core'] for c in configs]
stretch = [data_3b_ood_expected[c]['Stretch'] for c in configs]

x = np.arange(len(configs))
width = 0.25
bars1 = ax3.bar(x - width, overall, width, label='Overall', color='#2ecc71', alpha=0.8)
bars2 = ax3.bar(x, core, width, label='Core', color='#3498db', alpha=0.8)
bars3 = ax3.bar(x + width, stretch, width, label='Stretch', color='#e74c3c', alpha=0.8)

ax3.set_title('3B OOD EXPECTED (if pattern holds)\nPhase-Adapt wins', 
             fontsize=12, fontweight='bold', color='green')
ax3.set_xticks(x)
ax3.set_xticklabels(configs, rotation=15, ha='right')
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, 16)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)

# OOD if pattern breaks
ax4 = fig.add_subplot(gs[1, 2])
configs = list(data_3b_ood_broken.keys())
overall = [data_3b_ood_broken[c]['Overall'] for c in configs]
core = [data_3b_ood_broken[c]['Core'] for c in configs]
stretch = [data_3b_ood_broken[c]['Stretch'] for c in configs]

x = np.arange(len(configs))
bars1 = ax4.bar(x - width, overall, width, label='Overall', color='#2ecc71', alpha=0.8)
bars2 = ax4.bar(x, core, width, label='Core', color='#3498db', alpha=0.8)
bars3 = ax4.bar(x + width, stretch, width, label='Stretch', color='#e74c3c', alpha=0.8)

ax4.set_title('3B OOD if pattern BREAKS\nBaseline stays best', 
             fontsize=12, fontweight='bold', color='red')
ax4.set_xticks(x)
ax4.set_xticklabels(configs, rotation=15, ha='right')
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)
ax4.set_ylim(0, 16)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)

# === Row 3: Relative improvements comparison ===
ax5 = fig.add_subplot(gs[2, :])

scenarios = ['7B OOD\n(Actual)', '3B OOD\n(If Pattern Holds)', '3B OOD\n(If Pattern Breaks)']
baseline_improvements = [0, 0, 0]  # Baseline is reference
continuous_improvements = [
    ((16.90 - 16.23) / 16.23) * 100,  # 7B actual
    ((11.6 - 11.1) / 11.1) * 100,     # 3B expected
    ((10.2 - 11.1) / 11.1) * 100      # 3B broken
]
phaseadapt_improvements = [
    ((17.97 - 16.23) / 16.23) * 100,  # 7B actual
    ((12.3 - 11.1) / 11.1) * 100,     # 3B expected
    ((9.6 - 11.1) / 11.1) * 100       # 3B broken
]

x = np.arange(len(scenarios))
width = 0.35

bars1 = ax5.bar(x - width/2, continuous_improvements, width, 
               label='Continuous Zero', color='#e67e22', alpha=0.8)
bars2 = ax5.bar(x + width/2, phaseadapt_improvements, width,
               label='Phase-Adapt', color='#9b59b6', alpha=0.8)

ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax5.set_ylabel('Improvement over Baseline (%)', fontsize=12, fontweight='bold')
ax5.set_title('Relative Improvements: What We\'re Testing', fontsize=14, fontweight='bold', pad=20)
ax5.set_xticks(x)
ax5.set_xticklabels(scenarios)
ax5.legend(fontsize=11)
ax5.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.1f}%', ha='center', 
                va='bottom' if height >= 0 else 'top', 
                fontsize=10, fontweight='bold')

# Add annotations
ax5.text(0, 12, 'Known Pattern', ha='center', fontsize=11, 
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax5.text(1, 12, 'Strong Paper', ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax5.text(2, 5, 'Needs\nExplanation', ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

plt.suptitle('3B vs 7B Pattern Analysis: What Are We Testing?', 
            fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('/mnt/home/oana/projects/nano-grpo-envs/experiments_3b/pattern_hypothesis_visualization.png', 
           dpi=300, bbox_inches='tight')
print("✅ Saved visualization to experiments_3b/pattern_hypothesis_visualization.png")

# Create a second figure focusing on stretch task improvements
fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

# Stretch task improvements for 7B (actual)
ax = axes[0]
configs = ['Baseline', 'Continuous\nZero', 'Phase-Adapt']
stretch_7b = [8.18, 10.55, 12.73]
improvements_7b = [0, (10.55-8.18)/8.18*100, (12.73-8.18)/8.18*100]

colors = ['#2ecc71', '#e67e22', '#9b59b6']
bars = ax.bar(configs, stretch_7b, color=colors, alpha=0.8)
ax.set_ylabel('Pass@1 on Stretch Tasks (%)', fontsize=12, fontweight='bold')
ax.set_title('7B: Stretch Task Performance (ACTUAL)', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar, val, imp in zip(bars, stretch_7b, improvements_7b):
    ax.text(bar.get_x() + bar.get_width()/2., val,
           f'{val:.1f}%\n({imp:+.1f}%)', ha='center', va='bottom', 
           fontsize=10, fontweight='bold')

# Stretch task predictions for 3B
ax = axes[1]
stretch_3b_expected = [5.6, 7.2, 8.7]
stretch_3b_broken = [5.6, 4.9, 4.2]

x = np.arange(len(configs))
width = 0.35

bars1 = ax.bar(x - width/2, stretch_3b_expected, width, 
              label='If Pattern Holds', color=colors, alpha=0.8)
bars2 = ax.bar(x + width/2, stretch_3b_broken, width,
              label='If Pattern Breaks', color=colors, alpha=0.4, 
              edgecolor='red', linewidth=2)

ax.set_ylabel('Pass@1 on Stretch Tasks (%)', fontsize=12, fontweight='bold')
ax.set_title('3B: Stretch Task Predictions', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bar, val in zip(bars1, stretch_3b_expected):
    ax.text(bar.get_x() + bar.get_width()/2., val,
           f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar, val in zip(bars2, stretch_3b_broken):
    ax.text(bar.get_x() + bar.get_width()/2., val,
           f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

plt.suptitle('Stretch Task Analysis: Key Discriminator Between Hypotheses', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/mnt/home/oana/projects/nano-grpo-envs/experiments_3b/stretch_task_hypothesis.png', 
           dpi=300, bbox_inches='tight')
print("✅ Saved stretch task visualization to experiments_3b/stretch_task_hypothesis.png")

print("\n" + "="*80)
print("PATTERN HYPOTHESIS VISUALIZATIONS CREATED")
print("="*80)
print("\nThese visualizations show:")
print("1. The established 7B pattern (Phase-Adapt wins OOD)")
print("2. Current 3B in-domain results (Baseline wins)")
print("3. Two possible OOD outcomes for 3B")
print("4. Relative improvements comparison")
print("5. Focus on stretch tasks (key discriminator)")
print("\nThe stretch task performance is the KEY metric:")
print("- If 3B Phase-Adapt improves stretch tasks → Pattern holds")
print("- If 3B Baseline stays best on stretch → Pattern breaks")
