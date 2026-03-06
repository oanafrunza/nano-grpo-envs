"""
Create comprehensive paper-ready figures showing Phase-Adapt effectiveness
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Create output directory
output_dir = Path('paper_figures')
output_dir.mkdir(exist_ok=True)

print('='*80)
print('GENERATING PAPER FIGURES')
print('='*80)
print()

# ============================================================================
# FIGURE 1: In-Domain vs OOD Reversal Pattern
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 7B Results
models = ['Baseline', 'Continuous', 'Phase-Adapt']
colors = ['#95a5a6', '#e67e22', '#9b59b6']

# 7B In-Domain
ax = axes[0, 0]
indomain_7b = [38.36, 38.28, 34.68]
bars = ax.bar(models, indomain_7b, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Pass@1 (%)', fontweight='bold')
ax.set_title('7B: In-Domain (10 tasks)', fontweight='bold')
ax.set_ylim(0, 45)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, indomain_7b):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1,
           f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)
ax.axhline(y=indomain_7b[0], color='gray', linestyle='--', alpha=0.5, linewidth=1)

# 7B OOD Overall
ax = axes[0, 1]
ood_7b = [16.23, 16.90, 17.97]
bars = ax.bar(models, ood_7b, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Pass@1 (%)', fontweight='bold')
ax.set_title('7B: OOD Overall (31 tasks)', fontweight='bold')
ax.set_ylim(0, 22)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, ood_7b):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
           f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)
# Highlight Phase-Adapt as winner
bars[2].set_edgecolor('red')
bars[2].set_linewidth(3)

# 7B OOD Stretch
ax = axes[0, 2]
stretch_7b = [8.18, 10.55, 12.73]
bars = ax.bar(models, stretch_7b, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Pass@1 (%)', fontweight='bold')
ax.set_title('7B: OOD Stretch (10 hardest)', fontweight='bold', color='darkred')
ax.set_ylim(0, 16)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, stretch_7b):
    improvement = ((val - stretch_7b[0]) / stretch_7b[0] * 100) if val != stretch_7b[0] else 0
    label = f'{val:.1f}%' if improvement == 0 else f'{val:.1f}%\n(+{improvement:.0f}%)'
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
           label, ha='center', fontweight='bold', fontsize=10)
bars[2].set_edgecolor('red')
bars[2].set_linewidth(3)

# 3B In-Domain
ax = axes[1, 0]
indomain_3b = [26.20, 23.90, 22.48]
bars = ax.bar(models, indomain_3b, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Pass@1 (%)', fontweight='bold')
ax.set_title('3B: In-Domain (10 tasks)', fontweight='bold')
ax.set_ylim(0, 32)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, indomain_3b):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.8,
           f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)
ax.axhline(y=indomain_3b[0], color='gray', linestyle='--', alpha=0.5, linewidth=1)

# 3B OOD Overall
ax = axes[1, 1]
ood_3b = [15.35, 13.48, 11.84]
bars = ax.bar(models, ood_3b, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Pass@1 (%)', fontweight='bold')
ax.set_title('3B: OOD Overall (31 tasks)', fontweight='bold')
ax.set_ylim(0, 19)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, ood_3b):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
           f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)

# 3B OOD Stretch
ax = axes[1, 2]
stretch_3b = [9.18, 6.73, 10.55]
bars = ax.bar(models, stretch_3b, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Pass@1 (%)', fontweight='bold')
ax.set_title('3B: OOD Stretch (10 hardest)', fontweight='bold', color='darkred')
ax.set_ylim(0, 13)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, stretch_3b):
    improvement = ((val - stretch_3b[0]) / stretch_3b[0] * 100) if val != stretch_3b[0] else 0
    label = f'{val:.1f}%' if improvement == 0 else f'{val:.1f}%\n({improvement:+.0f}%)'
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.4,
           label, ha='center', fontweight='bold', fontsize=10)
bars[2].set_edgecolor('red')
bars[2].set_linewidth(3)

fig.suptitle('Phase-Adapt Consistently Improves OOD Stretch Tasks', 
            fontsize=16, fontweight='bold', y=0.995)

# Add annotations
fig.text(0.5, 0.48, '─'*140, ha='center', fontsize=8)
fig.text(0.18, 0.02, '7B: Baseline best in-domain → Phase-Adapt best OOD', 
        ha='center', fontsize=11, style='italic', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
fig.text(0.65, 0.02, '3B: Baseline best overall → Phase-Adapt best on stretch', 
        ha='center', fontsize=11, style='italic', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

plt.tight_layout(rect=[0, 0.04, 1, 0.99])
plt.savefig(output_dir / 'fig1_indomain_vs_ood_reversal.png', dpi=300)
print('✓ Saved: fig1_indomain_vs_ood_reversal.png')

# ============================================================================
# FIGURE 2: Stretch Task Improvement Comparison
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

categories = ['7B', '3B']
x = np.arange(len(categories))
width = 0.25

continuous_improvements = [28.9, -26.7]
phaseadapt_improvements = [55.6, 14.9]

bars1 = ax.bar(x - width/2, continuous_improvements, width, 
              label='Continuous Zero', color='#e67e22', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, phaseadapt_improvements, width,
              label='Phase-Adapt', color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_ylabel('Improvement vs Baseline (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Model Size', fontsize=13, fontweight='bold')
ax.set_title('Stretch Task Improvement: Phase-Adapt Wins Across Model Sizes', 
            fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(-35, 65)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:+.1f}%', ha='center', 
               va='bottom' if height >= 0 else 'top', 
               fontsize=11, fontweight='bold')

# Highlight Phase-Adapt bars
bars2[0].set_edgecolor('red')
bars2[0].set_linewidth(3)
bars2[1].set_edgecolor('red')
bars2[1].set_linewidth(3)

plt.tight_layout()
plt.savefig(output_dir / 'fig2_stretch_improvement.png', dpi=300)
print('✓ Saved: fig2_stretch_improvement.png')

# ============================================================================
# FIGURE 3: Core vs Stretch Performance
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 7B
ax = axes[0]
task_types = ['Core\n(21 tasks)', 'Stretch\n(10 tasks)']
x = np.arange(len(task_types))
width = 0.25

baseline_7b = [20.65, 8.18]
continuous_7b = [20.40, 10.55]
phaseadapt_7b = [20.85, 12.73]

bars1 = ax.bar(x - width, baseline_7b, width, label='Baseline', color='#95a5a6', alpha=0.8)
bars2 = ax.bar(x, continuous_7b, width, label='Continuous', color='#e67e22', alpha=0.8)
bars3 = ax.bar(x + width, phaseadapt_7b, width, label='Phase-Adapt', color='#9b59b6', alpha=0.8)

ax.set_ylabel('Pass@1 (%)', fontsize=12, fontweight='bold')
ax.set_title('7B: Core vs Stretch Performance', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(task_types, fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 24)

# Highlight stretch bars
bars1[1].set_edgecolor('black')
bars1[1].set_linewidth(2)
bars2[1].set_edgecolor('black')
bars2[1].set_linewidth(2)
bars3[1].set_edgecolor('red')
bars3[1].set_linewidth(3)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=9)

# 3B
ax = axes[1]
baseline_3b = [18.75, 9.18]
continuous_3b = [17.20, 6.73]
phaseadapt_3b = [12.55, 10.55]

bars1 = ax.bar(x - width, baseline_3b, width, label='Baseline', color='#95a5a6', alpha=0.8)
bars2 = ax.bar(x, continuous_3b, width, label='Continuous', color='#e67e22', alpha=0.8)
bars3 = ax.bar(x + width, phaseadapt_3b, width, label='Phase-Adapt', color='#9b59b6', alpha=0.8)

ax.set_ylabel('Pass@1 (%)', fontsize=12, fontweight='bold')
ax.set_title('3B: Core vs Stretch Performance', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(task_types, fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 22)

bars1[1].set_edgecolor('black')
bars1[1].set_linewidth(2)
bars2[1].set_edgecolor('black')
bars2[1].set_linewidth(2)
bars3[1].set_edgecolor('red')
bars3[1].set_linewidth(3)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Phase-Adapt Excels on Stretch Tasks Despite Lower Core Performance', 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'fig3_core_vs_stretch.png', dpi=300)
print('✓ Saved: fig3_core_vs_stretch.png')

# ============================================================================
# FIGURE 4: Training Stability
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

strategies = ['Baseline', 'Continuous\n(Mask)', 'Continuous\n(Zero)', 'Phase-Adapt\n(Old)', 'Phase-Adapt\n(7B)']
variances_3b = [1.30, 0.14, 2.29, 1.19, 0.06]
means_3b = [26.20, 25.18, 23.90, 21.36, 22.48]

colors_strat = ['#95a5a6', '#3498db', '#e67e22', '#c39bd3', '#9b59b6']

bars = ax.bar(strategies, variances_3b, color=colors_strat, alpha=0.8, edgecolor='black', linewidth=1.5)

# Highlight Phase-Adapt 7B as most stable
bars[4].set_edgecolor('red')
bars[4].set_linewidth(3)

ax.set_ylabel('Standard Deviation (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Strategy', fontsize=13, fontweight='bold')
ax.set_title('Training Stability (3B Models): Phase-Adapt Most Consistent', 
            fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 2.8)

for bar, var, mean in zip(bars, variances_3b, means_3b):
    ax.text(bar.get_x() + bar.get_width()/2., var + 0.1,
           f'σ={var:.2f}%\n({mean:.1f}%)', ha='center', fontsize=10, fontweight='bold')

ax.text(0.5, 0.95, 'Lower is better (more stable training)', 
       transform=ax.transAxes, ha='center', fontsize=11, style='italic',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig(output_dir / 'fig4_training_stability.png', dpi=300)
print('✓ Saved: fig4_training_stability.png')

print()
print('='*80)
print('ALL PAPER FIGURES GENERATED!')
print('='*80)
print()
print(f'Output directory: {output_dir.absolute()}')
print()
print('Files created:')
print('  1. fig1_indomain_vs_ood_reversal.png - Main result')
print('  2. fig2_stretch_improvement.png - Stretch task improvements')
print('  3. fig3_core_vs_stretch.png - Task type breakdown')
print('  4. fig4_training_stability.png - Training consistency')
print()
print('These figures are ready for paper inclusion!')
print('='*80)
