import pandas as pd

print('='*80)
print('3B OOD RESULTS - DETAILED BREAKDOWN')
print('='*80)
print()

# Load data
df_split = pd.read_csv('validation/results_3b_ood_best3/summary_per_split.csv')
df_model = pd.read_csv('validation/results_3b_ood_best3/summary_per_model.csv')

print('Overall Performance (All 31 Tasks):')
print('-'*80)
for _, row in df_model.iterrows():
    model = row['model'].replace('_seed0', '')
    pass1 = row['pass1_mean'] * 100
    print(f'{model:20s}: {pass1:6.2f}%')

print()
print('='*80)
print('Performance by Task Difficulty:')
print('='*80)
print()
print(f'{"Model":<25} {"Overall":>10} {"Core (21)":>12} {"Stretch (10)":>15}')
print('-'*80)

models = df_split['model'].unique()
for model in models:
    model_name = model.replace('_seed0', '')
    overall = df_model[df_model['model'] == model]['pass1_mean'].values[0] * 100
    core = df_split[(df_split['model'] == model) & (df_split['split'] == 'core')]['pass1_mean'].values[0] * 100
    stretch = df_split[(df_split['model'] == model) & (df_split['split'] == 'stretch')]['pass1_mean'].values[0] * 100
    print(f'{model_name:<25} {overall:>9.2f}%  {core:>11.2f}%  {stretch:>14.2f}%')

print()
print('='*80)
print('COMPARISON TO 7B:')
print('='*80)
print()
print(f'{"Model":<25} {"3B OOD":>10} {"7B OOD":>10} {"Difference":>12}')
print('-'*80)
print(f'{"Baseline":<25} {15.35:>9.2f}%  {16.23:>9.2f}%  {15.35-16.23:>11.2f}%')
print(f'{"Continuous Zero":<25} {13.48:>9.2f}%  {16.90:>9.2f}%  {13.48-16.90:>11.2f}%')
print(f'{"Phase-Adapt":<25} {11.84:>9.2f}%  {17.97:>9.2f}%  {11.84-17.97:>11.2f}%')

print()
print('='*80)
print('RELATIVE TO BASELINE (Within Size):')
print('='*80)
print()
baseline_3b = 15.35
baseline_7b = 16.23
print(f'{"3B:":<10} Continuous Zero: {((13.48-baseline_3b)/baseline_3b)*100:+.1f}%  |  Phase-Adapt: {((11.84-baseline_3b)/baseline_3b)*100:+.1f}%')
print(f'{"7B:":<10} Continuous Zero: {((16.90-baseline_7b)/baseline_7b)*100:+.1f}%  |  Phase-Adapt: {((17.97-baseline_7b)/baseline_7b)*100:+.1f}%')

print()
print('='*80)
print('STRETCH TASK COMPARISON:')
print('='*80)
print()
print(f'{"Model":<25} {"3B Stretch":>12} {"7B Stretch":>12} {"3B vs Baseline":>18}')
print('-'*80)
baseline_stretch_3b = 9.18
baseline_stretch_7b = 8.18
print(f'{"Baseline":<25} {baseline_stretch_3b:>11.2f}%  {baseline_stretch_7b:>11.2f}%  {"reference":>18s}')
print(f'{"Continuous Zero":<25} {6.73:>11.2f}%  {10.55:>11.2f}%  {((6.73-baseline_stretch_3b)/baseline_stretch_3b)*100:>17.1f}%')
print(f'{"Phase-Adapt":<25} {10.55:>11.2f}%  {12.73:>11.2f}%  {((10.55-baseline_stretch_3b)/baseline_stretch_3b)*100:>17.1f}%')

print()
print('='*80)
print('🔬 CRITICAL FINDING: PATTERN BREAKS!')
print('='*80)
print()
print('7B Pattern: Baseline → Phase-Adapt → Continuous (strategies WIN on OOD)')
print('3B Pattern: Baseline → Continuous → Phase-Adapt (strategies LOSE on OOD)')
print()
print('Key Observations:')
print('  1. 3B strategies HURT OOD performance (-12.2% to -22.9%)')
print('  2. 7B strategies HELP OOD performance (+4.1% to +10.7%)')
print('  3. Pattern REVERSES between model sizes!')
print('  4. Phase-Adapt does best on stretch tasks for 3B (+14.9%)')
print('     but overall still worse than baseline')
print()
print('Implication: Strategies require minimum model capacity (≥7B)')
print('='*80)
