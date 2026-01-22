import os
import sys
import json
from glob import glob
from typing import Dict, Any, Tuple

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analyze_science_suite import plot_overall_bar, plot_relative_to_baseline, plot_per_problem_absolute, to_csv_value

BASE_DIR = os.path.join(ROOT, 'exp_output', 'science2_suite')
CONT_DIR = os.path.join(ROOT, 'exp_output', 'science2_cont_suite')
PHASE_DIR = os.path.join(ROOT, 'exp_output', 'science2_phase_adapt_suite')


def load_summary(run_dir: str) -> Dict[str, Any]:
    p = os.path.join(run_dir, 'summary.json')
    if not os.path.isfile(p):
        return {}
    with open(p, 'r') as f:
        return json.load(f)


def best_baseline() -> Tuple[str, Dict[str, Any]]:
    best_key = None
    best_data: Dict[str, Any] = {}
    best_val = float('-inf')
    for d in glob(os.path.join(BASE_DIR, 'baseline_nomask_*')):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(os.path.normpath(d))
        s = load_summary(d)
        v = s.get('pass_at_k')
        if isinstance(v, (int, float)) and v > best_val:
            best_val = float(v)
            best_key = name
            best_data = s
    if not best_key:
        raise RuntimeError('No baseline_nomask_* runs found under science2_suite')
    return best_key, best_data


def best_continuous() -> Tuple[str, Dict[str, Any]]:
    best_key = None
    best_data: Dict[str, Any] = {}
    best_val = float('-inf')
    for d in glob(os.path.join(CONT_DIR, '*/')):
        name = os.path.basename(os.path.normpath(d))
        if name == '_analysis':
            continue
        s = load_summary(d)
        v = s.get('pass_at_k')
        if isinstance(v, (int, float)) and v > best_val:
            best_val = float(v)
            best_key = name
            best_data = s
    if not best_key:
        raise RuntimeError('No continuous runs with summary.json found under science2_cont_suite')
    return best_key, best_data


def best_phase() -> Tuple[str, Dict[str, Any]]:
    best_key = None
    best_data: Dict[str, Any] = {}
    best_val = float('-inf')
    for d in glob(os.path.join(PHASE_DIR, '*/')):
        name = os.path.basename(os.path.normpath(d))
        if name == '_analysis':
            continue
        s = load_summary(d)
        v = s.get('pass_at_k')
        if isinstance(v, (int, float)) and v > best_val:
            best_val = float(v)
            best_key = name
            best_data = s
    if not best_key:
        raise RuntimeError('No phase-adapt runs with summary.json found under science2_phase_adapt_suite')
    return best_key, best_data


def save_overall_csv_three(runs: Dict[str, Dict[str, Any]], out_dir: str) -> str:
    rows = ['run,step,pass_at_1,avg_format_reward,num_eval_problems']
    for name, r in runs.items():
        rows.append(','.join([
            name,
            to_csv_value(r.get('step')),
            to_csv_value(r.get('pass_at_k')),
            to_csv_value(r.get('avg_format_reward')),
            to_csv_value(r.get('num_eval_problems')),
        ]))
    path = os.path.join(out_dir, 'overall.csv')
    with open(path, 'w') as f:
        f.write('\n'.join(rows))
    return path


def main():
    out_dir = os.path.join(PHASE_DIR, '_analysis', 'best_three')
    os.makedirs(out_dir, exist_ok=True)

    base_key, base = best_baseline()
    cont_key, cont = best_continuous()
    phase_key, phase = best_phase()

    runs = {
        base_key: base,
        cont_key: cont,
        phase_key: phase,
    }

    with open(os.path.join(out_dir, 'selected_runs.json'), 'w') as f:
        json.dump({'baseline_key': base_key, 'cont_key': cont_key, 'phase_key': phase_key,
                   'baseline': base, 'cont': cont, 'phase': phase}, f, indent=2)

    _ = save_overall_csv_three(runs, out_dir)

    title_prefix = 'Science2: Best Baseline vs Best Continuous vs Best Phase'
    # Use distinct colors for clarity: baseline (blue), continuous (orange), phase (green)
    overall_plot = plot_overall_bar(runs, out_dir, title_prefix=title_prefix,
                                    colors=['#4C78A8', '#F58518', '#54A24B'])
    delta_plot = plot_relative_to_baseline(runs, out_dir, baseline_key=base_key,
                                           title_prefix='Science2: Δ vs Baseline (Best Three)')
    abs_plot = plot_per_problem_absolute(runs, out_dir, title_prefix='Science2: Best Baseline vs Continuous vs Phase')

    print('Analysis written to:', out_dir)
    print(' - overall.csv:', os.path.join(out_dir, 'overall.csv'))
    if overall_plot:
        print(' - overall_pass_at1.png:', overall_plot)
    if delta_plot:
        print(' - per_problem_delta_vs_baseline.png:', delta_plot)
    if abs_plot:
        print(' - per_problem_three_models.png:', abs_plot)


if __name__ == '__main__':
    main()
