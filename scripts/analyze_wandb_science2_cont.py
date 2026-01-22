import os
import re
import json
import argparse
from typing import Dict, Any, Optional, Tuple

try:
    import wandb
except Exception as e:
    wandb = None

# Reuse plotting from local analyzer
try:
    from analyze_science_suite import plot_overall_bar, plot_relative_to_baseline, to_csv_value
except Exception:
    plot_overall_bar = None
    plot_relative_to_baseline = None


def _get_metric(summary: Dict[str, Any], keys=(
    'eval/pass_at_1',
    'pass_at_1',
    'metrics/pass_at_1',
    'eval/pass@1',
)) -> Optional[float]:
    for k in keys:
        v = summary.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    # Sometimes nested dict under 'eval'
    eval_block = summary.get('eval')
    if isinstance(eval_block, dict):
        v = eval_block.get('pass_at_1') or eval_block.get('pass@1')
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _get_step(summary: Dict[str, Any]) -> Optional[int]:
    for k in ('eval/step', 'step', '_step'):
        v = summary.get(k)
        if isinstance(v, (int, float)):
            try:
                return int(v)
            except Exception:
                pass
    return None


def _get_per_problem(summary: Dict[str, Any], key='per_problem_type') -> Dict[str, Dict[str, float]]:
    val = summary.get(key)
    if isinstance(val, dict):
        return val
    # Sometimes nested under eval
    eval_block = summary.get('eval')
    if isinstance(eval_block, dict):
        pp = eval_block.get(key)
        if isinstance(pp, dict):
            return pp
    return {}


def _classify_run(name: str, tags: Optional[list]) -> str:
    tags = tags or []
    name_l = name.lower()
    tag_l = [t.lower() for t in tags]
    if 'baseline' in name_l or 'baseline_nomask' in name_l or 'baseline' in tag_l:
        return 'baseline'
    return 'cont'


def fetch_best_runs(entity: Optional[str], project: str, run_name_regex: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if wandb is None:
        raise RuntimeError("wandb is not installed. Install with: pip install wandb")

    api = wandb.Api()
    proj_path = f"{entity}/{project}" if entity else project
    runs = api.runs(proj_path)

    name_pat = re.compile(run_name_regex) if run_name_regex else None

    best_baseline = None  # (metric, run)
    best_cont = None

    for r in runs:
        if r.state != 'finished':
            continue
        if name_pat and not name_pat.search(r.name or r.id):
            continue

        s = dict(r.summary or {})
        metric = _get_metric(s)
        if metric is None:
            continue
        kind = _classify_run(r.name or r.id, getattr(r, 'tags', []))
        step = _get_step(s)
        per_problem = _get_per_problem(s)
        num_eval = s.get('num_eval_problems') or s.get('eval/num_eval_problems')

        record = {
            'run_path': r.path,
            'name': r.name or r.id,
            'step': step,
            'pass_at_k': float(metric),
            'avg_format_reward': s.get('avg_format_reward') or s.get('eval/avg_format_reward'),
            'num_eval_problems': int(num_eval) if isinstance(num_eval, (int, float)) else num_eval,
            'per_problem_type': per_problem,
        }

        if kind == 'baseline':
            if best_baseline is None or metric > best_baseline[0]:
                best_baseline = (metric, record)
        else:
            if best_cont is None or metric > best_cont[0]:
                best_cont = (metric, record)

    if not best_baseline or not best_cont:
        raise RuntimeError("Could not find both baseline and continuous runs with metrics in the specified project.")

    return best_baseline[1], best_cont[1]


def save_overall_csv_two(runs: Dict[str, Dict[str, Any]], out_dir: str) -> str:
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
    parser = argparse.ArgumentParser(description='Generate Science2 continuous plots from Weights & Biases runs')
    parser.add_argument('--entity', type=str, default=None, help='W&B entity (org/user). If omitted, uses default W&B config context.')
    parser.add_argument('--project', type=str, default='nano_grpo-science2-continuous', help='W&B project name')
    parser.add_argument('--name_regex', type=str, default=None, help='Optional regex to filter run names considered')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory for plots (default: exp_output/science2_cont_suite/_analysis/wandb_best)')

    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), '..', 'exp_output', 'science2_cont_suite', '_analysis', 'wandb_best')
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    baseline, cont = fetch_best_runs(args.entity, args.project, args.name_regex)

    # Compose minimal runs dict
    runs = {
        f"baseline:{baseline['name']}": baseline,
        f"cont:{cont['name']}": cont,
    }

    # Persist snapshot for traceability
    with open(os.path.join(out_dir, 'selected_runs.json'), 'w') as f:
        json.dump({'baseline': baseline, 'cont': cont}, f, indent=2)

    # Save CSV and plots
    _ = save_overall_csv_two(runs, out_dir)

    if plot_overall_bar is None or plot_relative_to_baseline is None:
        print('Warning: plotting functions unavailable; ensure analyze_science_suite.py is importable.')
        return

    overall_plot = plot_overall_bar(runs, out_dir, title_prefix='Science2 (W&B): Best Baseline vs Best Continuous')

    # Use the concrete baseline key
    baseline_key = [k for k in runs.keys() if k.startswith('baseline:')][0]
    delta_plot = plot_relative_to_baseline(runs, out_dir, baseline_key=baseline_key, title_prefix='Science2 (W&B)')

    print('W&B analysis written to:', out_dir)
    print(' - overall.csv:', os.path.join(out_dir, 'overall.csv'))
    if overall_plot:
        print(' - overall_pass_at1.png:', overall_plot)
    if delta_plot:
        print(' - per_problem_delta_vs_baseline.png:', delta_plot)


if __name__ == '__main__':
    main()
