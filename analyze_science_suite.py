import os
import json
import argparse
import re
from glob import glob
from typing import Dict, Any, List, Optional

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


ROOT = os.path.dirname(__file__)


def resolve_dirs(suite_name: str):
    # Allow analyzing any subfolder under exp_output
    sci_dir = os.path.join(ROOT, 'exp_output', suite_name)
    out_dir = os.path.join(sci_dir, '_analysis')
    os.makedirs(out_dir, exist_ok=True)
    return sci_dir, out_dir


def load_summary_or_last_eval(run_dir: str) -> Dict[str, Any]:
    summ_path = os.path.join(run_dir, 'summary.json')
    if os.path.isfile(summ_path):
        with open(summ_path, 'r') as f:
            return json.load(f)

    # Fallback: parse run_log.json and get the last eval block
    log_path = os.path.join(run_dir, 'run_log.json')
    if not os.path.isfile(log_path):
        return {}
    with open(log_path, 'r') as f:
        log = json.load(f)
    steps = log.get('steps', {})
    if not steps:
        return {}
    eval_steps = sorted([int(s) for s, v in steps.items() if isinstance(v, dict) and 'eval' in v])
    if not eval_steps:
        return {}
    last = str(eval_steps[-1])
    eval_block = steps[last]['eval']
    metrics = eval_block.get('metrics', {})
    pass_keys = [k for k in metrics.keys() if k.startswith('pass_at_')]
    pass_at = metrics[pass_keys[0]] if pass_keys else None
    return {
        'step': int(last),
        'pass_at_k': pass_at,
        'avg_format_reward': metrics.get('avg_format_reward'),
        'num_eval_problems': metrics.get('num_eval_problems'),
        'per_problem_type': metrics.get('per_problem_type', {}),
    }


def collect_runs(science_dir: str) -> Dict[str, Dict[str, Any]]:
    runs = {}
    for d in glob(os.path.join(science_dir, '*/')):
        name = os.path.basename(os.path.normpath(d))
        if name == '_analysis':
            continue
        data = load_summary_or_last_eval(d)
        if data:
            runs[name] = data
    return runs


def filter_runs_by_seed(runs: Dict[str, Dict[str, Any]], seed: Optional[int]) -> Dict[str, Dict[str, Any]]:
    if seed is None:
        return runs
    suffix = f"_seed{seed}"
    return {k: v for k, v in runs.items() if k.endswith(suffix)}


def filter_runs_by_includes(runs: Dict[str, Dict[str, Any]], includes: Optional[List[str]], seed: Optional[int]) -> Dict[str, Dict[str, Any]]:
    if not includes:
        return runs
    expected = []
    for base in includes:
        expected.append(f"{base}_seed{seed}" if seed is not None else base)
    return {k: v for k, v in runs.items() if k in expected}


def extract_seed(name: str) -> Optional[int]:
    m = re.search(r"_seed(\d+)$", name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def group_runs_by_seed(runs: Dict[str, Dict[str, Any]]) -> Dict[int, Dict[str, Dict[str, Any]]]:
    grouped: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for name, data in runs.items():
        s = extract_seed(name)
        if s is None:
            continue
        grouped.setdefault(s, {})[name] = data
    return grouped


def find_baseline_key(runs: Dict[str, Dict[str, Any]]) -> Optional[str]:
    candidates = [k for k in runs if k.startswith('baseline_nomask')]
    if not candidates:
        candidates = [k for k in runs if k.startswith('baseline')]
    if not candidates:
        return None
    def step_of(k: str) -> int:
        v = runs.get(k) or {}
        s = v.get('step')
        try:
            return int(s) if s is not None else -1
        except Exception:
            return -1
    return sorted(candidates, key=step_of, reverse=True)[0]


def write_json(path: str, obj: Any):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def to_csv_value(x: Any) -> str:
    if x is None:
        return ''
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def save_overall_csv(runs: Dict[str, Dict[str, Any]], out_dir: str) -> str:
    rows = ['run,step,pass_at_1,avg_format_reward,num_eval_problems']
    for name, r in sorted(runs.items()):
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


def save_per_problem_csv(runs: Dict[str, Dict[str, Any]], out_dir: str) -> str:
    problem_types = set()
    for r in runs.values():
        problem_types.update(r.get('per_problem_type', {}).keys())
    problem_types = sorted(problem_types)

    header = ['run'] + [f"{pt}:pass_at_1" for pt in problem_types]
    rows = [','.join(header)]
    for name, r in sorted(runs.items()):
        per = r.get('per_problem_type', {})
        vals = [name]
        for pt in problem_types:
            v = per.get(pt, {}).get('pass_at_1')
            vals.append(to_csv_value(v))
        rows.append(','.join(vals))
    path = os.path.join(out_dir, 'per_problem.csv')
    with open(path, 'w') as f:
        f.write('\n'.join(rows))
    return path


def plot_overall_bar(runs: Dict[str, Dict[str, Any]], out_dir: str, title_prefix: str = 'Science Suite', colors: Optional[List[str]] = None) -> Optional[str]:
    if not HAVE_MPL:
        return None
    names = []
    vals = []
    for name, r in sorted(runs.items()):
        names.append(name)
        vals.append(r.get('pass_at_k') or 0.0)
    plt.figure(figsize=(10, 4))
    # Use distinct colors for each bar if provided; else choose a qualitative palette
    if colors is None:
        palette = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2', '#EECA3B', '#B279A2', '#FF9DA7', '#9D755D', '#BAB0AC']
        use_colors = [palette[i % len(palette)] for i in range(len(names))]
    else:
        use_colors = colors
    bars = plt.bar(names, vals, color=use_colors)
    plt.ylabel('pass@1 (%)')
    plt.title(f'{title_prefix}: Overall pass@1 by Run')
    plt.xticks(rotation=30, ha='right')
    # Add value labels above each bar and give slight headroom
    try:
        ymax = max(vals) if vals else 0.0
        for b, v in zip(bars, vals):
            plt.text(
                b.get_x() + b.get_width() / 2.0,
                b.get_height() + (ymax * 0.02 if ymax else 0.02),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.5)
            )
        if ymax:
            plt.ylim(0, ymax * 1.10)
    except Exception:
        pass
    outp = os.path.join(out_dir, 'overall_pass_at1.png')
    plt.tight_layout()
    plt.savefig(outp)
    plt.close()
    return outp


def plot_relative_to_baseline(runs: Dict[str, Dict[str, Any]], out_dir: str, baseline_key: str = 'baseline_nomask', title_suffix: Optional[str] = None, title_prefix: str = 'Science Suite') -> Optional[str]:
    """Plot per-problem delta pass@1 relative to a baseline.

    Includes the baseline as a zero-delta series so all selected runs appear.
    """
    if not HAVE_MPL:
        print('Warning: matplotlib not available; skipping delta plot')
        return None

    # Resolve baseline: exact match or prefix match (for *_seedN)
    resolved_key: Optional[str] = None
    if baseline_key in runs:
        resolved_key = baseline_key
    else:
        candidates = [k for k in runs if k.startswith(baseline_key)]
        if candidates:
            candidates.sort(key=lambda k: (not k.endswith('seed1'), k))
            resolved_key = candidates[0]
            print(f"Info: baseline '{baseline_key}' not found; using '{resolved_key}' for delta plot")
    if not resolved_key:
        print(f"Warning: cannot resolve baseline '{baseline_key}'; skipping delta plot")
        return None

    base = runs[resolved_key]
    base_pt = base.get('per_problem_type', {})
    if not base_pt:
        print('Warning: baseline missing per_problem_type; skipping delta plot')
        return None

    problem_types = sorted(base_pt.keys())
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(problem_types))
    width = max(0.8 / max(1, len(runs)), 0.15)
    i = 0
    baseline_drawn_hline = False
    for name, r in sorted(runs.items()):
        if name == resolved_key:
            deltas = [0.0] * len(problem_types)
        else:
            deltas = []
            for pt in problem_types:
                b = base_pt.get(pt, {}).get('pass_at_1') or 0.0
                v = r.get('per_problem_type', {}).get(pt, {}).get('pass_at_1') or 0.0
                deltas.append(v - b)

        # Draw the bars for this series; suppress baseline label to avoid legend duplication
        positions = [xx + i * width for xx in x]
        bar_label = None if name == resolved_key else name
        bars = ax.bar(positions, deltas, width=width, label=bar_label)

        # If this is the baseline (all zeros), also draw a small marker/line so it is visible
        if name == resolved_key:
            # Draw a short horizontal line segment at y=0 for each category to make baseline visible
            # Label only once to avoid legend duplication
            base_label = name if not baseline_drawn_hline else None
            for pos in positions:
                ax.hlines(0, pos - width * 0.45, pos + width * 0.45,
                          colors=bars.patches[0].get_facecolor(), linewidth=2, label=base_label)
                base_label = None
            baseline_drawn_hline = True
        i += 1
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_xticks([xx + (i - 1) * width / 2 for xx in x])
    ax.set_xticklabels(problem_types, rotation=30, ha='right')
    ax.set_ylabel('Delta pass@1 vs baseline (pp)')
    title = f'{title_prefix}: Per-Problem Improvement vs Baseline (higher is better)'
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title)
    ax.legend()
    outp = os.path.join(out_dir, 'per_problem_delta_vs_baseline.png')
    plt.tight_layout()
    plt.savefig(outp)
    plt.close()
    return outp


def plot_per_problem_absolute(runs: Dict[str, Dict[str, Any]], out_dir: str, title_prefix: str = 'Science Suite') -> Optional[str]:
    """Plot per-problem absolute pass@1 for all provided runs (grouped bars per problem).

    Expects each run to have per_problem_type with pass_at_1 values.
    """
    if not HAVE_MPL:
        return None
    # Collect the union of problem types
    problem_types = set()
    for r in runs.values():
        problem_types.update((r.get('per_problem_type') or {}).keys())
    problem_types = sorted(problem_types)
    if not problem_types:
        return None

    # Prepare data matrix: problems x runs
    run_names = [name for name, _ in sorted(runs.items())]
    vals_by_run = []
    for name in run_names:
        per = runs[name].get('per_problem_type', {})
        vals = [(per.get(pt, {}).get('pass_at_1') or 0.0) for pt in problem_types]
        vals_by_run.append(vals)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = list(range(len(problem_types)))
    width = max(0.8 / max(1, len(run_names)), 0.20)

    # Distinct colors for each run
    palette = ['#4C78A8', '#F58518', '#54A24B', '#E45756', '#72B7B2', '#EECA3B']
    colors = [palette[i % len(palette)] for i in range(len(run_names))]

    for i, (name, vals) in enumerate(zip(run_names, vals_by_run)):
        ax.bar([xx + i * width for xx in x], vals, width=width, label=name, color=colors[i])

    ax.set_xticks([xx + (len(run_names) - 1) * width / 2 for xx in x])
    ax.set_xticklabels(problem_types, rotation=30, ha='right')
    ax.set_ylabel('pass@1 (%)')
    ax.set_title(f'{title_prefix}: Per-Problem Performance (grouped by model)')
    ax.legend()
    outp = os.path.join(out_dir, 'per_problem_three_models.png')
    plt.tight_layout()
    plt.savefig(outp)
    plt.close()
    return outp


def main():
    parser = argparse.ArgumentParser(description='Aggregate science results into CSVs and plots')
    parser.add_argument('--suite', default='science_suite', help='Which subfolder under exp_output to analyze (e.g., science_suite, science2_suite, science2_cont_suite, science2_phase_adapt_suite)')
    parser.add_argument('--seed', type=int, default=None, help='Filter runs to those ending with _seed{seed} (science2)')
    parser.add_argument('--include', type=str, default=None, help='Comma-separated base run names to include (e.g., baseline_nomask,softmask_every10_wt05,fullzero_every10_nothresh). Seed suffix is appended if --seed is set.')
    parser.add_argument('--baseline', default='auto', help="Baseline run key or 'auto' to detect baseline_nomask*")
    parser.add_argument('--per_seed', action='store_true', help='For science2: auto-generate per-seed delta plots comparing runs with the same seed')
    args = parser.parse_args()

    sci_dir, out_dir = resolve_dirs(args.suite)
    runs_all = collect_runs(sci_dir)

    if args.per_seed:
        # Auto-generate per-seed comparisons (use includes if provided)
        grouped = group_runs_by_seed(runs_all)
        seeds = [args.seed] if args.seed is not None else sorted(grouped.keys())
        if not seeds:
            print('No seed-suffixed runs found under', sci_dir)
            return
        for s in seeds:
            seed_dir = os.path.join(out_dir, f'seed_{s}')
            os.makedirs(seed_dir, exist_ok=True)
            runs_seed = grouped.get(s, {})
            if args.include:
                includes = [t.strip() for t in args.include.split(',') if t.strip()]
                runs_seed = filter_runs_by_includes(runs_seed, includes, s)
            if not runs_seed:
                print(f'No runs for seed {s} after filtering; skipping')
                continue
            write_json(os.path.join(seed_dir, 'runs_parsed.json'), runs_seed)
            overall_csv = save_overall_csv(runs_seed, seed_dir)
            per_problem_csv = save_per_problem_csv(runs_seed, seed_dir)
            _ = plot_overall_bar(runs_seed, seed_dir)
            base_key = f'baseline_nomask_seed{s}' if args.baseline == 'auto' else args.baseline
            delta_plot = plot_relative_to_baseline(runs_seed, seed_dir, baseline_key=base_key, title_suffix=f'seed {s}')
            print(f'Seed {s} analysis written to: {seed_dir}')
            print(' - overall.csv:', overall_csv)
            print(' - per_problem.csv:', per_problem_csv)
            if delta_plot:
                print(' - per_problem_delta_vs_baseline.png:', delta_plot)
        return

    # Non per-seed: aggregate or seed-filtered view
    runs = filter_runs_by_seed(runs_all, args.seed)
    if args.include:
        includes = [s.strip() for s in args.include.split(',') if s.strip()]
        runs = filter_runs_by_includes(runs, includes, args.seed)
    if not runs:
        print('No runs found in', sci_dir)
        return

    write_json(os.path.join(out_dir, 'runs_parsed.json'), runs)
    overall_csv = save_overall_csv(runs, out_dir)
    per_problem_csv = save_per_problem_csv(runs, out_dir)
    overall_plot = plot_overall_bar(runs, out_dir, title_prefix=args.suite)

    baseline_key = args.baseline
    if baseline_key == 'auto':
        baseline_key = find_baseline_key(runs) or 'baseline_nomask'
    delta_plot = plot_relative_to_baseline(runs, out_dir, baseline_key=baseline_key, title_prefix=args.suite) if baseline_key else None

    print('Analysis written to:', out_dir)
    print(' - overall.csv:', overall_csv)
    print(' - per_problem.csv:', per_problem_csv)
    if overall_plot:
        print(' - overall_pass_at1.png:', overall_plot)
    if delta_plot:
        print(' - per_problem_delta_vs_baseline.png:', delta_plot)
    else:
        print(' - Delta plot skipped (no baseline or plotting unavailable)')


if __name__ == '__main__':
    main()
