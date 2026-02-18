#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from statistics import mean


def read_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def collect_phase_adapt(exp_output_dir, include_lengths):
    root = os.path.join(exp_output_dir, "science2_phase_adapt_suite")
    if not os.path.isdir(root):
        return []
    pat = re.compile(r"^(phase_adapt_[a-z_]+|phase_split_masking)(?:_len512)?_seed(\d+)$")
    records = []
    for d in os.listdir(root):
        m = pat.match(d)
        if not m:
            continue
        length = 512 if "_len512_" in d else 1024
        if include_lengths and length not in include_lengths:
            continue
        summ = os.path.join(root, d, "summary.json")
        s = read_json(summ)
        if not s:
            continue
        records.append({
            "suite": "phase_adapt",
            "variant": m.group(1),
            "length": length,
            "seed": int(m.group(2)),
            "pass_at_1": s.get("pass_at_k"),
            "format": s.get("avg_format_reward"),
            "per_problem": s.get("per_problem_type", {}),
        })
    return records


def collect_continuous(exp_output_dir):
    root = os.path.join(exp_output_dir, "science2_cont_suite")
    if not os.path.isdir(root):
        return []
    pat = re.compile(r"^(cont_[a-z0-9_]+)_seed(\d+)$")
    records = []
    for d in os.listdir(root):
        m = pat.match(d)
        if not m:
            continue
        summ = os.path.join(root, d, "summary.json")
        s = read_json(summ)
        if not s:
            continue
        records.append({
            "suite": "continuous",
            "variant": m.group(1),
            "length": 512,
            "seed": int(m.group(2)),
            "pass_at_1": s.get("pass_at_k"),
            "format": s.get("avg_format_reward"),
            "per_problem": s.get("per_problem_type", {}),
        })
    return records


def collect_science2_baselines(exp_output_dir):
    root = os.path.join(exp_output_dir, "science2_suite")
    if not os.path.isdir(root):
        return []
    pat = re.compile(r"^([a-z0-9_]+)_seed(\d+)$")
    records = []
    for d in os.listdir(root):
        m = pat.match(d)
        if not m:
            continue
        summ = os.path.join(root, d, "summary.json")
        s = read_json(summ)
        if not s:
            continue
        records.append({
            "suite": "science2_baseline",
            "variant": m.group(1),
            "length": 512,
            "seed": int(m.group(2)),
            "pass_at_1": s.get("pass_at_k"),
            "format": s.get("avg_format_reward"),
            "per_problem": s.get("per_problem_type", {}),
        })
    return records


def aggregate_variants(records):
    by_key = defaultdict(list)
    for r in records:
        by_key[(r["suite"], r["variant"], r["length"])].append(r)
    rows = []
    for (suite, variant, length), group in sorted(by_key.items()):
        pa1 = [g["pass_at_1"] for g in group if g.get("pass_at_1") is not None]
        fmt = [g["format"] for g in group if g.get("format") is not None]
        rows.append({
            "suite": suite,
            "variant": variant,
            "length": length,
            "n_runs": len(group),
            "avg_pass_at_1": round(sum(pa1) / len(pa1), 2) if pa1 else None,
            "avg_format": round(sum(fmt) / len(fmt), 3) if fmt else None,
        })
    return rows, by_key


def write_csv(rows, out_csv):
    import csv
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["suite", "variant", "length", "n_runs", "avg_pass_at_1", "avg_format"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def per_task_avg(records):
    acc = defaultdict(list)
    for rec in records:
        for k, v in rec.get("per_problem", {}).items():
            acc[k].append(v.get("pass_at_1"))
    return {k: (sum(v) / len(v) if v else None) for k, v in acc.items()}


def plot_all_variants(rows, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    labels = [f"{r['suite']}\n{r['variant']}\nL{r['length']}" for r in rows]
    values = [r["avg_pass_at_1"] or 0 for r in rows]
    plt.figure(figsize=(14, 6))
    plt.bar(range(len(values)), values, color="#4C78A8")
    plt.xticks(range(len(values)), labels, rotation=45, ha="right")
    plt.ylabel("pass@1")
    plt.title("pass@1 comparison across suites/variants")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    return out_png


def plot_top_variants(records, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    def avg_by(key_fn, recs):
        groups = defaultdict(list)
        for r in recs:
            groups[key_fn(r)].append(r)
        return {k: mean([x["pass_at_1"] for x in v]) for k, v in groups.items() if v}

    phase = [r for r in records if r["suite"] == "phase_adapt"]
    cont = [r for r in records if r["suite"] == "continuous"]
    s2 = [r for r in records if r["suite"] == "science2_baseline"]

    pa_avg = avg_by(lambda r: (r["variant"], r["length"]), phase)
    if not pa_avg:
        return None
    pa_key = max(pa_avg.items(), key=lambda kv: kv[1])[0]
    pa_name = f"{pa_key[0]}|L{pa_key[1]}"
    pa_val = pa_avg[pa_key]

    cont_avg = avg_by(lambda r: r["variant"], cont)
    cont_key = max(cont_avg.items(), key=lambda kv: kv[1])[0]
    cont_name = f"{cont_key}|L512"
    cont_val = cont_avg[cont_key]

    s2_avg = avg_by(lambda r: r["variant"], s2)
    s2_key = max(s2_avg.items(), key=lambda kv: kv[1])[0]
    s2_name = f"{s2_key}|L512"
    s2_val = s2_avg[s2_key]

    labels = ["phase_adapt\n" + pa_name, "continuous\n" + cont_name, "science2\n" + s2_name]
    values = [pa_val, cont_val, s2_val]
    plt.figure(figsize=(7, 5))
    plt.bar(range(3), values, color=["#4C78A8", "#F58518", "#54A24B"])
    plt.xticks(range(3), labels)
    plt.ylabel("pass@1")
    plt.title("Top variant per suite (overall pass@1)")
    plt.ylim(0, max(values) * 1.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    return out_png


def plot_per_task_top3(records, baseline_choice, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return None

    def avg_by(key_fn, recs):
        groups = defaultdict(list)
        for r in recs:
            groups[key_fn(r)].append(r)
        return {k: mean([x["pass_at_1"] for x in v]) for k, v in groups.items() if v}

    phase = [r for r in records if r["suite"] == "phase_adapt"]
    cont = [r for r in records if r["suite"] == "continuous"]
    s2 = [r for r in records if r["suite"] == "science2_baseline"]

    pa_avg = avg_by(lambda r: (r["variant"], r["length"]), phase)
    if not pa_avg:
        return None
    pa_key = max(pa_avg.items(), key=lambda kv: kv[1])[0]
    pa_name = f"{pa_key[0]}|L{pa_key[1]}"
    pa_recs = [r for r in phase if (r["variant"], r["length"]) == pa_key]

    cont_avg = avg_by(lambda r: r["variant"], cont)
    cont_key = max(cont_avg.items(), key=lambda kv: kv[1])[0]
    cont_name = f"{cont_key}|L512"
    cont_recs = [r for r in cont if r["variant"] == cont_key]

    s2_key = baseline_choice
    s2_name = f"{s2_key}|L512"
    s2_recs = [r for r in s2 if r["variant"] == s2_key]
    if not s2_recs:
        # fallback to best baseline
        s2_avg = avg_by(lambda r: r["variant"], s2)
        s2_key = max(s2_avg.items(), key=lambda kv: kv[1])[0]
        s2_name = f"{s2_key}|L512"
        s2_recs = [r for r in s2 if r["variant"] == s2_key]

    def per_task_avg_simple(recs):
        acc = defaultdict(list)
        for rec in recs:
            for k, v in rec.get("per_problem", {}).items():
                acc[k].append(v.get("pass_at_1"))
        return {k: (sum(v) / len(v) if v else 0.0) for k, v in acc.items()}

    phase_task = per_task_avg_simple(pa_recs)
    cont_task = per_task_avg_simple(cont_recs)
    s2_task = per_task_avg_simple(s2_recs)
    tasks = sorted(set(list(phase_task.keys()) + list(cont_task.keys()) + list(s2_task.keys())))
    x = np.arange(len(tasks))
    width = 0.28
    plt.figure(figsize=(14, 6))
    plt.bar(x - width, [phase_task.get(t, 0.0) for t in tasks], width, label=pa_name, color="#4C78A8")
    plt.bar(x, [cont_task.get(t, 0.0) for t in tasks], width, label=cont_name, color="#F58518")
    plt.bar(x + width, [s2_task.get(t, 0.0) for t in tasks], width, label=s2_name, color="#54A24B")
    plt.xticks(x, tasks, rotation=45, ha="right")
    plt.ylabel("pass@1 per task")
    plt.title("Per-task pass@1: top variant from each suite")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    return out_png


def write_per_task_deltas(records, baseline_variant, out_csv):
    import csv
    # average baseline by task
    base_recs = [r for r in records if r["suite"] == "science2_baseline" and r["variant"] == baseline_variant]
    if not base_recs:
        return None
    base_task = per_task_avg(base_recs)
    by_key = defaultdict(list)
    for r in records:
        by_key[(r["suite"], r["variant"], r["length"])].append(r)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suite", "variant", "length", "task", "delta_pass_at_1"]) 
        for (suite, variant, length), group in by_key.items():
            if suite == "science2_baseline" and variant == baseline_variant:
                continue
            avg_task = per_task_avg(group)
            for (task, val) in avg_task.items():
                base_val = base_task.get(task)
                if val is None or base_val is None:
                    continue
                w.writerow([suite, variant, length, task, round(val - base_val, 2)])
    return out_csv


def analyze(exp_output_dir, out_dir, include_suites, include_lengths, baseline_variant):
    records = []
    if (not include_suites) or ("phase_adapt" in include_suites):
        records.extend(collect_phase_adapt(exp_output_dir, include_lengths))
    if (not include_suites) or ("continuous" in include_suites):
        records.extend(collect_continuous(exp_output_dir))
    if (not include_suites) or ("science2_baseline" in include_suites):
        records.extend(collect_science2_baselines(exp_output_dir))

    rows, _ = aggregate_variants(records)
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    out["metrics_csv"] = os.path.join(out_dir, "metrics_comparison.csv")
    write_csv(rows, out["metrics_csv"]) 

    out["all_variants_plot"] = os.path.join(out_dir, "pass_at1_comparison.png")
    plot_all_variants(rows, out["all_variants_plot"]) 

    out["top_variants_plot"] = os.path.join(out_dir, "top_variants_overall.png")
    plot_top_variants(records, out["top_variants_plot"]) 

    out["per_task_top3_plot"] = os.path.join(out_dir, "per_task_top3_comparison.png")
    plot_per_task_top3(records, baseline_variant, out["per_task_top3_plot"]) 

    out["per_task_deltas_csv"] = os.path.join(out_dir, "per_task_deltas_vs_baseline_nomask.csv")
    write_per_task_deltas(records, baseline_variant, out["per_task_deltas_csv"]) 

    return out


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Analyze Science2 results across suites and generate CSV/plots.")
    default_out = os.path.dirname(os.path.abspath(__file__))
    default_base = os.path.abspath(os.path.join(default_out, os.pardir))
    parser.add_argument("--base-dir", default=default_base, help="exp_output directory (default: parent of this script)")
    parser.add_argument("--out-dir", default=default_out, help="output directory for CSVs/plots (default: this folder)")
    parser.add_argument("--suites", nargs="*", choices=["phase_adapt", "continuous", "science2_baseline"], help="Subset of suites to include")
    parser.add_argument("--lengths", nargs="*", type=int, choices=[512, 1024], help="Phase-adapt lengths to include")
    parser.add_argument("--baseline-variant", default="baseline_nomask", help="Baseline variant for per-task deltas (default: baseline_nomask)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    out = analyze(
        exp_output_dir=os.path.abspath(args.base_dir),
        out_dir=os.path.abspath(args.out_dir),
        include_suites=set(args.suites) if args.suites else None,
        include_lengths=set(args.lengths) if args.lengths else None,
        baseline_variant=args.baseline_variant,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
