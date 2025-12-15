#!/usr/bin/env python3
import os
import json
import argparse
from typing import Dict, Any, List


def load_run_log(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "run_log.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing run_log.json in {run_dir}")
    with open(path, "r") as f:
        return json.load(f)


def summarize_run(run_dir: str) -> Dict[str, Any]:
    log = load_run_log(run_dir)
    # Default metrics
    overall_pass = None
    overall_format = None
    # Try to use last eval step metrics
    steps = log.get("steps", {})
    last_eval = None
    for step_key in sorted(steps.keys(), key=lambda x: int(x)):
        if "eval" in steps[step_key]:
            last_eval = steps[step_key]["eval"]
    if last_eval:
        metrics = last_eval.get("metrics", {})
        # find pass_at_k key dynamically
        pass_keys = [k for k in metrics.keys() if k.startswith("pass_at_")]
        if pass_keys:
            overall_pass = metrics[pass_keys[0]]
        overall_format = metrics.get("avg_format_reward", None)
        per_problem_type = metrics.get("per_problem_type", {})
    else:
        per_problem_type = {}

    # Fallback: compute avg format from train generations if needed
    if overall_format is None:
        fmt_vals = []
        for step in steps.values():
            train = step.get("train")
            if not train:
                continue
            for g in train.get("generations", []):
                if "format_reward" in g:
                    fmt_vals.append(float(g["format_reward"]))
        overall_format = sum(fmt_vals) / len(fmt_vals) if fmt_vals else None

    # Extract masking config snapshot
    cfg = {
        "reward_mask_strategy": log.get("steps", {}).get("0", {}).get("train", {}).get("reward_mask_strategy"),
    }

    return {
        "run_dir": run_dir,
        "overall_pass": overall_pass,
        "overall_format": overall_format,
        "per_task": per_problem_type,
        "config": cfg,
    }


def render_table(summaries: List[Dict[str, Any]]) -> str:
    # Build simple text table
    header = ["run", "overall_pass%", "avg_format", "tasks"]
    lines = ["\t".join(header)]
    for s in summaries:
        tasks = ",".join(sorted(s["per_task"].keys())) if s["per_task"] else "-"
        lines.append("\t".join([
            os.path.basename(s["run_dir"]),
            f"{s['overall_pass']:.2f}" if s['overall_pass'] is not None else "-",
            f"{s['overall_format']:.3f}" if s['overall_format'] is not None else "-",
            tasks
        ]))
    return "\n".join(lines)


def write_csv(summaries: List[Dict[str, Any]], out_path: str) -> None:
    import csv
    # Collect all task names
    task_names = set()
    for s in summaries:
        task_names.update(s["per_task"].keys())
    task_names = sorted(task_names)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["run", "overall_pass", "overall_format"] + [f"{t}_pass" for t in task_names] + [f"{t}_format" for t in task_names]
        w.writerow(header)
        for s in summaries:
            row = [os.path.basename(s["run_dir"]), s["overall_pass"], s["overall_format"]]
            # pass per task
            for t in task_names:
                v = s["per_task"].get(t, {}).get(next((k for k in s["per_task"].get(t, {}).keys() if k.startswith("pass_at_")), None))
                row.append(v)
            # format per task
            for t in task_names:
                v = s["per_task"].get(t, {}).get("avg_format_reward")
                row.append(v)
            w.writerow(row)


def try_plot(summaries: List[Dict[str, Any]], out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    names = [os.path.basename(s["run_dir"]) for s in summaries]
    passes = [s["overall_pass"] or 0.0 for s in summaries]
    formats = [s["overall_format"] or 0.0 for s in summaries]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], passes, width=0.4, label="overall_pass%")
    ax.bar([i + 0.2 for i in x], formats, width=0.4, label="avg_format")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()
    ax.set_title("Multi-run comparison")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "multi_run_overview.png"))


def main():
    p = argparse.ArgumentParser(description="Compare multiple runs in exp_output/")
    p.add_argument("--runs", nargs="+", help="List of run directories to compare (each containing run_log.json)")
    p.add_argument("--out_dir", default="exp_output", help="Where to write summary CSV and plot")
    args = p.parse_args()

    # Filter out checkpoint directories or any that start with 'checkpoint'
    candidate_runs = [r for r in args.runs if not os.path.basename(r).startswith(("checkpoint", "checkpoints_"))]
    summaries = []
    for r in candidate_runs:
        try:
            summaries.append(summarize_run(r))
        except FileNotFoundError:
            # Skip directories without run_log.json
            continue
    print(render_table(summaries))

    csv_path = os.path.join(args.out_dir, "multi_run_summary.csv")
    write_csv(summaries, csv_path)
    print(f"Wrote CSV: {csv_path}")

    try_plot(summaries, args.out_dir)
    print(f"If matplotlib is available, saved plot to {os.path.join(args.out_dir, 'multi_run_overview.png')}.")


if __name__ == "__main__":
    main()
