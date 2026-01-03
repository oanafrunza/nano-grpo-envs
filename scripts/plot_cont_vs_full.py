import os
import csv
import argparse
import matplotlib.pyplot as plt


def read_per_problem_delta(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "problem": r["problem"],
                "cont_best": float(r["cont_best"]),
                "baseline_avg": float(r["baseline_avg"]),
                "delta": float(r["delta(cont-baseline)"]),
            })
    return rows


def read_summary(path):
    # expects rows: label,run,step,pass_at_1,avg_format_reward
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            out[r["label"]] = {
                "run": r["run"],
                "step": int(r["step"]),
                "pass_at_1": float(r["pass_at_1"]),
                "avg_format_reward": float(r["avg_format_reward"]),
            }
    return out


def plot_per_problem_delta(rows, out_path):
    # sort by delta descending
    rows_sorted = sorted(rows, key=lambda r: r["delta"], reverse=True)
    labels = [r["problem"].split(":")[0] for r in rows_sorted]
    deltas = [r["delta"] for r in rows_sorted]

    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
    fig_h = max(4, 0.35 * len(labels))
    plt.figure(figsize=(10, fig_h))
    y = list(range(len(labels)))
    plt.barh(y, deltas, color=colors)
    plt.yticks(y, labels)
    plt.xlabel("Delta pass@1 (cont_best - baseline_avg)")
    plt.title("Continuous vs Full-set Baseline — Per-problem Delta")
    plt.axvline(0, color="#888", linewidth=1)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_overall(summary, out_path):
    labels = ["baseline_avg", "best_cont"]
    vals = [summary["baseline_avg"]["pass_at_1"], summary["best_cont"]["pass_at_1"]]
    colors = ["#ff7f0e", "#1f77b4"]
    if "best_adapt" in summary:
        labels.append("best_adapt")
        vals.append(summary["best_adapt"]["pass_at_1"])
        colors.append("#2ca02c")
    plt.figure(figsize=(6, 4))
    plt.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.5, f"{v:.2f}", ha="center")
    plt.ylabel("pass@1")
    plt.title("Overall pass@1: Baseline vs Continuous vs Phase-adapt")
    plt.ylim(0, max(vals) * 1.15)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison_dir", default="exp_output/cont_vs_full/_analysis")
    args = ap.parse_args()

    per_problem_path = os.path.join(args.comparison_dir, "per_problem_delta.csv")
    summary_path = os.path.join(args.comparison_dir, "summary.csv")

    rows = read_per_problem_delta(per_problem_path)
    summary = read_summary(summary_path)

    out1 = os.path.join(args.comparison_dir, "per_problem_delta.png")
    out2 = os.path.join(args.comparison_dir, "overall_bar.png")
    plot_per_problem_delta(rows, out1)
    plot_overall(summary, out2)
    print("Wrote:", out1)
    print("Wrote:", out2)


if __name__ == "__main__":
    main()
