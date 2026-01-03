import os
import csv
import argparse


def read_overall_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # coerce
            r["pass_at_1"] = float(r["pass_at_1"]) if r.get("pass_at_1") else None
            r["avg_format_reward"] = float(r["avg_format_reward"]) if r.get("avg_format_reward") else None
            r["step"] = int(r["step"]) if r.get("step") else None
            rows.append(r)
    return rows


def read_per_problem_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def pick_best_run(overall_rows):
    # Choose row with max pass@1; tie-break by higher step
    best = None
    for r in overall_rows:
        if best is None:
            best = r
            continue
        if (r["pass_at_1"], r["step"]) > (best["pass_at_1"], best["step"]):
            best = r
    return best


def average_baseline(overall_rows, name_prefix):
    vals = [r for r in overall_rows if r["run"].startswith(name_prefix)]
    if not vals:
        return None
    p = sum(r["pass_at_1"] for r in vals) / len(vals)
    fmt = sum(r["avg_format_reward"] for r in vals) / len(vals)
    steps = max(r["step"] for r in vals)
    return {"run": name_prefix+"_avg", "pass_at_1": p, "avg_format_reward": fmt, "step": steps}


def per_problem_map(per_problem_rows):
    # Map run -> {ptype: score}
    out = {}
    for row in per_problem_rows:
        run = row["run"]
        d = {}
        for k, v in row.items():
            if k == "run":
                continue
            d[k] = float(v) if v not in (None, "") else None
        out[run] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cont_suite", default="exp_output/science2_cont_suite/_analysis")
    ap.add_argument("--full_suite", default="exp_output/science2_suite/_analysis")
    ap.add_argument("--adapt_suite", default=None)
    ap.add_argument("--outdir", default="exp_output/cont_vs_full/_analysis")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load
    cont_overall = read_overall_csv(os.path.join(args.cont_suite, "overall.csv"))
    full_overall = read_overall_csv(os.path.join(args.full_suite, "overall.csv"))
    cont_perprob = read_per_problem_csv(os.path.join(args.cont_suite, "per_problem.csv"))
    full_perprob = read_per_problem_csv(os.path.join(args.full_suite, "per_problem.csv"))
    adapt_overall = []
    adapt_perprob = []
    if args.adapt_suite is not None:
        adapt_overall = read_overall_csv(os.path.join(args.adapt_suite, "overall.csv"))
        adapt_perprob = read_per_problem_csv(os.path.join(args.adapt_suite, "per_problem.csv"))

    # Pick best cont and average baseline_nomask from full
    best_cont = pick_best_run(cont_overall)
    base_avg = average_baseline(full_overall, "baseline_nomask")
    # Fallback: best baseline if no average available
    if base_avg is None:
        base_avg = pick_best_run([r for r in full_overall if r["run"].startswith("baseline_nomask")])

    # Write summary CSV
    with open(os.path.join(args.outdir, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "run", "step", "pass_at_1", "avg_format_reward"])
        w.writerow(["best_cont", best_cont["run"], best_cont["step"], f"{best_cont['pass_at_1']:.4f}", f"{best_cont['avg_format_reward']:.4f}"])
        w.writerow(["baseline_avg", base_avg["run"], base_avg["step"], f"{base_avg['pass_at_1']:.4f}", f"{base_avg['avg_format_reward']:.4f}"])
        if adapt_overall:
            best_adapt = pick_best_run(adapt_overall)
            w.writerow(["best_adapt", best_adapt["run"], best_adapt["step"], f"{best_adapt['pass_at_1']:.4f}", f"{best_adapt['avg_format_reward']:.4f}"])

    # Per-problem delta: best cont vs baseline average (or best)
    cont_map = per_problem_map(cont_perprob)
    full_map = per_problem_map(full_perprob)
    cont_scores = cont_map.get(best_cont["run"], {})
    adapt_map = per_problem_map(adapt_perprob) if adapt_perprob else {}
    best_adapt = pick_best_run(adapt_overall) if adapt_overall else None
    adapt_scores = adapt_map.get(best_adapt["run"], {}) if best_adapt else {}
    # Build baseline aggregate by averaging per-problem across all baseline runs
    base_runs = [k for k in full_map.keys() if k.startswith("baseline_nomask")] or list(full_map.keys())
    # Compute average per-problem over baselines
    base_scores = {}
    if base_runs:
        keys = [k for k in full_perprob[0].keys() if k != "run"]
        for k in keys:
            vals = [full_map[r].get(k) for r in base_runs]
            vals = [v for v in vals if v is not None]
            base_scores[k] = sum(vals)/len(vals) if vals else None

    # Write per-problem delta CSV
    with open(os.path.join(args.outdir, "per_problem_delta.csv"), "w", newline="") as f:
        w = csv.writer(f)
        header = ["problem", "baseline_avg", "cont_best", "delta(cont-baseline)"]
        if best_adapt:
            header += ["adapt_best", "delta(adapt-baseline)"]
        w.writerow(header)
        for k in sorted(base_scores.keys()):
            bb = base_scores.get(k)
            cb = cont_scores.get(k)
            row = [k, f"{bb:.4f}" if bb is not None else "", f"{cb:.4f}" if cb is not None else "", f"{(cb-bb):.4f}" if (cb is not None and bb is not None) else ""]
            if best_adapt:
                ab = adapt_scores.get(k)
                row += [f"{ab:.4f}" if ab is not None else "", f"{(ab-bb):.4f}" if (ab is not None and bb is not None) else ""]
            w.writerow(row)

    # Write quick markdown summary
    md_path = os.path.join(args.outdir, "summary.md")
    with open(md_path, "w") as f:
        f.write(f"# Continuous vs Full-set vs Phase-adapt\n\n")
        f.write(f"- Best continuous: {best_cont['run']} — pass@1 {best_cont['pass_at_1']:.2f}, format {best_cont['avg_format_reward']:.3f}, step {best_cont['step']}\n")
        f.write(f"- Baseline (avg): {base_avg['run']} — pass@1 {base_avg['pass_at_1']:.2f}, format {base_avg['avg_format_reward']:.3f}, step {base_avg['step']}\n")
        if best_adapt:
            f.write(f"- Best phase-adapt: {best_adapt['run']} — pass@1 {best_adapt['pass_at_1']:.2f}, format {best_adapt['avg_format_reward']:.3f}, step {best_adapt['step']}\n\n")
        else:
            f.write("\n")
        f.write(f"See per-problem deltas in per_problem_delta.csv.\n")

    print("Wrote:", os.path.join(args.outdir, "summary.csv"))
    print("Wrote:", os.path.join(args.outdir, "per_problem_delta.csv"))
    print("Wrote:", os.path.join(args.outdir, "summary.md"))


if __name__ == "__main__":
    main()
