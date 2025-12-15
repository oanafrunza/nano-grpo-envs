"""
Compare Nano-GRPO experiment outputs across multiple runs.

Usage:
  uv run compare_runs.py --dirs exp_output exp_output_probp_02_wt_05 [more...]

Outputs:
- Latest overall eval metrics (pass@k, format, problems)
- Per-task latest eval metrics
- Training summaries (last N steps: mean loss, mean reward)
- Optional CSV export
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np


def load_log(dir_path: Path) -> Dict[str, Any]:
    log_path = dir_path / "run_log.json"
    if not log_path.exists():
        raise FileNotFoundError(f"No run_log.json in {dir_path}")
    with open(log_path, "r") as f:
        return json.load(f)


def latest_eval_metrics(log: Dict[str, Any]) -> Dict[str, Any] | None:
    steps = sorted(int(s) for s in log.get("steps", {}) if "eval" in log["steps"][s])
    if not steps:
        return None
    eval_block = log["steps"][str(steps[-1])]["eval"]["metrics"]
    kkey = [k for k in eval_block.keys() if k.startswith("pass_at_")][0]
    return {
        "step": steps[-1],
        "pass_key": kkey,
        "pass_at_k": eval_block[kkey],
        "avg_format_reward": eval_block.get("avg_format_reward", None),
        "num_eval_problems": eval_block.get("num_eval_problems", None),
        "per_problem_type": eval_block.get("per_problem_type", {}),
    }


def train_summary(log: Dict[str, Any], last_n: int = 100) -> Dict[str, Any] | None:
    steps = sorted(int(s) for s in log.get("steps", {}) if "train" in log["steps"][s])
    if not steps:
        return None
    window = steps[-last_n:] if len(steps) > last_n else steps
    losses: List[float] = []
    rewards: List[float] = []
    masked: List[int] = []
    for s in window:
        tr = log["steps"][str(s)]["train"]
        losses.append(float(tr["loss"]))
        gens = tr.get("generations", [])
        if gens:
            rewards.append(float(np.mean([g.get("total_reward", 0.0) for g in gens])))
        masked.append(int(tr.get("num_masked_correct", 0)))
    return {
        "steps": len(window),
        "loss_mean": float(np.mean(losses)) if losses else None,
        "reward_mean": float(np.mean(rewards)) if rewards else None,
        "avg_masked_correct": float(np.mean(masked)) if masked else 0.0,
    }


def print_comparison(dir_paths: List[Path], last_n_train: int = 100) -> None:
    print("\n=== Experiment Comparison ===\n")
    rows = []
    for d in dir_paths:
        label = d.name
        try:
            log = load_log(d)
        except Exception as e:
            print(f"{label}: ERROR loading log -> {e}")
            continue
        evalm = latest_eval_metrics(log)
        trsum = train_summary(log, last_n=last_n_train)
        print(f"[{label}]\n  Output: {d}")
        if evalm:
            print(f"  Eval (step {evalm['step']}): {evalm['pass_key']}={evalm['pass_at_k']:.2f}% | format={evalm['avg_format_reward']:.3f} | problems={evalm['num_eval_problems']}")
            # Per-task
            if evalm["per_problem_type"]:
                for t, vals in evalm["per_problem_type"].items():
                    print(f"    - {t}: {evalm['pass_key']}={vals[evalm['pass_key']]:.2f}% | fmt={vals.get('avg_format_reward', 0.0):.3f} | n={vals.get('num_problems', 'NA')}")
        else:
            print("  Eval: none")
        if trsum:
            print(f"  Train (last {trsum['steps']} steps): loss_mean={trsum['loss_mean']:.4f} | reward_mean={trsum['reward_mean']:.3f} | avg_masked_correct={trsum['avg_masked_correct']:.2f}")
        else:
            print("  Train: none")
        print("")
        rows.append({"label": label, "eval": evalm, "train": trsum})

    # If we have at least two runs, print a delta summary comparing the first two
    if len(rows) >= 2 and rows[0]["eval"] and rows[1]["eval"]:
        a = rows[0]["eval"]
        b = rows[1]["eval"]
        pass_key = a["pass_key"]
        def fmt_pct(x):
            return f"{x:.2f}%" if isinstance(x, (int, float)) else "NA"
        def fmt_val(x):
            return f"{x:.3f}" if isinstance(x, (int, float)) else "NA"
        def adjective(delta):
            if delta >= 10: return "big jump"
            if delta >= 2: return "up"
            if delta >= 0.5: return "up a bit"
            if delta <= -10: return "big drop"
            if delta <= -2: return "down"
            if delta <= -0.5: return "down a bit"
            return "flat"

        overall_delta = float(b["pass_at_k"]) - float(a["pass_at_k"]) if (b["pass_at_k"] is not None and a["pass_at_k"] is not None) else None
        fmt_delta = (float(b["avg_format_reward"]) - float(a["avg_format_reward"])) if (b["avg_format_reward"] is not None and a["avg_format_reward"] is not None) else None

        print("=== Delta Summary (" + rows[0]["label"] + " → " + rows[1]["label"] + ") ===")
        if overall_delta is not None:
            arrow = "→"
            print(f"Overall: {fmt_pct(a['pass_at_k'])} {arrow} {fmt_pct(b['pass_at_k'])} ({overall_delta:+.2f} pts)")
        else:
            print("Overall: NA")

        # Per-task comparison: union of task names
        tasks = set()
        tasks.update(a.get("per_problem_type", {}).keys())
        tasks.update(b.get("per_problem_type", {}).keys())
        if tasks:
            print("Per task:")
            for t in sorted(tasks):
                av = a.get("per_problem_type", {}).get(t)
                bv = b.get("per_problem_type", {}).get(t)
                if av and bv and pass_key in av and pass_key in bv:
                    delta = float(bv[pass_key]) - float(av[pass_key])
                    adj = adjective(delta)
                    print(f"{t}: {fmt_pct(av[pass_key])} → {fmt_pct(bv[pass_key])} ({adj})")
                elif av and pass_key in av:
                    print(f"{t}: {fmt_pct(av[pass_key])} → NA")
                elif bv and pass_key in bv:
                    print(f"{t}: NA → {fmt_pct(bv[pass_key])}")
                else:
                    print(f"{t}: NA → NA")

        # Format reward comparison
        if fmt_delta is not None:
            note = ""
            # If per-task format exists, highlight extreme cases
            b_types = b.get("per_problem_type", {})
            extremes = [t for t, v in b_types.items() if abs(v.get("avg_format_reward", 0.0)) < 1e-6]
            if extremes:
                note = f"; {', '.join(extremes)} format is {0.000:.3f}"
            print(f"Format reward: {fmt_val(a['avg_format_reward'])} → {fmt_val(b['avg_format_reward'])} ({fmt_delta:+.3f}){note}")
        else:
            print("Format reward: NA")


def main():
    parser = argparse.ArgumentParser(description="Compare Nano-GRPO experiment outputs")
    parser.add_argument("--dirs", nargs="+", required=True, help="One or more output directories containing run_log.json")
    parser.add_argument("--last_n_train", type=int, default=100, help="Training steps window for train summary")
    args = parser.parse_args()

    dirs = [Path(d).resolve() for d in args.dirs]
    print_comparison(dirs, last_n_train=args.last_n_train)


if __name__ == "__main__":
    main()
