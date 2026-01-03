#!/usr/bin/env python3
import os
import json
from typing import List, Dict, Any

BASE = os.path.join(os.getcwd(), "exp_output", "science2_suite")
ANALYSIS_DIR = os.path.join(BASE, "_analysis")
OUT_JSON = os.path.join(ANALYSIS_DIR, "examples_showcase.json")
OUT_MD = os.path.join(ANALYSIS_DIR, "examples_showcase.md")

RUNS = [
    "baseline_nomask_seed1",
    "baseline_nomask_seed2",
    "softmask_every10_wt05_seed1",
    "softmask_every10_wt05_seed2",
    "fullzero_every10_nothresh_seed1",
    "fullzero_every10_nothresh_seed2",
]

MAX_EXAMPLES_PER_RUN = 6  # aim: mix correct/incorrect across problems


def load_run_log(run_dir: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, "run_log.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def extract_examples(run_name: str, run_log: Dict[str, Any]) -> List[Dict[str, Any]]:
    examples = []
    steps = run_log.get("steps", {})
    # Iterate steps, collect eval examples
    for step_key in sorted(steps.keys(), key=lambda x: int(x)):
        step = steps[step_key]
        eval_block = step.get("eval")
        if not eval_block:
            continue
        for ex in eval_block.get("examples", []):
            prompt = ex.get("prompt")
            question = ex.get("question")
            target = ex.get("target_answer")
            ptype = ex.get("problem_type")
            for comp in ex.get("completions", []):
                rec = {
                    "run": run_name,
                    "problem_type": ptype,
                    "prompt": prompt,
                    "question": question,
                    "target": target,
                    "text": comp.get("text"),
                    "extracted_answer": comp.get("extracted_answer"),
                    "correct": comp.get("correct"),
                    "format_reward": comp.get("format_reward"),
                }
                examples.append(rec)
        # Stop if we've gathered quite a few already to keep output manageable
        if len(examples) >= 1000:
            break
    return examples


def select_showcase(examples: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    # Prefer a balanced set: problems unique, mix correct/incorrect
    by_ptype: Dict[str, List[Dict[str, Any]]] = {}
    for ex in examples:
        by_ptype.setdefault(ex.get("problem_type", "unknown"), []).append(ex)

    showcase: List[Dict[str, Any]] = []
    # First pass: pick up to 2 correct and 2 incorrect per problem type where available
    for ptype, exs in by_ptype.items():
        correct = [e for e in exs if e.get("correct") == 1]
        incorrect = [e for e in exs if e.get("correct") == 0]
        for e in correct[:2]:
            showcase.append(e)
        for e in incorrect[:2]:
            showcase.append(e)
        if len(showcase) >= limit:
            break

    # If still under limit, fill remaining from any
    if len(showcase) < limit:
        remaining = [e for exs in by_ptype.values() for e in exs]
        # De-duplicate by text
        seen_text = set((s.get("text"), s.get("run")) for s in showcase)
        for e in remaining:
            key = (e.get("text"), e.get("run"))
            if key in seen_text:
                continue
            showcase.append(e)
            if len(showcase) >= limit:
                break
    return showcase


def write_md(records: List[Dict[str, Any]], md_path: str):
    lines = ["# Science2 Main Suite: Examples Showcase", ""]
    lines.append("Curated samples across runs showing correct and incorrect answers.")
    lines.append("")
    for i, r in enumerate(records, 1):
        status = "✅ correct" if r.get("correct") == 1 else "❌ incorrect"
        lines.append(f"## {i}. {status} — {r.get('problem_type')} — {r.get('run')}")
        lines.append("")
        q = (r.get("question") or "").strip()
        if q:
            lines.append("**Question:**")
            lines.append(q)
            lines.append("")
        target = r.get("target")
        if target:
            lines.append(f"**Target:** {target}")
            lines.append("")
        # Show completion (trim think if too long)
        text = (r.get("text") or "").strip()
        # Keep first 2000 chars to avoid huge blocks
        if len(text) > 2000:
            text = text[:2000] + "\n...[truncated]..."
        lines.append("**Model Output:**")
        lines.append(text)
        lines.append("")
        lines.append("---")
        lines.append("")
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    all_examples: List[Dict[str, Any]] = []
    for run in RUNS:
        run_dir = os.path.join(BASE, run)
        log = load_run_log(run_dir)
        if not log:
            continue
        exs = extract_examples(run, log)
        if not exs:
            continue
        # Limit per run to avoid huge outputs
        all_examples.extend(exs[:MAX_EXAMPLES_PER_RUN * 10])
    # Select final showcase
    showcase = select_showcase(all_examples, limit=24)
    # Write JSON and Markdown
    with open(OUT_JSON, "w") as jf:
        json.dump(showcase, jf, indent=2)
    write_md(showcase, OUT_MD)
    print(f"Wrote: {OUT_JSON}")
    print(f"Wrote: {OUT_MD}")


if __name__ == "__main__":
    main()
