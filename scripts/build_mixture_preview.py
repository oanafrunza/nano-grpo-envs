#!/usr/bin/env python3
"""
Builds a composite Reasoning Gym dataset using the same mechanism as training
(reasoning_envs.build_reasoning_envs) and writes a preview JSONL.

This mirrors how current experiments create datasets (composite of task specs)
so you can inspect samples before launching runs.

Example:
  python scripts/build_mixture_preview.py \
    --set core \
    --size 500 \
    --seed 42 \
    --out projects/nano-grpo-envs/validation/source_reasoning_gym_core.jsonl

Task sets:
- core: 20-task subset avoiding overlaps with prior wins
- stretch: +10 tasks to extend coverage

You can also pass explicit names/weights via --names/--weights.
"""
import argparse
import json
import os
import sys
import pathlib
from typing import List, Dict, Any

# Ensure repo root is on sys.path so we can import sibling modules like reasoning_envs
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import reasoning_envs
import reasoning_gym

CORE_TASKS = [
    # Algebra
    "simple_equations", "polynomial_multiplication",
    # Arithmetic
    "fraction_simplification", "decimal_arithmetic", "complex_arithmetic",
    # Algorithms
    "gcd", "prime_factorization", "rotate_matrix", "spiral_matrix", "palindrome_partitioning",
    # Cognition
    "figlet_font", "sentence_reordering",
    # Code
    "codeio",
    # Games
    "n_queens", "mini_sudoku",
    # Geometry
    "rectangle_count",
    # Graphs
    "shortest_path", "largest_island",
    # Induction
    "modulo_grid",
    # Logic
    "knights_knaves",
]

STRETCH_TASKS = [
    # Algebra
    "simple_integration",
    # Arithmetic
    "time_intervals",
    # Algorithms
    "group_anagrams", "string_synthesis",
    # Cognition
    "binary_matrix",
    # Games
    "word_ladder", "rush_hour",
    # Geometry
    "advanced_geometry",
    # Graphs
    "course_schedule", "graph_color",
    # Logic
    "circuit_logic",
]


def write_jsonl(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def default_weights(n: int) -> List[float]:
    return [round(1.0 / n, 6)] * n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["core", "stretch", "custom"], default="core")
    ap.add_argument("--names", nargs="*", default=None, help="Custom task names (use with --set custom)")
    ap.add_argument("--weights", nargs="*", default=None, help="Custom weights matching --names")
    ap.add_argument("--size", type=int, default=500)
    ap.add_argument("--per-task", type=int, dest="per_task", default=None, help="Exact examples per task; overrides --size and produces len(names)*per_task rows")
    ap.add_argument("--eval-size", type=int, default=0, help="Optional eval-only composite size (0 to skip)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.set == "core":
        names = CORE_TASKS
    elif args.set == "stretch":
        names = STRETCH_TASKS
    else:
        if not args.names:
            raise SystemExit("--set custom requires --names")
        names = args.names

    if args.weights:
        if len(args.weights) != len(names):
            raise SystemExit("--weights length must match number of names")
        weights = [float(w) for w in args.weights]
    else:
        weights = default_weights(len(names))

    rows: List[Dict[str, Any]] = []

    def pick(row: Dict[str, Any], keys: List[str], default: str = "") -> Any:
        for k in keys:
            if k and k in row and row[k] is not None:
                return row[k]
        return default

    if args.per_task is not None:
        # Build exactly N examples per task by constructing each task dataset separately
        for idx, name in enumerate(names):
            # Create each task dataset individually (no composite 'datasets' arg here)
            ds = reasoning_gym.create_dataset(name, size=args.per_task, seed=args.seed + idx)
            for i, ex in enumerate(ds):
                prompt = pick(ex, ["prompt", "question", "input", "query", "text", "instructions", "instruction"], "")
                answer = pick(ex, ["answer", "label", "output", "target", "response"], "")
                rows.append({
                    "id": f"{name}:{i}",
                    "prompt": str(prompt),
                    "answer": str(answer),
                    "format_regex": "<answer>.*</answer>",
                })
    else:
        # Original composite-based sampling (approximate counts per task)
        train_ds, eval_ds = reasoning_envs.build_reasoning_envs(
            train_names=names,
            train_weights=weights,
            train_size=args.size,
            seed=args.seed,
            eval_names=names if args.eval_size > 0 else [],
            eval_weights=weights if args.eval_size > 0 else [],
            eval_size=args.eval_size if args.eval_size > 0 else None,
        )

        for i, ex in enumerate(train_ds):
            prompt = pick(ex, ["prompt", "question", "input", "query", "text", "instructions", "instruction"], "")
            answer = pick(ex, ["answer", "label", "output", "target", "response"], "")
            rows.append({
                "id": f"train:{i}",
                "prompt": str(prompt),
                "answer": str(answer),
                "format_regex": "<answer>.*</answer>",
            })
        if args.eval_size and eval_ds is not None:
            for i, ex in enumerate(eval_ds):
                prompt = pick(ex, ["prompt", "question", "input", "query", "text", "instructions", "instruction"], "")
                answer = pick(ex, ["answer", "label", "output", "target", "response"], "")
                rows.append({
                    "id": f"eval:{i}",
                    "prompt": str(prompt),
                    "answer": str(answer),
                    "format_regex": "<answer>.*</answer>",
                })

    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} examples to {args.out} using {len(names)} tasks")


if __name__ == "__main__":
    main()
