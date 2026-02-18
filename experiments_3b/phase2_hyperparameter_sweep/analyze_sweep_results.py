#!/usr/bin/env python3
"""
Analyze 3B hyperparameter sweep results.
Compares all configs against baseline and identifies optimal settings.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

def load_sweep_results(sweep_dir: Path = Path("../../exp_output/science2_3b_sweep")) -> pd.DataFrame:
    """Load all summary.json files from sweep experiments."""
    results = []
    
    if not sweep_dir.exists():
        print(f"Warning: Sweep directory not found: {sweep_dir}")
        return pd.DataFrame()
    
    # Find all summary.json files
    summary_files = list(sweep_dir.rglob("summary.json"))
    print(f"Found {len(summary_files)} summary files")
    
    for summary_file in summary_files:
        try:
            with open(summary_file, "r") as f:
                data = json.load(f)
            
            # Extract config from directory name
            dir_name = summary_file.parent.name
            
            # Parse method and parameters
            if "continuous" in dir_name:
                method = "continuous"
            elif "phase_adapt" in dir_name:
                method = "phase_adapt"
            else:
                continue
            
            # Parse parameters from directory name
            # Format: {method}_n{every_n}_w{weight}_mw{mask_warmup}[_zw{zero_warmup}]_seed{seed}
            parts = dir_name.split("_")
            params = {}
            for part in parts:
                if part.startswith("n") and part[1:].replace(".", "").isdigit():
                    params["every_n"] = int(part[1:])
                elif part.startswith("w") and part[1:].replace(".", "").isdigit():
                    params["weight"] = float(part[1:])
                elif part.startswith("mw"):
                    params["mask_warmup"] = int(part[2:])
                elif part.startswith("zw"):
                    params["zero_warmup"] = int(part[2:])
                elif part.startswith("seed"):
                    params["seed"] = int(part[4:])
            
            # Extract performance metrics
            result = {
                "method": method,
                "every_n": params.get("every_n"),
                "weight": params.get("weight"),
                "mask_warmup": params.get("mask_warmup"),
                "zero_warmup": params.get("zero_warmup", None),
                "seed": params.get("seed"),
                "output_dir": str(summary_file.parent),
                "pass_at_1": data.get("overall_results", {}).get("pass@1", 0.0),
                "format_ok": data.get("overall_results", {}).get("format_ok", 0.0)
            }
            
            # Add per-task results if available
            if "problem_results" in data:
                for task, metrics in data["problem_results"].items():
                    result[f"task_{task}_pass@1"] = metrics.get("pass@1", 0.0)
            
            results.append(result)
            
        except Exception as e:
            print(f"Error loading {summary_file}: {e}")
            continue
    
    return pd.DataFrame(results)

def aggregate_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results across seeds for each config."""
    group_cols = ["method", "every_n", "weight", "mask_warmup"]
    if "zero_warmup" in df.columns:
        group_cols.append("zero_warmup")
    
    # Group by config (excluding seed)
    agg_df = df.groupby(group_cols, dropna=False).agg({
        "pass_at_1": ["mean", "std", "count"],
        "format_ok": "mean"
    }).reset_index()
    
    # Flatten column names
    agg_df.columns = [
        "_".join(col).strip("_") if col[1] else col[0]
        for col in agg_df.columns.values
    ]
    
    # Calculate seed variation
    agg_df["seed_variation"] = (agg_df["pass_at_1_std"] / agg_df["pass_at_1_mean"]) * 100
    
    return agg_df

def compare_to_baseline(df: pd.DataFrame, baseline_score: float = 26.2) -> pd.DataFrame:
    """Calculate improvement vs baseline."""
    df["vs_baseline"] = df["pass_at_1_mean"] - baseline_score
    df["vs_baseline_pct"] = (df["vs_baseline"] / baseline_score) * 100
    return df

def find_best_configs(df: pd.DataFrame, top_n: int = 5) -> Dict[str, pd.DataFrame]:
    """Find best configs for each method."""
    best = {}
    
    for method in df["method"].unique():
        method_df = df[df["method"] == method].copy()
        
        # Sort by pass@1 (descending) and seed_variation (ascending)
        method_df = method_df.sort_values(
            by=["pass_at_1_mean", "seed_variation"],
            ascending=[False, True]
        )
        
        best[method] = method_df.head(top_n)
    
    return best

def print_summary(df: pd.DataFrame, baseline_score: float = 26.2):
    """Print comprehensive sweep summary."""
    print("\n" + "="*80)
    print("3B HYPERPARAMETER SWEEP RESULTS")
    print("="*80)
    
    # Overall statistics
    print(f"\nTotal configs analyzed: {len(df)}")
    print(f"Baseline score: {baseline_score:.2f}%")
    
    # Method breakdown
    for method in df["method"].unique():
        method_df = df[df["method"] == method]
        print(f"\n{method.upper()} ({len(method_df)} configs):")
        print(f"  Mean: {method_df['pass_at_1_mean'].mean():.2f}%")
        print(f"  Best: {method_df['pass_at_1_mean'].max():.2f}%")
        print(f"  Worst: {method_df['pass_at_1_mean'].min():.2f}%")
        print(f"  Avg seed variation: {method_df['seed_variation'].mean():.2f}%")
    
    # Best configs per method
    print("\n" + "-"*80)
    print("TOP 5 CONFIGS PER METHOD")
    print("-"*80)
    
    best_configs = find_best_configs(df, top_n=5)
    
    for method, configs in best_configs.items():
        print(f"\n{method.upper()}:")
        print(configs[[
            "every_n", "weight", "mask_warmup", "zero_warmup",
            "pass_at_1_mean", "pass_at_1_std", "seed_variation", "vs_baseline"
        ]].to_string(index=False))
    
    # Overall best
    print("\n" + "-"*80)
    print("OVERALL BEST CONFIG")
    print("-"*80)
    
    best_overall = df.loc[df["pass_at_1_mean"].idxmax()]
    print(f"Method: {best_overall['method']}")
    print(f"Parameters:")
    print(f"  every_n: {best_overall['every_n']}")
    print(f"  weight: {best_overall['weight']}")
    print(f"  mask_warmup: {best_overall['mask_warmup']}")
    if pd.notna(best_overall.get("zero_warmup")):
        print(f"  zero_warmup: {best_overall['zero_warmup']}")
    print(f"Performance:")
    print(f"  pass@1: {best_overall['pass_at_1_mean']:.2f}% ± {best_overall['pass_at_1_std']:.2f}%")
    print(f"  vs baseline: {best_overall['vs_baseline']:.2f}% ({best_overall['vs_baseline_pct']:.1f}%)")
    print(f"  seed variation: {best_overall['seed_variation']:.2f}%")
    
    # Configs that beat baseline
    print("\n" + "-"*80)
    print("CONFIGS THAT BEAT BASELINE")
    print("-"*80)
    
    beat_baseline = df[df["pass_at_1_mean"] > baseline_score].sort_values(
        "pass_at_1_mean", ascending=False
    )
    
    if len(beat_baseline) > 0:
        print(f"\n{len(beat_baseline)} configs exceed baseline:")
        print(beat_baseline[[
            "method", "every_n", "weight", "mask_warmup", "zero_warmup",
            "pass_at_1_mean", "vs_baseline", "seed_variation"
        ]].to_string(index=False))
    else:
        print("\nNo configs exceed baseline.")
    
    # Stability analysis
    print("\n" + "-"*80)
    print("STABILITY ANALYSIS (Lowest Seed Variation)")
    print("-"*80)
    
    most_stable = df.sort_values("seed_variation").head(5)
    print(most_stable[[
        "method", "every_n", "weight", "mask_warmup", "zero_warmup",
        "pass_at_1_mean", "seed_variation"
    ]].to_string(index=False))

def save_results(df: pd.DataFrame, output_dir: Path = Path(".")):
    """Save analysis results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save full results
    df.to_csv(output_dir / "sweep_results_aggregated.csv", index=False)
    print(f"\n✓ Saved: {output_dir / 'sweep_results_aggregated.csv'}")
    
    # Save best configs
    best_configs = find_best_configs(df, top_n=5)
    for method, configs in best_configs.items():
        configs.to_csv(output_dir / f"best_{method}_configs.csv", index=False)
        print(f"✓ Saved: {output_dir / f'best_{method}_configs.csv'}")
    
    # Save summary statistics
    summary = {
        "total_configs": len(df),
        "baseline_score": 26.2,
        "best_overall": {
            "config": df.loc[df["pass_at_1_mean"].idxmax()].to_dict(),
            "performance": float(df["pass_at_1_mean"].max())
        },
        "method_stats": {}
    }
    
    for method in df["method"].unique():
        method_df = df[df["method"] == method]
        summary["method_stats"][method] = {
            "count": len(method_df),
            "mean": float(method_df["pass_at_1_mean"].mean()),
            "best": float(method_df["pass_at_1_mean"].max()),
            "worst": float(method_df["pass_at_1_mean"].min()),
            "avg_seed_variation": float(method_df["seed_variation"].mean())
        }
    
    with open(output_dir / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved: {output_dir / 'sweep_summary.json'}")

def main():
    print("Analyzing 3B hyperparameter sweep results...")
    
    # Load results
    df = load_sweep_results()
    
    if df.empty:
        print("\nNo results found. Make sure sweep has completed and generated summary.json files.")
        return
    
    # Aggregate by config (average across seeds)
    df_agg = aggregate_by_config(df)
    
    # Compare to baseline
    df_agg = compare_to_baseline(df_agg, baseline_score=26.2)
    
    # Print summary
    print_summary(df_agg, baseline_score=26.2)
    
    # Save results
    output_dir = Path("analysis")
    save_results(df_agg, output_dir)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()
