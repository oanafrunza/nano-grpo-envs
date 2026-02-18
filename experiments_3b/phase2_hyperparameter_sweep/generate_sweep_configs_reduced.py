#!/usr/bin/env python3
"""
Generate REDUCED sweep configs focusing on most promising hyperparameter ranges.
Reduces from 48 to 16 experiments (8 configs × 2 seeds).
"""

import json
from pathlib import Path
from itertools import product

# Load base sweep configuration
with open("sweep_config.json", "r") as f:
    sweep_config = json.load(f)

base_config = sweep_config["base_config"]
seeds = sweep_config["seeds"]

# Output directory for configs
config_dir = Path("configs_reduced")
config_dir.mkdir(exist_ok=True)

# Track all configs for manifest
all_configs = []

def generate_reduced_continuous_configs():
    """Generate reduced continuous masking sweep (most promising configs only)."""
    configs = []
    config_id = 0
    
    # Focus on most promising: aggressive masking (10), higher weight (0.9), earlier warmup (200)
    # Plus baseline comparison: moderate masking (20), standard weight (0.7)
    promising_combos = [
        (10, 0.9, 200),  # Most aggressive
        (10, 0.7, 200),  # Aggressive frequency, standard weight
        (20, 0.9, 400),  # Standard frequency, high weight
        (20, 0.7, 400),  # Current 7B setting (baseline)
    ]
    
    for every_n, weight, warmup in promising_combos:
        for seed in seeds:
            config = base_config.copy()
            config.update({
                "output_dir": f"{base_config['output_dir_base']}/continuous_n{every_n}_w{weight}_mw{warmup}_seed{seed}",
                "seed": seed,
                "reward_mask_every_n": every_n,
                "reward_mask_weight": weight,
                "mask_warmup_steps": warmup,
                "full_correct_zero_strategy": "disabled",
                "run_name": f"3b_continuous_n{every_n}_w{weight}_mw{warmup}_seed{seed}",
                "method": "continuous",
                "config_id": config_id
            })
            
            # Save individual config
            config_file = config_dir / f"continuous_{config_id:03d}.json"
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
            
            configs.append({
                "config_id": config_id,
                "method": "continuous",
                "file": str(config_file),
                "params": {
                    "every_n": every_n,
                    "weight": weight,
                    "warmup": warmup,
                    "seed": seed
                }
            })
            config_id += 1
    
    return configs

def generate_reduced_phase_adapt_configs():
    """Generate reduced phase-adaptive sweep (most promising configs only)."""
    configs = []
    config_id = 100  # Start at 100 to distinguish from continuous
    
    # Focus on most promising combinations
    promising_combos = [
        (10, 0.9, 200, 400),  # Aggressive masking, high weight, early warmups
        (10, 0.7, 200, 400),  # Aggressive masking, standard weight
        (20, 0.9, 400, 600),  # Standard frequency, high weight
        (20, 0.7, 400, 600),  # Current 7B setting (baseline)
    ]
    
    for every_n, weight, mask_warmup, zero_warmup in promising_combos:
        for seed in seeds:
            config = base_config.copy()
            config.update({
                "output_dir": f"{base_config['output_dir_base']}/phase_adapt_n{every_n}_w{weight}_mw{mask_warmup}_zw{zero_warmup}_seed{seed}",
                "seed": seed,
                "reward_mask_every_n": every_n,
                "reward_mask_weight": weight,
                "mask_warmup_steps": mask_warmup,
                "zero_warmup_steps": zero_warmup,
                "full_correct_zero_strategy": "late",
                "run_name": f"3b_phase_adapt_n{every_n}_w{weight}_mw{mask_warmup}_zw{zero_warmup}_seed{seed}",
                "method": "phase_adapt",
                "config_id": config_id,
                "gradient_checkpointing": True
            })
            
            # Save individual config
            config_file = config_dir / f"phase_adapt_{config_id:03d}.json"
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
            
            configs.append({
                "config_id": config_id,
                "method": "phase_adapt",
                "file": str(config_file),
                "params": {
                    "every_n": every_n,
                    "weight": weight,
                    "mask_warmup": mask_warmup,
                    "zero_warmup": zero_warmup,
                    "seed": seed
                }
            })
            config_id += 1
    
    return configs

def main():
    print("Generating REDUCED 3B hyperparameter sweep configs...")
    print("Focus: Most promising hyperparameter combinations")
    
    # Generate all configs
    continuous_configs = generate_reduced_continuous_configs()
    phase_adapt_configs = generate_reduced_phase_adapt_configs()
    
    all_configs = continuous_configs + phase_adapt_configs
    
    # Save manifest
    manifest = {
        "sweep_name": "3b_hyperparameter_optimization_reduced",
        "description": "Reduced sweep focusing on most promising configs (aggressive masking + baseline)",
        "total_configs": len(all_configs),
        "continuous_configs": len(continuous_configs),
        "phase_adapt_configs": len(phase_adapt_configs),
        "seeds": seeds,
        "configs": all_configs,
        "rationale": {
            "aggressive": "Test if 3B needs more aggressive masking (every_n=10, weight=0.9, warmup=200)",
            "baseline": "Compare against 7B-optimized settings (every_n=20, weight=0.7, warmup=400)",
            "reduction": "Reduced from 48 to 16 experiments (67% reduction) to conserve GPUs"
        }
    }
    
    with open("sweep_manifest_reduced.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Generated {len(all_configs)} configs:")
    print(f"  - Continuous: {len(continuous_configs)} (4 configs × 2 seeds)")
    print(f"  - Phase-Adapt: {len(phase_adapt_configs)} (4 configs × 2 seeds)")
    print(f"  - Config files: {config_dir}/")
    print(f"  - Manifest: sweep_manifest_reduced.json")
    
    print("\nReduction strategy:")
    print("  - Original: 48 experiments (8 continuous + 16 phase-adapt) × 2 seeds")
    print("  - Reduced: 16 experiments (4 continuous + 4 phase-adapt) × 2 seeds")
    print("  - Focus: Most aggressive (10, 0.9, 200) + baseline (20, 0.7, 400)")
    
    print("\nEstimated runtime:")
    print(f"  - Per experiment: ~3 hours")
    print(f"  - Total sequential: ~48 hours (2 days)")
    print(f"  - With 6 GPUs (1 job at a time)")
    
    print("\nNext steps:")
    print("  1. Review sweep_manifest_reduced.json")
    print("  2. Use reduced configs: MANIFEST='sweep_manifest_reduced.json' in run_sweep.sh")
    print("  3. Submit: sbatch run_sweep.sh")
    print("  4. Monitor: squeue -u $USER")

if __name__ == "__main__":
    main()
