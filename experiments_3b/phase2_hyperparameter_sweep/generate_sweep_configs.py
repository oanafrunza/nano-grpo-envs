#!/usr/bin/env python3
"""
Generate individual config files for 3B hyperparameter sweep.
Creates one config per setting combination for SLURM array job submission.
"""

import json
import os
from pathlib import Path
from itertools import product

# Load base sweep configuration
with open("sweep_config.json", "r") as f:
    sweep_config = json.load(f)

base_config = sweep_config["base_config"]
seeds = sweep_config["seeds"]

# Output directory for configs
config_dir = Path("configs")
config_dir.mkdir(exist_ok=True)

# Track all configs for manifest
all_configs = []

def generate_continuous_configs():
    """Generate configs for continuous masking sweep."""
    # Use reduced grid
    grid = sweep_config["reduced_grid"]["continuous"]["parameters"]
    
    configs = []
    config_id = 0
    
    for every_n, weight, warmup in product(
        grid["reward_mask_every_n"],
        grid["reward_mask_weight"],
        grid["mask_warmup_steps"]
    ):
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

def generate_phase_adapt_configs():
    """Generate configs for phase-adaptive sweep."""
    # Use reduced grid
    grid = sweep_config["reduced_grid"]["phase_adapt"]["parameters"]
    
    configs = []
    config_id = 100  # Start at 100 to distinguish from continuous
    
    for every_n, weight, mask_warmup, zero_warmup in product(
        grid["reward_mask_every_n"],
        grid["reward_mask_weight"],
        grid["mask_warmup_steps"],
        grid["zero_warmup_steps"]
    ):
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
                "gradient_checkpointing": True  # Enable for phase-adapt
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
    print("Generating 3B hyperparameter sweep configs...")
    
    # Generate all configs
    continuous_configs = generate_continuous_configs()
    phase_adapt_configs = generate_phase_adapt_configs()
    
    all_configs = continuous_configs + phase_adapt_configs
    
    # Save manifest
    manifest = {
        "sweep_name": sweep_config["sweep_name"],
        "total_configs": len(all_configs),
        "continuous_configs": len(continuous_configs),
        "phase_adapt_configs": len(phase_adapt_configs),
        "seeds": seeds,
        "configs": all_configs
    }
    
    with open("sweep_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Generated {len(all_configs)} configs:")
    print(f"  - Continuous: {len(continuous_configs)}")
    print(f"  - Phase-Adapt: {len(phase_adapt_configs)}")
    print(f"  - Config files: {config_dir}/")
    print(f"  - Manifest: sweep_manifest.json")
    
    print("\nNext steps:")
    print("  1. Review sweep_manifest.json")
    print("  2. Submit sweep: sbatch run_sweep.sh")
    print("  3. Monitor progress: squeue -u $USER")
    print("  4. Analyze results: python analyze_sweep_results.py")

if __name__ == "__main__":
    main()
