"""
Evaluate Top 3B Configs on 31 OOD Tasks
Compares with 7B OOD performance to verify if patterns hold
"""

import argparse
from pathlib import Path

# Best 3 configurations: Best baseline + Best continuous + Best phase-adapt
# Selected same way as 7B: 1 best seed per strategy
TOP_CONFIGS = [
    {
        'name': 'baseline_seed0',
        'checkpoint': 'exp_output/science2_3b_suite/baseline_len512_seed0',
        'description': 'Baseline (no masking/zeroing)',
        'strategy': 'Baseline',
        'in_domain': 27.12,
        'config_detail': 'No masking/zeroing'
    },
    {
        'name': 'continuous_zero_seed0',
        'checkpoint': 'exp_output/3b_7b_replication/continuous_fullzero_seed0',
        'description': 'Continuous Zero-only (every_n=20)',
        'strategy': 'Continuous',
        'in_domain': 25.52,
        'config_detail': 'Zero-only, every_n=20'
    },
    {
        'name': 'phase_adapt_7b_seed0',
        'checkpoint': 'exp_output/3b_7b_replication/phase_adapt_exact7b_seed0',
        'description': 'Phase-Adapt (7B config)',
        'strategy': 'Phase-Adapt',
        'in_domain': 22.52,
        'config_detail': 'Split masking + zero@200'
    },
]

def generate_eval_config():
    """Generate evaluation config JSON."""
    import json
    
    config = {
        "models": [],
        "evaluation": {
            "max_new_tokens": 512,
            "temperature": 0.0,
            "format_checker": "regex",
            "regex_pattern": "<answer>.*</answer>"
        }
    }
    
    for model in TOP_CONFIGS:
        # Use checkpoint_final if it exists
        checkpoint_path = Path(model['checkpoint'])
        if (checkpoint_path / "checkpoint_final").exists():
            final_path = str(checkpoint_path / "checkpoint_final")
        else:
            final_path = model['checkpoint']
        
        config["models"].append({
            "name": model['name'],
            "checkpoint": final_path
        })
    
    output_dir = Path("validation/results_3b_ood_best3")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    config_path = output_dir / "eval_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Generated config: {config_path}")
    return config_path


def create_slurm_script():
    """Create SLURM batch script for OOD evaluation."""
    script = """#!/bin/bash
#SBATCH --job-name=eval_3b_ood
#SBATCH --output=logs/slurm_eval_3b_ood_%j.out
#SBATCH --error=logs/slurm_eval_3b_ood_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gres=gpu:2
#SBATCH --partition=main

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Activate virtual environment
source .venv/bin/activate
echo "Virtual environment activated: .venv"

# Navigate to project directory
cd /mnt/home/oana/projects/nano-grpo-envs

echo "==== Evaluating 3B Best 3 Configs on 31-task OOD suite ===="
echo "Started: $(date)"
echo "Using Slurm-allocated GPUs: $CUDA_VISIBLE_DEVICES"

# Verify checkpoints exist
for checkpoint in exp_output/science2_3b_suite/baseline_len512_seed0 \\
                  exp_output/3b_7b_replication/continuous_fullzero_seed0 \\
                  exp_output/3b_7b_replication/phase_adapt_exact7b_seed0; do
    if [ ! -d "$checkpoint/checkpoint_final" ] && [ ! -f "$checkpoint/pytorch_model.bin" ] && [ ! -f "$checkpoint/model.safetensors" ]; then
        echo "WARNING: No checkpoint found in $checkpoint (looking for pytorch_model.bin, model.safetensors, or checkpoint-final/)"
    fi
done

# Create output directory
mkdir -p validation/results_3b_ood_best3

# Generate config if not exists
python experiments_3b/evaluate_3b_ood.py --generate-config

# Run evaluation using validation/evaluate_models.py
python validation/evaluate_models.py \\
    --config validation/results_3b_ood_best3/eval_config.json \\
    --dataset validation/source_reasoning_gym_30.jsonl \\
    --output validation/results_3b_ood_best3/results.csv \\
    --batch-size 16

echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="
"""
    
    script_path = Path("experiments_3b/run_3b_ood_eval.sh")
    with open(script_path, 'w') as f:
        f.write(script)
    script_path.chmod(0o755)
    
    print(f"✓ Generated SLURM script: {script_path}")
    return script_path


def main():
    parser = argparse.ArgumentParser(description='Evaluate top 3B configs on OOD tasks')
    parser.add_argument('--generate-config', action='store_true', 
                       help='Generate evaluation config only')
    parser.add_argument('--create-script', action='store_true',
                       help='Create SLURM script only')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("3B OOD EVALUATION SETUP")
    print("="*80 + "\n")
    
    print("Best 3 Configurations (1 best seed per strategy, same as 7B):\n")
    for i, model in enumerate(TOP_CONFIGS, 1):
        print(f"{i}. {model['description']}")
        print(f"   Checkpoint: {model['checkpoint']}")
        print(f"   Config: {model['config_detail']}")
        print(f"   In-domain: {model['in_domain']:.2f}%")
        print()
    
    if args.generate_config:
        config_path = generate_eval_config()
        print(f"\n✓ Config generated: {config_path}")
        return
    
    if args.create_script:
        script_path = create_slurm_script()
        print(f"\n✓ SLURM script created: {script_path}")
        print("\nTo submit job:")
        print(f"  sbatch {script_path}")
        return
    
    # Generate both
    config_path = generate_eval_config()
    script_path = create_slurm_script()
    
    print("\n" + "="*80)
    print("SETUP COMPLETE!")
    print("="*80)
    print("\nTo run OOD evaluation:")
    print(f"  sbatch {script_path}")
    print("\nOr run locally:")
    print("  python validation/evaluate_models.py \\")
    print("    --config validation/results_3b_ood_best3/eval_config.json \\")
    print("    --dataset validation/source_reasoning_gym_30.jsonl \\")
    print("    --out validation/results_3b_ood_best3/results.csv \\")
    print("    --backend hf \\")
    print("    --batch-size 16 \\")
    print("    --use-chat-template")
    print("\nAfter completion, analyze results with:")
    print("  python experiments_3b/analyze_3b_ood_results.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
