"""
Visualize GRPO training results with beautiful, retro-futuristic plots and detailed PDFs.

Usage:
    python visualize_results.py --output_dir output/
    python visualize_results.py --log_file output/run_log.json
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from datetime import datetime

# Set up the plotting style - clean and playful but professional
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': '#f5f5f5',
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'grid.color': '#cccccc',
    'grid.alpha': 0.7,
    'axes.grid': True,
    'grid.linestyle': ':',
    'lines.linewidth': 2.0,
    'axes.linewidth': 1.5,
})

# Color palette - muted but distinct
COLORS = {
    'primary': '#2E86AB',      # Steel blue
    'secondary': '#A23B72',    # Plum
    'tertiary': '#F18F01',     # Orange
    'quaternary': '#C73E1D',   # Red-orange
    'bg_light': '#f5f5f5',     # Light gray
    'text': '#333333',         # Dark gray
    'grid': '#cccccc',         # Medium gray
}


def load_log(log_path):
    """Load the run log JSON."""
    with open(log_path, 'r') as f:
        return json.load(f)


def extract_training_metrics(log):
    """Extract training metrics from log."""
    steps = []
    losses = []
    lrs = []
    rewards = []
    format_rewards = []
    correctness = []
    
    for step_str, step_data in log['steps'].items():
        if 'train' not in step_data:
            continue
        
        step = int(step_str)
        train_data = step_data['train']
        
        steps.append(step)
        losses.append(train_data['loss'])
        lrs.append(train_data['lr'])
        
        # Compute average reward metrics
        gens = train_data['generations']
        avg_total_reward = np.mean([g['total_reward'] for g in gens])
        avg_format = np.mean([g['format_reward'] for g in gens])
        avg_correct = np.mean([g['correct'] for g in gens])
        
        rewards.append(avg_total_reward)
        format_rewards.append(avg_format)
        correctness.append(avg_correct)
    
    return {
        'steps': np.array(steps),
        'losses': np.array(losses),
        'lrs': np.array(lrs),
        'rewards': np.array(rewards),
        'format_rewards': np.array(format_rewards),
        'correctness': np.array(correctness),
    }


def extract_eval_metrics(log):
    """Extract evaluation metrics from log."""
    steps = []
    pass_at_k = []
    format_rewards = []
    
    for step_str, step_data in log['steps'].items():
        if 'eval' not in step_data:
            continue
        
        step = int(step_str)
        eval_data = step_data['eval']
        metrics = eval_data['metrics']
        
        steps.append(step)
        
        # Get pass@k metric (key name depends on k value)
        pass_k_key = [k for k in metrics.keys() if k.startswith('pass_at_')][0]
        pass_at_k.append(metrics[pass_k_key])
        format_rewards.append(metrics['avg_format_reward'])
    
    return {
        'steps': np.array(steps),
        'pass_at_k': np.array(pass_at_k),
        'format_rewards': np.array(format_rewards),
        'pass_k_name': pass_k_key.replace('_', '@').replace('pass@at@', 'pass@'),
    }


def extract_per_task_eval_metrics(log):
    """Extract per-task evaluation metrics from log."""
    # Dictionary to hold metrics for each task type
    task_metrics = {}
    
    for step_str, step_data in log['steps'].items():
        if 'eval' not in step_data:
            continue
        
        step = int(step_str)
        eval_data = step_data['eval']
        metrics = eval_data['metrics']
        
        # Check if per_problem_type exists
        if 'per_problem_type' not in metrics:
            continue
        
        # Get pass@k key name
        pass_k_key = [k for k in metrics.keys() if k.startswith('pass_at_')][0]
        
        # Extract metrics for each problem type
        for problem_type, type_metrics in metrics['per_problem_type'].items():
            if problem_type not in task_metrics:
                task_metrics[problem_type] = {
                    'steps': [],
                    'pass_at_k': [],
                    'format_rewards': [],
                    'pass_k_name': pass_k_key.replace('_', '@').replace('pass@at@', 'pass@'),
                }
            
            task_metrics[problem_type]['steps'].append(step)
            task_metrics[problem_type]['pass_at_k'].append(type_metrics[pass_k_key])
            task_metrics[problem_type]['format_rewards'].append(type_metrics['avg_format_reward'])
    
    # Convert lists to numpy arrays
    for problem_type in task_metrics:
        task_metrics[problem_type]['steps'] = np.array(task_metrics[problem_type]['steps'])
        task_metrics[problem_type]['pass_at_k'] = np.array(task_metrics[problem_type]['pass_at_k'])
        task_metrics[problem_type]['format_rewards'] = np.array(task_metrics[problem_type]['format_rewards'])
    
    return task_metrics


def smooth(data, window=5):
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def create_training_plots(train_metrics, output_path, args_dict):
    """Create beautiful training metric plots."""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('white')
    
    # Main title
    fig.suptitle('Nano-GRPO Training Dynamics', fontsize=18, color=COLORS['text'], 
                 fontweight='bold', y=0.98)
    
    # Add metadata subtitle
    model_name = args_dict.get('model_name', 'Unknown').split('/')[-1]
    subtitle = f"Model: {model_name} | LR: {args_dict.get('learning_rate', 'N/A')} | Chains: {args_dict.get('num_chains', 'N/A')}"
    fig.text(0.5, 0.94, subtitle, ha='center', fontsize=9, color='gray', alpha=0.9)
    
    steps = train_metrics['steps']
    
    # 1. Loss plot
    ax1 = plt.subplot(2, 3, 1)
    losses_smooth = smooth(train_metrics['losses'], window=10)
    steps_smooth = steps[:len(losses_smooth)]
    ax1.plot(steps, train_metrics['losses'], alpha=0.3, color=COLORS['primary'], linewidth=1)
    ax1.plot(steps_smooth, losses_smooth, color=COLORS['primary'], linewidth=2.5, label='Loss (smoothed)')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss', fontweight='bold', pad=10)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Learning rate plot
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(steps, train_metrics['lrs'], color=COLORS['secondary'], linewidth=2.5)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule', fontweight='bold', pad=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 3. Total rewards
    ax3 = plt.subplot(2, 3, 3)
    rewards_smooth = smooth(train_metrics['rewards'], window=10)
    steps_smooth = steps[:len(rewards_smooth)]
    ax3.plot(steps, train_metrics['rewards'], alpha=0.3, color=COLORS['tertiary'], linewidth=1)
    ax3.plot(steps_smooth, rewards_smooth, color=COLORS['tertiary'], linewidth=2.5, label='Avg Reward (smoothed)')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Average Total Reward')
    ax3.set_title('Training Rewards', fontweight='bold', pad=10)
    ax3.legend(loc='lower right', framealpha=0.9)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 4. Correctness
    ax4 = plt.subplot(2, 3, 4)
    correctness_smooth = smooth(train_metrics['correctness'], window=10)
    steps_smooth = steps[:len(correctness_smooth)]
    ax4.plot(steps, train_metrics['correctness'] * 100, alpha=0.3, color=COLORS['quaternary'], linewidth=1)
    ax4.plot(steps_smooth, correctness_smooth * 100, color=COLORS['quaternary'], linewidth=2.5, label='Correctness (smoothed)')
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Correctness (%)')
    ax4.set_title('Training Correctness', fontweight='bold', pad=10)
    ax4.set_ylim(0, 100)
    ax4.legend(loc='lower right', framealpha=0.9)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # 5. Format rewards
    ax5 = plt.subplot(2, 3, 5)
    format_smooth = smooth(train_metrics['format_rewards'], window=10)
    steps_smooth = steps[:len(format_smooth)]
    ax5.plot(steps, train_metrics['format_rewards'], alpha=0.3, color=COLORS['secondary'], linewidth=1)
    ax5.plot(steps_smooth, format_smooth, color=COLORS['secondary'], linewidth=2.5, label='Format Reward (smoothed)')
    ax5.set_xlabel('Training Step')
    ax5.set_ylabel('Format Reward')
    ax5.set_title('Format Adherence', fontweight='bold', pad=10)
    ax5.legend(loc='lower right', framealpha=0.9)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    # 6. Combined view
    ax6 = plt.subplot(2, 3, 6)
    # Normalize metrics to 0-1 for comparison
    losses_norm = (train_metrics['losses'] - train_metrics['losses'].min()) / (train_metrics['losses'].max() - train_metrics['losses'].min() + 1e-8)
    rewards_norm = train_metrics['rewards'] / (train_metrics['rewards'].max() + 1e-8)
    correct_norm = train_metrics['correctness']
    
    # Smooth and align lengths
    losses_smooth = smooth(losses_norm, 10)
    rewards_smooth = smooth(rewards_norm, 10)
    correct_smooth = smooth(correct_norm, 10)
    steps_smooth = steps[:len(losses_smooth)]
    
    ax6.plot(steps_smooth, losses_smooth, color=COLORS['primary'], linewidth=2, label='Loss (norm)', alpha=0.7)
    ax6.plot(steps_smooth, rewards_smooth, color=COLORS['tertiary'], linewidth=2, label='Reward (norm)', alpha=0.7)
    ax6.plot(steps_smooth, correct_smooth, color=COLORS['quaternary'], linewidth=2, label='Correctness', alpha=0.7)
    ax6.set_xlabel('Training Step')
    ax6.set_ylabel('Normalized Metrics')
    ax6.set_title('Training Overview (Normalized)', fontweight='bold', pad=10)
    ax6.legend(loc='right', framealpha=0.9)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none')
    print(f"✓ Saved training plots to {output_path}")
    plt.close()


def create_eval_plots(eval_metrics, output_path, args_dict):
    """Create beautiful evaluation metric plots."""
    fig = plt.figure(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    
    # Main title
    fig.suptitle('Nano-GRPO Reasoning Envs Eval Performance', fontsize=18, color=COLORS['text'], 
                 fontweight='bold', y=0.98)
    
    # Add metadata subtitle
    model_name = args_dict.get('model_name', 'Unknown').split('/')[-1]
    eval_size = args_dict.get('eval_size', 'N/A')
    num_completions = args_dict.get('num_completions_eval', 'N/A')
    pass_k_value = args_dict.get('pass_at_k', 1)
    
    # Calculate starting and best scores
    starting_score = eval_metrics['pass_at_k'][0]
    best_score = eval_metrics['pass_at_k'].max()
    best_step = eval_metrics['steps'][eval_metrics['pass_at_k'].argmax()]
    
    subtitle = f"Model: {model_name} | Eval Size: {eval_size} | Probabilistic pass@{pass_k_value} from {num_completions} completions | Starting: {starting_score:.2f}% | Best ckpt (step {best_step}): {best_score:.2f}%"
    fig.text(0.5, 0.91, subtitle, ha='center', fontsize=9, color='gray', alpha=0.9)
    
    steps = eval_metrics['steps']
    
    # Pass@k plot (centered, single plot)
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(steps, eval_metrics['pass_at_k'], color=COLORS['primary'], 
             linewidth=3, marker='o', markersize=8, label=eval_metrics['pass_k_name'])
    ax1.fill_between(steps, 0, eval_metrics['pass_at_k'], color=COLORS['primary'], alpha=0.2)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel(f"{eval_metrics['pass_k_name'].capitalize()} (%)")
    ax1.set_ylim(0, 100)
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.89])
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none')
    print(f"✓ Saved evaluation plots to {output_path}")
    plt.close()


def create_per_task_eval_plot(task_name, task_metrics, output_path, args_dict):
    """Create evaluation plot for a specific task."""
    fig = plt.figure(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    
    # Main title with task name
    task_display_name = task_name.replace('_', ' ').title()
    fig.suptitle(f'Nano-GRPO: {task_display_name}', fontsize=18, color=COLORS['text'], 
                 fontweight='bold', y=0.98)
    
    # Add metadata subtitle
    model_name = args_dict.get('model_name', 'Unknown').split('/')[-1]
    num_completions = args_dict.get('num_completions_eval', 'N/A')
    pass_k_value = args_dict.get('pass_at_k', 1)
    
    # Calculate starting and best scores
    starting_score = task_metrics['pass_at_k'][0]
    best_score = task_metrics['pass_at_k'].max()
    best_step = task_metrics['steps'][task_metrics['pass_at_k'].argmax()]
    
    subtitle = f"Model: {model_name} | Probabilistic pass@{pass_k_value} from {num_completions} completions | Starting: {starting_score:.2f}% | Best ckpt (step {best_step}): {best_score:.2f}%"
    fig.text(0.5, 0.91, subtitle, ha='center', fontsize=9, color='gray', alpha=0.9)
    
    steps = task_metrics['steps']
    
    # Pass@k plot
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(steps, task_metrics['pass_at_k'], color=COLORS['primary'], 
             linewidth=3, marker='o', markersize=8, label=task_metrics['pass_k_name'])
    ax1.fill_between(steps, 0, task_metrics['pass_at_k'], color=COLORS['primary'], alpha=0.2)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel(f"{task_metrics['pass_k_name'].capitalize()} (%)")
    ax1.set_ylim(0, 100)
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.89])
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none')
    plt.close()


def wrap_text(text, width=100):
    """Wrap text to specified width."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length > width and current_line:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = word_length
        else:
            current_line.append(word)
            current_length += word_length
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


def create_training_pdf(log, output_path, max_steps=50):
    """Create a detailed PDF of training examples."""
    with PdfPages(output_path) as pdf:
        # Get all training steps
        train_steps = sorted([int(k) for k, v in log['steps'].items() if 'train' in v])
        
        # Sample steps if too many
        if len(train_steps) > max_steps:
            step_indices = np.linspace(0, len(train_steps) - 1, max_steps, dtype=int)
            train_steps = [train_steps[i] for i in step_indices]
        
        for step in train_steps:
            train_data = log['steps'][str(step)]['train']
            
            # Use axis for better text control
            fig, ax = plt.subplots(figsize=(11, 14))
            fig.patch.set_facecolor('white')
            ax.axis('off')
            
            y_pos = 0.98
            line_height = 0.012
            
            # Title
            ax.text(0.5, y_pos, f"Training Step {step}", ha='center', fontsize=14, 
                   fontweight='bold', color='black', transform=ax.transAxes)
            y_pos -= 0.025
            
            # Metadata
            metadata = f"Loss: {train_data['loss']:.6f} | LR: {train_data['lr']:.2e}"
            ax.text(0.5, y_pos, metadata, ha='center', fontsize=9, color='gray', transform=ax.transAxes)
            y_pos -= 0.03
            
            # Prompt
            ax.text(0.02, y_pos, "Prompt:", fontsize=10, fontweight='bold', 
                   color='black', transform=ax.transAxes)
            y_pos -= line_height * 1.5
            
            # Escape special characters in prompt
            safe_prompt = str(train_data['prompt']).replace('$', r'\$').replace('_', r'\_')
            prompt_lines = wrap_text(safe_prompt, width=120)
            for line in prompt_lines[:15]:  # Limit lines
                ax.text(0.02, y_pos, line, fontsize=7, color='black', 
                       family='monospace', transform=ax.transAxes)
                y_pos -= line_height
            y_pos -= line_height
            
            # Target answer
            ax.text(0.02, y_pos, "Target Answer:", fontsize=10, fontweight='bold', 
                   color='black', transform=ax.transAxes)
            y_pos -= line_height * 1.5
            safe_target = str(train_data['target_answer']).replace('$', r'\$').replace('_', r'\_')
            ax.text(0.02, y_pos, safe_target, fontsize=8, 
                   color='green', family='monospace', transform=ax.transAxes)
            y_pos -= line_height * 2
            
            # Generations
            for i, gen in enumerate(train_data['generations']):
                if y_pos < 0.15:  # Leave some margin at bottom
                    break
                
                # Generation header
                correct_marker = "✓" if gen['correct'] else "✗"
                header = f"Generation {i+1} {correct_marker}"
                ax.text(0.02, y_pos, header, fontsize=9, fontweight='bold',
                       color='green' if gen['correct'] else 'red', transform=ax.transAxes)
                y_pos -= line_height * 1.5
                
                # Metrics
                metrics = f"Correct: {gen['correct']} | Format: {gen['format_reward']:.2f} | Total: {gen['total_reward']:.2f}"
                ax.text(0.02, y_pos, metrics, fontsize=7, color='black', transform=ax.transAxes)
                y_pos -= line_height * 1.2
                
                # Extracted answer
                safe_answer = str(gen['extracted_answer']).replace('$', r'\$').replace('_', r'\_')
                ax.text(0.02, y_pos, f"Answer: {safe_answer}", fontsize=7, 
                       color='black', family='monospace', transform=ax.transAxes)
                y_pos -= line_height * 1.2
                
                # Completion text
                safe_text = str(gen['text']).replace('$', r'\$').replace('_', r'\_')
                completion_lines = wrap_text(safe_text, width=120)
                for line in completion_lines[:5]:  # Show first 5 lines
                    if y_pos < 0.05:
                        break
                    ax.text(0.04, y_pos, line, fontsize=6, family='monospace', 
                           color='black', alpha=0.7, transform=ax.transAxes)
                    y_pos -= line_height * 0.9
                
                y_pos -= line_height * 1.5  # Space between generations
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    print(f"✓ Saved training examples PDF to {output_path} ({len(train_steps)} steps)")


def escape_special_chars(text):
    """Escape special characters that can cause LaTeX parsing errors."""
    if not isinstance(text, str):
        text = str(text)
    # Replace problematic characters
    text = text.replace('$', '\\$')
    text = text.replace('_', '\\_')
    text = text.replace('%', '\\%')
    text = text.replace('&', '\\&')
    text = text.replace('#', '\\#')
    return text


def create_eval_pdf(log, output_path):
    """Create a detailed PDF of evaluation examples."""
    with PdfPages(output_path) as pdf:
        # Get all eval steps
        eval_steps = sorted([int(k) for k, v in log['steps'].items() if 'eval' in v])
        
        for step in eval_steps:
            eval_data = log['steps'][str(step)]['eval']
            metrics = eval_data['metrics']
            
            # Use axis for better text control
            fig, ax = plt.subplots(figsize=(11, 14))
            fig.patch.set_facecolor('white')
            ax.axis('off')
            
            y_pos = 0.98
            line_height = 0.012
            
            # Title
            ax.text(0.5, y_pos, f"Evaluation at Step {step}", ha='center', fontsize=14, 
                   color='black', family='monospace', transform=ax.transAxes)
            y_pos -= 0.025
            
            # Overall metrics
            pass_k_key = [k for k in metrics.keys() if k.startswith('pass_at_')][0]
            pass_k_value = metrics[pass_k_key]
            metadata = f"{pass_k_key.replace('_', ' ')}: {pass_k_value:.1f} percent | Format: {metrics['avg_format_reward']:.3f} | Problems: {metrics['num_eval_problems']}"
            ax.text(0.5, y_pos, metadata, ha='center', fontsize=9, color='gray', 
                   family='monospace', transform=ax.transAxes)
            y_pos -= 0.03
            
            # Show eval examples
            for i, example in enumerate(eval_data['examples'][:5]):  # First 5 examples
                if y_pos < 0.15:
                    break
                
                # Problem header
                ax.text(0.02, y_pos, f"Problem {i+1}:", fontsize=10, 
                       color='black', family='monospace', transform=ax.transAxes)
                y_pos -= line_height * 1.5
                
                # Prompt (full prompt, not just question)
                ax.text(0.02, y_pos, "Prompt:", fontsize=9, fontweight='bold', 
                       color='black', transform=ax.transAxes)
                y_pos -= line_height * 1.2
                # Escape special characters that cause LaTeX parsing issues
                safe_prompt = str(example['prompt']).replace('$', r'\$').replace('_', r'\_')
                prompt_lines = wrap_text(safe_prompt, width=120)
                for line in prompt_lines[:15]:  # Show more lines for full prompt
                    ax.text(0.02, y_pos, line, fontsize=6, color='black', 
                           family='monospace', transform=ax.transAxes)
                    y_pos -= line_height * 0.9
                y_pos -= line_height
                
                # Target
                safe_target = str(example['target_answer']).replace('$', r'\$').replace('_', r'\_')
                ax.text(0.02, y_pos, f"Target: {safe_target}", fontsize=8, 
                       color='green', family='monospace', transform=ax.transAxes)
                y_pos -= line_height * 1.5
                
                # Summary
                summary = f"Correct: {example['num_correct']}/{len(example['completions'])} | pass at k: {example['pass_at_k']:.3f} | Format: {example['avg_format_reward']:.3f}"
                ax.text(0.02, y_pos, summary, fontsize=7, color='black', 
                       family='monospace', transform=ax.transAxes)
                y_pos -= line_height * 1.5
                
                # Show completions - show all of them with full text
                for j, comp in enumerate(example['completions']):  # Show ALL completions
                    if y_pos < 0.05:
                        break
                    marker = "✓" if comp['correct'] else "✗"
                    # Show header with extracted answer
                    safe_answer = str(comp['extracted_answer']).replace('$', r'\$').replace('_', r'\_')
                    header = f"  {marker} Completion {j+1}: {safe_answer} (fmt: {comp['format_reward']:.2f})"
                    ax.text(0.04, y_pos, header, fontsize=7, 
                           color='green' if comp['correct'] else 'red', 
                           family='monospace', transform=ax.transAxes)
                    y_pos -= line_height * 1.2
                    
                    # Show full completion text
                    safe_text = str(comp['text']).replace('$', r'\$').replace('_', r'\_')
                    comp_lines = wrap_text(safe_text, width=115)
                    for line in comp_lines[:10]:  # Show up to 10 lines per completion
                        if y_pos < 0.05:
                            break
                        ax.text(0.06, y_pos, line, fontsize=6, color='black', 
                               family='monospace', alpha=0.7, transform=ax.transAxes)
                        y_pos -= line_height * 0.9
                    y_pos -= line_height
                
                y_pos -= line_height * 2  # Space between problems
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    print(f"✓ Saved evaluation examples PDF to {output_path} ({len(eval_steps)} eval points)")


def main():
    parser = argparse.ArgumentParser(description="Visualize GRPO training results")
    parser.add_argument('--output_dir', '--o', type=str, required=True, help='Output directory containing run_log.json')
    parser.add_argument('--max_train_steps', type=int, default=50, help='Max training steps to include in PDF')
    args = parser.parse_args()
    
    log_path = Path(args.output_dir) / 'run_log.json'

    
    # Load log
    print(f"Loading log from {log_path}...")
    log = load_log(log_path)
    
    # Extract metrics
    print("Extracting metrics...")
    train_metrics = extract_training_metrics(log)
    eval_metrics = extract_eval_metrics(log)
    per_task_metrics = extract_per_task_eval_metrics(log)
    
    # Create output directory for visualizations
    output_dir = Path(args.output_dir) / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    # Create per-task output directory
    per_task_dir = output_dir / 'per_task_visualization'
    per_task_dir.mkdir(exist_ok=True)
    
    # Create plots
    print("\nGenerating visualizations...")
    create_training_plots(train_metrics, output_dir / 'training_metrics.png', log['args'])
    create_eval_plots(eval_metrics, output_dir / 'eval_metrics.png', log['args'])
    
    # Create per-task plots
    if per_task_metrics:
        print("\nGenerating per-task visualizations...")
        for task_name, task_data in per_task_metrics.items():
            task_filename = f"{task_name}_eval.png"
            create_per_task_eval_plot(task_name, task_data, per_task_dir / task_filename, log['args'])
            print(f"  ✓ Saved {task_name} plot")
    
    # Create PDFs
    print("\nGenerating detailed PDFs...")
    create_training_pdf(log, output_dir / 'training_examples.pdf', max_steps=args.max_train_steps)
    create_eval_pdf(log, output_dir / 'eval_examples.pdf')
    
    print(f"\n{'='*60}")
    print(f"✓ All visualizations saved to: {output_dir}")
    print(f"{'='*60}")
    print(f"  • training_metrics.png  - Training dynamics plots")
    print(f"  • eval_metrics.png      - Evaluation performance plots")
    print(f"  • training_examples.pdf - Detailed training examples")
    print(f"  • eval_examples.pdf     - Detailed evaluation results")
    if per_task_metrics:
        print(f"\n  Per-task visualizations in: per_task_visualization/")
        for task_name in per_task_metrics.keys():
            print(f"    • {task_name}_eval.png")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

