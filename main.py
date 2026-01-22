
"""
GRPO training script for Reasoning Gym environments.

This is a simple Python-first implementation of GRPO tailored for reasoning_gym
datasets. It keeps the algorithm easy to read and modify, while supporting
industry-standard performance options:
  - vLLM: optional high-throughput generation via a server for sampling/logprobs
  - LigerKernel: optional fused kernels for faster, stable GRPO loss and model forward
  - Accelerate: multi-GPU ready via the Hugging Face ecosystem

Use --use_vllm to generate with vLLM, and --use_liger to enable the Liger model
and fused GRPO loss for local training.
"""

import os
import math
import json
import torch
import random
import argparse
from tqdm import tqdm
from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss


# Own modules 
import llms
import utils
import reasoning_envs
import vllm_client  as v_c



def _get_last_hidden_state_for_liger(model, input_ids, attention_mask, logits_to_keep: int):
    """
    Compute last hidden state aligned to completion tokens for Liger loss.

    Returns a tensor of shape (B, logits_to_keep, H).
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    # Prefer last_hidden_state if exposed; else derive from hidden_states
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        last_hidden = outputs.last_hidden_state  # (B, L, H)
    else:
        last_hidden = outputs.hidden_states[-1]
    # Exclude final time-step (next-token pred) and keep only completion window
    last_hidden = last_hidden[:, :-1, :]
    last_hidden = last_hidden[:, -logits_to_keep:, :]
    return last_hidden


def compute_liger_grpo_loss(model, prompt_ids, completion_ids, prompt_mask, completion_mask, advantages, args, liger_loss):
    """
    Liger kernel GRPO loss, mirroring TRL's usage of LigerFusedLinearGRPOLoss.
    """
    # Ensure all tensors are on the same device as the model
    device = model.device
    prompt_ids = prompt_ids.to(device)
    completion_ids = completion_ids.to(device)
    prompt_mask = prompt_mask.to(device)
    completion_mask = completion_mask.to(device)
    advantages = advantages.to(device)

    # Build full sequence and compute last hidden states for completion window
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)

    last_hidden_state = _get_last_hidden_state_for_liger(
        model, input_ids, attention_mask, logits_to_keep
    )

    # Align computation to the device of lm_head weights to avoid cross-device matmul
    target_device = model.lm_head.weight.device
    if last_hidden_state.device != target_device:
        last_hidden_state = last_hidden_state.to(target_device)
    if completion_ids.device != target_device:
        completion_ids = completion_ids.to(target_device)
    if completion_mask.device != target_device:
        completion_mask = completion_mask.to(target_device)
    if advantages.device != target_device:
        advantages = advantages.to(target_device)

    # Compute fused loss; we don't use ref/old logps in this simple setup
    loss, _metrics = liger_loss(
        _input=last_hidden_state,
        lin_weight=model.lm_head.weight,
        selected_token_ids=completion_ids,
        attention_mask=completion_mask,
        advantages=advantages,
        bias=getattr(model.lm_head, "bias", None),
        old_per_token_logps=None,
        ref_per_token_logps=None,
    )
    return loss


def generate_local(model, tokenizer, prompt_ids, prompt_mask, args):
    """Generate using local model (original method)"""
    # Repeat prompt for multiple parallel generations (chains)
    prompt_ids = prompt_ids.repeat(args.num_chains, 1).to(model.device)
    prompt_mask = prompt_mask.repeat(args.num_chains, 1).to(model.device)

    # Set up generation parameters
    generation_config = {
        "max_new_tokens": args.max_completion_length,  # Max tokens to generate
        "do_sample": True,  # Enable sampling (not greedy)
        "temperature": args.temperature,  # Sampling temperature
        "pad_token_id": tokenizer.pad_token_id,  # Padding token for batching
    }
    # Generate completions (disable gradients for inference)
    with torch.inference_mode():
        prompt_completion_ids = model.generate(prompt_ids, attention_mask=prompt_mask, **generation_config)

    # Split the full sequence back into prompt and completion parts
    prompt_len = prompt_ids.size(1)  # Length of original prompt
    prompt_ids = prompt_completion_ids[:, :prompt_len]  # Extract prompt portion
    completion_ids = prompt_completion_ids[:, prompt_len:]  # Extract completion portion

    # Create mask to handle EOS tokens properly
    is_eos = completion_ids == tokenizer.eos_token_id  # Find EOS tokens
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=model.device)  # Default to end
    has_eos = is_eos.any(dim=1)  # Check which sequences have EOS
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]  # Set EOS position for sequences that have it
    seq_idx = torch.arange(is_eos.size(1), device=model.device).expand_as(is_eos)  # Position indices
    completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()  # Mask: 1 for valid tokens, 0 after EOS

    # Combine prompt and completion attention masks
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    # Decode token IDs back to text
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completion_mask, completions_text


def generate_vllm(vllm_client, prompt_text, tokenizer, args, device):
    """Generate using vLLM server and return proper token IDs for GRPO training"""
    # Generate completions using vLLM server
    response = vllm_client.generate(
        prompts=[prompt_text] * args.num_chains,
        n=1,  # One completion per prompt
        temperature=args.temperature,
        max_tokens=args.max_completion_length,
    )
    
    # Extract data from response
    prompt_ids_list = response["prompt_ids"]  # List of prompt token IDs
    completion_ids_list = response["completion_ids"]  # List of completion token IDs
    
    # Convert to tensors with proper padding
    # First, pad all sequences to the same length
    max_prompt_len = max(len(ids) for ids in prompt_ids_list)
    max_completion_len = max(len(ids) for ids in completion_ids_list)
    
    # Pad prompt_ids
    padded_prompt_ids = []
    for ids in prompt_ids_list:
        padded = ids + [tokenizer.pad_token_id] * (max_prompt_len - len(ids))
        padded_prompt_ids.append(padded)
    prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)
    
    # Pad completion_ids
    padded_completion_ids = []
    for ids in completion_ids_list:
        padded = ids + [tokenizer.pad_token_id] * (max_completion_len - len(ids))
        padded_completion_ids.append(padded)
    completion_ids = torch.tensor(padded_completion_ids, dtype=torch.long, device=device)
    
    # Create full prompt+completion sequences
    prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    
    # Create attention masks
    attention_mask = torch.ones_like(prompt_completion_ids)
    completion_mask = torch.zeros_like(prompt_completion_ids)
    completion_mask[:, prompt_ids.shape[1]:] = 1  # Only completion tokens
    
    # Decode completions to text (use original unpadded sequences)
    completions_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in completion_ids_list]
    
    return prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completion_mask, completions_text


def generate(model, tokenizer, prompt_ids, prompt_mask, args, vllm_client=None, prompt_text=None):
    """Main generate function that routes to local or vLLM based on args"""
    if args.use_vllm and vllm_client is not None:
        return generate_vllm(vllm_client, prompt_text, tokenizer, args, model.device)
    else:
        return generate_local(model, tokenizer, prompt_ids, prompt_mask, args)


def compute_pass_at_k(n, c, k):
    """
    Calculate pass@k metric using the standard formula:
    pass@k = 1 - (n-c choose k) / (n choose k)
    
    Args:
        n: total number of samples
        c: number of correct samples
        k: k for pass@k
    
    Returns:
        pass@k probability (0.0 to 1.0)
    """
    if n - c < k:
        return 1.0
    
    # Calculate 1 - P(all k samples are wrong)
    # P(all k wrong) = product from i=0 to k-1 of (n-c-i)/(n-i)
    prob_all_wrong = 1.0
    for i in range(k):
        prob_all_wrong *= (n - c - i) / (n - i)
    
    return 1.0 - prob_all_wrong


def compute_grpo_loss(model, prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completion_mask, advantages, args=None):

    # DR-GRPO loss implementation
    # Number of completion tokens to compute loss over
    tokens_to_keep = completion_ids.size(1)

    # Reconstruct full input sequence (prompt + completion)
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    # Get per-token log probabilities from the current model
    logps = utils.get_per_token_logps(model, input_ids, attention_mask, tokens_to_keep)

    # Compute GRPO loss: -exp(logp - logp.detach()) * advantages
    # The exp(logp - logp.detach()) creates importance weights, advantages provide direction
    per_token_loss = -torch.exp(logps - logps.detach()) * advantages.unsqueeze(1)
    
    # Create a completion-only mask (extract the completion part from the full mask)
    completion_only_mask = completion_mask[:, -tokens_to_keep:]  # Take only the completion tokens
    
    # DR-GRPO loss: normalize by batch size and max completion length
    # This makes the loss scale-invariant to sequence length and batch size
    loss = (per_token_loss * completion_only_mask).sum() / (per_token_loss.size(0) * args.max_completion_length)
    
    return loss


def _build_reward_mask(correct_flags, step, args):
    """
    Build a boolean mask (1 keep reward, 0 mask reward) for each sample.
    Strategies:
      - none: keep all rewards
      - every_n: for correct answers, mask every N-th correct at the batch level
      - prob_p: for correct answers, mask with probability p
      - cosine: time-based schedule mask fraction f(step)=0.5*(1+cos(2π step / period)) * max_mask_frac
      - round_robin_k: partition chains into k buckets and mask one bucket per step (rotating)

    Masking applies only to correct answers to avoid punishing incorrect ones further.
    """
    strategy = getattr(args, "reward_mask_strategy", "none")
    if strategy == "none":
        return torch.ones(len(correct_flags), dtype=torch.float32)

    # Convert to tensor for convenience
    correct_t = torch.tensor(correct_flags, dtype=torch.float32)

    if strategy == "every_n":
        # Stateful across steps: keep a running count of correct samples seen
        if not hasattr(args, "_correct_seen"):
            args._correct_seen = 0
        n = max(1, int(getattr(args, "reward_mask_every_n", 10)))
        mask = torch.ones_like(correct_t)
        for i, c in enumerate(correct_t):
            if c.item() == 1.0:
                args._correct_seen += 1
                if args._correct_seen % n == 0:
                    mask[i] = 0.0
        return mask

    if strategy == "prob_p":
        p = float(getattr(args, "reward_mask_prob", 0.1))
        rand = torch.rand_like(correct_t)
        # If correct and rand < p -> mask
        mask = torch.where((correct_t == 1.0) & (rand < p), torch.zeros_like(correct_t), torch.ones_like(correct_t))
        return mask

    if strategy == "cosine":
        period = max(1, int(getattr(args, "reward_mask_period", 100)))
        max_frac = float(getattr(args, "reward_mask_max_frac", 0.5))
        # Fraction to mask this step
        frac = 0.5 * (1.0 + torch.cos(torch.tensor(2.0 * torch.pi * (step % period) / period))) * max_frac
        # Compute how many correct items to mask
        num_correct = int(correct_t.sum().item())
        num_to_mask = int(round(frac.item() * num_correct))
        mask = torch.ones_like(correct_t)
        if num_to_mask > 0:
            # Deterministic selection: take first num_to_mask correct examples
            count = 0
            for i, c in enumerate(correct_t):
                if c.item() == 1.0 and count < num_to_mask:
                    mask[i] = 0.0
                    count += 1
        return mask

    if strategy == "round_robin_k":
        k = max(1, int(getattr(args, "reward_mask_round_robin_k", 4)))
        bucket = step % k
        # Assign samples to buckets by position modulo k
        mask = torch.ones_like(correct_t)
        for i, c in enumerate(correct_t):
            if c.item() == 1.0 and (i % k) == bucket:
                mask[i] = 0.0
        return mask

    # Fallback
    return torch.ones(len(correct_flags), dtype=torch.float32)



def parse_args():
    parser = argparse.ArgumentParser(description="Nano GRPO with reasoning_gym composite datasets")

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base/reference model name")

    # Output and logging
    parser.add_argument("--output_dir", type=str, default="exp_output", help="Where to save logs")
    parser.add_argument("--use_wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="nano-grpo", help="W&B project name")
    parser.add_argument("--wandb_run", type=str, default="run", help="W&B run name")

    # Optimization
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.99, help="Adam beta2") 
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, help="Grad norm clip")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Grad accum steps")
    parser.add_argument("--warmup_percent", type=float, default=0.1, help="Warmup percent of iters")

    # Generation
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--num_chains", type=int, default=8, help="Parallel generations per prompt")
    parser.add_argument("--max_prompt_length", type=int, default=256, help="Max prompt tokens")
    parser.add_argument("--max_completion_length", type=int, default=512, help="Max completion tokens")
    
    # Liger loss options
    parser.add_argument("--use_liger", action="store_true", help="Use Liger kernel model and loss")
    parser.add_argument("--epsilon_low", type=float, default=0.2, help="Lower epsilon for clipping")
    parser.add_argument("--epsilon_high", type=float, default=None, help="Upper epsilon; defaults to epsilon_low if None")
    parser.add_argument("--beta", type=float, default=0.0, help="KL coefficient; 0 disables ref model pathway")
    parser.add_argument("--loss_type", type=str, default="dr_grpo", choices=["grpo", "bnpo", "dr_grpo"], help="Loss aggregation variant")
    
    # vLLM server option
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM server for generation instead of local model")
    parser.add_argument("--vllm_host", type=str, default="localhost", help="vLLM server host")
    parser.add_argument("--vllm_port", type=int, default=8000, help="vLLM server port")
    
    # Training
    parser.add_argument("--num_train_iters", type=int, default=1000, help="Training iterations")
    parser.add_argument("--seed", type=int, default=7111994, help="Random seed")
    parser.add_argument("--eval_every", type=int, default=20, help="Run evaluation every N steps")
    parser.add_argument("--save_every", type=int, default=50, help="Save model checkpoint every N steps")
    parser.add_argument("--disable_checkpoint_saving", action="store_true", help="Disable saving intermediate checkpoints to save disk space")
    parser.add_argument("--save_only_last", action="store_true", help="Only save a single final checkpoint at the end of training")
    
    # Evaluation
    parser.add_argument("--num_completions_eval", type=int, default=5, help="Number of completions to sample per eval problem for pass@k")
    parser.add_argument("--pass_at_k", type=int, default=1, help="k for pass@k metric")

    # Prompt formatting
    parser.add_argument("--think_tag", type=str, default="think", help="Name of think tag")
    parser.add_argument("--answer_tag", type=str, default="answer", help="Name of answer tag")

    # Dataset configuration (composite)
    parser.add_argument("--train-names", nargs="+", default=["leg_counting", "family_relationships"], help="Train dataset names")
    parser.add_argument("--train-weights", nargs="+", type=float, default=[.5,.5], help="Train dataset weights")
    parser.add_argument("--train-size", type=int, default=1000, help="Train dataset size")

    parser.add_argument("--eval-names", nargs="+", default=["leg_counting", "family_relationships", "coin_flip"], help="Eval dataset names")
    parser.add_argument("--eval-weights", nargs="+", type=float, default=[1/3, 1/3, 1/3], help="Eval dataset weights")
    parser.add_argument("--eval-size", type=int, default=60, help="Eval dataset size")

    # Reward masking (discipline of withholding rewards for some correct answers)
    parser.add_argument("--reward_mask_strategy", type=str, default="none", choices=["none","every_n","prob_p","cosine","round_robin_k"], help="Strategy to mask rewards for correct answers")
    parser.add_argument("--reward_mask_every_n", type=int, default=10, help="For every_n: mask every N-th correct example")
    parser.add_argument("--reward_mask_prob", type=float, default=0.1, help="For prob_p: probability of masking a correct reward")
    parser.add_argument("--reward_mask_period", type=int, default=100, help="For cosine: period in steps for cosine schedule")
    parser.add_argument("--reward_mask_max_frac", type=float, default=0.5, help="For cosine: maximum fraction of correct rewards to mask")
    parser.add_argument("--reward_mask_round_robin_k", type=int, default=4, help="For round_robin_k: number of buckets to rotate masking across")
    parser.add_argument("--reward_mask_weight", type=float, default=0.0, help="Scale for masked correct rewards (0.0 = drop, 1.0 = keep)")
    parser.add_argument("--mask_format", action="store_true", help="Apply reward masking to formatting component as well")
    # New: independently control masking for correctness vs formatting
    parser.add_argument("--mask_correctness", dest="mask_correctness", action="store_true", help="Apply reward masking to correctness component (default)")
    parser.add_argument("--no_mask_correctness", dest="mask_correctness", action="store_false", help="Disable reward masking for correctness component")
    parser.set_defaults(mask_correctness=True)
    parser.add_argument("--mask_warmup_steps", type=int, default=0, help="Disable reward masking for the first N training steps")

    # Multi-reward weights (combine components explicitly)
    parser.add_argument("--correctness_weight", type=float, default=1.0, help="Weight for correctness reward component")
    parser.add_argument("--format_weight", type=float, default=1.0, help="Weight for formatting reward component")

    # Full-correct zeroing (set reward to 0 for fully-correct samples occasionally)
    parser.add_argument("--full_correct_zero_strategy", type=str, default="none", choices=["none","every_n","prob_p","round_robin_k","cosine"], help="Strategy to zero rewards for fully-correct samples")
    parser.add_argument("--full_correct_zero_prob", type=float, default=0.1, help="Probability to zero fully-correct rewards (prob_p)")
    parser.add_argument("--full_correct_zero_every_n", type=int, default=10, help="Every N-th fully-correct gets zeroed (every_n)")
    parser.add_argument("--full_correct_zero_round_robin_k", type=int, default=4, help="Buckets for round-robin fully-correct zeroing")
    parser.add_argument("--full_correct_zero_period", type=int, default=200, help="Cosine period for fully-correct zeroing")
    parser.add_argument("--full_correct_zero_max_frac", type=float, default=0.3, help="Cosine max fraction to zero (fully-correct)")
    parser.add_argument("--format_full_threshold", type=float, default=0.9, help="Formatting threshold to consider sample fully-correct")
    parser.add_argument("--format_binary_threshold", type=float, default=None, help="If set, binarize formatting: 1 if format>=threshold else 0")
    parser.add_argument("--zero_warmup_steps", type=int, default=0, help="Disable full-correct zeroing for the first N training steps")

    return parser.parse_args()




if __name__ == "__main__":

    # Get all settings 
    args = parse_args()

    # Seed everything for reproducible results 
    utils.seed_everything(args.seed)

    # Setup logging 
    os.makedirs(args.output_dir, exist_ok=True)
    # Optional W&B
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    # Setup model and vLLM client (if needed)
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, use_liger_model=args.use_liger)
    vllm_client = None
    if args.use_vllm:
        base_url = f"http://{args.vllm_host}:{args.vllm_port}"
        vllm_client = v_c.VLLMClient(base_url=base_url)
        vllm_client.init_communicator(device=model.device)
        print(f"Connected to vLLM server at {args.vllm_host}:{args.vllm_port}")

    # Instantiate Liger loss once if requested
    if args.use_liger:
        liger_loss = LigerFusedLinearGRPOLoss(
            beta=getattr(args, "beta", 0.0),
            epsilon_low=getattr(args, "epsilon_low", 0.2),
            epsilon_high=(args.epsilon_high if getattr(args, "epsilon_high", None) is not None else getattr(args, "epsilon_low", 0.2)),
            temperature=args.temperature,
            use_ref_model=(getattr(args, "beta", 0.0) != 0.0),
            loss_type=getattr(args, "loss_type", "dr_grpo"),
            max_completion_length=args.max_completion_length,
        )

    # Build datasets
    train_ds, eval_ds = reasoning_envs.build_reasoning_envs(
        train_names=args.train_names,
        train_weights=args.train_weights,
        train_size=args.train_size,
        seed=args.seed,
        eval_names=args.eval_names,
        eval_weights=args.eval_weights,
        eval_size=args.eval_size,
    )



    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)
    warmup_steps = int(args.warmup_percent * args.num_train_iters)
    def get_lr(step):
        if step < warmup_steps:
            return (step / max(warmup_steps, 1))
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

    # Unified log structure - step-based with train/eval nested
    run_log = {
        "args": vars(args),
        "steps": {},  # {step: {"train": {...}, "eval": {...}}}
    }

    # Setup prompt 
    system_prompt = (
        f"Please solve the user's problem. "
        f"You must in the following way -  with <{args.think_tag}>Your step by step reasoning through the problem</{args.think_tag}> "
        f"followed by <{args.answer_tag}>your final answer</{args.answer_tag}>. This should be the final answer requested and nothing else, no other text. If the questions asks for a numerical answer you should always use the number, never spell it out."
    )


    # Training loop
    accumulated_loss = 0.0
    optimizer.zero_grad()
    for step in tqdm(range(args.num_train_iters), desc="Training"):
        entry = random.choice(list(train_ds))
        question = entry["question"]
        answer = entry["answer"]
        problem_type = entry.get("metadata", {}).get("source_dataset", "unknown")

        # Setup prompt
        prompt_text, prompt_ids, prompt_mask = utils.format_prompt(system_prompt, question, tokenizer)

        ##################
        ### GRPO LOOP ####
        ##################

        # Generate
        prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completion_mask, completions_text = generate(
            model, tokenizer, prompt_ids, prompt_mask, args, vllm_client, prompt_text
        )

        # Score
        extracted_answers = [utils.extract_answer(t, args.answer_tag) for t in completions_text]
        correctness = [float(train_ds.score_answer(answer=a, entry=entry) == 1.0) for a in extracted_answers]
        format_rewards = [utils.check_format(t, args.think_tag, args.answer_tag) for t in completions_text]
        
        # Combine correctness and format rewards (equal weight)
        # Compute base reward components
        correctness_t = torch.tensor(correctness, device=model.device, dtype=torch.float32)
        format_t = torch.tensor(format_rewards, device=model.device, dtype=torch.float32)
        # Optional: binarize formatting reward
        fbin_thresh = getattr(args, "format_binary_threshold", None)
        if fbin_thresh is not None:
            format_t = (format_t >= float(fbin_thresh)).float()

        # Apply reward masking ONLY to correctness component (optionally formatting too)
        # Warmup gating for reward masking
        if step < int(getattr(args, "mask_warmup_steps", 0)):
            reward_keep_mask = torch.ones_like(torch.tensor(correctness, device=model.device, dtype=torch.float32))
        else:
            reward_keep_mask = _build_reward_mask(correct_flags=correctness, step=step, args=args).to(model.device)
        reward_mask_weight = float(getattr(args, "reward_mask_weight", 0.0))
        effective_mask = reward_keep_mask + (1.0 - reward_keep_mask) * reward_mask_weight
        # Apply masks per component based on flags
        if getattr(args, "mask_correctness", True):
            correctness_masked = correctness_t * effective_mask
        else:
            correctness_masked = correctness_t

        if getattr(args, "mask_format", False):
            format_effective = format_t * effective_mask
        else:
            format_effective = format_t

        # Multi-reward combiner: weights for components (future-proof)
        # By default: correctness_weight=1.0, format_weight=1.0
        correctness_weight = float(getattr(args, "correctness_weight", 1.0))
        format_weight = float(getattr(args, "format_weight", 1.0))

        # Optional per-component masking flags (extendable): currently we only mask correctness
        # If needed: format_mask_flag = getattr(args, "mask_format", False)

        rewards = correctness_weight * correctness_masked + format_weight * format_effective

        # Full-correct zeroing: occasionally set total reward to 0 for a subset of fully-correct samples
        def _build_full_correct_zero_mask_tensor(is_fully_correct: torch.Tensor, step: int, args) -> torch.Tensor:
            strat = getattr(args, "full_correct_zero_strategy", "none")
            keep = torch.ones_like(is_fully_correct, dtype=torch.float32)
            if strat == "none":
                return keep
            if strat == "every_n":
                if not hasattr(args, "_fully_correct_seen"):
                    args._fully_correct_seen = 0
                n = max(1, int(getattr(args, "full_correct_zero_every_n", 10)))
                for i, fc in enumerate(is_fully_correct.tolist()):
                    if fc:
                        args._fully_correct_seen += 1
                        if args._fully_correct_seen % n == 0:
                            keep[i] = 0.0
                return keep
            if strat == "prob_p":
                p = max(0.0, min(1.0, float(getattr(args, "full_correct_zero_prob", 0.1))))
                rand = torch.rand_like(keep)
                keep = torch.where((is_fully_correct == 1) & (rand < p), torch.zeros_like(keep), keep)
                return keep
            if strat == "round_robin_k":
                k = max(2, int(getattr(args, "full_correct_zero_round_robin_k", 4)))
                bucket = step % k
                for i, fc in enumerate(is_fully_correct.tolist()):
                    if fc and (i % k) == bucket:
                        keep[i] = 0.0
                return keep
            if strat == "cosine":
                period = max(1, int(getattr(args, "full_correct_zero_period", 200)))
                max_frac = max(0.0, min(1.0, float(getattr(args, "full_correct_zero_max_frac", 0.3))))
                phase = (step % period) / float(period)
                frac = 0.5 * (1.0 - math.cos(2.0 * math.pi * phase)) * max_frac
                num_fc = int(is_fully_correct.sum().item())
                num_to_zero = int(round(frac * num_fc))
                if num_to_zero > 0:
                    count = 0
                    for i, fc in enumerate(is_fully_correct.tolist()):
                        if fc and count < num_to_zero:
                            keep[i] = 0.0
                            count += 1
                return keep
            return keep

        # Determine fully-correct flags
        format_full_thresh = float(getattr(args, "format_full_threshold", 0.9))
        is_fully_correct = (correctness_t >= 0.999) & (format_t >= format_full_thresh)
        # Build per-sample keep mask for fully-correct zeroing
        # Warmup gating for full-correct zeroing
        if step < int(getattr(args, "zero_warmup_steps", 0)):
            full_keep_mask = torch.ones_like(is_fully_correct, dtype=torch.float32).to(rewards.device)
        else:
            full_keep_mask = _build_full_correct_zero_mask_tensor(is_fully_correct.float(), step, args).to(rewards.device)
        # Apply full-correct zeroing to combined rewards (only affects fully-correct samples)
        rewards = rewards * full_keep_mask
        total_rewards = (correctness_t + format_t).tolist()
        # Multi-reward weights (combine components explicitly)
        # parser.add_argument("--correctness_weight", type=float, default=1.0, help="Weight for correctness reward component")
        # parser.add_argument("--format_weight", type=float, default=1.0, help="Weight for formatting reward component")
        # Apply reward masking to correct answers per strategy
        # (already applied above)

        # Advantages
        grouped = rewards.view(-1, args.num_chains)
        mean_group = grouped.mean(dim=1).repeat_interleave(args.num_chains)
        std_group = grouped.std(dim=1).repeat_interleave(args.num_chains)
        
        advantages = (rewards - mean_group) / (std_group + 1e-4)

        # Normalize masks for loss computation
        # Build a batched prompt mask matching `prompt_ids` (B, prompt_len)
        prompt_mask_batched = torch.ones_like(prompt_ids, device=model.device)
        # Ensure completion_mask covers only completion tokens (B, completion_len)
        if completion_mask.shape[1] != completion_ids.shape[1]:
            completion_mask_for_loss = completion_mask[:, -completion_ids.size(1):]
        else:
            completion_mask_for_loss = completion_mask

        # Compute loss (Liger fused if enabled)
        if args.use_liger:
            loss = compute_liger_grpo_loss(
                model,
                prompt_ids,
                completion_ids,
                prompt_mask_batched,
                completion_mask_for_loss,
                advantages,
                args,
                liger_loss,
            )
        else:
            loss = compute_grpo_loss(
                model,
                prompt_completion_ids,
                prompt_ids,
                completion_ids,
                attention_mask,
                completion_mask_for_loss,
                advantages,
                args,
            )
        (loss / args.gradient_accumulation_steps).backward()
        accumulated_loss += loss.item()

        # Optim step
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            
            # Update vLLM server model parameters if using vLLM
            if args.use_vllm and vllm_client is not None:
                vllm_client.update_model_params(model)
        scheduler.step()

        # Log per step
        if step not in run_log["steps"]:
            run_log["steps"][step] = {}
        
        run_log["steps"][step]["train"] = {
            "prompt": prompt_text,
            "question": question,
            "target_answer": answer,
            "problem_type": problem_type,
            "generations": [
                {
                    "text": t,
                    "extracted_answer": ea,
                    "correct": int(c),
                    "format_reward": float(f),
                    "total_reward": float(tr),
                    "reward_kept": int(rkm),
                    "effective_mask": float(em)
                } for t, ea, c, f, tr, rkm, em in zip(
                    completions_text,
                    extracted_answers,
                    correctness,
                    format_rewards,
                    total_rewards,
                    reward_keep_mask.tolist(),
                    effective_mask.tolist()
                )
            ],
            "loss": loss.item(),
            "lr": scheduler.get_last_lr()[0],
            "num_masked_correct": int((1.0 - reward_keep_mask).sum().item()),
            "reward_mask_strategy": getattr(args, "reward_mask_strategy", "none"),
            "reward_mask_weight": reward_mask_weight,
            "full_correct_zero_strategy": getattr(args, "full_correct_zero_strategy", "none"),
            "num_full_correct_zeroed": int(((is_fully_correct.float() * (1.0 - full_keep_mask))).sum().item()),
        }

        if args.use_wandb:
            import wandb
            # Aggregate simple per-step averages for visibility in W&B
            avg_format = float(format_t.mean().item())
            avg_format_effective = float((format_effective.mean().item()))
            avg_correct = float(correctness_t.mean().item())
            avg_total_reward = float(rewards.mean().item())
            wandb.log({
                "train/loss": loss.item(),
                "lr": scheduler.get_last_lr()[0],
                "train/avg_format_reward": avg_format,
                "train/avg_format_reward_effective": avg_format_effective,
                "train/avg_correctness": avg_correct,
                "train/avg_total_reward": avg_total_reward,
                "train/num_masked_correct": run_log["steps"][step]["train"]["num_masked_correct"],
                "train/num_full_correct_zeroed": run_log["steps"][step]["train"]["num_full_correct_zeroed"],
            }, step=step)

        # Periodic evaluation with pass@k
        if step % args.eval_every == 0 and eval_ds is not None:
            model.eval()  # Set model to eval mode
            pass_at_k_scores = []
            format_total = 0
            eval_count = 0
            eval_examples = []
            
            # Temporarily modify args for eval generation
            original_num_chains = args.num_chains
            args.num_chains = args.num_completions_eval
            
            with torch.no_grad():  # Disable gradients during eval
                for i, eval_entry in enumerate(eval_ds):
                    if i >= args.eval_size:
                        break
                    q = eval_entry["question"]
                    a = eval_entry["answer"]
                    eval_problem_type = eval_entry.get("metadata", {}).get("source_dataset", "unknown")
                    prompt_text, prompt_ids, prompt_mask = utils.format_prompt(system_prompt, q, tokenizer)
                    
                    # Generate multiple completions for this eval problem
                    _, prompt_ids, completion_ids, attention_mask, completion_mask, completions_text = generate(
                        model, tokenizer, prompt_ids, prompt_mask, args, vllm_client, prompt_text
                    )
                    
                    # Score all completions
                    extracted_answers = [utils.extract_answer(t, args.answer_tag) for t in completions_text]
                    correctness = [float(eval_ds.score_answer(answer=ea, entry=eval_entry) == 1.0) for ea in extracted_answers]
                    format_rewards = [utils.check_format(t, args.think_tag, args.answer_tag) for t in completions_text]
                    
                    # Compute pass@k for this problem
                    num_correct = sum(correctness)
                    pass_at_k = compute_pass_at_k(
                        n=args.num_completions_eval,
                        c=int(num_correct),
                        k=args.pass_at_k
                    )
                    pass_at_k_scores.append(pass_at_k)
                    
                    # Average format reward across completions
                    avg_format_for_problem = sum(format_rewards) / len(format_rewards)
                    format_total += avg_format_for_problem
                    eval_count += 1
                    
                    # Log this eval example
                    eval_examples.append({
                        "prompt": prompt_text,
                        "question": q,
                        "target_answer": a,
                        "problem_type": eval_problem_type,
                        "completions": [
                            {
                                "text": t,
                                "extracted_answer": ea,
                                "correct": int(c),
                                "format_reward": float(f)
                            } for t, ea, c, f in zip(completions_text, extracted_answers, correctness, format_rewards)
                        ],
                        "num_correct": int(num_correct),
                        "pass_at_k": pass_at_k,
                        "avg_format_reward": avg_format_for_problem,
                    })
            
            # Restore original num_chains and training mode
            args.num_chains = original_num_chains
            model.train()  # Set model back to train mode
            
            # Clear CUDA cache to prevent OOM
            torch.cuda.empty_cache()
            
            # Aggregate overall metrics
            avg_pass_at_k = (sum(pass_at_k_scores) / max(eval_count, 1)) * 100
            avg_format = (format_total / max(eval_count, 1))
            
            # Compute per-problem-type metrics
            problem_type_metrics = {}
            for example in eval_examples:
                ptype = example["problem_type"]
                if ptype not in problem_type_metrics:
                    problem_type_metrics[ptype] = {
                        "pass_at_k_scores": [],
                        "format_rewards": []
                    }
                problem_type_metrics[ptype]["pass_at_k_scores"].append(example["pass_at_k"])
                problem_type_metrics[ptype]["format_rewards"].append(example["avg_format_reward"])
            
            # Calculate averages per problem type
            per_problem_type = {}
            for ptype, data in problem_type_metrics.items():
                per_problem_type[ptype] = {
                    f"pass_at_{args.pass_at_k}": (sum(data["pass_at_k_scores"]) / len(data["pass_at_k_scores"])) * 100,
                    "avg_format_reward": sum(data["format_rewards"]) / len(data["format_rewards"]),
                    "num_problems": len(data["pass_at_k_scores"])
                }
            
            # Log to step-based structure
            if step not in run_log["steps"]:
                run_log["steps"][step] = {}
            
            run_log["steps"][step]["eval"] = {
                "examples": eval_examples,
                "metrics": {
                    f"pass_at_{args.pass_at_k}": avg_pass_at_k,
                    "avg_format_reward": avg_format,
                    "num_eval_problems": eval_count,
                    "per_problem_type": per_problem_type,
                }
            }
            # Write a concise eval summary to summary.json for external consumption
            try:
                summary_out = {
                    "step": step,
                    "pass_at_k": avg_pass_at_k,
                    "avg_format_reward": avg_format,
                    "num_eval_problems": eval_count,
                    "per_problem_type": per_problem_type,
                }
                with open(os.path.join(args.output_dir, "summary.json"), "w") as sf:
                    json.dump(summary_out, sf, indent=2)
            except Exception as e:
                print(f"Warning: failed to write summary.json at step {step}: {e}")
            
            if args.use_wandb:
                import wandb
                wandb.log({
                    f"eval/pass_at_{args.pass_at_k}": avg_pass_at_k,
                    "eval/avg_format_reward": avg_format,
                    "eval/num_eval_problems": eval_count,
                }, step=step)

        # Periodic model saving (skipped if disabled or saving only last)
        if (step + 1) % args.save_every == 0 and not getattr(args, "disable_checkpoint_saving", False) and not getattr(args, "save_only_last", False):
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_step_{step+1}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            print(f"Saved checkpoint at step {step+1} to {checkpoint_path}")

        # Persist log
        with open(os.path.join(args.output_dir, "run_log.json"), "w") as f:
            json.dump(run_log, f, indent=2)

    # Optionally save only a single final checkpoint at the end
    if getattr(args, "save_only_last", False) and not getattr(args, "disable_checkpoint_saving", False):
        final_ckpt = os.path.join(args.output_dir, "checkpoint_final")
        os.makedirs(final_ckpt, exist_ok=True)
        model.save_pretrained(final_ckpt)
        tokenizer.save_pretrained(final_ckpt)
        print(f"Saved final checkpoint to {final_ckpt}")













#