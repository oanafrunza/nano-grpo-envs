
"""
Phase-adaptive GRPO trainer for Reasoning Gym.

This trainer extends main.py with optional phase-adaptive scheduling and
instrumentation. All new features are default-off so existing behavior is
unchanged unless flags are provided.

Adds:
- Completion NLL (entropy proxy) logging and moving-average phase detection
- Phase switch actions: length scaling and optional diversity bonus

"""
import os
import math
import json
import torch
import random
import argparse
from collections import deque
from tqdm import tqdm
from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

# Own modules
import llms
import utils
import reasoning_envs
import vllm_client as v_c


def _get_last_hidden_state_for_liger(model, input_ids, attention_mask, logits_to_keep: int):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        last_hidden = outputs.last_hidden_state
    else:
        last_hidden = outputs.hidden_states[-1]
    last_hidden = last_hidden[:, :-1, :]
    last_hidden = last_hidden[:, -logits_to_keep:, :]
    return last_hidden


def compute_liger_grpo_loss(model, prompt_ids, completion_ids, prompt_mask, completion_mask, advantages, args, liger_loss):
    device = model.device
    prompt_ids = prompt_ids.to(device)
    completion_ids = completion_ids.to(device)
    prompt_mask = prompt_mask.to(device)
    completion_mask = completion_mask.to(device)
    advantages = advantages.to(device)

    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)

    last_hidden_state = _get_last_hidden_state_for_liger(
        model, input_ids, attention_mask, logits_to_keep
    )

    target_device = model.lm_head.weight.device
    if last_hidden_state.device != target_device:
        last_hidden_state = last_hidden_state.to(target_device)
    if completion_ids.device != target_device:
        completion_ids = completion_ids.to(target_device)
    if completion_mask.device != target_device:
        completion_mask = completion_mask.to(target_device)
    if advantages.device != target_device:
        advantages = advantages.to(target_device)

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
    prompt_ids = prompt_ids.repeat(args.num_chains, 1).to(model.device)
    prompt_mask = prompt_mask.repeat(args.num_chains, 1).to(model.device)

    generation_config = {
        "max_new_tokens": args.max_completion_length,
        "do_sample": True,
        "temperature": args.temperature,
        "pad_token_id": tokenizer.pad_token_id,
    }
    with torch.inference_mode():
        prompt_completion_ids = model.generate(prompt_ids, attention_mask=prompt_mask, **generation_config)

    prompt_len = prompt_ids.size(1)
    prompt_ids = prompt_completion_ids[:, :prompt_len]
    completion_ids = prompt_completion_ids[:, prompt_len:]

    is_eos = completion_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=model.device)
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    seq_idx = torch.arange(is_eos.size(1), device=model.device).expand_as(is_eos)
    completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()

    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completion_mask, completions_text


def generate_vllm(vllm_client, prompt_text, tokenizer, args, device):
    response = vllm_client.generate(
        prompts=[prompt_text] * args.num_chains,
        n=1,
        temperature=args.temperature,
        max_tokens=args.max_completion_length,
    )
    prompt_ids_list = response["prompt_ids"]
    completion_ids_list = response["completion_ids"]

    max_prompt_len = max(len(ids) for ids in prompt_ids_list)
    max_completion_len = max(len(ids) for ids in completion_ids_list)

    padded_prompt_ids = []
    for ids in prompt_ids_list:
        padded = ids + [tokenizer.pad_token_id] * (max_prompt_len - len(ids))
        padded_prompt_ids.append(padded)
    prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

    padded_completion_ids = []
    for ids in completion_ids_list:
        padded = ids + [tokenizer.pad_token_id] * (max_completion_len - len(ids))
        padded_completion_ids.append(padded)
    completion_ids = torch.tensor(padded_completion_ids, dtype=torch.long, device=device)

    prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

    attention_mask = torch.ones_like(prompt_completion_ids)
    completion_mask = torch.zeros_like(prompt_completion_ids)
    completion_mask[:, prompt_ids.shape[1]:] = 1

    completions_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in completion_ids_list]

    return prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completion_mask, completions_text


def generate(model, tokenizer, prompt_ids, prompt_mask, args, vllm_client=None, prompt_text=None):
    if args.use_vllm and vllm_client is not None:
        return generate_vllm(vllm_client, prompt_text, tokenizer, args, model.device)
    else:
        return generate_local(model, tokenizer, prompt_ids, prompt_mask, args)


def compute_pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    prob_all_wrong = 1.0
    for i in range(k):
        prob_all_wrong *= (n - c - i) / (n - i)
    return 1.0 - prob_all_wrong


def compute_grpo_loss_and_logps(model, prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completion_mask, advantages, args=None):
    tokens_to_keep = completion_ids.size(1)
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    logps = utils.get_per_token_logps(model, input_ids, attention_mask, tokens_to_keep)
    per_token_loss = -torch.exp(logps - logps.detach()) * advantages.unsqueeze(1)
    completion_only_mask = completion_mask[:, -tokens_to_keep:]
    loss = (per_token_loss * completion_only_mask).sum() / (per_token_loss.size(0) * args.max_completion_length)
    return loss, logps, completion_only_mask


def _build_reward_mask(correct_flags, step, args):
    strategy = getattr(args, "reward_mask_strategy", "none")
    if strategy == "none":
        return torch.ones(len(correct_flags), dtype=torch.float32)
    correct_t = torch.tensor(correct_flags, dtype=torch.float32)
    if strategy == "every_n":
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
        mask = torch.where((correct_t == 1.0) & (rand < p), torch.zeros_like(correct_t), torch.ones_like(correct_t))
        return mask
    if strategy == "cosine":
        period = max(1, int(getattr(args, "reward_mask_period", 100)))
        max_frac = float(getattr(args, "reward_mask_max_frac", 0.5))
        frac = 0.5 * (1.0 + torch.cos(torch.tensor(2.0 * torch.pi * (step % period) / period))) * max_frac
        num_correct = int(correct_t.sum().item())
        num_to_mask = int(round(frac.item() * num_correct))
        mask = torch.ones_like(correct_t)
        if num_to_mask > 0:
            count = 0
            for i, c in enumerate(correct_t):
                if c.item() == 1.0 and count < num_to_mask:
                    mask[i] = 0.0
                    count += 1
        return mask
    if strategy == "round_robin_k":
        k = max(1, int(getattr(args, "reward_mask_round_robin_k", 4)))
        bucket = step % k
        mask = torch.ones_like(correct_t)
        for i, c in enumerate(correct_t):
            if c.item() == 1.0 and (i % k) == bucket:
                mask[i] = 0.0
        return mask
    return torch.ones(len(correct_flags), dtype=torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase-adaptive Nano GRPO with reasoning_gym composite datasets")
    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    # Output and logging
    parser.add_argument("--output_dir", type=str, default="exp_output")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="nano-grpo")
    parser.add_argument("--wandb_run", type=str, default="run")
    # Optimization
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_percent", type=float, default=0.1)
    # Generation
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--num_chains", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--max_completion_length_cap", type=int, default=None, help="Optional hard cap on max_completion_length (useful to avoid OOM when scaling post-phase)")
    # Liger loss options
    parser.add_argument("--use_liger", action="store_true")
    parser.add_argument("--epsilon_low", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=None)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--loss_type", type=str, default="dr_grpo", choices=["grpo","bnpo","dr_grpo"])
    # vLLM server option
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_host", type=str, default="localhost")
    parser.add_argument("--vllm_port", type=int, default=8000)
    # Training
    parser.add_argument("--num_train_iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7111994)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--disable_checkpoint_saving", action="store_true")
    parser.add_argument("--save_only_last", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to reduce memory usage")
    # Evaluation
    parser.add_argument("--num_completions_eval", type=int, default=5)
    parser.add_argument("--pass_at_k", type=int, default=1)
    # Prompt formatting
    parser.add_argument("--think_tag", type=str, default="think")
    parser.add_argument("--answer_tag", type=str, default="answer")
    # Dataset configuration (composite)
    parser.add_argument("--train-names", nargs="+", default=["leg_counting","family_relationships"])
    parser.add_argument("--train-weights", nargs="+", type=float, default=[.5,.5])
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--eval-names", nargs="+", default=["leg_counting","family_relationships","coin_flip"])
    parser.add_argument("--eval-weights", nargs="+", type=float, default=[1/3,1/3,1/3])
    parser.add_argument("--eval-size", type=int, default=60)
    # Reward masking
    parser.add_argument("--reward_mask_strategy", type=str, default="none", choices=["none","every_n","prob_p","cosine","round_robin_k"])
    parser.add_argument("--reward_mask_every_n", type=int, default=10)
    parser.add_argument("--reward_mask_prob", type=float, default=0.1)
    parser.add_argument("--reward_mask_period", type=int, default=100)
    parser.add_argument("--reward_mask_max_frac", type=float, default=0.5)
    parser.add_argument("--reward_mask_round_robin_k", type=int, default=4)
    parser.add_argument("--reward_mask_weight", type=float, default=0.0)
    parser.add_argument("--mask_format", action="store_true")
    parser.add_argument("--mask_warmup_steps", type=int, default=0)
    # Multi-reward weights
    parser.add_argument("--correctness_weight", type=float, default=1.0)
    parser.add_argument("--format_weight", type=float, default=1.0)
    # Full-correct zeroing
    parser.add_argument("--full_correct_zero_strategy", type=str, default="none", choices=["none","every_n","prob_p","round_robin_k","cosine"])
    parser.add_argument("--full_correct_zero_prob", type=float, default=0.1)
    parser.add_argument("--full_correct_zero_every_n", type=int, default=10)
    parser.add_argument("--full_correct_zero_round_robin_k", type=int, default=4)
    parser.add_argument("--full_correct_zero_period", type=int, default=200)
    parser.add_argument("--full_correct_zero_max_frac", type=float, default=0.3)
    parser.add_argument("--format_full_threshold", type=float, default=0.9)
    parser.add_argument("--format_binary_threshold", type=float, default=None)
    parser.add_argument("--zero_warmup_steps", type=int, default=0)
    # Phase-adaptive options (default-off)
    parser.add_argument("--enable_phase_adaptive", action="store_true", help="Enable phase detection and adaptive scheduling")
    parser.add_argument("--phase_target_nll", type=float, default=2.0, help="Target mean NLL (entropy proxy) to consider consolidation complete")
    parser.add_argument("--phase_window", type=int, default=5, help="Moving average window for NLL")
    parser.add_argument("--phase_patience", type=int, default=3, help="Number of consecutive windows below target to switch phase")
    parser.add_argument("--post_phase_max_len_factor", type=float, default=1.3, help="Multiply max_completion_length after phase switch")
    parser.add_argument("--diversity_weight", type=float, default=0.0, help="Weight of distinct-2 diversity bonus (applied after phase switch)")
    parser.add_argument("--post_phase_chain_scale", type=float, default=1.0, help="Scale factor to apply to num_chains after phase switch (e.g., 0.5 halves chains)")
    # Problem-specific overrides (optional JSON file)
    parser.add_argument("--problem_overrides_file", type=str, default=None, help="Path to JSON mapping problem_type -> overrides {format_weight, mask_format, num_chains, max_completion_length}")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    utils.seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    model, tokenizer = llms.get_llm_tokenizer(args.model_name, use_liger_model=args.use_liger)
    if getattr(args, "gradient_checkpointing", False):
        try:
            model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing")
        except Exception as e:
            print(f"Warning: failed to enable gradient checkpointing: {e}")
    vllm_client = None
    if args.use_vllm:
        base_url = f"http://{args.vllm_host}:{args.vllm_port}"
        vllm_client = v_c.VLLMClient(base_url=base_url)
        vllm_client.init_communicator(device=model.device)
        print(f"Connected to vLLM server at {args.vllm_host}:{args.vllm_port}")

    liger_loss = None
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

    train_ds, eval_ds = reasoning_envs.build_reasoning_envs(
        train_names=args.train_names,
        train_weights=args.train_weights,
        train_size=args.train_size,
        seed=args.seed,
        eval_names=args.eval_names,
        eval_weights=args.eval_weights,
        eval_size=args.eval_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)
    warmup_steps = int(args.warmup_percent * args.num_train_iters)
    def get_lr(step):
        if step < warmup_steps:
            return (step / max(warmup_steps, 1))
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

    run_log = {
        "args": vars(args),
        "steps": {},
    }

    system_prompt = (
        f"Please solve the user's problem. "
        f"You must in the following way -  with <{args.think_tag}>Your step by step reasoning through the problem</{args.think_tag}> "
        f"followed by <{args.answer_tag}>your final answer</{args.answer_tag}>. This should be the final answer requested and nothing else, no other text. If the questions asks for a numerical answer you should always use the number, never spell it out."
    )

    # Phase-adaptive state
    args._phase = "consolidation"  # Phase ①
    nll_window = deque(maxlen=max(1, int(getattr(args, "phase_window", 5))))
    below_target_windows = 0
    post_phase_len_factor = float(getattr(args, "post_phase_max_len_factor", 1.0))
    post_phase_chain_scale = float(getattr(args, "post_phase_chain_scale", 1.0))
    max_len_cap = getattr(args, "max_completion_length_cap", None)

    # Optional per-problem overrides
    problem_overrides = {}
    if getattr(args, "problem_overrides_file", None):
        try:
            with open(args.problem_overrides_file, "r") as pf:
                problem_overrides = json.load(pf)
        except Exception as e:
            print(f"Warning: failed to load problem_overrides_file {args.problem_overrides_file}: {e}")

    def get_overrides_for(ptype: str):
        ov = problem_overrides.get(ptype, {}) if isinstance(problem_overrides, dict) else {}
        eff = {
            "format_weight": ov.get("format_weight", None),
            "mask_format": ov.get("mask_format", None),
            "num_chains": ov.get("num_chains", None),
            "max_completion_length": ov.get("max_completion_length", None),
        }
        return eff

    accumulated_loss = 0.0
    optimizer.zero_grad()
    for step in tqdm(range(args.num_train_iters), desc="Training"):
        entry = random.choice(list(train_ds))
        question = entry["question"]
        answer = entry["answer"]
        problem_type = entry.get("metadata", {}).get("source_dataset", "unknown")

        prompt_text, prompt_ids, prompt_mask = utils.format_prompt(system_prompt, question, tokenizer)

        # Determine per-problem overrides
        ov = get_overrides_for(problem_type)
        orig_chains = args.num_chains
        orig_max_len = args.max_completion_length
        # Never exceed current global max (liger configured); allow reducing
        if ov.get("num_chains") is not None:
            try:
                args.num_chains = int(ov["num_chains"])
            except Exception:
                pass
        if ov.get("max_completion_length") is not None:
            try:
                desired_len = int(ov["max_completion_length"])
                args.max_completion_length = min(desired_len, orig_max_len)
            except Exception:
                pass

        # Generate with OOM backoff
        def _apply_memory_backoff():
            try:
                # Preserve completion length for apples-to-apples; reduce chains only
                args.num_chains = max(1, int(round(args.num_chains * 0.75)))
                torch.cuda.empty_cache()
                print(f"[Backoff] Reducing num_chains to {args.num_chains} while preserving max_completion_length={args.max_completion_length}")
            except Exception as _:
                pass

        try:
            prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completion_mask, completions_text = generate(
                model, tokenizer, prompt_ids, prompt_mask, args, vllm_client, prompt_text
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM during generation at step {step}: {e}. Applying backoff and retrying once.")
            _apply_memory_backoff()
            try:
                prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completion_mask, completions_text = generate(
                    model, tokenizer, prompt_ids, prompt_mask, args, vllm_client, prompt_text
                )
            except torch.cuda.OutOfMemoryError as e2:
                print(f"Retry OOM during generation at step {step}: {e2}. Skipping step.")
                # Restore global args before continue
                args.num_chains = orig_chains
                args.max_completion_length = orig_max_len
                continue

        # Restore global args after generation
        args.num_chains = orig_chains
        args.max_completion_length = orig_max_len

        # Score
        extracted_answers = [utils.extract_answer(t, args.answer_tag) for t in completions_text]
        correctness = [float(train_ds.score_answer(answer=a, entry=entry) == 1.0) for a in extracted_answers]
        format_rewards = [utils.check_format(t, args.think_tag, args.answer_tag) for t in completions_text]

        # Compute loss (and logps for NLL proxy) with OOM backoff
        prompt_mask_batched = torch.ones_like(prompt_ids, device=model.device)
        try:
            if args.use_liger:
                loss = compute_liger_grpo_loss(
                    model,
                    prompt_ids,
                    completion_ids,
                    prompt_mask_batched,
                    (completion_mask[:, -completion_ids.size(1):] if completion_mask.shape[1] != completion_ids.shape[1] else completion_mask),
                    torch.zeros(completion_ids.size(0), device=model.device),  # placeholder; will recompute advantages below
                    args,
                    liger_loss,
                )
                # For Liger path, compute logps separately for NLL proxy
                tokens_to_keep = completion_ids.size(1)
                input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
                logps = utils.get_per_token_logps(model, input_ids, attention_mask, tokens_to_keep)
                completion_only_mask = (completion_mask[:, -tokens_to_keep:]) if completion_mask.shape[1] != completion_ids.shape[1] else completion_mask
            else:
                loss_tmp, logps, completion_only_mask = compute_grpo_loss_and_logps(
                    model,
                    prompt_completion_ids,
                    prompt_ids,
                    completion_ids,
                    attention_mask,
                    completion_mask,
                    torch.zeros(completion_ids.size(0), device=model.device),  # placeholder; will recompute advantages below
                    args,
                )
                loss = loss_tmp
        except torch.cuda.OutOfMemoryError as e:
            print(f"OOM during loss forward at step {step}: {e}. Applying backoff and retrying once.")
            _apply_memory_backoff()
            try:
                # Re-generate with reduced settings
                prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completion_mask, completions_text = generate(
                    model, tokenizer, prompt_ids, prompt_mask, args, vllm_client, prompt_text
                )
                if args.use_liger:
                    loss = compute_liger_grpo_loss(
                        model,
                        prompt_ids,
                        completion_ids,
                        prompt_mask_batched,
                        (completion_mask[:, -completion_ids.size(1):] if completion_mask.shape[1] != completion_ids.shape[1] else completion_mask),
                        torch.zeros(completion_ids.size(0), device=model.device),
                        args,
                        liger_loss,
                    )
                    tokens_to_keep = completion_ids.size(1)
                    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
                    logps = utils.get_per_token_logps(model, input_ids, attention_mask, tokens_to_keep)
                    completion_only_mask = (completion_mask[:, -tokens_to_keep:]) if completion_mask.shape[1] != completion_ids.shape[1] else completion_mask
                else:
                    loss_tmp, logps, completion_only_mask = compute_grpo_loss_and_logps(
                        model,
                        prompt_completion_ids,
                        prompt_ids,
                        completion_ids,
                        attention_mask,
                        completion_mask,
                        torch.zeros(completion_ids.size(0), device=model.device),
                        args,
                    )
                    loss = loss_tmp
            except torch.cuda.OutOfMemoryError as e2:
                print(f"Retry OOM during loss forward at step {step}: {e2}. Skipping step.")
                # Restore global args after generation adjustments
                args.num_chains = orig_chains
                args.max_completion_length = orig_max_len
                continue

        # NLL proxy (entropy): mean of -log p(token) across completion tokens
        with torch.no_grad():
            nll_per_token = -logps
            denom = (completion_only_mask.sum(dim=1).clamp_min(1)).float()
            nll_mean = (nll_per_token.sum(dim=1) / denom).mean().item()
            nll_window.append(nll_mean)

        # Combine correctness and format rewards
        correctness_t = torch.tensor(correctness, device=model.device, dtype=torch.float32)
        format_t = torch.tensor(format_rewards, device=model.device, dtype=torch.float32)
        fbin_thresh = getattr(args, "format_binary_threshold", None)
        if fbin_thresh is not None:
            format_t = (format_t >= float(fbin_thresh)).float()

        # Reward masking warmup
        if step < int(getattr(args, "mask_warmup_steps", 0)):
            reward_keep_mask = torch.ones_like(torch.tensor(correctness, device=model.device, dtype=torch.float32))
        else:
            reward_keep_mask = _build_reward_mask(correct_flags=correctness, step=step, args=args).to(model.device)
        reward_mask_weight = float(getattr(args, "reward_mask_weight", 0.0))
        effective_mask = reward_keep_mask + (1.0 - reward_keep_mask) * reward_mask_weight
        correctness_masked = correctness_t * effective_mask
        format_effective = (format_t * effective_mask) if getattr(args, "mask_format", False) else format_t

        correctness_weight = float(getattr(args, "correctness_weight", 1.0))
        # Apply per-problem format weight and mask_format if specified
        format_weight_eff = float(ov["format_weight"]) if ov.get("format_weight") is not None else float(getattr(args, "format_weight", 1.0))
        mask_format_eff = bool(ov["mask_format"]) if ov.get("mask_format") is not None else bool(getattr(args, "mask_format", False))
        format_effective = (format_t * effective_mask) if mask_format_eff else format_t
        rewards = correctness_weight * correctness_masked + format_weight_eff * format_effective

        # Full-correct zeroing
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

        format_full_thresh = float(getattr(args, "format_full_threshold", 0.9))
        is_fully_correct = (correctness_t >= 0.999) & (format_t >= format_full_thresh)
        if step < int(getattr(args, "zero_warmup_steps", 0)):
            full_keep_mask = torch.ones_like(is_fully_correct, dtype=torch.float32).to(rewards.device)
        else:
            full_keep_mask = _build_full_correct_zero_mask_tensor(is_fully_correct.float(), step, args).to(rewards.device)
        rewards = rewards * full_keep_mask

        # Diversity bonus (distinct-2 within each completion), applied after phase switch
        def _distinct2_ratio(text: str) -> float:
            toks = text.split()
            if len(toks) < 2:
                return 0.0
            bigrams = [tuple(toks[i:i+2]) for i in range(len(toks)-1)]
            return len(set(bigrams)) / float(len(bigrams))
        if getattr(args, "enable_phase_adaptive", False) and args._phase == "planning" and float(getattr(args, "diversity_weight", 0.0)) > 0.0:
            divs = torch.tensor([_distinct2_ratio(t) for t in completions_text], device=model.device, dtype=torch.float32)
            rewards = rewards + float(getattr(args, "diversity_weight", 0.0)) * divs

        # Advantages
        grouped = rewards.view(-1, args.num_chains)
        mean_group = grouped.mean(dim=1).repeat_interleave(args.num_chains)
        std_group = grouped.std(dim=1).repeat_interleave(args.num_chains)
        advantages = (rewards - mean_group) / (std_group + 1e-4)

        # Compute final loss using advantages
        completion_mask_for_loss = completion_mask[:, -completion_ids.size(1):] if completion_mask.shape[1] != completion_ids.shape[1] else completion_mask
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
            loss = compute_grpo_loss_and_logps(
                model,
                prompt_completion_ids,
                prompt_ids,
                completion_ids,
                attention_mask,
                completion_mask_for_loss,
                advantages,
                args,
            )[0]
        (loss / args.gradient_accumulation_steps).backward()
        accumulated_loss += loss.item()

        # Phase switch check (after warmup): use moving-average NLL
        if getattr(args, "enable_phase_adaptive", False) and step >= int(getattr(args, "mask_warmup_steps", 0)):
            if len(nll_window) == nll_window.maxlen and (sum(nll_window) / len(nll_window)) < float(getattr(args, "phase_target_nll", 2.0)):
                below_target_windows += 1
            else:
                below_target_windows = 0
            if args._phase == "consolidation" and below_target_windows >= int(getattr(args, "phase_patience", 3)):
                args._phase = "planning"
                # Length scaling
                new_len = int(args.max_completion_length * post_phase_len_factor)
                if max_len_cap is not None:
                    try:
                        cap_val = int(max_len_cap)
                        new_len = min(new_len, cap_val)
                    except Exception:
                        pass
                args.max_completion_length = max(1, new_len)
                # Optionally reduce num_chains to keep token budget stable
                if post_phase_chain_scale != 1.0:
                    try:
                        new_chains = max(1, int(round(args.num_chains * post_phase_chain_scale)))
                        print(f"[Phase-Adapt] Switching to planning: max_len={args.max_completion_length}, num_chains {args.num_chains} -> {new_chains}")
                        args.num_chains = new_chains
                    except Exception as e:
                        print(f"Warning: failed to scale num_chains after phase switch: {e}")
                # If using Liger, reinitialize fused loss with new max length to avoid shape issues
                if args.use_liger:
                    try:
                        nonlocal_liger = LigerFusedLinearGRPOLoss(
                            beta=getattr(args, "beta", 0.0),
                            epsilon_low=getattr(args, "epsilon_low", 0.2),
                            epsilon_high=(args.epsilon_high if getattr(args, "epsilon_high", None) is not None else getattr(args, "epsilon_low", 0.2)),
                            temperature=args.temperature,
                            use_ref_model=(getattr(args, "beta", 0.0) != 0.0),
                            loss_type=getattr(args, "loss_type", "dr_grpo"),
                            max_completion_length=args.max_completion_length,
                        )
                        liger_loss = nonlocal_liger
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"Warning: failed to reinit Liger loss after phase switch: {e}")

        # Optim step
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
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
                } for t, ea, c, f in zip(
                    completions_text,
                    extracted_answers,
                    correctness,
                    format_rewards,
                )
            ],
            "loss": loss.item(),
            "lr": scheduler.get_last_lr()[0],
            "avg_completion_nll": float(sum(nll_window)/len(nll_window)) if len(nll_window)>0 else None,
            "phase": args._phase,
        }
        if args.use_wandb:
            import wandb
            avg_format = float(torch.tensor(format_rewards, dtype=torch.float32).mean().item())
            avg_correct = float(torch.tensor(correctness, dtype=torch.float32).mean().item())
            wandb.log({
                "train/loss": loss.item(),
                "lr": scheduler.get_last_lr()[0],
                "train/avg_format_reward": avg_format,
                "train/avg_correctness": avg_correct,
                "train/avg_completion_nll": run_log["steps"][step]["train"]["avg_completion_nll"],
                "train/phase": 0 if args._phase=="consolidation" else 1,
            }, step=step)

        # Periodic evaluation
        if step % args.eval_every == 0 and eval_ds is not None:
            model.eval()
            pass_at_k_scores = []
            format_total = 0
            eval_count = 0
            eval_examples = []

            original_num_chains = args.num_chains
            args.num_chains = args.num_completions_eval

            with torch.no_grad():
                for i, eval_entry in enumerate(eval_ds):
                    if i >= args.eval_size:
                        break
                    q = eval_entry["question"]
                    a = eval_entry["answer"]
                    eval_problem_type = eval_entry.get("metadata", {}).get("source_dataset", "unknown")
                    prompt_text, prompt_ids, prompt_mask = utils.format_prompt(system_prompt, q, tokenizer)
                    # Apply per-problem overrides for eval generation as well (bounded by current global max)
                    eval_ov = get_overrides_for(eval_problem_type)
                    save_chains, save_maxlen = args.num_chains, args.max_completion_length
                    if eval_ov.get("num_chains") is not None:
                        try:
                            args.num_chains = int(eval_ov["num_chains"])
                        except Exception:
                            pass
                    if eval_ov.get("max_completion_length") is not None:
                        try:
                            desired_len_eval = int(eval_ov["max_completion_length"])
                            args.max_completion_length = min(desired_len_eval, save_maxlen)
                        except Exception:
                            pass
                    _, prompt_ids, completion_ids, attention_mask, completion_mask, completions_text = generate(
                        model, tokenizer, prompt_ids, prompt_mask, args, vllm_client, prompt_text
                    )
                    args.num_chains, args.max_completion_length = save_chains, save_maxlen
                    extracted_answers = [utils.extract_answer(t, args.answer_tag) for t in completions_text]
                    correctness = [float(eval_ds.score_answer(answer=ea, entry=eval_entry) == 1.0) for ea in extracted_answers]
                    format_rewards = [utils.check_format(t, args.think_tag, args.answer_tag) for t in completions_text]

                    num_correct = sum(correctness)
                    pass_at_k = compute_pass_at_k(n=args.num_completions_eval, c=int(num_correct), k=args.pass_at_k)
                    pass_at_k_scores.append(pass_at_k)

                    avg_format_for_problem = sum(format_rewards) / len(format_rewards)
                    format_total += avg_format_for_problem
                    eval_count += 1
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

            args.num_chains = original_num_chains
            model.train()
            torch.cuda.empty_cache()

            avg_pass_at_k = (sum(pass_at_k_scores) / max(eval_count, 1)) * 100
            avg_format = (format_total / max(eval_count, 1))

            problem_type_metrics = {}
            for example in eval_examples:
                ptype = example["problem_type"]
                if ptype not in problem_type_metrics:
                    problem_type_metrics[ptype] = {"pass_at_k_scores": [], "format_rewards": []}
                problem_type_metrics[ptype]["pass_at_k_scores"].append(example["pass_at_k"])
                problem_type_metrics[ptype]["format_rewards"].append(example["avg_format_reward"])

            per_problem_type = {}
            for ptype, data in problem_type_metrics.items():
                per_problem_type[ptype] = {
                    f"pass_at_{args.pass_at_k}": (sum(data["pass_at_k_scores"]) / len(data["pass_at_k_scores"])) * 100,
                    "avg_format_reward": sum(data["format_rewards"]) / len(data["format_rewards"]),
                    "num_problems": len(data["pass_at_k_scores"])
                }

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

        if (step + 1) % args.save_every == 0 and not getattr(args, "disable_checkpoint_saving", False) and not getattr(args, "save_only_last", False):
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_step_{step+1}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            print(f"Saved checkpoint at step {step+1} to {checkpoint_path}")

        with open(os.path.join(args.output_dir, "run_log.json"), "w") as f:
            json.dump(run_log, f, indent=2)

    if getattr(args, "save_only_last", False) and not getattr(args, "disable_checkpoint_saving", False):
        final_ckpt = os.path.join(args.output_dir, "checkpoint_final")
        os.makedirs(final_ckpt, exist_ok=True)
        model.save_pretrained(final_ckpt)
        tokenizer.save_pretrained(final_ckpt)
        print(f"Saved final checkpoint to {final_ckpt}")
