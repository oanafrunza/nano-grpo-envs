import os
import torch
import random
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple

import re


####################
## MISC FUNCTIONS ##
####################


def seed_everything(seed: int) -> None:
    """
    Set random seed for reproducibility across multiple libraries.
    
    This function sets consistent random seeds for Python's random module,
    NumPy, PyTorch (both CPU and CUDA), and configures CUDNN for deterministic
    operation. This ensures reproducible results across multiple runs.

    Args:
        seed: The random seed to use for all random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Additional settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

########################
##  String Formatting ##
########################

def format_prompt(system_prompt: str, question: str, tokenizer) -> Tuple[str, torch.Tensor, torch.Tensor]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    return prompt_text, inputs["input_ids"], inputs["attention_mask"]


def extract_answer(text: str, answer_tag: str) -> str:
    start = f"<{answer_tag}>"
    end = f"</{answer_tag}>"
    if start in text and end in text:
        return text.split(start, 1)[1].split(end, 1)[0].strip()
    return text.strip()


def check_format(text: str, think_tag: str, answer_tag: str) -> float:
    """Check if text follows the exact format <think>content</think><answer>content</answer> with no other text.
    Returns a soft reward between 0 and 0.4 based on how well it matches the pattern."""
    import re
    
    # Expected pattern: <think>content</think> (optional whitespace) <answer>content</answer>
    pattern = f"<{think_tag}>(.*?)</{think_tag}>\\s*<{answer_tag}>(.*?)</{answer_tag}>"
    match = re.search(pattern, text, re.DOTALL)
    
    if not match:
        return 0.0
    
    think_content, answer_content = match.groups()
    
    # Check if there's any text before or after the pattern
    before_pattern = text[:match.start()].strip()
    after_pattern = text[match.end():].strip()
    
    # Base reward for having the right structure
    reward = 0.2
    
    # Bonus for no extra text
    if not before_pattern and not after_pattern:
        reward += 0.2
    
    # Bonus for non-empty content in both tags
    if think_content.strip():
        reward += 0.1
    if answer_content.strip():
        reward += 0.1
    
    return min(reward, 0.4)


####################################################################################
## Copied Directly from TRL -> generate log probs per token                 ########
## https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py ########
####################################################################################

def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps

def get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

    input_ids = input_ids[:, -logits_to_keep:]
    # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
    # See https://github.com/huggingface/trl/issues/2770
    logits = logits[:, -logits_to_keep:]
    return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens
