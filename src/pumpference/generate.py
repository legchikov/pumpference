"""
Greedy text generation 
"""

import torch

from .model import Qwen3Model


@torch.no_grad()
def generate(
    model: Qwen3Model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    Greedy autoregressive generation.

    Args:
        model:          Qwen3Model in eval mode.
        input_ids:      Prompt token ids, shape [1, seq_len].
        max_new_tokens: How many tokens to generate.
        eos_token_id:   Stop early when this token is produced (optional).

    Returns:
        Token ids including the prompt, shape [1, seq_len + generated].
    """
    model.eval()
    tokens = input_ids

    for _ in range(max_new_tokens):
        logits = model(tokens)           # [1, seq_len, vocab_size]
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)  # [1, 1]

        tokens = torch.cat([tokens, next_token], dim=1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return tokens
