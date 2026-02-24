"""
Autoregressive text generation with greedy and sampling decoding.
"""

import torch
import torch.nn.functional as F

from .model import KVCache, Qwen3Model


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Select the next token from logits using greedy or sampled decoding.

    When temperature == 0 (default), falls back to argmax (greedy).
    Otherwise applies temperature scaling, then optional top-k and top-p
    filtering, then draws one sample via torch.multinomial.

    The order of operations is: temperature → top-k → top-p → sample.

    Args:
        logits:      Raw logits for the next position, shape [batch, vocab_size].
        temperature: Softmax temperature. 0.0 = greedy argmax.
        top_k:       Keep only the k highest-probability tokens (0 = disabled).
        top_p:       Nucleus sampling threshold; keep the smallest set of tokens
                     whose cumulative probability >= p (1.0 = disabled).

    Returns:
        Selected token indices, shape [batch, 1].
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits.float() / temperature

    # Top-k: mask all but the k largest logits.
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold = logits.topk(k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    # Top-p (nucleus): mask tokens beyond the cumulative-probability cutoff.
    if top_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        # Shift right so the token that pushes us over the threshold is kept.
        remove = (cumulative - sorted_probs) >= top_p
        # Always keep the highest-probability token to avoid an all-zero distribution.
        remove[..., 0] = False
        sorted_probs[remove] = 0.0
        # Scatter back to original vocab ordering.
        probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
        logits = probs.log()  # multinomial works on unnormalized weights too

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: Qwen3Model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    use_cache: bool = True,
) -> torch.Tensor:
    """
    Autoregressive generation loop.

    When *use_cache* is True (default) the loop uses a KV-cache to make each
    decode step O(n) instead of O(n²):

      Phase 1 — Prefill:  feed the full prompt once, fill the cache.
      Phase 2 — Decode:   feed only the single new token each step; the
                          cache supplies the past K/V context.

    When *use_cache* is False the original full-sequence recompute path is
    used (useful for correctness comparison and debugging).

    Args:
        model:          Qwen3Model in eval mode.
        input_ids:      Prompt token ids, shape [1, seq_len].
        max_new_tokens: How many tokens to generate.
        eos_token_id:   Stop early when this token is produced (optional).
        temperature:    Sampling temperature (0.0 = greedy argmax).
        top_k:          Top-k filtering (0 = disabled).
        top_p:          Nucleus sampling threshold (1.0 = disabled).
        use_cache:      Use KV-cache for O(n) decode steps (default: True).

    Returns:
        Token ids including the prompt, shape [1, seq_len + generated].
    """
    model.eval()

    if not use_cache:
        # Original naive path: re-feed the full growing sequence every step.
        tokens = input_ids
        for _ in range(max_new_tokens):
            logits = model(tokens)
            next_token = sample_next_token(
                logits[:, -1],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            tokens = torch.cat([tokens, next_token], dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        return tokens

    # -----------------------------------------------------------------------
    # Cached path
    # -----------------------------------------------------------------------
    cache = KVCache()

    # Phase 1: Prefill — process the full prompt, populate the cache.
    logits = model(input_ids, kv_cache=cache)   # [1, prompt_len, vocab_size]
    next_token = sample_next_token(
        logits[:, -1],
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )                                            # [1, 1]

    generated = [next_token]

    if eos_token_id is not None and next_token.item() == eos_token_id:
        return torch.cat([input_ids, *generated], dim=1)

    # Phase 2: Decode — feed one token at a time; cache grows by one each step.
    for _ in range(max_new_tokens - 1):
        logits = model(next_token, kv_cache=cache)   # [1, 1, vocab_size]
        next_token = sample_next_token(
            logits[:, -1],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated.append(next_token)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return torch.cat([input_ids, *generated], dim=1)
