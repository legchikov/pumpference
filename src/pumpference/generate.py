"""
Autoregressive text generation with greedy and sampling decoding.

Includes standard KV-cached generation and speculative decoding.
"""

from dataclasses import dataclass

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


# ---------------------------------------------------------------------------
# Speculative Decoding
# ---------------------------------------------------------------------------

@dataclass
class SpeculativeStats:
    """Per-run statistics for speculative decoding."""

    num_rounds: int
    total_draft_tokens: int
    total_accepted_tokens: int
    acceptance_rate: float
    tokens_per_round: float  # mean accepted+bonus tokens produced per round


def _get_probs(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """
    Convert a logits vector to a probability distribution.

    Applies the same temperature / top-k / top-p filtering used by
    sample_next_token, but returns the full distribution rather than sampling
    from it.  Used by rejection sampling to compare draft vs target probs.

    Args:
        logits: [vocab_size] — raw logits for a single position.

    Returns:
        [vocab_size] float32 probability vector (sums to 1).
    """
    logits = logits.float()
    if temperature > 0:
        logits = logits / temperature
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold = logits.topk(k).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = probs.sort(descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        remove = (cumulative - sorted_probs) >= top_p
        remove[0] = False
        sorted_probs[remove] = 0.0
        probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
        return probs / probs.sum()
    return F.softmax(logits, dim=-1)


@torch.no_grad()
def speculative_generate(
    target_model: Qwen3Model,
    draft_model: Qwen3Model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    num_speculative_tokens: int = 5,
    eos_token_id: int | None = None,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, SpeculativeStats]:
    """
    Speculative decoding: draft model proposes tokens, target model verifies.

    Each speculation round:
      1. Draft K tokens with *draft_model* (K serial single-token forwards).
      2. Verify with *target_model* in ONE forward pass over K+1 tokens
         (the last accepted token plus the K drafts).
      3. Accept tokens greedily or via rejection sampling, depending on
         whether *temperature* is 0.

    Acceptance rules
    ----------------
    Greedy (temperature == 0):
        Accept draft token d_i if target argmax at position i matches d_i.
        First mismatch: take the target's argmax and start a new round.
        All K accepted: take bonus token from target logits at position K.

    Sampling (temperature > 0):
        Accept d_i with probability min(1, p_target(d_i) / p_draft(d_i)).
        First rejection: sample corrected token from norm(max(0, p - q)).
        All K accepted: sample bonus token from target distribution at K.

    KV-cache handling
    -----------------
    - Both caches are prefilled with the prompt in one forward pass each.
    - On full acceptance: draft cache is one step behind (it saw d_1..d_{K-1}
      but not d_K), so one extra draft forward syncs it.
    - On partial acceptance (n_accepted < K): both caches are truncated to
      prompt_len + previously_accepted + n_accepted + 1.

    Args:
        target_model:           Large, slow, authoritative model.
        draft_model:            Small, fast model that proposes candidates.
        input_ids:              Prompt token ids, shape [1, prompt_len].
        max_new_tokens:         Maximum new tokens to generate.
        num_speculative_tokens: K — draft tokens proposed per round.
        eos_token_id:           Stop when this token is produced.
        temperature:            Sampling temperature (0.0 = greedy).
        top_k:                  Top-k filtering (0 = disabled).
        top_p:                  Nucleus sampling threshold (1.0 = disabled).

    Returns:
        (token_ids [1, prompt_len + generated], SpeculativeStats)
    """
    target_model.eval()
    draft_model.eval()

    target_cache = KVCache()
    draft_cache = KVCache()

    # -----------------------------------------------------------------------
    # Prefill both models with the prompt.
    # -----------------------------------------------------------------------
    target_logits = target_model(input_ids, kv_cache=target_cache)  # [1, P, V]
    draft_model(input_ids, kv_cache=draft_cache)                     # populate draft cache

    # Sample the very first token from the target.
    first_token = sample_next_token(
        target_logits[:, -1],
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    # Do NOT feed first_token to draft here.  The draft loop feeds last_token
    # at the start of every round, advancing draft_cache one step each time.
    # Invariant: draft_cache.seq_len == target_cache.seq_len == P + n_gen - 1,
    # where n_gen is len(generated).  Both caches do NOT yet contain last_token.

    generated: list[torch.Tensor] = [first_token]
    if eos_token_id is not None and first_token.item() == eos_token_id:
        stats = SpeculativeStats(
            num_rounds=0,
            total_draft_tokens=0,
            total_accepted_tokens=0,
            acceptance_rate=0.0,
            tokens_per_round=0.0,
        )
        return torch.cat([input_ids, *generated], dim=1), stats

    # -----------------------------------------------------------------------
    # Speculation rounds.
    # -----------------------------------------------------------------------
    num_rounds = 0
    total_draft_tokens = 0
    total_accepted_tokens = 0
    K = num_speculative_tokens

    while len(generated) < max_new_tokens:
        last_token = generated[-1]  # [1, 1] — last accepted token

        # --- Phase 1: draft K tokens ----------------------------------------
        draft_tokens: list[torch.Tensor] = []
        draft_token_logits: list[torch.Tensor] = []  # [vocab_size] per token

        token = last_token
        for _ in range(K):
            if len(generated) + len(draft_tokens) >= max_new_tokens:
                break
            d_logits = draft_model(token, kv_cache=draft_cache)  # [1, 1, V]
            token = sample_next_token(
                d_logits[:, -1],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            draft_tokens.append(token)
            draft_token_logits.append(d_logits[0, -1])  # [V]

        if not draft_tokens:
            break

        num_proposed = len(draft_tokens)
        total_draft_tokens += num_proposed
        num_rounds += 1

        # --- Phase 2: target verifies in one forward pass --------------------
        # Feed [last_token, d_1, ..., d_K] — K+1 tokens together.
        verify_ids = torch.cat([last_token, *draft_tokens], dim=1)  # [1, K+1]
        target_v_logits = target_model(verify_ids, kv_cache=target_cache)  # [1, K+1, V]

        # target_v_logits[:, i, :] predicts position (cache_len + i + 1).
        # Position 0 logits → what comes after last_token → checks d_1.
        # Position i logits → what comes after d_i       → checks d_{i+1}.
        # Position K logits → what comes after d_K       → bonus token.

        # Record the pre-verification cache length (excludes the K+1 new tokens).
        cache_len_before = target_cache.seq_len - (num_proposed + 1)

        # --- Phase 3: accept / reject ----------------------------------------
        n_accepted = 0

        if temperature == 0.0:
            # Greedy acceptance: target argmax must match draft token.
            for i, d_tok in enumerate(draft_tokens):
                target_pred = target_v_logits[0, i].argmax().item()
                if target_pred == d_tok.item():
                    n_accepted += 1
                else:
                    break
        else:
            # Rejection sampling: accept d_i with prob min(1, p(d_i)/q(d_i)).
            for i, (d_tok, d_log) in enumerate(
                zip(draft_tokens, draft_token_logits)
            ):
                p = _get_probs(target_v_logits[0, i], temperature, top_k, top_p)
                q = _get_probs(d_log, temperature, top_k, top_p)
                d_idx = int(d_tok.item())
                accept_prob = min(1.0, (p[d_idx] / (q[d_idx] + 1e-10)).item())
                if torch.rand(1).item() < accept_prob:
                    n_accepted += 1
                else:
                    break

        total_accepted_tokens += n_accepted

        # --- Phase 4: collect accepted tokens + corrected/bonus --------------
        round_tokens: list[torch.Tensor] = list(draft_tokens[:n_accepted])
        all_accepted = n_accepted == num_proposed

        if all_accepted:
            # Bonus token from target logits at the last position (d_K).
            bonus = sample_next_token(
                target_v_logits[:, num_proposed],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            round_tokens.append(bonus)
            # Sync draft cache: the loop fed last_token + d_1..d_{K-1} (K items)
            # but not d_K itself.  Feed d_K so both caches reach seq_len
            # P + len(generated after extend) - 1, preserving the invariant.
            draft_model(draft_tokens[-1], kv_cache=draft_cache)
        else:
            # Rejected at position n_accepted: sample corrected token.
            if temperature == 0.0:
                corrected = target_v_logits[:, n_accepted].argmax(dim=-1, keepdim=True)
            else:
                p = _get_probs(
                    target_v_logits[0, n_accepted], temperature, top_k, top_p
                )
                q = _get_probs(
                    draft_token_logits[n_accepted], temperature, top_k, top_p
                )
                adjusted = (p - q).clamp(min=0.0)
                s = adjusted.sum()
                adjusted = (
                    adjusted / s if s > 0
                    else torch.ones_like(adjusted) / adjusted.numel()
                )
                corrected = torch.multinomial(adjusted, num_samples=1).unsqueeze(0)
            round_tokens.append(corrected)

            # Roll back both caches so they do NOT contain last_token or any
            # draft tokens beyond what was accepted.  After truncation:
            #   target_cache.seq_len = cache_len_before + 1 + n_accepted
            #                        = P + len(generated so far) - 1  ← invariant
            # The corrected token is NOT fed here; it becomes last_token for
            # the next round and is fed at the start of that round's draft loop.
            new_cache_len = cache_len_before + 1  # = P + 1 (prefix + last_token)
            target_cache.truncate(new_cache_len + n_accepted)
            draft_cache.truncate(new_cache_len + n_accepted)

        generated.extend(round_tokens)

        # Check for EOS in the newly added tokens.
        if eos_token_id is not None:
            eos_hit = False
            for j, tok in enumerate(round_tokens):
                if tok.item() == eos_token_id:
                    # Trim generated list to include everything up to EOS.
                    keep = len(generated) - len(round_tokens) + j + 1
                    generated = generated[:keep]
                    eos_hit = True
                    break
            if eos_hit:
                break

        if len(generated) >= max_new_tokens:
            generated = generated[:max_new_tokens]
            break

    # -----------------------------------------------------------------------
    # Build stats.
    # -----------------------------------------------------------------------
    acc_rate = (
        total_accepted_tokens / total_draft_tokens
        if total_draft_tokens > 0
        else 0.0
    )
    tpr = (
        (len(generated) - 1) / num_rounds  # -1: first token came from prefill
        if num_rounds > 0
        else 0.0
    )
    stats = SpeculativeStats(
        num_rounds=num_rounds,
        total_draft_tokens=total_draft_tokens,
        total_accepted_tokens=total_accepted_tokens,
        acceptance_rate=round(acc_rate, 4),
        tokens_per_round=round(tpr, 2),
    )

    return torch.cat([input_ids, *generated], dim=1), stats
