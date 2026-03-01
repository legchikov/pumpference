"""
Tests for speculative decoding and the supporting model primitives.

All tests use Qwen3-0.6B as *both* draft and target so no additional model
download is required.  When draft == target, greedy acceptance rate is 100%
and the output must be identical to standard KV-cached generation.
"""

import pytest
import torch

from pumpference.generate import generate, speculative_generate, SpeculativeStats
from pumpference.model import (
    QWEN3_0_6B_CONFIG,
    KVCache,
    Qwen3Model,
    download_and_load_weights,
)

REPO_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cpu"
PROMPT = "Give me a short introduction to large language models."
MAX_NEW_TOKENS = 20
K = 4  # speculative draft window


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def our_model() -> Qwen3Model:
    model = Qwen3Model(QWEN3_0_6B_CONFIG)
    download_and_load_weights(model, repo_id=REPO_ID)
    model.to(DEVICE)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    from pumpference.tokenizer import download_tokenizer
    return download_tokenizer(repo_id=REPO_ID)


@pytest.fixture(scope="module")
def input_ids(tokenizer) -> torch.Tensor:
    return torch.tensor([tokenizer.encode(PROMPT)], device=DEVICE)


# ---------------------------------------------------------------------------
# KVCache.truncate
# ---------------------------------------------------------------------------

def test_kv_cache_truncate_seq_len() -> None:
    """After truncation the reported seq_len must match the requested length."""
    cache = KVCache()
    # Simulate a cache with seq_len=10 by injecting dummy tensors.
    batch, kv_heads, seq_len, head_dim = 1, 8, 10, 64
    for _ in range(28):
        k = torch.randn(batch, kv_heads, seq_len, head_dim)
        v = torch.randn(batch, kv_heads, seq_len, head_dim)
        cache._cache.append((k, v))

    assert cache.seq_len == seq_len
    cache.truncate(6)
    assert cache.seq_len == 6


def test_kv_cache_truncate_tensor_shape() -> None:
    """Truncated tensors must have the correct shape along dim=2."""
    cache = KVCache()
    batch, kv_heads, seq_len, head_dim = 1, 8, 12, 64
    originals_k = []
    for _ in range(4):
        k = torch.randn(batch, kv_heads, seq_len, head_dim)
        v = torch.randn(batch, kv_heads, seq_len, head_dim)
        cache._cache.append((k, v))
        originals_k.append(k)

    new_len = 5
    cache.truncate(new_len)
    for layer_idx, (k, v) in enumerate(cache._cache):
        assert k.shape == (batch, kv_heads, new_len, head_dim), (
            f"Layer {layer_idx}: expected K shape {(batch, kv_heads, new_len, head_dim)}, "
            f"got {k.shape}"
        )
        assert v.shape == (batch, kv_heads, new_len, head_dim)
        # Values must equal the original prefix, not garbage.
        assert torch.equal(k, originals_k[layer_idx][:, :, :new_len, :])


# ---------------------------------------------------------------------------
# Causal mask fix: multi-token decode with non-empty KV cache
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_multi_token_decode_mask_matches_sequential(our_model, input_ids) -> None:
    """
    Feeding K tokens in one call with a non-empty cache must produce the same
    logits as K sequential single-token calls.

    This validates the causal mask fix for speculative verification:
    the unified mask code must correctly handle past_len > 0 and q_len > 1.
    """
    K_TOKENS = 5
    prompt_len = input_ids.shape[1]

    # --- Sequential path: prefill then K single-token steps ---------------
    seq_cache = KVCache()
    our_model(input_ids, kv_cache=seq_cache)                 # prefill
    first_token = our_model(input_ids, kv_cache=KVCache())[:, -1].argmax(dim=-1, keepdim=True)

    seq_logits = []
    tok = first_token.clone()
    tmp_cache = KVCache()
    our_model(input_ids, kv_cache=tmp_cache)                 # re-prefill for sequential path
    for _ in range(K_TOKENS):
        logits = our_model(tok, kv_cache=tmp_cache)          # [1, 1, V]
        seq_logits.append(logits[:, -1].clone())
        tok = logits[:, -1].argmax(dim=-1, keepdim=True)

    # --- Batch path: prefill then feed K tokens at once -------------------
    batch_cache = KVCache()
    our_model(input_ids, kv_cache=batch_cache)               # re-prefill

    # Collect the same K tokens by replaying sequential argmax decisions.
    batch_tokens = [first_token]
    t = first_token.clone()
    replay_cache = KVCache()
    our_model(input_ids, kv_cache=replay_cache)
    for _ in range(K_TOKENS - 1):
        out = our_model(t, kv_cache=replay_cache)
        t = out[:, -1].argmax(dim=-1, keepdim=True)
        batch_tokens.append(t)

    multi_input = torch.cat(batch_tokens, dim=1)             # [1, K]
    batch_out = our_model(multi_input, kv_cache=batch_cache) # [1, K, V]

    # Logits at position i in the batch output should match sequential step i.
    for i in range(K_TOKENS):
        seq_argmax   = seq_logits[i].argmax(dim=-1)
        batch_argmax = batch_out[:, i, :].argmax(dim=-1)
        assert torch.equal(seq_argmax, batch_argmax), (
            f"Position {i}: sequential argmax {seq_argmax.item()} != "
            f"batch argmax {batch_argmax.item()}"
        )


# ---------------------------------------------------------------------------
# Speculative decoding correctness (greedy, draft == target)
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_speculative_greedy_matches_standard(our_model, input_ids, tokenizer) -> None:
    """
    When draft == target, greedy acceptance rate is 100% and output must be
    identical to standard KV-cached generation.
    """
    standard_output = generate(
        our_model,
        input_ids.clone(),
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    spec_output, stats = speculative_generate(
        target_model=our_model,
        draft_model=our_model,
        input_ids=input_ids.clone(),
        max_new_tokens=MAX_NEW_TOKENS,
        num_speculative_tokens=K,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.0,
    )

    assert standard_output[0].tolist() == spec_output[0].tolist(), (
        f"Speculative (draft==target, greedy) differs from standard generation!\n"
        f"  Standard:    {standard_output[0].tolist()}\n"
        f"  Speculative: {spec_output[0].tolist()}"
    )
    # With identical draft and target models, every draft token should be accepted.
    assert stats.acceptance_rate == 1.0, (
        f"Expected 100% acceptance rate when draft==target, got {stats.acceptance_rate:.1%}"
    )


# ---------------------------------------------------------------------------
# Sampling smoke test
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_speculative_sampling_smoke(our_model, input_ids, tokenizer) -> None:
    """
    Speculative decoding with sampling must run without errors and produce a
    non-trivial output sequence.
    """
    output, stats = speculative_generate(
        target_model=our_model,
        draft_model=our_model,
        input_ids=input_ids.clone(),
        max_new_tokens=MAX_NEW_TOKENS,
        num_speculative_tokens=K,
        eos_token_id=tokenizer.eos_token_id,
        temperature=1.0,
        top_k=50,
    )

    generated_len = output.shape[1] - input_ids.shape[1]
    assert generated_len > 0, "Expected at least one generated token"
    assert output.shape[0] == 1, "Batch size should be 1"


# ---------------------------------------------------------------------------
# SpeculativeStats sanity
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_speculative_stats_populated(our_model, input_ids, tokenizer) -> None:
    """SpeculativeStats fields must be internally consistent."""
    _, stats = speculative_generate(
        target_model=our_model,
        draft_model=our_model,
        input_ids=input_ids.clone(),
        max_new_tokens=MAX_NEW_TOKENS,
        num_speculative_tokens=K,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.0,
    )

    assert isinstance(stats, SpeculativeStats)
    assert stats.num_rounds >= 0
    assert stats.total_draft_tokens >= stats.total_accepted_tokens
    assert 0.0 <= stats.acceptance_rate <= 1.0
    assert stats.tokens_per_round >= 0.0
    if stats.num_rounds > 0:
        assert stats.total_draft_tokens > 0
