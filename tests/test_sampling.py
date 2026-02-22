"""
End-to-end tests for sampling decoding strategies.
"""

import torch

from pumpference.generate import sample_next_token


def test_temperature_zero_matches_argmax() -> None:
    """temperature=0 must return the exact same token as argmax across all parameter combos."""
    torch.manual_seed(0)
    logits = torch.randn(1, 151936)  # full Qwen3 vocab size
    expected = logits.argmax(dim=-1, keepdim=True)

    for top_k, top_p in [(0, 1.0), (50, 1.0), (0, 0.9), (50, 0.9)]:
        result = sample_next_token(logits, temperature=0.0, top_k=top_k, top_p=top_p)
        assert torch.equal(result, expected), (
            f"Greedy mismatch with top_k={top_k}, top_p={top_p}: "
            f"got {result.item()}, expected {expected.item()}"
        )


def test_sampling_pipeline_respects_filters() -> None:
    """
    With top_k=3 and top_p=0.9, only the top-3 tokens may ever be sampled.

    We construct logits so token 0 (prob=0.60) and tokens 1-2 (prob=0.20 each)
    cover the vocabulary. Tokens 3-9 are forbidden by top_k=3. We verify this
    holds over 500 draws with different seeds, and confirm the output is always
    shape [1, 1].
    """
    probs = [0.60, 0.20, 0.20] + [0.0] * 7   # 10-token vocab for clarity
    logits = torch.tensor(probs, dtype=torch.float32).log().unsqueeze(0)
    allowed = {0, 1, 2}

    sampled = set()
    for seed in range(500):
        torch.manual_seed(seed)
        result = sample_next_token(logits, temperature=1.0, top_k=3, top_p=0.9)
        assert result.shape == (1, 1), f"Wrong shape: {result.shape}"
        token = result.item()
        assert token in allowed, (
            f"Sampled token {token} not in allowed set {allowed} (seed={seed})"
        )
        sampled.add(token)

    # All three candidates must appear in 500 draws (avoids degenerate collapse).
    assert sampled == allowed, f"Not all candidates were sampled: missing {allowed - sampled}"
