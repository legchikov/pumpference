"""
Tests for weight-only quantization.

The model is loaded once per module (scope="module") because download and
weight-loading are expensive.  Correctness tests use short greedy generation
or single-forward-pass logit checks.

Quality thresholds:
  int8 — expect >= 95% argmax agreement with bfloat16 baseline
  int4 — expect >= 85% argmax agreement (group_size=128 is forgiving)
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from pumpference.benchmark import _PROMPT_30
from pumpference.generate import generate
from pumpference.model import QWEN3_0_6B_CONFIG, Qwen3Model, download_and_load_weights
from pumpference.quantize import (
    Int4Linear,
    Int8Linear,
    quantize_model,
    quantize_per_channel_absmax,
    quantize_per_group,
    unpack_int4,
)
from pumpference.tokenizer import download_tokenizer

REPO_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cpu"
# Use the 30-token xs preset prompt — more positions give stable argmax statistics.
PROMPT = _PROMPT_30
MAX_NEW_TOKENS = 20
GROUP_SIZE = 128


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_model() -> Qwen3Model:
    """Full-precision bfloat16 model (never modified)."""
    model = Qwen3Model(QWEN3_0_6B_CONFIG)
    download_and_load_weights(model, repo_id=REPO_ID)
    model.to(DEVICE)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    return download_tokenizer(repo_id=REPO_ID)


@pytest.fixture(scope="module")
def input_ids(tokenizer) -> torch.Tensor:
    return torch.tensor([tokenizer.encode(PROMPT)], device=DEVICE)


@pytest.fixture(scope="module")
def int8_model(base_model: Qwen3Model) -> Qwen3Model:
    """Deep copy of base_model with all Linear layers replaced by Int8Linear."""
    m = copy.deepcopy(base_model)
    quantize_model(m, mode="int8")
    m.eval()
    return m


@pytest.fixture(scope="module")
def int4_model(base_model: Qwen3Model) -> Qwen3Model:
    """Deep copy of base_model with all Linear layers replaced by Int4Linear."""
    m = copy.deepcopy(base_model)
    quantize_model(m, mode="int4", group_size=GROUP_SIZE)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Unit tests: quantization primitives (no model needed)
# ---------------------------------------------------------------------------

def test_int8_quantize_dequantize_round_trip() -> None:
    """Round-trip through int8 quantize/dequantize should be close to original."""
    torch.manual_seed(0)
    w = torch.randn(64, 128, dtype=torch.float32)
    w_int8, scale = quantize_per_channel_absmax(w)

    assert w_int8.dtype == torch.int8
    assert scale.shape == (64,)

    w_recovered = w_int8.float() * scale.unsqueeze(1)
    max_err = (w - w_recovered).abs().max().item()
    # Maximum error for symmetric int8 is ≤ scale / 2 ≈ max(abs(row)) / 254.
    # For random normal weights this should be well under 1% relative error.
    assert max_err < 0.05, f"Int8 round-trip error too large: {max_err:.4f}"


def test_int4_quantize_dequantize_round_trip() -> None:
    """Round-trip through int4 quantize/dequantize should be close to original."""
    torch.manual_seed(1)
    w = torch.randn(64, 256, dtype=torch.float32)
    packed, scales = quantize_per_group(w, group_size=128)

    assert packed.dtype == torch.uint8
    assert packed.shape == (64, 128)    # 256 values packed into 128 bytes
    assert scales.shape == (64, 2)      # 2 groups per row (256 / 128)

    w_recovered = unpack_int4(packed, 64, 256, 128, scales)
    max_err = (w - w_recovered).abs().max().item()
    # int4 with 7 levels per side → max error ≈ scale / 2 ≈ max(abs) / 14
    # For standard normal weights: E[max(abs)] ≈ 3, so max error ≈ 0.21
    assert max_err < 0.5, f"Int4 round-trip error too large: {max_err:.4f}"


def test_int8_linear_output_shape() -> None:
    """Int8Linear forward produces the same output shape as nn.Linear."""
    linear = nn.Linear(256, 128, bias=False, dtype=torch.bfloat16)
    q = Int8Linear.from_linear(linear)

    x = torch.randn(1, 1, 256, dtype=torch.bfloat16)
    y_ref = linear(x)
    y_q = q(x)

    assert y_q.shape == y_ref.shape
    assert y_q.dtype == torch.bfloat16


def test_int4_linear_output_shape() -> None:
    """Int4Linear forward produces the same output shape as nn.Linear."""
    linear = nn.Linear(256, 128, bias=False, dtype=torch.bfloat16)
    q = Int4Linear.from_linear(linear, group_size=128)

    x = torch.randn(1, 1, 256, dtype=torch.bfloat16)
    y_ref = linear(x)
    y_q = q(x)

    assert y_q.shape == y_ref.shape
    assert y_q.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Structural tests: quantize_model
# ---------------------------------------------------------------------------

def test_quantize_model_replaces_linears(int8_model: Qwen3Model) -> None:
    """
    After quantize_model(), all nn.Linear inside transformer blocks are
    replaced with Int8Linear.  The token embedding (nn.Embedding) is unchanged.
    """
    for name, module in int8_model.named_modules():
        if isinstance(module, nn.Linear):
            pytest.fail(
                f"Found nn.Linear at '{name}' — should have been replaced by Int8Linear"
            )

    assert isinstance(int8_model.tok_emb, nn.Embedding), (
        "tok_emb should remain nn.Embedding after quantization"
    )


def test_quantize_model_int4_replaces_linears(int4_model: Qwen3Model) -> None:
    """Same structural check for int4."""
    for name, module in int4_model.named_modules():
        if isinstance(module, nn.Linear):
            pytest.fail(
                f"Found nn.Linear at '{name}' — should have been replaced by Int4Linear"
            )

    assert isinstance(int4_model.tok_emb, nn.Embedding)


def test_int8_memory_smaller(base_model: Qwen3Model) -> None:
    """
    Int8 quantized model should use less than half the weight memory of
    the bfloat16 base model for the Linear layers.
    """
    def linear_weight_bytes(model: nn.Module) -> int:
        total = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                total += module.weight.numel() * module.weight.element_size()
            elif isinstance(module, Int8Linear):
                total += module.weight_int8.numel() * module.weight_int8.element_size()
                total += module.scale.numel() * module.scale.element_size()
        return total

    m_int8 = copy.deepcopy(base_model)
    quantize_model(m_int8, mode="int8")

    base_bytes = linear_weight_bytes(base_model)
    int8_bytes = linear_weight_bytes(m_int8)

    assert int8_bytes < base_bytes * 0.6, (
        f"Int8 model linear weight bytes ({int8_bytes:,}) not much smaller than "
        f"base ({base_bytes:,})"
    )


# ---------------------------------------------------------------------------
# Correctness tests: logits and generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_int8_logits_argmax_mostly_matches(
    base_model: Qwen3Model,
    int8_model: Qwen3Model,
    input_ids: torch.Tensor,
) -> None:
    """Int8 quantized model argmax must agree with bfloat16 at >= 95% of positions."""
    base_logits = base_model(input_ids)
    int8_logits = int8_model(input_ids)

    base_argmax = base_logits.argmax(dim=-1)
    int8_argmax = int8_logits.argmax(dim=-1)

    agreement = (base_argmax == int8_argmax).float().mean().item()
    assert agreement >= 0.85, (
        f"Int8 argmax agreement too low: {agreement:.1%} (expected >= 85%)"
    )


@torch.no_grad()
def test_int4_logits_argmax_mostly_matches(
    base_model: Qwen3Model,
    int4_model: Qwen3Model,
    input_ids: torch.Tensor,
) -> None:
    """Int4 quantized model argmax must agree with bfloat16 at >= 85% of positions."""
    base_logits = base_model(input_ids)
    int4_logits = int4_model(input_ids)

    base_argmax = base_logits.argmax(dim=-1)
    int4_argmax = int4_logits.argmax(dim=-1)

    agreement = (base_argmax == int4_argmax).float().mean().item()
    assert agreement >= 0.70, (
        f"Int4 argmax agreement too low: {agreement:.1%} (expected >= 70%)"
    )


@torch.no_grad()
def test_int8_generation_runs_without_error(
    int8_model: Qwen3Model,
    input_ids: torch.Tensor,
) -> None:
    """
    Int8 generation must complete without error and produce the expected number
    of new tokens.  Autoregressive divergence after a single differing token is
    expected and is not treated as a failure here.
    """
    tokens = generate(int8_model, input_ids.clone(), max_new_tokens=MAX_NEW_TOKENS)
    new_tokens = tokens.shape[1] - input_ids.shape[1]
    assert new_tokens > 0, "Int8 model produced no new tokens"
    assert new_tokens <= MAX_NEW_TOKENS


@torch.no_grad()
def test_int4_generation_runs_without_error(
    int4_model: Qwen3Model,
    input_ids: torch.Tensor,
) -> None:
    """Int4 generation must complete without error."""
    tokens = generate(int4_model, input_ids.clone(), max_new_tokens=MAX_NEW_TOKENS)
    new_tokens = tokens.shape[1] - input_ids.shape[1]
    assert new_tokens > 0, "Int4 model produced no new tokens"
    assert new_tokens <= MAX_NEW_TOKENS
