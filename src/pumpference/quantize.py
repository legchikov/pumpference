"""
Weight-only quantization for Qwen3Model.

Two schemes, both W×A16 (quantized weights, bfloat16 activations):

  Int8Linear   — W8A16, symmetric per-output-channel quantization.
                 Weight stored as int8.  Forward: dequantize to bf16 → F.linear.
                 ~2× memory reduction, near-lossless for most layers.

  Int4Linear   — W4A16, symmetric group-wise quantization (group_size=128).
                 Two int4 values packed into each uint8 byte.
                 Forward: unpack → dequantize per group → F.linear.
                 ~4× memory reduction; small quality loss on some layers.

Entry point:

  quantize_model(model, mode="int8" | "int4", group_size=128)

Replaces every nn.Linear inside the model (attention projections, FFN, output
head) with the corresponding quantized module.  The token embedding
(nn.Embedding) is not a Linear and is left in bfloat16.

Weight tying note:
  Qwen3-0.6B ties tok_emb.weight and out_head.weight.  Quantizing out_head
  makes a separate int8/int4 copy of those weights.  The embedding lookup
  stays bfloat16; only the output projection uses the quantized copy.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# int8 helpers
# ---------------------------------------------------------------------------

def quantize_per_channel_absmax(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-output-channel int8 quantization.

    Each output channel (row of weight) gets its own scale:
        scale[i] = max(abs(weight[i])) / 127

    Returns:
        weight_int8  shape [out, in],  dtype torch.int8
        scale        shape [out],      dtype torch.float32
    """
    # Compute per-row scale from absolute maximum.
    scale = weight.float().abs().amax(dim=1) / 127.0
    scale = scale.clamp(min=1e-8)              # guard against all-zero rows

    # Quantize: divide by scale (broadcast over columns), round, clamp to [-128, 127].
    weight_int8 = (weight.float() / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return weight_int8, scale


# ---------------------------------------------------------------------------
# int4 helpers
# ---------------------------------------------------------------------------

def quantize_per_group(
    weight: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric group-wise int4 quantization.

    The weight is divided into groups of *group_size* elements along the
    input dimension (dim=1).  Each group gets its own scale.  Values are
    clamped to [-7, 7] (using 7 not 8 avoids asymmetry at the cost of one
    code point, which keeps dequantization exact).

    Two int4 values are packed into one uint8 byte (low nibble = even index,
    high nibble = odd index).

    Returns:
        weight_packed  shape [out, in // 2],       dtype torch.uint8
        scales         shape [out, n_groups],       dtype torch.float32

    n_groups = in_features // group_size.
    """
    out_features, in_features = weight.shape
    assert in_features % group_size == 0, (
        f"in_features={in_features} must be divisible by group_size={group_size}"
    )
    n_groups = in_features // group_size

    w = weight.float().view(out_features, n_groups, group_size)

    # Per-group scale: max(abs) / 7 (clamped to avoid div-by-zero)
    scales = w.abs().amax(dim=2) / 7.0          # [out, n_groups]
    scales = scales.clamp(min=1e-8)

    # Quantize each group.
    w_q = (w / scales.unsqueeze(2)).round().clamp(-7, 7)  # [out, n_groups, group_size]
    w_q = w_q.view(out_features, in_features).to(torch.int8)   # int8 for arithmetic

    # Pack: two int4s per byte.  values are in [-7, 7], shift to [0, 15] for unsigned packing.
    w_shifted = (w_q + 8).to(torch.uint8)       # [out, in_features], values in [1, 15]
    even = w_shifted[:, 0::2]                    # [out, in // 2]
    odd  = w_shifted[:, 1::2]                    # [out, in // 2]
    packed = (odd << 4) | even                   # pack: odd in high nibble, even in low nibble

    return packed, scales.to(torch.float32)


def unpack_int4(
    packed: torch.Tensor,
    out_features: int,
    in_features: int,
    group_size: int,
    scales: torch.Tensor,
) -> torch.Tensor:
    """
    Unpack and dequantize a weight packed by *quantize_per_group*.

    Returns a float32 weight of shape [out_features, in_features].
    """
    # Unpack nibbles.
    even = (packed & 0x0F).to(torch.int8) - 8      # low  nibble → int4 in [-7, 7]
    odd  = ((packed >> 4) & 0x0F).to(torch.int8) - 8  # high nibble

    # Interleave back to original order.
    w_q = torch.empty(out_features, in_features, dtype=torch.int8, device=packed.device)
    w_q[:, 0::2] = even
    w_q[:, 1::2] = odd

    # Dequantize per group.
    n_groups = in_features // group_size
    w = w_q.float().view(out_features, n_groups, group_size)
    w = w * scales.unsqueeze(2)                 # broadcast scale over group elements
    return w.view(out_features, in_features)


# ---------------------------------------------------------------------------
# Quantized linear modules
# ---------------------------------------------------------------------------

class Int8Linear(nn.Module):
    """
    Drop-in replacement for nn.Linear (no bias) with int8 weight storage.

    Forward pass:
        1. Dequantize: w_bf16 = weight_int8 * scale   (element-wise per row)
        2. F.linear(x, w_bf16)                        (bfloat16 matmul)
    """

    def __init__(self, weight_int8: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        # Store int8 weight as a non-parameter buffer (not updated by optimizers).
        self.register_buffer("weight_int8", weight_int8)
        # Scale per output channel.  float32 for precision.
        self.register_buffer("scale", scale)
        self.out_features, self.in_features = weight_int8.shape

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Int8Linear":
        """Quantize an existing nn.Linear and return an Int8Linear."""
        weight_int8, scale = quantize_per_channel_absmax(linear.weight.data)
        return cls(weight_int8, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weight to bfloat16 for the matmul.
        # scale: [out] → unsqueeze to [out, 1] for broadcast over in_features.
        w = self.weight_int8.to(x.dtype) * self.scale.to(x.dtype).unsqueeze(1)
        return F.linear(x, w)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, scheme=int8"


class Int4Linear(nn.Module):
    """
    Drop-in replacement for nn.Linear (no bias) with int4 weight storage.

    Weights are packed two-per-byte (4-bit symmetric, group-wise scaling).

    Forward pass:
        1. Unpack nibbles and dequantize per group.
        2. F.linear(x, w_bf16)
    """

    def __init__(
        self,
        weight_packed: torch.Tensor,
        scales: torch.Tensor,
        out_features: int,
        in_features: int,
        group_size: int,
    ) -> None:
        super().__init__()
        self.register_buffer("weight_packed", weight_packed)
        self.register_buffer("scales", scales)
        self.out_features = out_features
        self.in_features = in_features
        self.group_size = group_size

    @classmethod
    def from_linear(cls, linear: nn.Linear, group_size: int = 128) -> "Int4Linear":
        """Quantize an existing nn.Linear and return an Int4Linear."""
        packed, scales = quantize_per_group(linear.weight.data, group_size)
        out_features, in_features = linear.weight.shape
        return cls(packed, scales, out_features, in_features, group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = unpack_int4(
            self.weight_packed,
            self.out_features,
            self.in_features,
            self.group_size,
            self.scales,
        ).to(x.dtype)
        return F.linear(x, w)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"scheme=int4, group_size={self.group_size}"
        )


# ---------------------------------------------------------------------------
# Model-level quantization
# ---------------------------------------------------------------------------

def quantize_model(
    model: nn.Module,
    mode: Literal["int8", "int4"] = "int8",
    group_size: int = 128,
) -> nn.Module:
    """
    Replace every nn.Linear in *model* with Int8Linear or Int4Linear.

    The replacement is done in-place and the model is returned for chaining.
    nn.Embedding layers are untouched (they are not Linear modules).

    Weight-tying:
        Qwen3-0.6B ties tok_emb.weight to out_head.weight.  After this
        function runs, out_head is an Int8/Int4Linear with its own weight
        copy.  The embedding still holds the original bfloat16 weights.

    Args:
        model:      Any nn.Module (typically Qwen3Model).
        mode:       "int8" for per-channel int8, "int4" for group-wise int4.
        group_size: Group size for int4 quantization (ignored for int8).

    Returns:
        The same model object, modified in-place.
    """
    _replace_linears(model, mode=mode, group_size=group_size)
    return model


def _replace_linears(
    module: nn.Module,
    mode: str,
    group_size: int,
    parent: nn.Module | None = None,
    attr_name: str = "",
) -> None:
    """Recursively walk the module tree and replace nn.Linear with quantized variants."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if mode == "int8":
                replacement = Int8Linear.from_linear(child)
            else:
                replacement = Int4Linear.from_linear(child, group_size=group_size)
            setattr(module, name, replacement)
        else:
            _replace_linears(child, mode=mode, group_size=group_size, parent=module, attr_name=name)
