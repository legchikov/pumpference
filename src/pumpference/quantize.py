"""
Weight-only quantization for Qwen3Model.

Two base RTN (Round-To-Nearest) schemes, both W×A16 (quantized weights,
bfloat16 activations):

  Int8Linear   — W8A16, symmetric per-output-channel quantization.
                 Weight stored as int8.  Forward: dequantize to bf16 → F.linear.
                 ~2× memory reduction, near-lossless for most layers.

  Int4Linear   — W4A16, symmetric group-wise quantization (group_size=128).
                 Two int4 values packed into each uint8 byte.
                 Forward: unpack → dequantize per group → F.linear.
                 ~4× memory reduction; small quality loss on some layers.

AWQ (Activation-Aware Weight Quantization):
  Calibration-based method that improves quantization quality by protecting
  weight channels that are multiplied by large-magnitude activations.

  Algorithm per (norm, linears) group:
    1. Run calibration data through the model; record per-channel mean |activation|.
    2. Grid-search α ∈ [0, 1] to find s = act_stats^α that minimises the
       activation-weighted quantization reconstruction error.
    3. Multiply each downstream linear's weight columns by s.
    4. Divide the preceding RMSNorm's scale by s (scale absorption — keeps the
       computation mathematically equivalent while the quantization grid is finer
       for salient channels).

  Groups processed per TransformerBlock:
    norm1 → att.W_query, att.W_key, att.W_value   (scale absorbed into norm1)
    norm2 → ff.fc1, ff.fc2                          (scale absorbed into norm2)
    att.out_proj, ff.fc3 — no preceding norm; standard RTN is applied.

Entry points:

  quantize_model(model, mode="int8"|"int4"|"awq_int8"|"awq_int4", ...)
    Replaces every nn.Linear with Int8Linear or Int4Linear.  For AWQ modes,
    calibration_ids (tokenized calibration sequences) must be provided;
    calibrate_awq() is called automatically before replacement.

  calibrate_awq(model, calibration_ids, mode="int8"|"int4", ...)
    Standalone calibration: modifies norm scales and weight data in-place.
    Model stays bfloat16 after this call.  Follow with quantize_model().

Weight tying note:
  Qwen3-0.6B ties tok_emb.weight and out_head.weight.  Quantizing out_head
  makes a separate int8/int4 copy of those weights.  The embedding lookup
  stays bfloat16; only the output projection uses the quantized copy.
"""

from __future__ import annotations

from typing import Callable, Literal

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
    mode: Literal["int8", "int4", "awq_int8", "awq_int4"] = "int8",
    group_size: int = 128,
    calibration_ids: list[torch.Tensor] | None = None,
) -> nn.Module:
    """
    Replace every nn.Linear in *model* with Int8Linear or Int4Linear.

    The replacement is done in-place and the model is returned for chaining.
    nn.Embedding layers are untouched (they are not Linear modules).

    For AWQ modes ("awq_int8", "awq_int4"), calibration_ids is required.
    AWQ calibration runs first (modifying norm scales and weight data in-place),
    then the standard RTN quantization replaces all nn.Linear modules.

    Weight-tying:
        Qwen3-0.6B ties tok_emb.weight to out_head.weight.  After this
        function runs, out_head is an Int8/Int4Linear with its own weight
        copy.  The embedding still holds the original bfloat16 weights.

    Args:
        model:           Any nn.Module (typically Qwen3Model).
        mode:            Quantization scheme. "int8"/"int4" for plain RTN;
                         "awq_int8"/"awq_int4" for calibration-based AWQ + RTN.
        group_size:      Group size for int4 quantization (ignored for int8).
        calibration_ids: Required for AWQ modes. List of input_ids tensors
                         (one per calibration sequence) used to collect
                         activation statistics.

    Returns:
        The same model object, modified in-place.
    """
    if mode in ("awq_int8", "awq_int4"):
        if calibration_ids is None:
            raise ValueError(
                f"calibration_ids is required for AWQ mode '{mode}'. "
                "Tokenize representative sequences and pass them as a list of tensors."
            )
        base_mode: Literal["int8", "int4"] = mode[4:]  # type: ignore[assignment]
        calibrate_awq(model, calibration_ids, mode=base_mode, group_size=group_size)
        _replace_linears(model, mode=base_mode, group_size=group_size)
    else:
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


# ---------------------------------------------------------------------------
# AWQ: Activation-Aware Weight Quantization
# ---------------------------------------------------------------------------

def _rtn_dequant_int8(weight: torch.Tensor) -> torch.Tensor:
    """Quantize and dequantize using int8 RTN (used during AWQ grid search)."""
    w_int8, scale = quantize_per_channel_absmax(weight)
    return w_int8.float() * scale.unsqueeze(1)


def _rtn_dequant_int4(weight: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Quantize and dequantize using int4 group-wise RTN (used during AWQ grid search)."""
    out_features, in_features = weight.shape
    packed, scales = quantize_per_group(weight, group_size)
    return unpack_int4(packed, out_features, in_features, group_size, scales)


def _collect_norm_activation_stats(
    model: nn.Module,
    calibration_ids: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Run calibration sequences through the model and collect per-channel mean
    absolute activation magnitudes at each RMSNorm output in transformer blocks.

    Keys: "{block_idx}.norm1", "{block_idx}.norm2".
    Values: float32 tensors of shape [emb_dim] — mean |activation| per channel,
    averaged over all calibration samples and all token positions.

    norm1 output feeds W_query, W_key, W_value.
    norm2 output feeds ff.fc1, ff.fc2.
    """
    stats: dict[str, list[torch.Tensor]] = {}
    hooks = []

    for block_idx, block in enumerate(model.trf_blocks):  # type: ignore[attr-defined]
        for norm_name in ("norm1", "norm2"):
            key = f"{block_idx}.{norm_name}"
            norm_module = getattr(block, norm_name)

            def make_hook(k: str) -> Callable:
                def hook(module: nn.Module, inp: tuple, output: torch.Tensor) -> None:
                    # output: [batch, seq_len, emb_dim] → average to [emb_dim]
                    stats.setdefault(k, []).append(
                        output.detach().float().abs().mean(dim=(0, 1))
                    )
                return hook

            hooks.append(norm_module.register_forward_hook(make_hook(key)))

    model.eval()
    with torch.no_grad():
        for ids in calibration_ids:
            model(ids)

    for h in hooks:
        h.remove()

    return {
        key: torch.stack(tensors).mean(dim=0)
        for key, tensors in stats.items()
    }


def _search_optimal_scale(
    weights: list[torch.Tensor],
    act_stats: torch.Tensor,
    dequant_fn: Callable[[torch.Tensor], torch.Tensor],
    n_grid: int = 20,
) -> torch.Tensor:
    """
    Grid-search for the per-channel scale s = act_stats^alpha (alpha in [0, 1])
    that minimises the activation-weighted quantization reconstruction error.

    The error for a group of weight matrices at a given alpha:

      s = act_stats^alpha                       [in_features]
      For each W in weights:
        W_scaled  = W * s                       (column-wise)
        W_q       = dequant(W_scaled)           (quantize → dequantize round-trip)
        col_err   = mean_row((W_q - W_scaled)²) [in_features]
        loss     += mean( col_err * (act_stats/s)² )

    The (act_stats/s)² term weights each column by the importance of that
    channel in the output (channels with large activations contribute more).

    Returns the best scale tensor of shape [in_features].
    """
    best_error = float("inf")
    best_s = torch.ones(act_stats.shape, dtype=torch.float32, device=act_stats.device)

    for i in range(n_grid + 1):
        alpha = i / n_grid
        s = act_stats.pow(alpha).clamp(min=1e-6)
        channel_weight = (act_stats / s).pow(2)  # [in_features]

        total_error = 0.0
        for W in weights:
            W_f = W.float()
            W_scaled = W_f * s                                    # [out, in]
            W_q = dequant_fn(W_scaled)                            # [out, in] float32
            col_err = (W_q - W_scaled).pow(2).mean(dim=0)        # [in_features]
            total_error += (col_err * channel_weight).mean().item()

        if total_error < best_error:
            best_error = total_error
            best_s = s.clone()

    return best_s


def calibrate_awq(
    model: nn.Module,
    calibration_ids: list[torch.Tensor],
    mode: Literal["int8", "int4"] = "int4",
    group_size: int = 128,
    n_grid: int = 20,
) -> nn.Module:
    """
    Apply AWQ calibration to a Qwen3Model in-place.

    For each (RMSNorm, downstream linears) group in every transformer block:
      1. Collect per-channel mean |activation| from calibration data.
      2. Grid-search alpha in [0, 1] for optimal scale s = act_stats^alpha.
      3. Multiply weight columns by s  (salient channels → finer quant grid).
      4. Divide norm.scale by s         (scale absorption — output unchanged).

    The model remains bfloat16 after this call.  Call quantize_model()
    afterward to apply the actual int8 or int4 RTN quantization.

    Groups per block:
      norm1 → att.W_query, att.W_key, att.W_value   (absorbed into norm1)
      norm2 → ff.fc1, ff.fc2                          (absorbed into norm2)

    att.out_proj and ff.fc3 receive no AWQ treatment (no adjacent preceding norm).
    Standard RTN quantization will be applied to them by quantize_model().

    This function is Qwen3-specific: it directly accesses model.trf_blocks and
    the named sub-modules within each TransformerBlock.

    Args:
        model:           Qwen3Model in eval mode with bfloat16 weights.
        calibration_ids: List of input_ids tensors (one per calibration sample).
        mode:            Target quantization format used during grid search.
        group_size:      Group size for int4 grid search (ignored for int8).
        n_grid:          Number of alpha values to evaluate (higher = better
                         quality, slower calibration; default 20).

    Returns:
        The same model object, modified in-place.
    """
    if mode == "int8":
        dequant_fn: Callable[[torch.Tensor], torch.Tensor] = _rtn_dequant_int8
    else:
        dequant_fn = lambda w: _rtn_dequant_int4(w, group_size)  # noqa: E731

    act_stats = _collect_norm_activation_stats(model, calibration_ids)

    for block_idx, block in enumerate(model.trf_blocks):  # type: ignore[attr-defined]
        # --- Group 1: norm1 → W_query, W_key, W_value -----------------------
        stats1 = act_stats[f"{block_idx}.norm1"]
        att_weights = [
            block.att.W_query.weight.data.float(),
            block.att.W_key.weight.data.float(),
            block.att.W_value.weight.data.float(),
        ]
        s1 = _search_optimal_scale(att_weights, stats1, dequant_fn, n_grid)
        s1_bf = s1.to(block.att.W_query.weight.dtype)
        block.att.W_query.weight.data.mul_(s1_bf)
        block.att.W_key.weight.data.mul_(s1_bf)
        block.att.W_value.weight.data.mul_(s1_bf)
        # Absorb inverse scale into norm1 so the output is unchanged.
        block.norm1.scale.data.div_(s1.to(block.norm1.scale.dtype))

        # --- Group 2: norm2 → fc1, fc2 ---------------------------------------
        stats2 = act_stats[f"{block_idx}.norm2"]
        ff_weights = [
            block.ff.fc1.weight.data.float(),
            block.ff.fc2.weight.data.float(),
        ]
        s2 = _search_optimal_scale(ff_weights, stats2, dequant_fn, n_grid)
        s2_bf = s2.to(block.ff.fc1.weight.dtype)
        block.ff.fc1.weight.data.mul_(s2_bf)
        block.ff.fc2.weight.data.mul_(s2_bf)
        block.norm2.scale.data.div_(s2.to(block.norm2.scale.dtype))

    return model
