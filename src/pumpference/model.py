"""
Qwen3 0.6B model architecture 

Based on: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/standalone-qwen3.ipynb

Architecture components:
  RMSNorm, RoPE, FeedForward (SwiGLU), GroupedQueryAttention, KVCache
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Config for Qwen3-0.6B
# ---------------------------------------------------------------------------

@dataclass
class Qwen3Config:
    """Typed, self-documenting model configuration."""

    repo_id: str = "Qwen/Qwen3-0.6B"
    vocab_size: int = 151_936
    context_length: int = 40_960
    emb_dim: int = 1024
    n_heads: int = 16
    n_layers: int = 28
    hidden_dim: int = 3072
    head_dim: int = 128
    qk_norm: bool = True
    n_kv_groups: int = 8
    rope_base: float = 1_000_000.0
    dtype: torch.dtype = field(default=torch.bfloat16)
    use_flash_attn: bool = False


QWEN3_0_6B_CONFIG = Qwen3Config()


# ---------------------------------------------------------------------------
# KV-Cache
# ---------------------------------------------------------------------------

class KVCache:
    """
    Per-layer key/value cache for autoregressive generation.

    During the prefill pass (full prompt), the model stores K and V for every
    token in this cache.  On subsequent decode steps each layer appends the
    single new token's K/V and returns the full accumulated tensor so the
    query can attend to the entire past context.

    Tensors are stored at KV-head granularity (num_kv_groups heads) BEFORE
    the repeat_interleave expansion.  This is intentional:
      - K and V are cached *after* QK-norm and *after* RoPE so positional
        encoding is already baked in and never needs to be re-applied.
      - Storing pre-expansion saves memory (8 heads instead of 16 for GQA).

    Shape of each cached tensor: [batch, num_kv_groups, seq_len, head_dim]
    """

    def __init__(self) -> None:
        # One (K, V) tuple per transformer layer, populated lazily.
        self._cache: list[tuple[torch.Tensor, torch.Tensor]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append *keys* and *values* for *layer_idx* and return the full
        accumulated K/V tensors for that layer.

        On the first call for a given layer (prefill) the tensors are stored
        as-is.  On subsequent calls (decode) the new single-token slice is
        concatenated along dim=2 (the sequence dimension).
        """
        if layer_idx < len(self._cache):
            prev_k, prev_v = self._cache[layer_idx]
            keys = torch.cat([prev_k, keys], dim=2)
            values = torch.cat([prev_v, values], dim=2)
            self._cache[layer_idx] = (keys, values)
        else:
            self._cache.append((keys, values))
        return keys, values

    def reset(self) -> None:
        """Clear all cached tensors (call before a new generation request)."""
        self._cache.clear()

    @property
    def seq_len(self) -> int:
        """Number of tokens currently stored in the cache (0 if empty)."""
        return self._cache[0][0].shape[2] if self._cache else 0


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square layer normalization (no bias, no re-centering)."""

    def __init__(self, emb_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x = x.to(torch.float32)                          # upcast for stability
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.scale).to(original_dtype)


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# ---------------------------------------------------------------------------

def compute_rope_params(
    head_dim: int,
    context_length: int,
    theta_base: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute cos/sin tables for RoPE."""
    assert head_dim % 2 == 0, "head_dim must be even"

    inv_freq = 1.0 / (
        theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    )                                                        # [head_dim // 2]
    positions = torch.arange(context_length).float()         # [context_length]
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # [ctx, head_dim // 2]
    angles = torch.cat([angles, angles], dim=1)              # [ctx, head_dim]

    return torch.cos(angles), torch.sin(angles)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE to a tensor of shape [batch, heads, seq_len, head_dim].

    *cos* and *sin* must already be sliced to the correct position range by
    the caller (shape [seq_len, head_dim]).  This allows the same function to
    handle both the full-sequence prefill pass and the single-token decode
    pass without any internal position arithmetic.
    """
    _, _, _, head_dim = x.shape

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    # Broadcast over batch and head dimensions.
    cos = cos.unsqueeze(0).unsqueeze(0)   # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)

    rotated = torch.cat((-x2, x1), dim=-1)
    return ((x * cos) + (rotated * sin)).to(x.dtype)


# ---------------------------------------------------------------------------
# Feed-Forward Network (SwiGLU)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """SwiGLU feed-forward:  silu(fc1(x)) * fc2(x)  →  fc3."""

    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        dtype = cfg.dtype
        self.fc1 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, bias=False, dtype=dtype)
        self.fc2 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, bias=False, dtype=dtype)
        self.fc3 = nn.Linear(cfg.hidden_dim, cfg.emb_dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(nn.functional.silu(self.fc1(x)) * self.fc2(x))


# ---------------------------------------------------------------------------
# Flash Attention (pure PyTorch, tiled)
# ---------------------------------------------------------------------------

def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    block_size: int = 64,
) -> torch.Tensor:
    """
    Flash Attention: tiled attention with online softmax.

    Computes the same result as standard scaled dot-product attention but
    uses O(n) memory instead of O(n²) by processing Q and KV in blocks and
    accumulating the output without ever materialising the full N×N score
    matrix.

    The algorithm maintains three per-row running accumulators across the
    inner KV loop:
        m — running row-max of scores seen so far (for numerical stability)
        l — running sum of exp(score − m) values (the softmax denominator)
        O — running weighted sum of V vectors (the softmax numerator)

    When a new KV tile arrives with a higher row-max, a correction factor
    exp(m_old − m_new) rescales both l and O to the new baseline before
    adding the new tile's contribution.

    Args:
        q:          [batch, heads, q_len, head_dim]
        k:          [batch, heads, kv_len, head_dim]
        v:          [batch, heads, kv_len, head_dim]
        is_causal:  apply causal mask (token i cannot attend to j > i)
        block_size: tokens per tile along both Q and KV dimensions

    Returns:
        [batch, heads, q_len, head_dim], same dtype as q
    """
    batch, heads, q_len, head_dim = q.shape
    kv_len = k.shape[2]
    scale = head_dim ** -0.5
    orig_dtype = q.dtype

    # Accumulate in float32 — exp() calls need the extra range; mirrors the
    # existing eager path which also uses dtype=torch.float32 for softmax.
    q, k, v = q.float(), k.float(), v.float()

    output = torch.zeros(batch, heads, q_len, head_dim, dtype=torch.float32, device=q.device)

    for q_start in range(0, q_len, block_size):
        q_end   = min(q_start + block_size, q_len)
        q_block = q[:, :, q_start:q_end, :]           # [B, H, Bq, head_dim]
        Bq      = q_end - q_start

        # Per-row running accumulators — reset fresh for each Q block.
        # running_max starts at -inf so the first tile's max is accepted unconditionally.
        acc_out = torch.zeros(batch, heads, Bq, head_dim, dtype=torch.float32, device=q.device)
        acc_lse = torch.zeros(batch, heads, Bq, 1,        dtype=torch.float32, device=q.device)
        running_max = torch.full((batch, heads, Bq, 1), float('-inf'), dtype=torch.float32, device=q.device)

        for k_start in range(0, kv_len, block_size):
            k_end = min(k_start + block_size, kv_len)

            # Causal skip: if every key in this KV block is strictly in the
            # future relative to every query in this Q block, all scores will
            # be -inf and the tile contributes nothing — skip it entirely.
            # KV blocks are in ascending order so we can break, not continue.
            if is_causal and k_start > q_end - 1:
                break

            k_block = k[:, :, k_start:k_end, :]       # [B, H, Bk, head_dim]
            v_block = v[:, :, k_start:k_end, :]       # [B, H, Bk, head_dim]

            # Scaled dot-product scores for this tile: [B, H, Bq, Bk]
            s = (q_block @ k_block.transpose(-2, -1)) * scale

            # Causal mask within the tile: mask positions where key > query.
            # Broadcasting: q_pos [Bq, 1] vs k_pos [1, Bk] → [Bq, Bk] mask.
            if is_causal:
                q_pos = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)
                k_pos = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)
                s = s.masked_fill(k_pos > q_pos, float('-inf'))

            # Online softmax update ------------------------------------------
            new_max    = torch.maximum(running_max, s.amax(dim=-1, keepdim=True))  # [B, H, Bq, 1]
            correction = torch.exp(running_max - new_max)                           # [B, H, Bq, 1]
            p          = torch.exp(s - new_max)                                     # [B, H, Bq, Bk]

            # Rescale old accumulators to new max baseline, then add new tile.
            acc_lse = correction * acc_lse + p.sum(dim=-1, keepdim=True)
            acc_out = correction * acc_out + p @ v_block
            running_max = new_max

        # Divide by accumulated normalizer to get the final softmax-weighted sum.
        output[:, :, q_start:q_end, :] = acc_out / acc_lse

    return output.to(orig_dtype)


# ---------------------------------------------------------------------------
# Grouped-Query Attention (GQA)
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    """Multi-head attention with fewer KV heads (grouped-query attention)."""

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int,
        qk_norm: bool = False,
        dtype: torch.dtype | None = None,
        use_flash_attn: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.group_size = num_heads // num_kv_groups   # how many Q heads per KV head
        self.use_flash_attn = use_flash_attn

        d_out = num_heads * head_dim

        self.W_query = nn.Linear(emb_dim, d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(emb_dim, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(emb_dim, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, emb_dim, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)
        else:
            self.q_norm = self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Project to Q, K, V -------------------------------------------------
        queries = self.W_query(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.W_key(x).view(batch, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(batch, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # Optional QK normalization -------------------------------------------
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        # Apply RoPE ----------------------------------------------------------
        # cos/sin are already sliced to the correct position window by the
        # model's forward — no further slicing needed here.
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # KV-cache: store K/V after RoPE (positional encoding baked in) and
        # before GQA expansion (saves memory at KV-head granularity).
        # update() appends the new slice and returns the full past+current K/V.
        if kv_cache is not None:
            keys, values = kv_cache.update(layer_idx, keys, values)

        # Expand KV heads to match Q heads ------------------------------------
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Attention -----------------------------------------------------------
        q_len = queries.shape[2]
        d_out = self.num_heads * self.head_dim

        # Flash path: tiled O(n) memory — only worth it when q_len > 1.
        # During KV-cached decode q_len == 1 so the scores matrix is already
        # a single row; flash would add Python loop overhead with no benefit.
        if self.use_flash_attn and q_len > 1:
            context = flash_attention(queries, keys, values, is_causal=True)
        else:
            # Eager path: materialise the full [B, H, q_len, kv_len] scores matrix.
            scores = queries @ keys.transpose(-2, -1)
            scores = scores / self.head_dim ** 0.5
            scores = scores.masked_fill(mask, -torch.inf)
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(queries.dtype)
            context = weights @ values

        context = context.transpose(1, 2).reshape(batch, q_len, d_out)
        return self.out_proj(context)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm transformer block: norm → attention → residual → norm → FFN → residual."""

    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.att = GroupedQueryAttention(
            emb_dim=cfg.emb_dim,
            num_heads=cfg.n_heads,
            num_kv_groups=cfg.n_kv_groups,
            head_dim=cfg.head_dim,
            qk_norm=cfg.qk_norm,
            dtype=cfg.dtype,
            use_flash_attn=cfg.use_flash_attn,
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg.emb_dim)
        self.norm2 = RMSNorm(cfg.emb_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        x = x + self.att(self.norm1(x), mask, cos, sin, kv_cache=kv_cache, layer_idx=layer_idx)
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full Qwen3 Model
# ---------------------------------------------------------------------------

class Qwen3Model(nn.Module):
    """Qwen3 causal language model (decoder-only transformer)."""

    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim, dtype=cfg.dtype)
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = RMSNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype)

        # Pre-compute RoPE cos/sin tables
        cos, sin = compute_rope_params(cfg.head_dim, cfg.context_length, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Pre-allocate causal mask once (sliced in forward)
        causal_mask = torch.triu(
            torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """
        Return logits of shape [batch, seq_len, vocab_size].

        When *kv_cache* is provided the model operates in one of two modes:

        Prefill  (cache is empty, input_ids is the full prompt):
          - Processes all P tokens in parallel as usual.
          - Each layer stores its K/V tensors in the cache.
          - Returns logits for all P positions (caller uses only the last one).

        Decode  (cache has P tokens, input_ids is a single new token):
          - Processes only the 1 new token.
          - cos/sin are sliced to position P (the new token's true position).
          - The attention mask is all-False: the single query can attend to
            the full cached context — no future tokens exist to mask.
          - Each layer appends the new K/V to the cache.
          - Returns logits for the 1 new position.
        """
        x = self.tok_emb(input_ids)

        q_len = x.shape[1]

        # Compute the position offset: 0 for the first (prefill) call,
        # cache.seq_len for every subsequent decode call.
        past_len = kv_cache.seq_len if kv_cache is not None else 0

        # Pre-slice RoPE tables to the exact position window [past_len, past_len + q_len).
        # Both prefill (past_len=0, q_len=P) and decode (past_len=P, q_len=1) work
        # correctly without any further slicing inside apply_rope.
        cos = self.cos[past_len : past_len + q_len]
        sin = self.sin[past_len : past_len + q_len]

        # Build the attention mask for scores of shape [q_len, kv_len].
        kv_len = past_len + q_len
        if past_len == 0:
            # Standard causal mask: token i cannot attend to token j > i.
            mask = self.causal_mask[:q_len, :kv_len]
        else:
            # Single new token during decode: it can attend to all past tokens
            # plus itself — no future tokens exist to block, so mask is all-False.
            mask = torch.zeros(q_len, kv_len, dtype=torch.bool, device=x.device)

        for layer_idx, block in enumerate(self.trf_blocks):
            x = block(x, mask, cos, sin, kv_cache=kv_cache, layer_idx=layer_idx)

        x = self.final_norm(x)
        return self.out_head(x.to(self.cfg.dtype))


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_weights_into_qwen(
    model: Qwen3Model,
    weights: dict[str, torch.Tensor],
) -> None:
    """Copy HuggingFace safetensors weights into our Qwen3Model in-place."""

    def copy_(dst: nn.Parameter, src: torch.Tensor) -> None:
        with torch.no_grad():
            dst.copy_(src if isinstance(src, torch.Tensor) else torch.as_tensor(src))

    copy_(model.tok_emb.weight, weights["model.embed_tokens.weight"])

    for layer_idx in range(model.cfg.n_layers):
        block = model.trf_blocks[layer_idx]
        prefix = f"model.layers.{layer_idx}"

        # Attention projections
        copy_(block.att.W_query.weight,  weights[f"{prefix}.self_attn.q_proj.weight"])
        copy_(block.att.W_key.weight,    weights[f"{prefix}.self_attn.k_proj.weight"])
        copy_(block.att.W_value.weight,  weights[f"{prefix}.self_attn.v_proj.weight"])
        copy_(block.att.out_proj.weight, weights[f"{prefix}.self_attn.o_proj.weight"])

        # QK norms
        if block.att.q_norm is not None:
            copy_(block.att.q_norm.scale, weights[f"{prefix}.self_attn.q_norm.weight"])
        if block.att.k_norm is not None:
            copy_(block.att.k_norm.scale, weights[f"{prefix}.self_attn.k_norm.weight"])

        # Layer norms
        copy_(block.norm1.scale, weights[f"{prefix}.input_layernorm.weight"])
        copy_(block.norm2.scale, weights[f"{prefix}.post_attention_layernorm.weight"])

        # Feed-forward
        copy_(block.ff.fc1.weight, weights[f"{prefix}.mlp.gate_proj.weight"])
        copy_(block.ff.fc2.weight, weights[f"{prefix}.mlp.up_proj.weight"])
        copy_(block.ff.fc3.weight, weights[f"{prefix}.mlp.down_proj.weight"])

    copy_(model.final_norm.scale, weights["model.norm.weight"])

    # Qwen3-0.6B uses weight tying (no separate lm_head.weight)
    if "lm_head.weight" in weights:
        copy_(model.out_head.weight, weights["lm_head.weight"])
    else:
        model.out_head.weight = model.tok_emb.weight


def download_and_load_weights(
    model: Qwen3Model,
    repo_id: str = "Qwen/Qwen3-0.6B",
) -> None:
    """Download weights from HuggingFace Hub and load them into the model."""
    local_dir = Path(repo_id).parts[-1]

    # Qwen3-0.6B has a single safetensors file
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
        local_dir=local_dir,
    )
    weights = load_file(weights_path)
    load_weights_into_qwen(model, weights)
    del weights
