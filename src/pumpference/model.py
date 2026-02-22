"""
Qwen3 0.6B model architecture 

Based on: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/standalone-qwen3.ipynb

Architecture components:
  RMSNorm, RoPE, FeedForward (SwiGLU), GroupedQueryAttention
"""

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


QWEN3_0_6B_CONFIG = Qwen3Config()


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
    """Apply RoPE to a tensor of shape [batch, heads, seq_len, head_dim]."""
    _, _, seq_len, head_dim = x.shape

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)   # [1, 1, seq_len, head_dim]
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

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
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.group_size = num_heads // num_kv_groups   # how many Q heads per KV head

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
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Expand KV heads to match Q heads ------------------------------------
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Scaled dot-product attention ----------------------------------------
        scores = queries @ keys.transpose(-2, -1)
        scores = scores / self.head_dim ** 0.5
        scores = scores.masked_fill(mask, -torch.inf)
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(queries.dtype)

        # Combine heads -------------------------------------------------------
        d_out = self.num_heads * self.head_dim
        context = (weights @ values).transpose(1, 2).reshape(batch, seq_len, d_out)
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
    ) -> torch.Tensor:
        x = x + self.att(self.norm1(x), mask, cos, sin)
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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits of shape [batch, seq_len, vocab_size]."""
        x = self.tok_emb(input_ids)

        seq_len = x.shape[1]
        mask = self.causal_mask[:seq_len, :seq_len]

        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)

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
