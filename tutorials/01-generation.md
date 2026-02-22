# Tutorial 1: Let's Build LLM Inference From Scratch

*Pumpference series — from zero to a working inference framework in plain PyTorch*

---

There's a moment during this project that I keep thinking about. You've been staring at matrix shapes for days. You've debugged RoPE conventions and weight name mappings and bfloat16 precision traps. And then you type a prompt, hit enter, and your from-scratch implementation — your code, talking directly to the weight tensors — prints a coherent English sentence.

No `transformers`. No magic. Just PyTorch matmuls all the way down.

That moment is what this tutorial is about. We're going to build a complete LLM inference engine from scratch. By the end, you'll understand *every single operation* that happens between a text prompt and the first generated word. Not at a "I've read the diagram" level — at a "I wrote this code and debugged why line 147 produces NaN" level.

The implementation is in `src/pumpference/` on main. This article explains the why and the how. Let's get into it.

---

## Why bother?

HuggingFace `transformers` is incredible. `vLLM` is incredible. `llama.cpp` is incredible. They're also enormous. When you call `model.generate()`, somewhere between 50 and several thousand things happen invisibly: KV-cache management, attention masking, tokenization, sampling, quantization, kernel dispatch. When something goes wrong — or when you want to add something new — you're lost in a labyrinth of abstractions.

Building from scratch forces you to answer questions that production frameworks never make explicit:

- What *exact shape* is the attention mask, and why?
- Why does normalization need to upcast to float32?
- What happens when the number of KV heads doesn't match the number of Q heads?
- How do pretrained weight names (which are someone else's naming convention) map to your architecture?

I want to be clear about what this is not. This is not a performance-engineering exercise. The implementation we build here is intentionally naive — no KV-cache, no batching, no quantization. It re-computes everything from scratch on every forward pass. That's the point. Clarity first, speed later.

---

## Choosing a model

The first question is: which model? We need something small enough to run on a MacBook CPU (roughly 1-2 GB of weights), modern enough to have all the techniques worth learning, and well-documented enough that we can verify our work.

Qwen3-0.6B is essentially perfect for this. It's ~1.2 GB in bfloat16, it fits in RAM on any modern laptop, and it uses every important modern LLM architectural idea: RMSNorm, Rotary Positional Embeddings (RoPE), SwiGLU feed-forward networks, Grouped-Query Attention, and QK normalization. If you understand the Qwen3-0.6B architecture, you understand the vast majority of what's happening inside LLaMA, Mistral, Gemma, and their relatives. Sebastian Raschka's [from-scratch Qwen3 walkthrough](https://sebastianraschka.com/blog/2025/qwen3-from-scratch.html) was my primary reference baseline.

Here are the numbers from the model's `config.json`:

| Parameter | Value |
|---|---|
| Parameters | ~600M |
| Embedding dim | 1024 |
| Attention heads | 16 |
| KV heads | 8 |
| Head dim | 128 |
| Layers | 28 |
| FFN hidden dim | 3072 |
| Vocab size | 151,936 |
| Context length | 40,960 |
| Positional encoding | RoPE (θ=1,000,000) |
| Activation | SwiGLU |
| QK normalization | Yes |
| Weight tying | Yes |

One more thing I appreciated: Qwen3-0.6B has a single `model.safetensors` file. No sharding, no complicated weight merging. Just one file.

---

## Setup

I use [`uv`](https://docs.astral.sh/uv/) for package management. It's fast, handles virtual environments correctly, and locks dependencies properly. The project is a proper Python package with a `src/` layout:

```toml
[project]
name = "pumpference"
requires-python = ">=3.11"
dependencies = [
  "torch>=2.0",
  "safetensors",
  "huggingface_hub",
  "tokenizers",
]

[dependency-groups]
dev = [
  "pytest>=8.0.0",
  "transformers>=4.51.0",
]
```

Notice the dependency split. The runtime stack is minimal: PyTorch, safetensors for loading weights, `huggingface_hub` for downloading them, and `tokenizers` (the lightweight HuggingFace library, not `transformers`) for BPE encoding. HuggingFace `transformers` is a dev dependency only — used exclusively in tests to verify our implementation is correct. At runtime, we never touch it.

The directory structure:

```
pumpference/
├── src/pumpference/
│   ├── model.py      # All architecture + weight loading (~340 lines)
│   ├── generate.py   # Greedy generation loop (~40 lines)
│   └── tokenizer.py  # Tokenizer wrapper (~80 lines)
└── tests/
    └── test_model.py # Comparison tests vs HuggingFace
```

One design decision worth flagging up front: I initially split the model into many small files — `config.py`, `attention.py`, `rope.py`, `weights.py`, inside a `qwen3/` subpackage. I later consolidated everything into a single `model.py`. Why? At ~340 lines, the whole model fits in one file without losing readability, and having everything in one place makes it dramatically easier to follow the data flow. Premature modularization is a real antipattern in educational code.

---

## The architecture at a glance

Before we write any code, let's look at what we're building:

```
input_ids [batch, seq_len]
    │
    ▼
Token Embedding          # vocab_size → emb_dim (1024)
    │
    ▼
TransformerBlock × 28
  ├─ RMSNorm
  ├─ GroupedQueryAttention   ─┐
  ├─ + residual               │
  ├─ RMSNorm                  │ repeated 28 times
  ├─ FeedForward (SwiGLU)  ─┐ │
  └─ + residual              │ │
    │
    ▼
Final RMSNorm
    │
    ▼
Linear Head              # emb_dim → vocab_size (151,936)
    │
    ▼
logits [batch, seq_len, 151936]
```

Every decoder-only transformer — GPT-2, LLaMA, Qwen, Mistral — follows this structure. The differences between models live in the *details* of each block. Let's build each one.

---

## The config

Every architecture constant — embedding dimension, number of heads, layer count — needs to come from somewhere. I use a Python dataclass:

```python
@dataclass
class Qwen3Config:
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
```

All values come from the model's `config.json` on HuggingFace. The mapping isn't always obvious — HuggingFace calls the embedding dimension `hidden_size` (1024) and the FFN width `intermediate_size` (3072). Their naming is confusing because in most ML literature "hidden size" means the FFN's internal width. I renamed them `emb_dim` and `hidden_dim` to be less ambiguous.

Why bfloat16? The weights are distributed in bfloat16. Using the same dtype avoids precision loss from casting and keeps memory identical to the original. bfloat16 has the same exponent range as float32 (so no overflow) but 8 bits of mantissa instead of 24 — plenty for inference.

---

## RMSNorm

Ok, the simplest building block first. Modern transformers normalize activations between sub-layers — this stabilizes training and prevents values from exploding or vanishing as they pass through 28 layers. RMSNorm (Root Mean Square Normalization) is the normalization used in Qwen3, LLaMA, Mistral, and most modern models.

The idea is simple: normalize each vector by its root mean square, then scale by a learned parameter. The formula:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$$

Compare this to LayerNorm, which also subtracts the mean before normalizing. RMSNorm skips that step entirely. Empirically, the re-centering doesn't help, and removing it makes the computation ~10-15% faster. Fewer parameters too — no bias term.

The code is almost trivially short:

```python
class RMSNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x * self.scale).to(original_dtype)
```

There's one thing in here that isn't obvious: the `x.to(torch.float32)` upcast. This was my first subtle bug. If you compute the variance in bfloat16, you lose precision — squaring and averaging small values accumulates rounding error. The model will still *run*, but its logits will diverge from the reference implementation. The fix is to upcast to float32 for the RMS computation, then cast back. Two lines. Hours of confusion before I found it.

---

## Rotary Positional Embeddings (RoPE)

Here's the fundamental problem: a transformer processes all positions in parallel, with no built-in sense of order. Position 1 and position 100 look the same to the model unless we explicitly encode position information somewhere.

The classic solution (used in original GPT models) is to add positional embeddings to the token embeddings. RoPE takes a different approach: instead of adding position information to the input, it *rotates* the query and key vectors in attention. The rotation angle depends on the position, so tokens far apart in the sequence will have their Q and K vectors rotated by different amounts — making their dot products naturally depend on their relative distance.

The math: for each pair of dimensions $(2i, 2i+1)$ in a head vector, we apply a 2D rotation by angle $\theta_i \cdot m$, where $m$ is the token position and $\theta_i = \text{base}^{-2i/d}$. The frequencies decrease geometrically with dimension index, so different dimensions encode position at different scales — a bit like how clocks use seconds, minutes, and hours.

We pre-compute the cos and sin tables once, for all positions up to the context length:

```python
def compute_rope_params(
    head_dim: int,
    context_length: int,
    theta_base: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (
        theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    )                                                        # [head_dim // 2]
    positions = torch.arange(context_length).float()
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # [ctx, head_dim // 2]
    angles = torch.cat([angles, angles], dim=1)              # [ctx, head_dim]
    return torch.cos(angles), torch.sin(angles)
```

And applying the rotation is surprisingly clean — we don't need explicit rotation matrices. Split the vector into first and second halves, negate the second half, concatenate them in reverse order, then combine with cos/sin:

```python
def apply_rope(x, cos, sin):
    _, _, seq_len, head_dim = x.shape
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    return ((x * cos) + (rotated * sin)).to(x.dtype)
```

Two gotchas worth flagging:

**The theta_base value.** Qwen3 uses `rope_base = 1,000,000` — far higher than the original RoPE paper's 10,000. A higher base stretches the positional wavelengths, enabling the model to generalize to longer sequences. If you use the default 10,000 (which I initially did), the model generates garbage on any sequence longer than a few hundred tokens. And crucially, even short sequences don't match the reference — different theta means different rotations everywhere.

**Concat vs interleave.** There are two conventions for how rotation pairs dimensions:
- *Interleaved*: pairs are $(0,1), (2,3), (4,5)...$ — the original RoPE paper.
- *Split-half*: pairs are $(0, d/2), (1, d/2+1)...$ — LLaMA, Qwen, and most modern implementations.

These produce completely different results. Qwen3 uses split-half, which is what `torch.cat([-x2, x1], dim=-1)` implements. If you port code from a different implementation, check which convention it uses.

---

## SwiGLU Feed-Forward Network

Each transformer block has a feed-forward network that applies after attention. The standard design (from "Attention Is All You Need") uses two linear layers: project up → apply nonlinearity → project back down. SwiGLU adds a twist: use three linear layers instead of two.

$$\text{FFN}(x) = W_3 \cdot (\text{SiLU}(W_1 \cdot x) \odot W_2 \cdot x)$$

Where $\odot$ is elementwise multiplication. The idea: project to a higher dimension *twice* — one projection passes through a SiLU activation, the other is a "gate" — then multiply them together before projecting back down. The gate learns to suppress or amplify different features before passing them through the nonlinearity.

In code it's delightfully concise:

```python
class FeedForward(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, bias=False, dtype=cfg.dtype)
        self.fc2 = nn.Linear(cfg.emb_dim, cfg.hidden_dim, bias=False, dtype=cfg.dtype)
        self.fc3 = nn.Linear(cfg.hidden_dim, cfg.emb_dim, bias=False, dtype=cfg.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(nn.functional.silu(self.fc1(x)) * self.fc2(x))
```

Yes, `fc1` and `fc2` both take the same input `x`. They're independent projections — neither depends on the other. The gate mechanism (`* self.fc2(x)`) is purely multiplicative gating after the activation.

One gotcha that bit me: the HuggingFace weight names are `gate_proj`, `up_proj`, `down_proj`. In SwiGLU, `gate_proj` is the branch that goes through SiLU (`fc1` in my naming), and `up_proj` is the gate branch (`fc2`). I had them swapped initially. The model ran fine — matrix multiplications don't complain — but the outputs were nonsense. The test suite caught it immediately, which is why writing tests against a reference implementation is non-negotiable.

Notice also `bias=False` on every linear layer. Modern LLMs (LLaMA, Qwen, Mistral, Gemma) have dropped bias terms across the board. Empirically, biases don't help at scale and removing them simplifies the architecture.

---

## Grouped-Query Attention

This is the most complex component. Let's build up to it.

Standard multi-head attention projects queries, keys, and values all to the same number of heads. GQA reduces the number of key-value heads while keeping the full count of query heads. Qwen3-0.6B has 16 query heads but only 8 KV heads. Each KV head is shared by 2 query heads.

Why does this matter? The key-value cache (which we'll implement in a later tutorial) stores the keys and values for every token in the sequence. With 16 full KV heads, the cache is 2× larger than it needs to be. With 8 KV heads, we halve the cache memory while keeping the expressiveness of 16 query heads. At scale, this is a very attractive trade.

For our naive implementation (no KV-cache yet), GQA just means asymmetric projection dimensions:

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, num_kv_groups, head_dim,
                 qk_norm=False, dtype=None):
        super().__init__()
        self.num_heads = num_heads          # 16
        self.num_kv_groups = num_kv_groups  # 8
        self.head_dim = head_dim            # 128
        self.group_size = num_heads // num_kv_groups  # 2

        d_out = num_heads * head_dim        # 16 × 128 = 2048

        self.W_query = nn.Linear(emb_dim, d_out, bias=False, dtype=dtype)
        self.W_key   = nn.Linear(emb_dim, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(emb_dim, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, emb_dim, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)
```

`W_query` projects to 2048, `W_key` and `W_value` project to 1024. That asymmetry is the whole GQA idea.

The forward pass proceeds in a specific order that matters:

```python
def forward(self, x, mask, cos, sin):
    batch, seq_len, _ = x.shape

    # 1. Project → reshape → [batch, heads, seq_len, head_dim]
    queries = self.W_query(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    keys    = self.W_key(x).view(batch, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
    values  = self.W_value(x).view(batch, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)

    # 2. QK normalization (Qwen3-specific — applied AFTER projection, BEFORE RoPE)
    if self.q_norm:
        queries = self.q_norm(queries)
    if self.k_norm:
        keys = self.k_norm(keys)

    # 3. Apply RoPE to Q and K (NOT to values)
    queries = apply_rope(queries, cos, sin)
    keys    = apply_rope(keys, cos, sin)

    # 4. Expand KV heads to match Q heads (AFTER RoPE, not before)
    keys   = keys.repeat_interleave(self.group_size, dim=1)    # 8 → 16 heads
    values = values.repeat_interleave(self.group_size, dim=1)

    # 5. Scaled dot-product attention
    scores = queries @ keys.transpose(-2, -1)
    scores = scores / self.head_dim ** 0.5
    scores = scores.masked_fill(mask, -torch.inf)
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(queries.dtype)

    # 6. Aggregate and project back
    d_out = self.num_heads * self.head_dim
    context = (weights @ values).transpose(1, 2).reshape(batch, seq_len, d_out)
    return self.out_proj(context)
```

Several things in here deserve explanation:

**QK normalization.** Qwen3 applies RMSNorm to queries and keys *after* projection but *before* RoPE. Most models (LLaMA, Mistral) don't do this. QK-norm stabilizes attention scores and prevents them from growing too large in deep models. Skip it and the logits won't match the reference. It's a Qwen3-specific detail that's easy to miss when you're working from a generic transformer blueprint.

**Softmax in float32.** The line `torch.softmax(scores, dim=-1, dtype=torch.float32)` is not optional. Softmax involves exponentiation, and in bfloat16, large attention scores can overflow to infinity. Casting just the softmax to float32 (and then back) is cheap and prevents NaN attention weights.

**`-torch.inf` for the causal mask.** The causal mask zeros out future positions so each token can only attend to past tokens. I use `-torch.inf` rather than `-1e9`. In bfloat16, a value like `-1e9` might not be sufficiently negative to produce *exact* zeros after softmax — there can be tiny amounts of attention leakage to future positions. `-torch.inf` guarantees exact zeros, period.

---

## Putting it together: Transformer Block and full model

Each of the 28 blocks follows the pre-norm pattern — normalize *before* each sub-layer, not after:

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.att = GroupedQueryAttention(...)
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg.emb_dim)
        self.norm2 = RMSNorm(cfg.emb_dim)

    def forward(self, x, mask, cos, sin):
        x = x + self.att(self.norm1(x), mask, cos, sin)  # norm → attn → residual
        x = x + self.ff(self.norm2(x))                   # norm → FFN → residual
        return x
```

Pre-norm (normalize before the sub-layer) is the modern standard. The original "Attention Is All You Need" used post-norm (normalize after), but pre-norm trains more stably, especially at depth.

The full model assembles the pieces:

```python
class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim, dtype=cfg.dtype)
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype)

        # Pre-compute RoPE tables — registered as buffers so they move with the model
        cos, sin = compute_rope_params(cfg.head_dim, cfg.context_length, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Pre-allocate full causal mask, sliced in forward
        causal_mask = torch.triu(
            torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        seq_len = x.shape[1]
        mask = self.causal_mask[:seq_len, :seq_len]
        for block in self.trf_blocks:
            x = block(x, mask, self.cos, self.sin)
        x = self.final_norm(x)
        return self.out_head(x.to(self.cfg.dtype))
```

A few deliberate choices here:

`register_buffer` with `persistent=False` means these tensors move with the model when you call `.to(device)` — so they automatically land on the right device — but they're not saved in state_dict checkpoints (they can always be recomputed from config). The RoPE tables and causal mask are constants derived from hyperparameters; treating them as parameters would be wrong.

Pre-allocating the full causal mask and slicing it in `forward` (`self.causal_mask[:seq_len, :seq_len]`) avoids creating a new tensor on every forward call. Small optimization, but correct practice.

---

## Loading real weights

We have an architecture. Now we need to load actual trained weights into it.

The weights live on HuggingFace as a safetensors file. Downloading is easy — `huggingface_hub` handles caching so the ~1.2 GB file is only downloaded once:

```python
def download_and_load_weights(model: Qwen3Model, repo_id: str = "Qwen/Qwen3-0.6B") -> None:
    local_dir = Path(repo_id).parts[-1]
    weights_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
        local_dir=local_dir,
    )
    weights = load_file(weights_path)
    load_weights_into_qwen(model, weights)
    del weights  # free the raw dict immediately — it's a full copy of all parameters
```

The tricky part is `load_weights_into_qwen`. HuggingFace uses its own naming convention:

```
model.embed_tokens.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.self_attn.q_norm.weight
model.layers.0.self_attn.k_norm.weight
model.layers.0.input_layernorm.weight
model.layers.0.post_attention_layernorm.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.mlp.down_proj.weight
...
model.norm.weight
```

We need to copy each tensor into the right parameter in our model:

```python
def load_weights_into_qwen(model, weights):
    def copy_(dst, src):
        with torch.no_grad():
            dst.copy_(src)

    copy_(model.tok_emb.weight, weights["model.embed_tokens.weight"])

    for layer_idx in range(model.cfg.n_layers):
        block = model.trf_blocks[layer_idx]
        prefix = f"model.layers.{layer_idx}"

        copy_(block.att.W_query.weight,  weights[f"{prefix}.self_attn.q_proj.weight"])
        copy_(block.att.W_key.weight,    weights[f"{prefix}.self_attn.k_proj.weight"])
        copy_(block.att.W_value.weight,  weights[f"{prefix}.self_attn.v_proj.weight"])
        copy_(block.att.out_proj.weight, weights[f"{prefix}.self_attn.o_proj.weight"])

        if block.att.q_norm is not None:
            copy_(block.att.q_norm.scale, weights[f"{prefix}.self_attn.q_norm.weight"])
            copy_(block.att.k_norm.scale, weights[f"{prefix}.self_attn.k_norm.weight"])

        copy_(block.norm1.scale, weights[f"{prefix}.input_layernorm.weight"])
        copy_(block.norm2.scale, weights[f"{prefix}.post_attention_layernorm.weight"])

        copy_(block.ff.fc1.weight, weights[f"{prefix}.mlp.gate_proj.weight"])
        copy_(block.ff.fc2.weight, weights[f"{prefix}.mlp.up_proj.weight"])
        copy_(block.ff.fc3.weight, weights[f"{prefix}.mlp.down_proj.weight"])

    copy_(model.final_norm.scale, weights["model.norm.weight"])

    # Weight tying: Qwen3-0.6B doesn't have a separate lm_head.weight
    if "lm_head.weight" in weights:
        copy_(model.out_head.weight, weights["lm_head.weight"])
    else:
        model.out_head.weight = model.tok_emb.weight
```

That last block — weight tying — caught me off guard. Qwen3-0.6B shares the output projection weights with the token embedding table. There is no `lm_head.weight` key in the safetensors file. If you create a separate `nn.Linear` for the output head (as I do), you have to explicitly detect this absence and point `out_head.weight` to `tok_emb.weight`. Miss this and your output head has random initialized weights and the model outputs complete garbage — and it will do so confidently, because it has 600M parameters worth of good hidden representations feeding into a randomly initialized projection.

---

## Tokenizer

Text in, token IDs out. Token IDs in, text out. That's the tokenizer's job.

Qwen3 uses a BPE tokenizer, and its definition lives in `tokenizer.json` on HuggingFace. We could load it with `AutoTokenizer` from the `transformers` library, but that would bring `transformers` into the runtime dependency stack, which I want to avoid. Instead, we use the lightweight `tokenizers` library (also by HuggingFace) which can load `tokenizer.json` directly.

The tricky part is special tokens. Qwen3 has many of them: `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`, and more. The raw BPE tokenizer sometimes splits these into sub-pieces instead of treating them atomically. The fix: split the input text at special token boundaries first, handle specials by direct ID lookup, and BPE-encode the rest:

```python
_SPECIAL_TOKENS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", ...)
_SPLIT_RE = re.compile("(" + "|".join(re.escape(t) for t in _SPECIAL_TOKENS) + ")")

class Qwen3Tokenizer:
    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for part in filter(None, _SPLIT_RE.split(text)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids
```

One more thing: the EOS token for Qwen3 is `<|im_end|>`, *not* `<|endoftext|>`. Qwen3 uses the ChatML message format where `<|im_end|>` marks the end of an assistant turn. If you stop at `<|endoftext|>` instead, generation just... keeps going. Depending on your `max_tokens` limit, you might not notice for a while.

---

## The generation loop

Here's the entire autoregressive generation loop:

```python
@torch.no_grad()
def generate(model, input_ids, max_new_tokens, eos_token_id=None):
    model.eval()
    tokens = input_ids

    for _ in range(max_new_tokens):
        logits = model(tokens)                                  # [1, seq_len, vocab]
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True) # [1, 1]
        tokens = torch.cat([tokens, next_token], dim=1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return tokens
```

That's it. That's the whole thing.

On every step: run the full forward pass over the entire sequence, look at only the last position's logits, pick the highest-probability token (argmax = greedy decoding), append it to the sequence, repeat.

This is "naive" in a precise sense: we're feeding the *entire sequence* — prompt plus all generated tokens so far — through all 28 layers on every single step. For token #100, we re-compute the attention for tokens #1–99 even though they haven't changed. This is O(n²) in sequence length: each new token makes the next step proportionally slower.

The fix is a KV-cache: store the keys and values computed during prefill, and on each subsequent step, only compute the new token's query against the cached keys and values. That turns decode from O(n) per step to O(1) per step. We'll implement it in the next tutorial. For now, we accept the slowness and appreciate the clarity.

`argmax` always picks the highest-probability token, which is deterministic and "safe" but can be repetitive. Temperature, top-k, and top-p sampling (which add controlled randomness) are Tutorial 3.

---

## How do we know it's correct?

This is, I'd argue, the most important part of the whole project.

Without tests, you have no idea whether your implementation is correct. The model might output plausible-looking text while getting every attention score slightly wrong. Visual inspection can't catch precision bugs. You need a reference.

The approach: load both our model and the HuggingFace `transformers` version of Qwen3-0.6B with identical weights, feed them the same input, and compare outputs.

```python
def test_logits_argmax_matches_single_forward(our_model, hf_model, input_ids):
    our_logits = our_model(input_ids)
    hf_logits  = hf_model(input_ids).logits

    assert torch.equal(our_logits.argmax(dim=-1), hf_logits.argmax(dim=-1))

    diff = (our_logits.float() - hf_logits.float()).abs()
    assert diff.max().item() < 1.0
```

Why compare argmax rather than exact logit values? Because bfloat16 arithmetic is not associative — the order of operations affects the result. Our implementation and HuggingFace's perform the same mathematical operations in slightly different orders (different code paths, different fused kernels). Raw logit values will differ by small amounts, typically well under 0.5. But the *argmax* — which token has the highest probability — should be identical. If it's not, we have a real correctness bug.

We also run a generation comparison test: 20-token greedy generation, token-by-token equality. If single-forward argmax matches, generation should too. Running both catches edge cases.

**The one-keyword-argument bug that cost me hours.** When loading the HuggingFace model for testing, you must specify:

```python
hf_model = AutoModelForCausalLM.from_pretrained(
    REPO_ID,
    dtype=torch.bfloat16,
    attn_implementation="eager",  # <-- this
)
```

By default, HuggingFace uses `sdpa` (PyTorch's fused `scaled_dot_product_attention` kernel) which uses different internal precision and operation ordering than manual matmul + softmax. With `sdpa`, the argmax mismatches in ~2-3% of positions. I spent a long time checking my RoPE implementation, my normalization math, my weight loading — everything looked correct. The actual fix was adding `attn_implementation="eager"` to force HuggingFace down the same explicit code path we use. One argument. Discovered after far too many hours.

Fixtures use `scope="module"` — models are loaded once per test file, not per test function. Loading a 600M parameter model takes several seconds; you don't want to pay that cost for every test.

---

## Everything that went wrong

Let me consolidate all the precision and correctness traps from above. If you're following along and your outputs don't match the reference, this is where to look:

| Issue | Symptom | Fix |
|---|---|---|
| Missing float32 upcast in RMSNorm | Logits diverge slightly | Cast to float32 for variance computation, cast back |
| Wrong `rope_base` (10k vs 1M) | Garbage on longer sequences; short sequence logit mismatch too | Use value from `config.json`: 1,000,000 |
| Interleaved vs split-half RoPE | Completely wrong attention patterns | Use split-half: `cat([-x2, x1], dim=-1)` |
| Swapped `gate_proj` / `up_proj` | Model generates nonsense, runs fine | `fc1=gate_proj`, `fc2=up_proj`, `fc3=down_proj` |
| Missing QK normalization | Logits don't match HF; instability on long sequences | Apply RMSNorm to Q and K after projection, before RoPE |
| Missing weight tying | Output head has random weights; coherent hidden states → garbage output | Point `out_head.weight` to `tok_emb.weight` |
| Wrong EOS token | Generation never stops | `<|im_end|>` is the EOS, not `<|endoftext|>` |
| HF using SDPA by default in tests | Argmax mismatches ~2-3% of positions | `attn_implementation="eager"` in the HF model fixture |
| Softmax in bfloat16 | Overflow to inf, NaN attention weights | `torch.softmax(..., dtype=torch.float32)` |
| `-1e9` instead of `-torch.inf` for causal mask | Subtle attention leakage to future tokens | Use `-torch.inf` |
| Not deleting weight dict after loading | Out of memory on lower-RAM machines | `del weights` after `load_weights_into_qwen` |

I hit all eleven of these. The good news: having a reference implementation to test against means none of them can hide. They all surface as test failures. Write the tests first.

---

## How fast (or slow) is it?

Let's see the damage. The benchmark harness in `src/pumpference/benchmark.py` measures prefill throughput, decode throughput, time-to-first-token (TTFT), per-token decode latency (p50/p90/p99), and peak memory. All numbers below are from a single run on an Apple M3 Pro CPU, bfloat16 weights, 100 generated tokens per preset.

| Preset | Prompt tokens | Prefill TPS | TTFT | Decode TPS | Latency mean | P50 | P99 | Peak memory |
|--------|:---:|---:|---:|---:|---:|---:|---:|---:|
| xs     |  30  | 102.1 tok/s |  294 ms | 1.3 tok/s |  777 ms/tok |  762 ms |  1750 ms | 4122 MB |
| short  | 115  | 105.2 tok/s | 1093 ms | 0.6 tok/s | 1749 ms/tok | 1696 ms |  2784 ms | 4013 MB |
| medium | 218  |  90.9 tok/s | 2398 ms | 0.3 tok/s | 3323 ms/tok | 3252 ms |  6003 ms | 3738 MB |
| long   | 372  |  70.7 tok/s | 5264 ms | 0.2 tok/s | 6321 ms/tok | 6284 ms |  7612 ms | 3783 MB |

![Baseline benchmark — decode throughput, per-token latency, and TTFT across context lengths](assets/baseline-benchmark.png)

**Prefill is relatively fast.** Processing 372 tokens takes 5264 ms — about 17.9× longer than 30 tokens (294 ms). Slightly superlinear but not dramatically so.

**Decode is catastrophically slow and clearly O(n²).** The per-token decode latency goes from 777 ms at a 30-token context to 6321 ms at a 372-token context — an 8× slowdown for a 12× increase in context length. That's O(n²) behavior: each new token makes every future token proportionally more expensive. At 0.2 tok/s on a 372-token prompt, this implementation is unusable for anything practical.

**The P99 spike on xs tells the story clearly.** For the xs preset, P50 latency is 762 ms but P99 jumps to 1750 ms — a 2.3× spread. This makes perfect sense: the first decode step processes 31 tokens (fast), the last processes 129 tokens (slow). P50 captures the typical middle; P99 reflects the worst late-generation steps when the context is longest. For the long preset, the spread narrows (P50=6284 ms, P99=7612 ms, 1.2×) because all steps start from an already-large context and the proportional growth per step is smaller.

**Peak memory is ~4 GB regardless of context.** Almost all of that is model weights (~1.2 GB in bfloat16) amplified by intermediate activations and PyTorch's allocator overhead. The KV-cache we'll add next will *increase* memory — that's the trade: more memory for dramatically faster decode.

These numbers are our baseline. Every subsequent optimization tutorial will report new numbers against this table.

---

## What's next

- **Tutorial 2**: Sampling — temperature, top-k, and top-p (nucleus) sampling. Argmax always picks the single most probable token, which produces deterministic but often repetitive output. Sampling strategies let you trade some probability mass for diversity, and getting the details right (correct temperature scaling, proper renormalization after top-k truncation) is trickier than it looks.

---

*This article is part of the Pumpference tutorial series. The full source code is at [github.com/legchikov/pumpference](https://github.com/legchikov/pumpference).*

*Found an error? Open an issue or PR.*

---

## References

- Sebastian Raschka's from-scratch Qwen3 implementation was the primary reference baseline: [notebook](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/standalone-qwen3.ipynb) and [blog post](https://sebastianraschka.com/blog/2025/qwen3-from-scratch.html).
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) — the original paper describing the architecture and training details.
