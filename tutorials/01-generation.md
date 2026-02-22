# Part 1: Building an LLM Inference Engine From Scratch

*Pumpference series — from zero to a running language model in plain PyTorch*

---

Somewhere around hour 16 of this project, deep into a debugging session that should have taken 30 minutes, I found the bug. Everything in my implementation looked right. The architecture matched the paper. The weight loading looked correct. The model generated text. But the logits were off by a small, consistent margin — just enough to fail the comparison test.

I had checked RMSNorm twice. I had re-read the RoPE math. I had gone through the weight-loading function line by line three times. I had added print statements everywhere. Nothing obviously wrong.

The bug was two words: `attn_implementation="eager"`. By default, HuggingFace uses a fused attention kernel that produces slightly different numerical results than explicit matrix operations. My implementation was perfectly correct. My *test* was comparing it against the wrong thing.

I fixed it in 30 seconds. I found it in 4 hours.

This is what building LLM inference from scratch actually feels like. Not a smooth progression from diagram to working code. Just a long series of small collisions with reality, each one making the system slightly more legible. This article documents all of them.

---

## Why bother?

HuggingFace `transformers` handles all of this. `vLLM` does it faster. `llama.cpp` runs it on your phone. So why build from scratch?

Because every time you use a black box, you're trading understanding for convenience. For LLMs specifically, the black box is unusually opaque. When you call `model.generate()`, somewhere between 50 and several thousand things happen invisibly: KV-cache management, attention masking, tokenization, sampling, quantization, kernel dispatch. You get text out. You have no idea what decisions were made on the way.

Building from scratch forces a different relationship. You can't have a wrong mental model about something you wrote every line of. The questions that production frameworks never make explicit — what exact shape is the attention mask, why does normalization upcast to float32, what happens when KV heads don't match Q heads — have to be answered, because your code won't run until you answer them.

I want to be precise about what this is not. This is not a performance project. The implementation here is intentionally naive: no KV-cache, no batching, no quantization, no clever kernels. It recomputes everything from scratch on every forward pass. That's the point. Clarity first. Speed later (Tutorial 4).

---

## Choosing a model

I needed something small enough to run on a MacBook CPU — under 2 GB of weights — modern enough to include all the architectural ideas worth learning, and simple enough that I could verify correctness against a reference.

Qwen3-0.6B is close to perfect. It's ~1.2 GB in bfloat16. It runs on any modern laptop with 8+ GB of RAM. And it uses every important technique in modern decoder-only transformers: RMSNorm, Rotary Positional Embeddings, SwiGLU feed-forward networks, Grouped-Query Attention, QK normalization. If you understand this model's internals, you understand roughly 90% of what's happening inside LLaMA 3, Mistral, Gemma, and their relatives. The architecture family has substantially converged on these components.

The numbers from `config.json`:

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

One thing I appreciated: Qwen3-0.6B has a single `model.safetensors` file. No sharding, no complicated weight merging. Just one file.

My primary reference was Sebastian Raschka's [from-scratch Qwen3 walkthrough](https://sebastianraschka.com/blog/2025/qwen3-from-scratch.html). Highly recommended alongside this one — his approach and mine differ in places, and the differences are instructive.

---

## Setup

I'm using [`uv`](https://docs.astral.sh/uv/) for package management. Fast, correct virtual environment handling, real lockfiles.

The dependency split matters:

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

The runtime stack is minimal: PyTorch, `safetensors` for loading weights, `huggingface_hub` for downloading them, and `tokenizers` — the lightweight HuggingFace library, not `transformers` — for BPE encoding. HuggingFace `transformers` is dev-only, used exclusively in tests to verify our implementation is correct. At runtime we never touch it.

The directory structure:

```
pumpference/
├── src/pumpference/
│   ├── model.py      # All architecture + weight loading (~340 lines)
│   ├── generate.py   # Generation loop (~60 lines)
│   └── tokenizer.py  # Tokenizer wrapper (~80 lines)
└── tests/
    └── test_model.py # Comparison tests vs HuggingFace
```

One design decision I want to flag: I initially split the model into many files — `config.py`, `attention.py`, `rope.py`, `weights.py`, inside a `qwen3/` subpackage. I later merged everything into a single `model.py`. The model is ~340 lines. Having everything in one place makes it dramatically easier to trace the data flow. Premature modularization is a real antipattern in educational code — you get the complexity of multiple files without any of the benefits that justify having them.

---

## Architecture overview

Before writing any code, here's the full picture of what we're building:

```
input_ids [batch, seq_len]
    │
    ▼
Token Embedding          # vocab_size → emb_dim (1024)
    │
    ▼
TransformerBlock × 28
  ├─ RMSNorm
  ├─ GroupedQueryAttention   # 16 Q-heads, 8 KV-heads
  ├─ + residual
  ├─ RMSNorm
  ├─ FeedForward (SwiGLU)
  └─ + residual
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

Every decoder-only transformer — GPT-2, LLaMA, Qwen, Mistral — follows this exact structure. The differences between models live in the *details* of each block, not in the overall shape. Let's build each piece.

---

## The config

Every architecture constant needs to come from somewhere. A Python dataclass is the cleanest option:

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

All values come from `config.json` on HuggingFace. The naming translation isn't always obvious — HuggingFace calls the embedding dimension `hidden_size` (1024) and the FFN width `intermediate_size` (3072). Their naming is confusing because in most ML literature "hidden" means the FFN's internal width, not the main model dimension. I renamed them `emb_dim` and `hidden_dim` to be less ambiguous.

bfloat16 throughout: same exponent range as float32 (no overflow), 8 bits of mantissa instead of 24. Plenty for inference. Matching the distribution dtype avoids casting overhead and keeps everything numerically consistent with the original.

---

## RMSNorm

Let's start with the simplest component.

Modern transformers normalize activations between sub-layers to prevent values from growing or vanishing as they pass through 28 layers. RMSNorm (Root Mean Square Normalization) is the choice in Qwen3, LLaMA, Mistral, and most current models.

The formula:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$$

Compare to LayerNorm: LayerNorm subtracts the mean before normalizing; RMSNorm skips that step entirely. The argument is that re-centering doesn't help at scale — the network can work around any constant offset through its weights. RMSNorm is ~10–15% faster and has fewer parameters. Fewer moving parts.

The code is almost embarrassingly short:

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

One thing in here that's not obvious: `x.to(torch.float32)`. This was my first precision bug. If you compute variance in bfloat16, you lose precision — squaring small values and averaging them accumulates rounding error. The model runs, the outputs look reasonable, but the logits diverge from the reference by just enough to fail the test. The fix is to upcast to float32 for the computation, then cast back to bfloat16 before returning. Two lines. An embarrassingly long time to find.

---

## Rotary Positional Embeddings (RoPE)

A transformer processes all positions in parallel. Without explicit position information, the token at position 1 and the token at position 100 look identical to the model — same input representation, same operations, same output. We need to encode position somewhere.

The classic approach adds a position-dependent vector to each token embedding before the transformer layers. RoPE takes a more elegant approach: instead of adding position information to the input, it *rotates* the query and key vectors in each attention head. The rotation angle depends on position and dimension index — tokens far apart in the sequence have their Q and K vectors rotated by different amounts, so their dot product naturally encodes relative distance.

The math: for each pair of dimensions $(2i, 2i+1)$ in a head vector, apply a 2D rotation by angle $\theta_i \cdot m$, where $m$ is the token position and $\theta_i = \text{base}^{-2i/d}$. The frequencies decrease geometrically with dimension index — the first few dimensions change rapidly with position, the last few change slowly. It's a little like how a clock represents time with three hands at very different speeds.

We precompute the cos and sin tables once, for all positions up to the context length:

```python
def compute_rope_params(
    head_dim: int,
    context_length: int,
    theta_base: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (
        theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    )
    positions = torch.arange(context_length).float()
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # [ctx, head_dim // 2]
    angles = torch.cat([angles, angles], dim=1)              # [ctx, head_dim]
    return torch.cos(angles), torch.sin(angles)
```

Applying the rotation is clean — split each head vector into two halves, swap with a sign flip, combine with cos/sin:

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

Two gotchas that cost me meaningful time:

**The theta base.** Qwen3 uses `rope_base = 1,000,000` — far higher than the original RoPE paper's 10,000. A higher base stretches the positional wavelengths, letting the model generalize better to long sequences. If you use 10,000 — which I initially did, because that's the default in most blog posts about RoPE — the logits don't match the reference even on short sequences. Every attention score is computed with the wrong rotation angle.

**Interleaved vs split-half pairing.** There are two conventions for how to pair dimensions for rotation:
- *Interleaved*: pairs are (0,1), (2,3), (4,5)... — the original RoPE paper.
- *Split-half*: pairs are (0, d/2), (1, d/2+1)... — LLaMA, Qwen, and most modern implementations.

These produce completely different results. Qwen3 uses split-half, which is what `torch.cat([-x2, x1], dim=-1)` implements. This has no type error, no shape mismatch, no error message of any kind. It just silently produces wrong rotations. Check which convention your source material uses before porting anything.

---

## SwiGLU Feed-Forward

Every transformer block has a feed-forward network that runs after attention. The classical design (from "Attention Is All You Need") is two linear layers: project up, apply a nonlinearity, project back down. SwiGLU replaces this with three:

$$\text{FFN}(x) = W_3 \cdot (\text{SiLU}(W_1 \cdot x) \odot W_2 \cdot x)$$

You project the input twice, independently. One projection goes through SiLU. The other is a "gate" — it multiplies the activated projection element-wise before projecting back down. The gate learns to suppress or amplify different features. Empirically this outperforms a standard MLP at the same parameter count.

In code it's unreasonably concise:

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

`fc1` and `fc2` both take the same input `x`. They're independent projections. The gate mechanism is purely multiplicative after the activation.

A gotcha worth documenting: HuggingFace weight names are `gate_proj`, `up_proj`, `down_proj`. In SwiGLU, `gate_proj` is the branch that goes through SiLU — that's `fc1` in my naming. `up_proj` is the gate branch — `fc2`. I had them swapped in an early version. The model ran perfectly fine (matrix multiplication is indifferent to which branch has which weights), but the outputs were completely wrong. The test suite caught this immediately. This is exactly the kind of bug that passes visual inspection and needs a comparison test to surface.

Also: `bias=False` on every linear layer. Modern LLMs (LLaMA, Qwen, Mistral, Gemma) have universally dropped bias terms. They don't improve training at scale; removing them simplifies the architecture.

---

## Grouped-Query Attention

This is the most complex component. Let me build up to it.

Standard multi-head attention projects queries, keys, and values all to the same number of heads. Grouped-Query Attention (GQA) reduces the KV head count while keeping the full Q head count. Qwen3-0.6B has 16 Q heads but only 8 KV heads. Each KV head is shared by 2 Q heads.

Why? Memory. The key-value cache (Tutorial 4) stores key and value tensors for every token in context. With 16 full KV heads, that cache is 2× larger than it needs to be. With 8 KV heads, you halve the KV cache memory at a tiny quality cost — empirically, barely measurable. For our naive implementation without a KV-cache, GQA just means asymmetric projection dimensions:

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

`W_query` projects to 2048 dimensions (16 heads × 128). `W_key` and `W_value` project to 1024 (8 heads × 128). That asymmetry is the entire GQA idea.

The forward pass has a specific operation order that matters:

```python
def forward(self, x, mask, cos, sin):
    batch, seq_len, _ = x.shape

    queries = self.W_query(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    keys    = self.W_key(x).view(batch, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
    values  = self.W_value(x).view(batch, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)

    # QK normalization: Qwen3-specific, applied AFTER projection, BEFORE RoPE
    if self.q_norm:
        queries = self.q_norm(queries)
    if self.k_norm:
        keys = self.k_norm(keys)

    # Apply RoPE to Q and K (not to values)
    queries = apply_rope(queries, cos, sin)
    keys    = apply_rope(keys, cos, sin)

    # Expand KV heads to match Q heads — AFTER RoPE, not before
    keys   = keys.repeat_interleave(self.group_size, dim=1)
    values = values.repeat_interleave(self.group_size, dim=1)

    scores = queries @ keys.transpose(-2, -1)
    scores = scores / self.head_dim ** 0.5
    scores = scores.masked_fill(mask, -torch.inf)
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(queries.dtype)

    d_out = self.num_heads * self.head_dim
    context = (weights @ values).transpose(1, 2).reshape(batch, seq_len, d_out)
    return self.out_proj(context)
```

Four things in here deserve explanation:

**QK normalization.** Qwen3 applies RMSNorm to queries and keys *after* projection but *before* RoPE. Most models don't do this at all. QK-norm stabilizes attention scores in deep models. The order matters: apply it before RoPE or you get wrong results. This is a Qwen3-specific detail that's easy to miss from a generic transformer diagram.

**Softmax in float32.** `dtype=torch.float32` in softmax is not optional. Softmax involves exponentiation, and in bfloat16 large attention scores can overflow to infinity, producing NaN attention weights that propagate through everything downstream. Casting just the softmax to float32 — then casting back — costs almost nothing and prevents this.

**`-torch.inf` for the causal mask.** The causal mask zeros future positions so each token only attends to past ones. Using `-torch.inf` rather than `-1e9` guarantees exact zeros after softmax. In bfloat16, `-1e9` might not be sufficiently negative — tiny residual attention can leak to future positions. `-torch.inf` makes softmax produce exactly 0. No floating-point surprises.

**KV head expansion after RoPE.** The `repeat_interleave` that expands 8 KV heads to 16 happens *after* applying RoPE. Expanding before RoPE and then rotating means you're rotating already-duplicated tensors — twice the compute for the same result.

---

## Putting it together

Each of the 28 transformer blocks follows the pre-norm pattern: normalize *before* each sub-layer, not after:

```python
class TransformerBlock(nn.Module):
    def forward(self, x, mask, cos, sin):
        x = x + self.att(self.norm1(x), mask, cos, sin)  # norm → attention → residual
        x = x + self.ff(self.norm2(x))                   # norm → FFN → residual
        return x
```

Pre-norm is the modern standard. The original "Attention Is All You Need" used post-norm, but pre-norm trains more stably at depth. With pre-norm, the residual connection carries the raw representation without normalization — gradient flow is cleaner.

The full model:

```python
class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim, dtype=cfg.dtype)
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype)

        cos, sin = compute_rope_params(cfg.head_dim, cfg.context_length, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

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

`register_buffer` with `persistent=False`: these tensors move with the model when you call `.to(device)` — they automatically land on the right device — but they're not saved in checkpoints, because they can always be recomputed from config. The RoPE tables and causal mask are constants, not parameters.

Pre-allocating the full causal mask and slicing it in `forward` avoids allocating a new tensor on every forward call. Minor optimization, correct practice.

---

## Loading real weights

We have an architecture. Now we need actual trained weights.

The weights live on HuggingFace as a safetensors file. Downloading is handled by `huggingface_hub`:

```python
def download_and_load_weights(model: Qwen3Model, repo_id: str = "Qwen/Qwen3-0.6B") -> None:
    local_dir = Path(repo_id).parts[-1]
    weights_path = hf_hub_download(
        repo_id=repo_id, filename="model.safetensors", local_dir=local_dir,
    )
    weights = load_file(weights_path)
    load_weights_into_qwen(model, weights)
    del weights  # free the raw dict — it's a full copy of all parameters
```

The `del weights` is not cosmetic. The raw weights dict holds a full copy of ~600M parameters. If you don't free it after loading, you're keeping two copies of the model in memory simultaneously. On a machine with 8 GB of RAM, that hurts.

The tricky part is `load_weights_into_qwen`. HuggingFace uses its own naming convention:

```
model.embed_tokens.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.up_proj.weight
...
```

We copy each tensor into the right parameter with explicit mappings:

```python
def load_weights_into_qwen(model, weights):
    def copy_(dst, src):
        with torch.no_grad():
            dst.copy_(src)

    copy_(model.tok_emb.weight, weights["model.embed_tokens.weight"])

    for i in range(model.cfg.n_layers):
        block = model.trf_blocks[i]
        p = f"model.layers.{i}"

        copy_(block.att.W_query.weight,  weights[f"{p}.self_attn.q_proj.weight"])
        copy_(block.att.W_key.weight,    weights[f"{p}.self_attn.k_proj.weight"])
        copy_(block.att.W_value.weight,  weights[f"{p}.self_attn.v_proj.weight"])
        copy_(block.att.out_proj.weight, weights[f"{p}.self_attn.o_proj.weight"])
        copy_(block.att.q_norm.scale,    weights[f"{p}.self_attn.q_norm.weight"])
        copy_(block.att.k_norm.scale,    weights[f"{p}.self_attn.k_norm.weight"])
        copy_(block.norm1.scale,         weights[f"{p}.input_layernorm.weight"])
        copy_(block.norm2.scale,         weights[f"{p}.post_attention_layernorm.weight"])
        copy_(block.ff.fc1.weight,       weights[f"{p}.mlp.gate_proj.weight"])
        copy_(block.ff.fc2.weight,       weights[f"{p}.mlp.up_proj.weight"])
        copy_(block.ff.fc3.weight,       weights[f"{p}.mlp.down_proj.weight"])

    copy_(model.final_norm.scale, weights["model.norm.weight"])

    if "lm_head.weight" in weights:
        copy_(model.out_head.weight, weights["lm_head.weight"])
    else:
        model.out_head.weight = model.tok_emb.weight
```

That last block — weight tying — caught me completely off guard. Qwen3-0.6B shares the output projection weights with the token embedding table. There is no `lm_head.weight` key in the safetensors file. If you create a separate `nn.Linear` for the output head and forget to detect this, your output head has randomly initialized weights. The model produces beautiful intermediate representations through all 28 layers and then destroys them with a random projection at the end. The output is total garbage. The hidden states are completely fine. I spent an embarrassing amount of time looking at the wrong thing before I found this.

---

## Tokenizer

Text in, token IDs out. Token IDs in, text out.

Qwen3 uses BPE, and the tokenizer lives in `tokenizer.json` on HuggingFace. We could use `AutoTokenizer` from `transformers`, but that would bring `transformers` into the runtime dependency stack. Instead, we use the lighter `tokenizers` library which can load `tokenizer.json` directly.

The one wrinkle: special tokens. Qwen3 has many — `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`, and more. The raw BPE tokenizer sometimes splits these into sub-pieces instead of treating them as atomic units. The fix: split the input text at special token boundaries first, handle those by direct ID lookup, BPE-encode everything else:

```python
_SPLIT_RE = re.compile("(" + "|".join(re.escape(t) for t in _SPECIAL_TOKENS) + ")")

def encode(self, text: str) -> list[int]:
    ids: list[int] = []
    for part in filter(None, _SPLIT_RE.split(text)):
        if part in self._special_to_id:
            ids.append(self._special_to_id[part])
        else:
            ids.extend(self._tok.encode(part).ids)
    return ids
```

One more thing: the EOS token for Qwen3 is `<|im_end|>`, not `<|endoftext|>`. Qwen3 uses the ChatML message format where `<|im_end|>` marks the end of an assistant turn. If you stop on `<|endoftext|>` instead, generation just keeps going. Depending on your `max_tokens` limit, you might not notice for a while — outputs just get progressively longer and never terminate naturally.

---

## The generation loop

Here's the complete autoregressive generation loop:

```python
@torch.no_grad()
def generate(model, input_ids, max_new_tokens, eos_token_id=None):
    model.eval()
    tokens = input_ids

    for _ in range(max_new_tokens):
        logits = model(tokens)                                   # [1, seq_len, vocab]
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True) # [1, 1]
        tokens = torch.cat([tokens, next_token], dim=1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return tokens
```

That's it. That's the whole thing.

On every step: full forward pass over the entire sequence, look at only the last position's logits, pick the highest-probability token, append it to the sequence, repeat.

This is "naive" in a precise computational sense. On step 100, we're recomputing keys and values for positions 1 through 99 even though they haven't changed. Every additional token in context makes every future step proportionally more expensive. This is O(n²) — and we'll see exactly how bad that is in the benchmarks section.

`argmax` always picks the single most probable token. Deterministic, correct, produces reasonable output for factual queries. But it has a failure mode: it always picks the most "expected" continuation. For open-ended generation, this tends toward safe, repetitive output. We address this in Tutorial 2 with sampling.

---

## How do we know it's correct?

This is the most important section in the tutorial.

Without a reference implementation to test against, you have no idea whether your model is actually correct. The model might output plausible-looking text while getting every attention score slightly wrong. Logits can look reasonable even when the computations have bugs. Visual inspection is hopeless. You need a ground truth.

The approach: load both our model and HuggingFace's `transformers` version with identical weights, feed them the same input, compare outputs:

```python
def test_logits_argmax_matches_single_forward(our_model, hf_model, input_ids):
    our_logits = our_model(input_ids)
    hf_logits  = hf_model(input_ids).logits

    assert torch.equal(our_logits.argmax(dim=-1), hf_logits.argmax(dim=-1))

    diff = (our_logits.float() - hf_logits.float()).abs()
    assert diff.max().item() < 1.0
```

Why compare argmax rather than exact logit values? Because bfloat16 arithmetic is not associative — the order of floating-point operations affects the result. Our implementation and HuggingFace's perform mathematically identical operations in slightly different orders (different code paths, different loop ordering). Raw logit values will differ by small amounts, typically under 0.5. But the *argmax* — which token has the highest probability — should be identical. If it's not, that's a real correctness bug.

We also run a generation comparison: 20-token greedy generation, token-by-token equality. Both tests together cover single-forward accuracy and sequential accumulation.

**The hidden bug that opened this article.** Loading the HuggingFace model for comparison requires:

```python
hf_model = AutoModelForCausalLM.from_pretrained(
    REPO_ID,
    dtype=torch.bfloat16,
    attn_implementation="eager",  # ← this is critical
)
```

By default, HuggingFace uses PyTorch's fused `scaled_dot_product_attention` kernel, which uses different internal precision than explicit matmul + softmax. With the default setting, the argmax mismatches in ~2–3% of positions. I spent the better part of a day checking my RoPE implementation, normalization math, weight loading. All correct. The actual fix was one argument, forcing HuggingFace to use the same explicit attention code path we use. One keyword argument. Found after far too long.

Fixtures use `scope="module"` — models load once per test file. Loading a 600M-parameter model takes several seconds; you don't want to pay that per test function.

---

## All the things that went wrong

Let me consolidate everything. If your implementation doesn't match the reference, start here.

| Issue | Symptom | Fix |
|---|---|---|
| Missing float32 upcast in RMSNorm | Logits diverge slightly | Cast to float32 for variance computation, cast back |
| Wrong `rope_base` (10k instead of 1M) | Any sequence mismatch; garbage on longer sequences | Use value from `config.json`: 1,000,000 |
| Interleaved vs split-half RoPE | Completely wrong attention patterns | Use split-half: `cat([-x2, x1], dim=-1)` |
| Swapped `gate_proj` / `up_proj` | Runs fine; nonsense output | `fc1=gate_proj`, `fc2=up_proj`, `fc3=down_proj` |
| Missing QK normalization | Logit mismatch; instability on long sequences | Apply RMSNorm to Q and K after projection, before RoPE |
| Missing weight tying | Coherent hidden states → garbage output | Point `out_head.weight` to `tok_emb.weight` |
| Wrong EOS token | Generation never stops | `<|im_end|>` is EOS, not `<|endoftext|>` |
| HF using SDPA by default in tests | Argmax mismatches ~2–3% of positions | `attn_implementation="eager"` |
| Softmax in bfloat16 | Overflow to inf, NaN attention weights | `torch.softmax(..., dtype=torch.float32)` |
| `-1e9` instead of `-torch.inf` for mask | Subtle attention leakage to future tokens | Use `-torch.inf` |
| Not freeing weight dict | OOM on low-RAM machines | `del weights` after `load_weights_into_qwen` |

I hit all eleven. Having a reference implementation to test against means none of them can hide — they all surface as test failures. Write the tests first.

---

## How fast is it?

Let's see the damage.

The benchmark harness in `src/pumpference/benchmark.py` measures prefill throughput, decode throughput, time-to-first-token (TTFT), per-token decode latency (p50/p90/p99), and peak memory. Four prompt presets at increasing context lengths: `xs` (~30 tokens), `short` (~115), `medium` (~218), `long` (~372). All numbers from Apple M3 Pro CPU, bfloat16, 100 generated tokens per preset.

| Preset | Prompt tok | Prefill TPS | TTFT | Decode TPS | Mean latency | P50 | P99 | Peak memory |
|--------|:---:|---:|---:|---:|---:|---:|---:|---:|
| xs     |  30  | 102.1 tok/s |  294 ms | 1.3 tok/s |  777 ms/tok |  762 ms |  1750 ms | 4122 MB |
| short  | 115  | 105.2 tok/s | 1093 ms | 0.6 tok/s | 1749 ms/tok | 1696 ms |  2784 ms | 4013 MB |
| medium | 218  |  90.9 tok/s | 2398 ms | 0.3 tok/s | 3323 ms/tok | 3252 ms |  6003 ms | 3738 MB |
| long   | 372  |  70.7 tok/s | 5264 ms | 0.2 tok/s | 6321 ms/tok | 6284 ms |  7612 ms | 3783 MB |

![Baseline benchmark — decode throughput, per-token latency, and TTFT across context lengths](assets/baseline-benchmark.png)

**Prefill is reasonable.** From 30 to 372 tokens (12.4×), TTFT goes from 294 ms to 5264 ms (17.9×). Slightly superlinear — attention is O(n²) even for prefill — but prefill processes all tokens in parallel, so memory bandwidth dominates at these sizes, not sequence-length iteration.

**Decode is catastrophically O(n²).** Per-token latency goes from 777 ms at a 30-token context to 6321 ms at 372 tokens. That's an **8× slowdown for a 12× increase in context length**. The signature of quadratic scaling. At 0.2 tok/s on a 372-token prompt, this is unusable for anything interactive.

**The P99 spike on `xs` is the most revealing number in the table.** P50 decode latency for `xs` is 762 ms. P99 is 1750 ms — a 2.3× spread within a single 100-token run. This isn't noise. It's the O(n²) cost structure in plain sight: the first decode step processes 31 tokens (fast), the last processes 129 tokens (slow). The median step is somewhere around 80 tokens. The distribution isn't a distribution — it's a slope. For `long`, the spread narrows (P50=6284, P99=7612, 1.2×) because all decode steps start from an already-large context and the proportional growth per additional step is smaller.

**Memory is ~4 GB everywhere.** Model weights are ~1.2 GB in bfloat16. The rest is PyTorch's allocator, Python runtime, and intermediate activation buffers. It barely varies with context length because we're not caching anything. When we add a KV-cache in Tutorial 4, this number will go up. That's the explicit trade: more memory for dramatically faster decode.

These are our baseline numbers. Every subsequent optimization tutorial reports against this table.

---

## What's next

The model works and it's slow. Before worrying about the slowness, let's make it more interesting to talk to.

**Tutorial 2**: Sampling. Right now `argmax` always picks the single most probable token — correct, deterministic, and tends toward safe, predictable output. Temperature, top-k, and top-p sampling add controlled randomness, letting you trade some probability mass for diversity. Getting the implementation right, especially top-p, is more interesting than it looks.

---

*This is part of the Pumpference tutorial series. Source code: [github.com/legchikov/pumpference](https://github.com/legchikov/pumpference).*

*Found an error? Open an issue.*

---

## References

- Sebastian Raschka's from-scratch Qwen3: [notebook](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/standalone-qwen3.ipynb) and [blog post](https://sebastianraschka.com/blog/2025/qwen3-from-scratch.html).
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388).
