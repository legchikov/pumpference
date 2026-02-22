# Architecture

## Directory structure

```
pumpference/
├── pyproject.toml              # Project config, dependencies, pytest settings
├── uv.lock                     # Locked dependency versions
├── Makefile                    # lint / format / test / bench shortcuts
├── src/
│   └── pumpference/
│       ├── __init__.py         # Public API: Qwen3Model, QWEN3_0_6B_CONFIG, generate
│       ├── __main__.py         # CLI entry point (argparse, device detection, load, generate)
│       ├── model.py            # All architecture components + config + weight loading (~340 lines)
│       ├── generate.py         # Greedy + sampling generation loop (~110 lines)
│       ├── tokenizer.py        # Qwen3Tokenizer wrapper over HF tokenizers lib (~80 lines)
│       └── benchmark.py        # Benchmark harness: TPS, TTFT, memory, JSON output (~440 lines)
├── benchmarks/                 # Auto-created; JSON results from benchmark runs (gitignored)
├── tests/
│   ├── conftest.py             # sys.path setup for src layout
│   └── test_model.py           # Comparison tests vs HuggingFace transformers
└── tutorials/
    ├── 01-generation.md        # Tutorial 1: building naive inference from scratch (full walkthrough)
    ├── 02-sampling.md          # Tutorial 2: temperature, top-k, top-p decoding (dev-log)
    └── 03-benchmarking.md      # Tutorial 3: benchmarking harness (dev-log)
```

## Model architecture (Qwen3-0.6B)

Decoder-only transformer with the following components in `model.py`:

| Class / Function | Purpose |
|---|---|
| `Qwen3Config` | Dataclass holding all hyperparameters (vocab_size, n_heads, n_layers, etc.) |
| `QWEN3_0_6B_CONFIG` | Singleton config instance for Qwen3-0.6B |
| `RMSNorm` | Root Mean Square normalization (upcasts to float32 internally) |
| `compute_rope_params()` | Pre-computes cos/sin tables for Rotary Positional Embeddings |
| `apply_rope()` | Applies RoPE rotation to Q/K tensors (split-half convention) |
| `FeedForward` | SwiGLU FFN: silu(fc1(x)) * fc2(x) → fc3 (three linear layers) |
| `GroupedQueryAttention` | GQA with 16 Q-heads, 8 KV-heads, optional QK-norm, causal masking |
| `TransformerBlock` | Pre-norm block: norm→attn→residual, norm→FFN→residual |
| `Qwen3Model` | Full model: token embedding → 28 transformer blocks → final norm → linear head |
| `load_weights_into_qwen()` | Maps HuggingFace weight names to our parameter names |
| `download_and_load_weights()` | Downloads safetensors from HF Hub and calls load function |

## Data flow

```
input_ids [batch, seq_len]
  → tok_emb (Embedding)
  → 28× TransformerBlock (each: RMSNorm → GQA → residual → RMSNorm → SwiGLU → residual)
  → final_norm (RMSNorm)
  → out_head (Linear → vocab_size)
  → logits [batch, seq_len, 151936]
```

Generation loop (in `generate.py`): feeds full sequence on every step (no KV-cache). `sample_next_token` selects the next token — greedy argmax when `temperature=0`, otherwise applies temperature scaling → optional top-k filter → optional top-p (nucleus) filter → `torch.multinomial`.

## Key design decisions

1. **Single `model.py` file**: all architecture in one file (~340 lines) for readability — premature modularization was tried and reverted
2. **No bias terms**: all `nn.Linear` layers use `bias=False` (matches modern LLM convention)
3. **bfloat16 by default**: matches the distribution format, avoids precision loss from casting
4. **Pre-allocated buffers**: RoPE cos/sin tables and causal mask are `register_buffer(persistent=False)` — move with model to device, not saved in checkpoints
5. **Weight tying**: output head shares weights with token embedding (no separate lm_head in Qwen3-0.6B)
6. **`transformers` is dev-only**: the HF library is used only in tests for comparison, not at runtime

## Critical implementation details

- RMSNorm upcasts to float32 for numerical stability
- RoPE uses split-half convention (not interleaved) with theta_base=1,000,000
- Softmax in attention is computed in float32 to prevent overflow
- Causal mask uses `-torch.inf` (not `-1e9`) for exact zeros after softmax
- QK normalization (RMSNorm on Q and K) applied after projection, before RoPE
- KV heads expanded via `repeat_interleave` after RoPE application
- FFN weight mapping: fc1=gate_proj, fc2=up_proj, fc3=down_proj
- Tokenizer handles special tokens via regex splitting before BPE encoding
- EOS token is `<|im_end|>` (not `<|endoftext|>`)

## Test architecture

Tests in `test_model.py` use `scope="module"` fixtures (models loaded once). Two tests:
1. `test_logits_argmax_matches_single_forward` — compares argmax at every position + checks max logit diff < 1.0
2. `test_greedy_generation_matches_hf` — 20-token greedy generation, token-by-token equality

`test_sampling.py` — 17 unit tests for sampling strategies (no model load required): greedy equivalence, reproducibility, top-k / top-p correctness, composability, output shape.

HF model must use `attn_implementation="eager"` for numerical consistency with manual attention.
