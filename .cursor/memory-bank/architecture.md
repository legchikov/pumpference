# Architecture

## Directory structure

```
pumpference/
├── pyproject.toml              # Project config, dependencies, pytest settings
├── uv.lock                     # Locked dependency versions
├── Makefile                    # lint / format / test / bench / profile shortcuts
├── src/
│   └── pumpference/
│       ├── __init__.py         # Public API: Qwen3Model, QWEN3_0_6B_CONFIG, generate, quantize_model
│       ├── __main__.py         # CLI entry point (argparse, device detection, load, generate, --quantize)
│       ├── model.py            # All architecture components + config + weight loading (~580 lines, incl. flash_attention)
│       ├── generate.py         # Greedy + sampling generation loop (~110 lines)
│       ├── tokenizer.py        # Qwen3Tokenizer wrapper over HF tokenizers lib (~80 lines)
│       ├── benchmark.py        # Benchmark harness: TPS, TTFT, memory, JSON output; --quantize flag
│       ├── profile.py          # Profiling harness: hook-based per-layer timing + torch.profiler
│       └── quantize.py         # Weight-only quantization: Int8Linear, Int4Linear, quantize_model
├── benchmarks/                 # Auto-created; JSON results from benchmark runs (gitignored)
├── tests/
│   ├── conftest.py             # sys.path setup for src layout
│   ├── test_model.py           # Comparison tests vs HuggingFace transformers
│   ├── test_sampling.py        # Sampling strategy unit tests (no model)
│   └── test_quantize.py        # Quantization correctness + structural tests
└── tutorials/
    ├── 01-generation.md        # Tutorial 1: building naive inference from scratch (full walkthrough)
    ├── 02-sampling.md          # Tutorial 2: temperature, top-k, top-p decoding (dev-log)
    ├── 03-benchmarking.md      # Tutorial 3: benchmarking harness (dev-log)
    ├── 04-kv-cache.md          # Tutorial 4: KV-cache (dev-log)
    ├── 05-profiling.md         # Tutorial 5: profiling decode step (dev-log)
    ├── 06-quantization.md      # Tutorial 6: weight-only RTN quantization (dev-log)
    ├── 06b-awq-quantization.md # Tutorial 6b: AWQ calibration-based quantization (dev-log)
    └── 07-flash-attention.md   # Tutorial 7: Flash Attention — tiled O(n) attention (dev-log)
```

## Model architecture (Qwen3-0.6B)

Decoder-only transformer with the following components in `model.py`:

| Class / Function | Purpose |
|---|---|
| `Qwen3Config` | Dataclass holding all hyperparameters; includes `use_flash_attn: bool = False` |
| `QWEN3_0_6B_CONFIG` | Singleton config instance for Qwen3-0.6B |
| `KVCache` | Per-layer K/V cache: `update(layer_idx, k, v)` appends and returns accumulated tensors; `seq_len` property; `reset()` |
| `RMSNorm` | Root Mean Square normalization (upcasts to float32 internally) |
| `compute_rope_params()` | Pre-computes cos/sin tables for Rotary Positional Embeddings |
| `apply_rope()` | Applies RoPE rotation to Q/K tensors (split-half); accepts pre-sliced cos/sin from caller |
| `FeedForward` | SwiGLU FFN: silu(fc1(x)) * fc2(x) → fc3 (three linear layers) |
| `flash_attention()` | Tiled attention with online softmax; O(n) memory; is_causal + block_size args; auto-bypassed when q_len=1 |
| `GroupedQueryAttention` | GQA with 16 Q-heads, 8 KV-heads, optional QK-norm; accepts kv_cache + layer_idx; use_flash_attn flag |
| `TransformerBlock` | Pre-norm block: norm→attn→residual, norm→FFN→residual; threads kv_cache through |
| `Qwen3Model` | Full model: embedding → 28 blocks → final norm → linear head; accepts kv_cache |
| `load_weights_into_qwen()` | Maps HuggingFace weight names to our parameter names |
| `download_and_load_weights()` | Downloads safetensors from HF Hub and calls load function |

**Quantization (`quantize.py`):**

| Class / Function | Purpose |
|---|---|
| `quantize_per_channel_absmax()` | Symmetric per-row int8 quantization; returns `(int8_weight, scale)` |
| `quantize_per_group()` | Symmetric group-wise int4 quantization; packs two values per byte; returns `(packed_uint8, scales)` |
| `unpack_int4()` | Unpacks and dequantizes int4 back to float32 |
| `Int8Linear` | Drop-in for `nn.Linear`; stores int8 weight + float32 scale; dequantizes on forward |
| `Int4Linear` | Drop-in for `nn.Linear`; stores packed uint8 weight + float32 group scales; unpacks on forward |
| `quantize_model()` | Replaces all `nn.Linear` in model in-place; accepts `mode="int8"\|"int4"\|"awq_int8"\|"awq_int4"`; AWQ modes require `calibration_ids` |
| `calibrate_awq()` | AWQ calibration: collects activation stats, grid-searches optimal per-channel scale α, absorbs scale into RMSNorm; model stays bfloat16 |
| `_collect_norm_activation_stats()` | Hook-based: records mean \|activation\| at each block's norm1/norm2 output |
| `_search_optimal_scale()` | Grid search over α ∈ [0,1]; minimises activation-weighted quant reconstruction error |
| `_rtn_dequant_int8/int4()` | Quantize+dequantize round-trip helpers used in the grid search |

**Profiling (`profile.py`):**

| Function | Purpose |
|---|---|
| `_coarse_profile()` | Registers forward hooks on each module; times 10 decode steps; returns mean ms per module |
| `_fine_profile()` | Runs `torch.profiler.profile` over 5 decode steps; prints top-20 operators; optionally exports Chrome trace |

## Data flow

```
input_ids [batch, seq_len]
  → tok_emb (Embedding)
  → 28× TransformerBlock (each: RMSNorm → GQA → residual → RMSNorm → SwiGLU → residual)
  → final_norm (RMSNorm)
  → out_head (Linear → vocab_size)
  → logits [batch, seq_len, 151936]
```

Generation loop (in `generate.py`): two-phase when `use_cache=True` (default): phase 1 prefills the full prompt and fills the `KVCache`, phase 2 feeds one token per step. `use_cache=False` retains the original full-sequence recompute path. `sample_next_token` selects the next token — greedy argmax when `temperature=0`, otherwise applies temperature scaling → optional top-k filter → optional top-p (nucleus) filter → `torch.multinomial`.

## Key design decisions

1. **Single `model.py` file**: all architecture in one file (~580 lines) for readability — premature modularization was tried and reverted
2. **No bias terms**: all `nn.Linear` layers use `bias=False` (matches modern LLM convention)
3. **bfloat16 by default**: matches the distribution format, avoids precision loss from casting
4. **Pre-allocated buffers**: RoPE cos/sin tables and causal mask are `register_buffer(persistent=False)` — move with model to device, not saved in checkpoints
5. **Weight tying**: output head shares weights with token embedding (no separate lm_head in Qwen3-0.6B)
6. **`transformers` is dev-only**: the HF library is used only in tests for comparison, not at runtime
7. **Flash attention is opt-in via config flag**: `Qwen3Config.use_flash_attn=False` by default; switched via `dataclasses.replace()` to avoid mutating the module-level singleton; auto-bypassed for `q_len=1` decode steps

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
- Flash attention accumulates V-weighted sum in float32 (more precise than eager's bfloat16 V matmul); can produce different argmax than eager on borderline logit pairs — correct behaviour, not a bug

## Test architecture

Tests in `test_model.py` use `scope="module"` fixtures (models loaded once). Seven tests:
1. `test_logits_argmax_matches_single_forward` — compares argmax at every position + checks max logit diff < 1.0
2. `test_greedy_generation_matches_hf` — 20-token greedy generation, token-by-token equality
3. `test_kv_cache_prefill_logits_match` — prefill with empty cache produces identical logits to uncached forward
4. `test_kv_cache_generation_matches_no_cache` — cached generation produces bit-identical tokens to naive path
5. `test_kv_cache_generation_matches_hf` — cached generation matches HuggingFace
6. `test_flash_attention_logits_match_eager` — flash argmax identical to eager at every position (ground truth correctness proof)
7. `test_flash_attention_logits_close_across_steps` — flash and eager logit values within 1.0 for 5 growing-sequence steps; flash computes in float32 (more precise than eager's bfloat16 V matmul), so token-equality is not the right bar

`test_sampling.py` — unit tests for sampling strategies (no model load required).

`test_quantize.py` — tests for quantization: RTN primitive round-trip, output shapes, structural (no nn.Linear remaining), memory reduction, argmax agreement (RTN int8 ≥ 85%, int4 ≥ 70%), generation smoke tests; plus AWQ tests: argmax quality (awq_int8 ≥ 88%, awq_int4 ≥ 75%), AWQ-vs-RTN comparison, generation smoke tests, convenience API, and error on missing calibration_ids.

Total: 20 tests. HF model must use `attn_implementation="eager"` for numerical consistency with manual attention.
