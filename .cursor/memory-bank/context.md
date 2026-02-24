# Context

## Current state

The **benchmark harness is complete**. The naive inference is verified correct against HuggingFace. Benchmark preset aliases (`xs`, `short`, `medium`, `long`) were added to replace bare token counts.

## What has been built

- Full Qwen3-0.6B architecture: RMSNorm, RoPE, SwiGLU FFN, Grouped-Query Attention, Transformer Block, full model
- Weight loading from HuggingFace safetensors (with weight-tying detection)
- Custom tokenizer wrapper (uses `tokenizers` library, handles special tokens via regex splitting)
- Greedy autoregressive generation loop (no KV-cache)
- CLI entry point with device auto-detection (CUDA / MPS / CPU)
- Test suite comparing single-forward logits argmax and multi-step greedy generation against HuggingFace
- Benchmark harness (`benchmark.py`): prefill/decode TPS, TTFT, peak memory, per-token latency (p50/p90/p99), JSON output
- Benchmark preset aliases: `xs`≈30tok, `short`≈115tok, `medium`≈218tok, `long`≈373tok — used via `make bench PRESET=short`
- Tutorial 1 written: `tutorials/01-generation.md` — comprehensive walkthrough of the entire implementation, **including baseline benchmark results** (section 17) measured on Apple M3 Pro CPU

## Baseline numbers (naive inference, CPU bfloat16, 100 tokens generated)

| Preset | Prompt tok | Prefill TPS | TTFT | Decode TPS | Mean latency | Peak memory |
|--------|-----------|-------------|------|------------|--------------|-------------|
| xs (30 tok) | 30 | 102 tok/s | 294 ms | 1.3 tok/s | 777 ms/tok | 4122 MB |
| short (115 tok) | 115 | 105 tok/s | 1093 ms | 0.6 tok/s | 1749 ms/tok | 4013 MB |
| medium (218 tok) | 218 | 91 tok/s | 2398 ms | 0.3 tok/s | 3323 ms/tok | 3738 MB |
| long (372 tok) | 372 | 71 tok/s | 5264 ms | 0.2 tok/s | 6321 ms/tok | 3783 MB |

## Current work focus

KV-cache is complete. Tutorial 4 written.

## What has been added (KV-cache)

- `KVCache` class in `model.py`: per-layer K/V storage, `update(layer_idx, keys, values)` → appends and returns full accumulated tensors, `reset()`, `seq_len` property
- `apply_rope` simplified: internal `cos[:seq_len]` slicing removed; caller pre-slices to the correct position window
- `GroupedQueryAttention.forward()` extended: accepts `kv_cache` and `layer_idx`, calls `kv_cache.update()` after RoPE and QK-norm, before GQA `repeat_interleave` expansion
- `TransformerBlock.forward()` threads `kv_cache` and `layer_idx` down to attention
- `Qwen3Model.forward()` extended: accepts `kv_cache`, computes `past_len = kv_cache.seq_len`, pre-slices RoPE tables to `[past_len, past_len+q_len)`, builds all-False decode mask when `past_len > 0`
- `generate()` extended: `use_cache=True` default runs two-phase loop (prefill full prompt → decode single token per step); `use_cache=False` retains naive path for comparison
- `benchmark.py` `timed_generate()` updated to use the two-phase cached approach
- `KVCache` exported from `__init__.py`
- 3 new tests in `test_model.py`: prefill logits identity, cached-vs-naive token identity, cached-vs-HF token identity
- Tutorial 4 written: `tutorials/04-kv-cache.md`

## KV-cache benchmark results (CPU bfloat16, 100 tokens generated)

| Preset | Prompt tok | Decode TPS (naive) | Decode TPS (cached) | Speedup |
|--------|-----------|--------------------|--------------------|---------|
| xs (30 tok) | 30 | 1.3 | 12.0 | 9.2× |
| short (115 tok) | 115 | 0.6 | 12.6 | 21× |
| medium (218 tok) | 218 | 0.3 | 8.5 | 28× |
| long (372 tok) | 372 | 0.2 | 9.2 | 46× |

## Next steps (from roadmap)

1. Profiling and optimization deep-dive

## Known issues

None.
