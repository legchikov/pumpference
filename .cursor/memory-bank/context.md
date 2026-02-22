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

Sampling decoding is complete. Tutorial 2 written.

## What has been added (sampling)

- `sample_next_token(logits, temperature, top_k, top_p)` function in `generate.py`
  - `temperature=0` → greedy argmax (backward-compatible default)
  - top-k: masks all but the k highest-logit tokens before sampling
  - top-p (nucleus): cumulative-probability threshold with shift-right logic; always keeps position 0
  - Final softmax in float32; `torch.multinomial` for stochastic draw
- `generate()` updated to accept and pass through `temperature`, `top_k`, `top_p`
- CLI (`__main__.py`) exposes `--temperature`, `--top-k`, `--top-p` flags
- `sample_next_token` exported from `__init__.py`
- 17 unit tests in `tests/test_sampling.py` (no model load required): greedy equivalence, reproducibility, top-k/top-p correctness, composability, output shape
- Tutorial 2 written: `tutorials/02-sampling.md`

## Tutorial format decision

Tutorials 2+ are **dev-log articles**, not step-by-step code walkthroughs. Code always lives in `src/pumpference/` on `main`. Tutorials cover: current state + baseline metric, motivation, key decisions (why, not what), gotchas table, results table with real numbers, and optional focused code snippet. No "before/after" diffs, no git tags. Full format documented in `AGENTS.md` and `.cursor/plans/tutorial_writing_format_dc7867c5.plan.md`.

## Next steps (from roadmap)

1. **KV-cache**: make generation O(n) instead of O(n²) — biggest single speedup
2. Profiling and optimization deep-dive

## Known issues

None.
