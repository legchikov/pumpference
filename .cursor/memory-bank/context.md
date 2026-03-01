# Context

## Current state

**AWQ quantization is complete.** Tutorials 5 (profiling), 6 (RTN quantization), and 6b (AWQ quantization) are written. 25 tests pass. AWQ improves int4 argmax agreement from 80% → 83.3% and int8 from 90% → 93.3% (+3.3pp each) using a 2-sequence calibration dataset and 20-step grid search.

## What has been built

- Full Qwen3-0.6B architecture: RMSNorm, RoPE, SwiGLU FFN, Grouped-Query Attention, Transformer Block, full model
- Weight loading from HuggingFace safetensors (with weight-tying detection)
- Custom tokenizer wrapper (uses `tokenizers` library, handles special tokens via regex splitting)
- Greedy autoregressive generation loop with KV-cache (O(n) per step)
- Sampling: temperature, top-k, top-p (nucleus) decoding
- CLI entry point with device auto-detection (CUDA / MPS / CPU)
- Test suite comparing single-forward logits argmax and multi-step greedy generation against HuggingFace
- Benchmark harness (`benchmark.py`): prefill/decode TPS, TTFT, peak memory, per-token latency (p50/p90/p99), JSON output; supports `--quantize none|int8|int4|awq_int8|awq_int4`
- Benchmark preset aliases: `xs`≈30tok, `short`≈115tok, `medium`≈218tok, `long`≈373tok
- **Profile harness** (`profile.py`): hook-based per-layer coarse timing + `torch.profiler` operator breakdown; `make profile`
- **Weight-only quantization** (`quantize.py`): `Int8Linear` (W8A16, per-channel), `Int4Linear` (W4A16, group_size=128), `quantize_model(model, mode, group_size)`; exported from `__init__.py`
- **AWQ calibration** (`quantize.py`): `calibrate_awq(model, calibration_ids, mode, group_size, n_grid)` — per-channel scale search (α grid), absorption into RMSNorm scales; `quantize_model(mode="awq_int8"|"awq_int4", calibration_ids=...)` convenience wrapper

## Baseline numbers (KV-cached decode, CPU bfloat16, 100 tokens generated)

| Preset | Prompt tok | Decode TPS | Mean latency |
|--------|-----------|------------|--------------|
| xs (30 tok) | 30 | 12.4 | 80.5 ms/tok |
| short (115 tok) | 115 | 12.6 | ~79 ms/tok |
| medium (218 tok) | 218 | 8.5 | ~118 ms/tok |
| long (372 tok) | 372 | 9.2 | ~109 ms/tok |

## Profiling results (CPU, xs preset, decode step)

- 28 transformer blocks: 92% of total decode time
- `aten::mm` + `aten::bmm`: 65–70% of self-CPU time
- `out_head`: 4.8% (vocab-size projection, 311 MB weights)
- `aten::cat` (KV cache concatenation): 2.8%

## Quantization results (CPU, xs preset, dequantize-on-forward)

| Mode | Decode TPS | vs bfloat16 | Model bytes |
|------|-----------|-------------|-------------|
| bfloat16 | 12.4 | — | 3,223 MB |
| int8 | 3.7 | 3.4× slower | 2,629 MB |
| int4 | 1.8 | 6.9× slower | 2,348 MB |

Key finding: dequantize-on-forward triples memory bandwidth (load int8 → write fp16 temp → read fp16 for matmul). Requires fused kernels for actual speedup.

## AWQ quality results (30-token prompt, argmax agreement with bfloat16)

| Mode | Argmax agreement | vs RTN |
|------|-----------------|--------|
| RTN int8 | 90.0% | — |
| AWQ int8 | 93.3% | +3.3pp |
| RTN int4 | 80.0% | — |
| AWQ int4 | 83.3% | +3.3pp |

## Current work focus

Speculative Decoding (chapter 8) is complete. 26 tests pass.

## What was added (Speculative Decoding)

- `QWEN3_1_7B_CONFIG` in `model.py`: config for the target model (emb_dim=2048, hidden_dim=6144, same n_layers/heads/vocab as 0.6B)
- `KVCache.truncate(new_seq_len)` in `model.py`: slices cached tensors along dim=2; used for cache rollback on rejection
- Causal mask fix in `Qwen3Model.forward()`: unified `mask = zeros(q_len, kv_len)` + `mask[:, past_len:] = causal_mask[:q_len, :q_len]` when `q_len > 1`; handles prefill, single-token decode, and multi-token speculative verification with one code path
- Sharded weight loading in `download_and_load_weights`: tries single `model.safetensors` first, falls back to reading `model.safetensors.index.json` and downloading each shard (needed for Qwen3-1.7B's 2 shards)
- `SpeculativeStats` dataclass and `speculative_generate()` in `generate.py`:
  - Greedy acceptance (argmax comparison) and sampling (rejection sampling that provably preserves target distribution)
  - KV-cache invariant: both caches have `seq_len = P + n_gen - 1`; draft loop feeds `last_token` as first input each round
  - Full acceptance: feed d_K to draft to sync caches; bonus token from target position K
  - Partial acceptance: truncate both caches to `cache_len_before + 1 + n_accepted`; corrected token fed in next round's draft loop
- `_get_probs()` helper: converts logits to probability distribution respecting temperature/top-k/top-p (used for rejection sampling)
- `--speculative` and `--draft-k` flags in `__main__.py` and `benchmark.py`
- `speculative: bool`, `draft_k: int`, `acceptance_rate: float` fields in `BenchmarkResult`
- `timed_speculative_generate()` in `benchmark.py`
- 6 new tests in `test_speculative.py`: cache truncate (shape + values), multi-token mask correctness, greedy correctness (draft==target → 100% acceptance), sampling smoke, stats sanity
- Tutorial `tutorials/08-speculative-decoding.md` in dev-log format

## Key bug found and fixed

Feeding first_token to draft in init then feeding last_token again in the draft loop caused double-counting at wrong positions → <100% acceptance even with draft==target. The fix: don't advance draft_cache past the prompt during init; the draft loop handles last_token itself.

## Next steps (from roadmap)

1. Fused int8 kernels — bypass fp16 temporary for actual quantization speedup
2. Chapter 9: Continuous Batching + Paged Attention

## Known issues

None.
