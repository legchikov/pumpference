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

Flash Attention (chapter 7) is complete. 20 tests pass.

## What was added (Flash Attention)

- `flash_attention(q, k, v, is_causal, block_size)` in `model.py`: tiled online-softmax, O(n) memory, float32 accumulation, causal tile skipping
- `use_flash_attn: bool = False` field on `Qwen3Config`; wired into `GroupedQueryAttention` and `TransformerBlock`
- Flash is auto-bypassed when `q_len == 1` (KV-cache decode step); eager used instead
- `--flash-attn` flag added to `__main__.py` and `benchmark.py`; `dataclasses.replace()` pattern for safe config copy
- `flash_attn: bool` field added to `BenchmarkResult`; shown in formatted output and JSON
- Two new tests in `test_model.py`: argmax match for prompt (ground truth correctness); logit values close (< 1.0 diff) for 5 growing-sequence steps

## Precision note

Flash accumulates `p @ V` in float32 (higher precision than eager's bfloat16 V matmul). Both are mathematically correct; they differ by ≤1 ULP in bfloat16, which can flip argmax when two logits are extremely close. Token-equality tests are not appropriate for this reason.

## Next steps (from roadmap)

1. Speculative decoding — reduce total decode steps (chapter 8)
2. Fused int8 kernels — bypass fp16 temporary for actual quantization speedup

## Known issues

None.
