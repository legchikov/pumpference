# Context

## Current state

**Profiling and quantization are complete.** Tutorial 5 (profiling) and Tutorial 6 (quantization) are written and committed. 18 tests pass. The key finding: naive dequantize-on-forward quantization is 3–7× slower than bfloat16 on CPU because it materialises a full fp16 weight tensor before each matmul, adding bandwidth rather than removing it. The storage reduction is real but modest due to the large pre-allocated causal mask.

## What has been built

- Full Qwen3-0.6B architecture: RMSNorm, RoPE, SwiGLU FFN, Grouped-Query Attention, Transformer Block, full model
- Weight loading from HuggingFace safetensors (with weight-tying detection)
- Custom tokenizer wrapper (uses `tokenizers` library, handles special tokens via regex splitting)
- Greedy autoregressive generation loop with KV-cache (O(n) per step)
- Sampling: temperature, top-k, top-p (nucleus) decoding
- CLI entry point with device auto-detection (CUDA / MPS / CPU)
- Test suite comparing single-forward logits argmax and multi-step greedy generation against HuggingFace
- Benchmark harness (`benchmark.py`): prefill/decode TPS, TTFT, peak memory, per-token latency (p50/p90/p99), JSON output; supports `--quantize none|int8|int4`
- Benchmark preset aliases: `xs`≈30tok, `short`≈115tok, `medium`≈218tok, `long`≈373tok
- **Profile harness** (`profile.py`): hook-based per-layer coarse timing + `torch.profiler` operator breakdown; `make profile`
- **Weight-only quantization** (`quantize.py`): `Int8Linear` (W8A16, per-channel), `Int4Linear` (W4A16, group_size=128), `quantize_model(model, mode, group_size)`; exported from `__init__.py`

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

## Current work focus

Completed tutorials 05 and 06. Ready for next step.

## Next steps (from roadmap)

1. Fused int8 kernels — bypass the fp16 temporary tensor to realize the bandwidth savings from quantization
2. Calibration-based quantization — GPTQ or AWQ for better int4 quality
3. Continuous batching / serving optimizations

## Known issues

None.
