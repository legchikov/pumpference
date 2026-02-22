## [2026.2.22] — 2026-02-22

### Changes
- Full Qwen3-0.6B decoder-only transformer implemented from scratch in plain PyTorch (`model.py`): RMSNorm, RoPE (split-half, θ=1 000 000), SwiGLU FFN, Grouped-Query Attention (16 Q-heads / 8 KV-heads), QK-norm, causal masking
- Weight loading from HuggingFace safetensors with automatic weight-tying detection (`load_weights_into_qwen`, `download_and_load_weights`)
- Greedy autoregressive generation loop — full-sequence re-computation per step, no KV-cache (`generate.py`)
- Custom `Qwen3Tokenizer` wrapper over the `tokenizers` library with special-token regex splitting (`tokenizer.py`)
- CLI entry point with automatic device selection (CUDA / MPS / CPU) (`__main__.py`)
- Public API surface: `Qwen3Model`, `QWEN3_0_6B_CONFIG`, `generate` (`__init__.py`)
- Test suite comparing logits argmax and 20-token greedy generation against HuggingFace `transformers` (eager attention)
- Benchmark harness (`benchmark.py`): prefill/decode TPS, TTFT, peak memory, per-token latency (p50/p90/p99), JSON output; preset aliases `xs` / `short` / `medium` / `long`
- Tutorial: `tutorials/01-generation.md` — full walkthrough of the implementation with baseline benchmark results on Apple M3 Pro CPU
