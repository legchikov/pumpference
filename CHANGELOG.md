## [2026.2.28] — 2026-02-28

### Changes
- Flash Attention (`flash_attention()` in `model.py`): tiled outer Q-loop + inner KV-loop with online softmax (running max, running normaliser, output accumulator); causal tile-skip with `break`; float32 accumulation throughout; O(n) peak memory for score matrices vs O(n²) eager
- `use_flash_attn: bool = False` field on `Qwen3Config`; stored on `GroupedQueryAttention` via `use_flash_attn=cfg.use_flash_attn`; forward branches on `self.use_flash_attn and q_len > 1` — auto-falls back to eager during KV-cached decode
- `--flash-attn` flag added to `__main__.py` and `benchmark.py`; both use `dataclasses.replace()` to derive a config from the singleton safely
- `flash_attn: bool` field added to `BenchmarkResult`; shown in formatted output and saved in JSON
- Two new tests in `test_model.py`: `test_flash_attention_logits_match_eager` (argmax identity at every position) and `test_flash_attention_logits_close_across_steps` (max logit diff < 1.0 for 5 growing-sequence steps)
- Tutorial `tutorials/07-flash-attention.md` written with measured prefill speedup (1.4–1.9× on CPU, 218–372 token prompts)

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
