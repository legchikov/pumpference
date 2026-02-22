# Pumpference

**LLM inference from scratch in PyTorch** — focused on *learning-by-implementing*.

### Features

- Full Qwen3-0.6B architecture in plain PyTorch (~340 lines, one file)
- Greedy autoregressive generation verified token-for-token against HuggingFace `transformers`
- Sampling decoding — temperature, top-k, top-p (nucleus), composable; `temperature=0` falls back to greedy
- Benchmark harness — prefill/decode TPS, TTFT, peak memory, per-token latency (p50/p90/p99)
- Custom tokenizer wrapper (uses `tokenizers` library, no `transformers` at runtime)
- CLI with device auto-detection (CUDA / MPS / CPU)

### Reference baseline

This project re-implements **Qwen3** architecture from scratch:

- Raschka notebook: [`standalone-qwen3.ipynb`](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/standalone-qwen3.ipynb)
- Raschka blog: [Understanding and Implementing Qwen3 From Scratch](https://sebastianraschka.com/blog/2025/qwen3-from-scratch.html)
- Qwen3 report: [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)

### Quickstart

```bash
pipx install uv
git clone https://github.com/legchikov/pumpference.git
cd pumpference
uv venv
source .venv/bin/activate
uv pip install -U pip
```

Install PyTorch (CPU/CUDA) using the official selector: `https://pytorch.org/get-started/locally/`.

Then install the package:

```bash
uv pip install -e ".[dev]"
```

### Running inference

```bash
uv run python -m pumpference --prompt "Explain how transformers work"
uv run python -m pumpference --help   # all options
```

Downloads Qwen3-0.6B (~1.2 GB) on first run.

**Sampling flags:**

```bash
# Greedy (default)
uv run python -m pumpference --prompt "Tell me a joke"

# Temperature sampling
uv run python -m pumpference --prompt "Tell me a joke" --temperature 0.8

# Top-k + temperature
uv run python -m pumpference --prompt "Tell me a joke" --temperature 0.8 --top-k 50

# Nucleus (top-p) sampling
uv run python -m pumpference --prompt "Tell me a joke" --temperature 0.9 --top-p 0.95
```

### Benchmarking

```bash
make bench                   # default: xs preset (~30 prompt tokens)
make bench PRESET=short      # ~115 tokens
make bench PRESET=medium     # ~218 tokens
make bench PRESET=long       # ~373 tokens
```

Results are printed to stdout and saved as JSON under `benchmarks/`.

### Running tests

```bash
uv run pytest
```

Compares our implementation against HuggingFace `transformers` for correctness. Also includes 17 unit tests for sampling strategies (no model load required).

### Tutorials

The implementation is accompanied by a written series explaining *why* decisions were made, recording concrete metrics, and documenting gotchas. Code always lives in `src/` on `main`; tutorials provide context and reasoning.

| # | Title |
|---|---|
| 1 | [Building an LLM Inference Engine From Scratch](tutorials/01-generation.md) |
| 2 | [Sampling — Temperature, Top-k, and Nucleus Decoding](tutorials/02-sampling.md) |
| 3 | [Benchmarking — Knowing What You're Measuring](tutorials/03-benchmarking.md) |
| 4 | KV-Cache *(coming soon)* |

### Roadmap

- [x] Naive impl. for Qwen3-0.6B — compare generation against HuggingFace
- [x] Sampling — temperature, top-k, top-p decoding
- [x] Benchmarking baseline — TPS, TTFT, peak memory, per-token latency
- [ ] KV-cache — reduce generation from O(n²) to O(n)

### Contributing

TBD
