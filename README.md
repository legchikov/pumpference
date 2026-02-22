# Pumpference

**LLM inference from scratch in PyTorch** — focused on *learning-by-implementing*.

### Features

- Full Qwen3-0.6B architecture in plain PyTorch (~340 lines, one file)
- Greedy autoregressive generation verified token-for-token against HuggingFace `transformers`
- Benchmark harness — prefill/decode TPS, TTFT, peak memory, per-token latency
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

### Running inference

```bash
uv run python -m pumpference --prompt "Explain how transformers work"
uv run python -m pumpference --help   # all options
```

Downloads Qwen3-0.6B (~1.2 GB) on first run.

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

Compares our implementation against HuggingFace `transformers` for correctness.

### Roadmap

- [x] Naive impl. for Qwen3-0.6B — compare generation against HuggingFace
- [ ] Sampling — temperature, top-k, top-p decoding
- [ ] Benchmarking baseline — TPS, TTFT, peak memory, per-token latency
- [ ] KV-cache — reduce generation from O(n²) to O(n)

### Contributing

TBD

