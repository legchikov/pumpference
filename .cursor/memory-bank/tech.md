# Tech

## Language & runtime

- **Python >=3.11** (uses modern syntax: `X | Y` union types, dataclass features)
- **PyTorch >=2.0** — all neural network code uses `torch.nn`, `torch.Tensor`, `@torch.no_grad()`

## Runtime dependencies

| Package | Purpose |
|---|---|
| `torch>=2.0` | Neural network framework, tensor operations |
| `safetensors` | Fast, safe weight loading (no pickle) |
| `huggingface_hub` | Download model weights and tokenizer from HF Hub |
| `tokenizers` | BPE tokenizer (lightweight alternative to full `transformers`) |

## Dev dependencies

| Package | Purpose |
|---|---|
| `pytest>=8.0.0` | Test runner |
| `transformers>=4.51.0` | Reference HF model for comparison tests only |

## Package manager & build

- **uv** — fast Python package manager, handles venv and lockfile (`uv.lock`)
- **hatchling>=1.27.0** — build backend (configured in `pyproject.toml`)
- **src layout**: package lives under `src/pumpference/`, installed as editable

## Dev environment setup

```bash
pipx install uv
git clone https://github.com/legchikov/pumpference.git
cd pumpference
uv venv
source .venv/bin/activate
uv pip install -U pip
# Install PyTorch separately via https://pytorch.org/get-started/locally/
```

## Running

```bash
uv run python -m pumpference --prompt "Your prompt" --max-tokens 200 --device auto
uv run pytest          # run tests (downloads ~1.2GB model on first run)
```

## Makefile commands

| Target | Command |
|---|---|
| `make lint` | `ruff check` + `mypy --strict` on `src tests` |
| `make format` | `ruff check --fix` on `src tests` |
| `make test` | `pytest --cov` |
| `make bench [PRESET=<alias>]` | Run benchmark; `PRESET` accepts named aliases or token counts |

**Benchmark presets** (`PRESET` variable, default `xs`):

| Alias | Token count |
|---|---|
| `xs` | ~30 |
| `short` | ~115 |
| `medium` | ~218 |
| `long` | ~373 |

## Model weights

- Qwen3-0.6B weights (~1.2GB) auto-downloaded from HuggingFace Hub on first run
- Stored locally in `Qwen3-0.6B/` directory (gitignored)
- Single `model.safetensors` file (no sharding)
- Tokenizer: `tokenizer.json` downloaded separately to same directory

## Technical constraints

- No GPU required — runs on CPU (also supports CUDA and MPS)
- bfloat16 precision throughout (matches HF distribution)
- Current generation is O(n²) per token (no KV-cache yet)
- Sampling decoding: temperature, top-k, top-p (nucleus) — all composable; temperature=0 falls back to greedy
- Single-sequence inference only (batch size = 1)

## Testing strategy

- Compare against HuggingFace `transformers` as ground truth
- HF model must use `attn_implementation="eager"` for fair comparison
- Module-scoped fixtures avoid re-downloading/loading model per test
- Tests verify: logits argmax equality, max logit diff < 1.0, greedy generation token equality
- `tests/test_sampling.py`: 17 unit tests for sampling strategies (no model load required)
