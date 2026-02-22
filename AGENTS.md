# Pumpference — Agent Guidelines

Educational LLM inference framework built from scratch in plain PyTorch.
Reference model: **Qwen3-0.6B** (decoder-only transformer).

## Repository layout

```
src/pumpference/
  __init__.py       # Public API: Qwen3Model, QWEN3_0_6B_CONFIG, generate
  __main__.py       # CLI entry point (argparse, device auto-detection)
  model.py          # Full architecture + config dataclass + weight loading
  generate.py       # Greedy autoregressive generation loop
  tokenizer.py      # Qwen3Tokenizer wrapper over HF `tokenizers` library
tests/
  conftest.py       # sys.path setup for src layout
  test_model.py     # Comparison tests vs HuggingFace transformers
tutorials/          # Written walkthrough series
```

## Setup & running

Requires **Python >=3.11** and **uv** as the package manager.

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"           # or: uv pip install -e . && uv pip install transformers pytest
uv run python -m pumpference --prompt "Hello"   # inference (~1.2 GB model auto-downloaded on first run)
uv run pytest                                    # tests (also downloads model on first run)
```

Makefile shortcuts: `make lint`, `make format`, `make test`.

## Architecture at a glance

All model code lives in **one file** (`model.py`, ~340 lines) by design — premature modularisation was tried and reverted. The key components:

| Component | Role |
|---|---|
| `Qwen3Config` | Dataclass with all hyperparameters |
| `RMSNorm` | Layer norm without bias/re-centering; upcasts to float32 internally |
| `compute_rope_params` / `apply_rope` | Rotary Positional Embeddings (split-half, θ=1 000 000) |
| `FeedForward` | SwiGLU: `silu(fc1(x)) * fc2(x) → fc3` |
| `GroupedQueryAttention` | 16 Q-heads, 8 KV-heads, optional QK-norm, causal masking |
| `TransformerBlock` | Pre-norm: norm→attn→residual→norm→FFN→residual |
| `Qwen3Model` | Embedding → 28 blocks → final norm → linear head (weight-tied) |
| `load_weights_into_qwen` | Maps HF safetensors weight names → our parameter names |

Generation (`generate.py`) feeds the **full sequence** on every step — no KV-cache yet.

## Coding conventions

- **No bias terms** in any `nn.Linear` — matches modern LLM convention.
- **bfloat16 by default** everywhere (set in `Qwen3Config.dtype`).
- **Pre-allocated buffers**: RoPE cos/sin tables and causal mask use `register_buffer(persistent=False)`.
- `transformers` is a **dev-only** dependency — runtime code must never import it.
- Keep architecture components in `model.py` unless a split becomes clearly justified.
- Avoid redundant code comments that merely narrate what the code does.

## Numerical precision gotchas

These are easy to break during refactoring — keep them intact:

1. `RMSNorm` upcasts to **float32** before computing variance.
2. Softmax in attention is computed in **float32** (`dtype=torch.float32`), then cast back.
3. Causal mask uses **`-torch.inf`** (not `-1e9`) so softmax produces exact zeros.
4. QK normalization is applied **after** projection, **before** RoPE.
5. KV heads are expanded via `repeat_interleave` **after** RoPE, not before.

## Testing

Tests compare our implementation against HuggingFace `transformers` as ground truth.

- HF model **must** use `attn_implementation="eager"` for numerical consistency.
- Fixtures are `scope="module"` — models are loaded once per test run.
- Two tests: (1) logits argmax equality at every position, (2) 20-token greedy generation identity.
- Run with: `uv run pytest` (or `make test` for coverage).

When adding new functionality, add a test that verifies output matches HF under equivalent settings.

## Weight mapping reference

Our parameter names → HuggingFace names (useful when extending to new layers):

| Ours | HuggingFace |
|---|---|
| `tok_emb.weight` | `model.embed_tokens.weight` |
| `trf_blocks[i].att.W_query.weight` | `model.layers.{i}.self_attn.q_proj.weight` |
| `trf_blocks[i].att.W_key.weight` | `model.layers.{i}.self_attn.k_proj.weight` |
| `trf_blocks[i].att.W_value.weight` | `model.layers.{i}.self_attn.v_proj.weight` |
| `trf_blocks[i].att.out_proj.weight` | `model.layers.{i}.self_attn.o_proj.weight` |
| `trf_blocks[i].att.q_norm.scale` | `model.layers.{i}.self_attn.q_norm.weight` |
| `trf_blocks[i].att.k_norm.scale` | `model.layers.{i}.self_attn.k_norm.weight` |
| `trf_blocks[i].norm1.scale` | `model.layers.{i}.input_layernorm.weight` |
| `trf_blocks[i].norm2.scale` | `model.layers.{i}.post_attention_layernorm.weight` |
| `trf_blocks[i].ff.fc1.weight` | `model.layers.{i}.mlp.gate_proj.weight` |
| `trf_blocks[i].ff.fc2.weight` | `model.layers.{i}.mlp.up_proj.weight` |
| `trf_blocks[i].ff.fc3.weight` | `model.layers.{i}.mlp.down_proj.weight` |
| `final_norm.scale` | `model.norm.weight` |
| `out_head.weight` | `lm_head.weight` (or tied to `tok_emb`) |

## Changelog Release Notes

When cutting a release and publishing to GitHub:

**Tag format:** `vYYYY.M.D` for stable, `vYYYY.M.D-beta.N` for pre-releases (e.g. `v2026.2.22-beta.1`).

**Steps:**

1. Update `CHANGELOG.md` — add a version section at the top with the release date.
2. Bump `version` in `pyproject.toml` to match the tag.
3. Commit: `git commit -m "release vYYYY.M.D"`.
4. Tag from the release commit: `git tag vYYYY.M.D`.
5. Push tag: `git push origin vYYYY.M.D`.
6. Create GitHub release via CLI:
   ```bash
   gh release create vYYYY.M.D --title "pumpference YYYY.M.D" \
     --notes "$(sed -n '/^## \[YYYY.M.D\]/,/^## /{ /^## \[YYYY.M.D\]/d; /^## /d; p }' CHANGELOG.md)"
   ```
   Add `--prerelease` flag for beta tags.

**`CHANGELOG.md` entry format:**

```markdown
## [YYYY.M.D] — YYYY-MM-DD

### Changes
- ...

### Fixes
- ...
```

Sort entries by impact: user-visible changes first, internal fixes last. Do not duplicate the version heading as the release title.

## Tutorial writing format

Tutorials are published as **dev-log articles** — not step-by-step code walkthroughs. Code always lives in `src/pumpference/` on `main` and is always up to date. Tutorials explain *why* decisions were made, record concrete metrics, and document gotchas. Readers go to the repo for code; they come to the article for context and reasoning.

**Tutorial 1** (`tutorials/01-naive-inference.md`) is an exception — it is a full from-scratch walkthrough because it introduces the entire codebase from zero.

**Tutorials 2+** follow this structure:

1. **Where we are** — 2-3 sentences: state of the framework at the start of this chapter + baseline metric
2. **What we're adding and why** — motivation and concrete goal
3. **Key decisions** — bullet list of non-trivial choices with reasoning (not "what the code does" — "why we did it this way")
4. **The tricky parts** — table: Issue / Symptom / Fix
5. **Results** — before/after table with real numbers (tok/s, memory, etc.) — mandatory when performance changes
6. **Worth looking at in the code** — optional; 1-2 focused snippets only when prose cannot convey the point; link to `src/` for the full picture
7. **What's next** — one-sentence preview

Code snippets in tutorials are used sparingly — only when a non-obvious point is hard to explain in prose. No "before/after" diffs, no full file listings.

**Workflow:** implement on `main` → run benchmarks → write tutorial markdown → commit. No git tags required.

## Roadmap

Current priority order for upcoming work:

1. **KV-cache** — make generation O(n) per step instead of O(n²)
2. **Sampling** — temperature, top-k, top-p (nucleus) decoding
3. Profiling & further optimisations
