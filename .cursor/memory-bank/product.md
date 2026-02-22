# Product

## Why this project exists

Pumpference is an **educational LLM inference framework** built from scratch in plain PyTorch. Production frameworks (HuggingFace `transformers`, vLLM, llama.cpp) are powerful but opaque — hundreds of things happen behind a single `model.generate()` call. Pumpference strips everything down to the bare minimum so every matrix multiply between a text prompt and the first generated word is explicit and understandable.

## What problems it solves

- **Learning gap**: most LLM tutorials explain concepts but skip implementation details (weight mapping, precision pitfalls, special-token handling). Pumpference forces the builder to confront every detail.
- **Reference implementation**: provides a clean, minimal, tested codebase that can be used as a baseline before layering on optimizations (KV-cache, sampling, batching).
- **Tutorial series**: accompanies a published tutorial series ("Pumpference series"). Tutorial 1 is a full from-scratch walkthrough. Tutorials 2+ are dev-log articles — they explain design decisions, record concrete metrics (tok/s, memory), and document gotchas. Code always lives in `src/` and is always up to date; tutorials link to the repo rather than duplicating code.

## How it works

1. User provides a text prompt via CLI (`python -m pumpference --prompt "..."`)
2. The tokenizer encodes the prompt into token IDs (using HuggingFace `tokenizers` library, not `transformers`)
3. The from-scratch Qwen3 model (decoder-only transformer) processes token IDs through 28 transformer blocks
4. A greedy generation loop autoregressively produces new tokens one at a time (argmax of logits at each step)
5. Generation stops at EOS token (`<|im_end|>`) or `--max-tokens` limit
6. The tokenizer decodes generated IDs back to text

## User experience goals

- **Clarity over performance**: every architectural component is explicit — no hidden abstractions
- **One-command setup**: `uv` handles environment and dependencies; first run auto-downloads model weights from HuggingFace Hub
- **Verified correctness**: test suite compares output against HuggingFace `transformers` to prove the implementation is correct
- **Progressive complexity**: the project starts naive (no KV-cache, no sampling) and will layer optimizations in subsequent tutorials
