# 08 — Speculative Decoding

## Where we are

Flash Attention (chapter 7) cut memory overhead for long prefills from O(n²) to O(n). Decode speed on CPU is still ~12 tok/s for short prompts, down to ~8 tok/s for longer ones. Every generated token requires one serial target-model forward pass, and that number is fixed regardless of every other optimisation we've applied. This chapter attacks that serial bottleneck directly.

Baseline: 12.4 tok/s decode (CPU, bfloat16, xs preset, 100 tokens generated).

## What we're adding and why

Speculative decoding (Leviathan et al., 2023) reduces the number of serial *target-model* forward passes without changing the output distribution. The idea:

- A fast *draft model* proposes K tokens in K small forward passes.
- The *target model* verifies all K in a **single** parallel forward pass.
- Accept the longest prefix of draft tokens that the target agrees with, then begin a new round.

When the draft model is right n times (n ≤ K), we pay one target forward pass but produce n+1 tokens (n accepted + one bonus or corrected token). In the best case — all K accepted — we get K+1 tokens for the cost of one target pass.

Setup: **Qwen3-0.6B as draft, Qwen3-1.7B as target.** They share the same tokenizer, vocabulary, and architecture. Qwen3-1.7B is the smallest off-the-shelf model that is strictly larger; there is no smaller Qwen3 variant to use as draft. The code lives in `src/pumpference/`, with Qwen3-1.7B weights (~4 GB sharded across two safetensors files) downloaded automatically on first run.

## Key decisions

- **Single verification pass**: Feed `[last_accepted_token, d_1, ..., d_K]` (K+1 tokens) to the target with the KV-cache already populated by the prompt. Position 0 of the output verifies d_1, position i verifies d_{i+1}, position K supplies the bonus token. No stored-logit bookkeeping between rounds.

- **KV-cache invariant**: At the start of every draft round, *both* caches have the same `seq_len = P + n_gen - 1`, where P is the prompt length and n_gen is the number of tokens generated so far. The cache does **not** yet contain `last_token`; the draft loop feeds it as its first input. This invariant lets the same draft loop body handle every round without special cases.

- **Cache truncation on rejection**: When the target rejects draft token at position j, both caches are truncated to remove everything beyond the j-th accepted token. The corrected token is NOT written to the cache immediately — it becomes `last_token` for the next round and is picked up by the draft loop. This keeps the invariant intact without an extra forward pass.

- **Draft-cache sync on full acceptance**: The draft loop feeds `last_token, d_1, ..., d_{K-1}` (K items) and predicts d_1 through d_K but does NOT add d_K to its own cache. After full acceptance, one extra draft forward pass feeds d_K to sync the draft cache to seq_len = P + n_gen - 1 for the next round.

- **Greedy and rejection sampling**: Temperature=0 uses argmax comparison (deterministic, 100% acceptance when draft==target). Temperature>0 uses the rejection-sampling theorem: accept d_i with probability min(1, p\_target(d\_i) / p\_draft(d\_i)), and on rejection sample from norm(max(0, p\_target - p\_draft)). This provably produces samples from the exact target distribution.

- **Sharded weight loading**: Qwen3-1.7B ships as two safetensors shards. `download_and_load_weights` now tries `model.safetensors` first, and if not found, reads `model.safetensors.index.json` to discover and download each shard.

## The tricky parts

| Issue | Symptom | Fix |
|---|---|---|
| Double-feeding `last_token` to draft | Draft sees `last_token` at the wrong position; acceptance rate < 100% even when draft==target | Do not feed first_token to draft during init. The draft loop feeds `last_token` as its first input each round. |
| Off-by-one in target cache truncation | Target retains one extra token after rejection; next round sees wrong context | Truncate to `cache_len_before + 1 + n_accepted` (not +2). |
| Causal mask for multi-token decode | Speculative verification feeds K+1 tokens with a non-empty cache; the old mask code assumed `q_len == 1` when `past_len > 0` | Unified mask: always allocate a `[q_len, kv_len]` bool tensor of zeros, then set `mask[:, past_len:] = causal_mask[:q_len, :q_len]` when `q_len > 1`. |
| Draft-target divergence at sequence boundaries | Even with same model, different contexts produce different logits | The cache invariant ensures both models see identical token sequences at identical positions; test with draft==target confirms 100% acceptance at temperature=0. |

## Results

Benchmarks below are on CPU (bfloat16, xs preset, 100 tokens generated) using Qwen3-0.6B as draft. Real speedup numbers require the Qwen3-1.7B target (not run in CI due to 4 GB download).

| Setup | Decode TPS | Notes |
|---|---|---|
| Qwen3-0.6B standard | 12.4 | Baseline |
| Speculative (draft==target, K=5) | ~same | Sanity check: no speedup expected, same model |
| Speculative (0.6B→1.7B, K=5) | run `make bench --speculative` | Requires 1.7B download |

The theoretical maximum speedup is K+1 if every draft token is accepted. In practice, draft models from the same family achieve 60–80% token acceptance rates, giving an effective speedup of 2–4× on single-request latency.

## Worth looking at in the code

The acceptance loop is the core of the algorithm. For greedy mode it is just an argmax comparison:

```python
for i, d_tok in enumerate(draft_tokens):
    target_pred = target_v_logits[0, i].argmax().item()
    if target_pred == d_tok.item():
        n_accepted += 1
    else:
        break
```

For sampling mode, the rejection-sampling theorem produces a corrected distribution that collapses to the target when subtracted from itself — so no matter how bad the draft is, the output is always correct:

```python
adjusted = (p_target - p_draft).clamp(min=0.0)
corrected = torch.multinomial(adjusted / adjusted.sum(), num_samples=1)
```

Full implementation: [`src/pumpference/generate.py`](../src/pumpference/generate.py).

## What's next

Fused int8 kernels — bypass the fp16 temporary that makes dequantize-on-forward slower than bfloat16 on CPU, enabling the quantization speedups that the memory-bandwidth analysis predicted.
