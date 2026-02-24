# 4. KV-Cache: from O(n²) to O(n) Decode

## Where we are

Pumpference can generate text, sample with temperature/top-k/top-p, and measure performance. The baseline decode speed on Apple M3 Pro CPU tells a bleak story: 1.3 tok/s on a 30-token prompt, dropping to 0.2 tok/s once the context grows to 372 tokens. That is not a hardware limitation — it is a fundamental algorithmic waste we are about to fix.

## What we are adding and why

The KV-cache is the single highest-impact optimisation in autoregressive generation. Without it, every decode step re-feeds the entire growing sequence through all 28 transformer layers. With it, each decode step processes exactly one token — the new one — while reusing the key/value tensors already computed for all past tokens.

The concrete goal: make decode cost O(n) per step instead of O(n²), where n is the current sequence length.

## Key decisions

**Cache after RoPE, not before.**
Rotary positional embeddings encode absolute position into each key vector. If you cache K before applying RoPE, you would need to re-apply RoPE on every decode step to every cached key, which both defeats the purpose and introduces a subtle bug: cached keys would get their positions rotated again and accumulate incorrect encodings. Caching after RoPE means the positional information is already baked in permanently — the cache entry for token at position 47 will always carry the rotation for position 47, regardless of when it gets attended to.

**Cache at KV-head granularity, not Q-head granularity.**
Qwen3-0.6B uses grouped-query attention (GQA): 16 query heads sharing 8 KV heads, with each KV head serving 2 query heads. The `repeat_interleave` expansion that inflates 8 KV heads to 16 happens in the forward pass, not in the cache. Caching at 8 heads instead of 16 cuts cache memory by half. For a 1000-token sequence: `28 layers × 2 (K+V) × 8 KV-heads × 128 head_dim × 2 bytes = ~114 MB` instead of ~228 MB.

**Cache after QK-norm.**
Qwen3 applies an RMSNorm to Q and K (QK-norm) before RoPE. Because RMSNorm is a purely local, position-independent operation, there is no mathematical reason to re-apply it to cached keys. Caching after QK-norm is consistent and correct.

**Pre-slice the RoPE tables in the model, not inside `apply_rope`.**
The original `apply_rope` sliced `cos[:seq_len]` internally, assuming positions always start at 0. With a cache, the new token at decode step `t` lives at position `P + t` (where P is the prompt length), not at position 0. Moving the slice to `Qwen3Model.forward()` — which has full knowledge of the cache's current length — means `apply_rope` can stay simple: it receives exactly the right position window and just broadcasts over batch and head dimensions.

**All-false mask during decode.**
The attention mask exists to prevent a token from attending to future tokens — a constraint that is mechanically enforced by the upper-triangular mask during prefill. During decode there is only one query token. There are no future tokens in the sequence. The mask for a `[1, kv_len]` attention matrix should be all-False: the single new token is allowed to attend to everything. Creating a `[1, kv_len]` all-False tensor is cheap and avoids any confusion from reusing the triangular mask.

**`use_cache=False` as an escape hatch.**
The `generate()` function accepts `use_cache=False` to fall back to the naive full-sequence recompute path. This exists solely for debugging and correctness comparison — it lets you run both paths on the same prompt and verify they produce identical tokens. In practice, the cached path should always be used.

## The tricky parts

| Issue | Symptom | Fix |
|-------|---------|-----|
| Caching K before RoPE | Text becomes incoherent after the first token; positional encodings accumulate incorrectly across steps | Call `kv_cache.update()` strictly after `apply_rope()` |
| Wrong position offset on new token | New token attends to the wrong context; generation diverges from no-cache baseline | Read `kv_cache.seq_len` at the top of `Qwen3Model.forward()` *before* calling any layer, then slice `cos[past_len : past_len + q_len]` |
| Using causal mask during decode | The single query is partially or fully masked; model only "sees" a subset of the context | During decode (`past_len > 0`), create a fresh all-False `[q_len, kv_len]` tensor on the correct device |
| Expanding KV heads before caching | 2× memory usage for the cache | `kv_cache.update()` is called before `keys.repeat_interleave(...)` |
| Stale cache across requests | Cached K/V from a previous generation leaks into the next one | `KVCache` is instantiated fresh at the start of each `generate()` call; `reset()` exists for manual control |
| Float32 softmax not maintained | Long-sequence attention can overflow or produce NaN in bfloat16 | The `dtype=torch.float32` argument to `torch.softmax` is unchanged — it applies regardless of cache mode |

## Results

### CPU (Apple M3 Pro, bfloat16, 100 tokens generated)

| Preset | Prompt tokens | Decode TPS (naive) | Decode TPS (KV-cache) | Speedup |
|--------|---------------|--------------------|-----------------------|---------|
| xs | 30 | 1.3 | 12.0 | 9.2× |
| short | 115 | 0.6 | 12.6 | 21× |
| medium | 218 | 0.3 | 8.5 | 28× |
| long | 372 | 0.2 | 9.2 | 46× |

The speedup grows with prompt length because the naive path scales as O(P²) per decode step — doubling the prompt roughly quadruples the per-token cost. The cached path's decode cost scales as O(P) per step, so the baseline penalty grows faster than the cache penalty, widening the gap.

Decode TPS plateaus around 9–13 tok/s across prompt lengths (with some noise from the test system). This is the signature of the O(n) regime: cost grows slowly with sequence length rather than catastrophically.

### GPU (CUDA, bfloat16, 100 tokens generated)

| Preset | Prompt tokens | Prefill TPS | TTFT | Decode TPS | Mean lat | P99 lat | Peak memory |
|--------|---------------|-------------|------|------------|----------|---------|-------------|
| xs | 30 | 1 010 | 29.7 ms | 40.0 | 25.0 ms | 38.5 ms | 3 178 MB |
| short | 115 | 3 602 | 31.9 ms | 38.2 | 26.2 ms | 40.1 ms | 3 463 MB |
| medium | 218 | 6 650 | 32.8 ms | 37.8 | 26.5 ms | 48.3 ms | 3 874 MB |
| long | 372 | 10 996 | 33.8 ms | 37.6 | 26.6 ms | 39.6 ms | 4 587 MB |

**Decode TPS is essentially flat**: 40.0 → 37.6 tok/s across 30 to 372 prompt tokens — a drop of only 6% while the context grows 12×. This is the O(n) signature made visible. On CPU the same experiment produced a 7× difference (12.0 vs 1.7 tok/s); on GPU it barely registers.

**Prefill TPS scales with prompt length**: 1 010 → 10 996 tok/s, an 11× increase as the prompt grows 12×. This is the GPU's parallelism at work in the opposite direction — at 30 tokens the attention matrices are tiny (30×1024) and most CUDA cores sit idle. At 372 tokens the matrices fill the hardware much better. This is why GPU batch inference becomes proportionally more efficient at longer inputs.

**TTFT is nearly constant**: 29.7 ms for 30 tokens, 33.8 ms for 372 tokens — only 4 ms more to process 12× more input. GPU parallelism absorbs the larger prefill without meaningful wall-clock cost.

### The surprising result: KV-cache is near-zero speedup on GPU at these lengths

Comparing the new cached GPU numbers against the naive GPU benchmarks from before the cache was implemented:

| Preset | Decode TPS (naive GPU) | Decode TPS (cached GPU) | Speedup |
|--------|------------------------|-------------------------|---------|
| medium (218) | 36.6 | 37.8 | 1.03× |
| long (372) | 37.8 | 37.6 | 1.00× |

The cache provides essentially no speedup on GPU at these sequence lengths. This is counterintuitive after the dramatic CPU results, but the explanation is straightforward: at 200–500 total tokens, the O(n²) attention work is small enough that the GPU handles it without effort. The bottleneck at these lengths is not the attention computation but the weight-loading bandwidth — iterating over the ~1.2 GB of model parameters on every token. That cost is identical with and without a cache, and it dominates.

The KV-cache's GPU benefit becomes significant at longer sequences (thousands of tokens) where the attention matrices grow large enough to compete with weight-loading as a bottleneck. At production scale, systems like vLLM typically work with sequences 10–100× longer than what we tested here.

### CPU vs GPU, cached

| Preset | CPU cached (tok/s) | GPU cached (tok/s) | GPU advantage |
|--------|--------------------|--------------------|---------------|
| xs | 12.0 | 40.0 | 3.3× |
| short | 12.6 | 38.2 | 3.0× |
| medium | 8.5 | 37.8 | 4.4× |
| long | 9.2 | 37.6 | 4.1× |

With the cache in place, GPU is 3–4× faster than CPU in absolute terms. Without the cache, GPU was already fast at these lengths; it was CPU that suffered catastrophically from the quadratic compute pattern.

## Comparison: with and without KV-cache

### Decode throughput across all modes

All numbers are decode tok/s, 100 tokens generated, bfloat16.

| Prompt tokens | CPU naive | CPU cached | GPU naive | GPU cached |
|---------------|-----------|------------|-----------|------------|
| 30 | 1.3 | **12.0** | 37.9 | **40.0** |
| 115 | 0.6 | **12.6** | 37.0 | **38.2** |
| 218 | 0.3 | **8.5** | 36.6 | **37.8** |
| 372 | 0.2 | **9.2** | 37.8 | **37.6** |

The two devices tell opposite stories.

On CPU, the cache is the difference between a usable system and a broken one. Decode goes from 0.2 tok/s (one token every 5 seconds) to 9.2 tok/s at the long preset — a 46× speedup. The naive CPU numbers aren't just slow in absolute terms, they get worse with every token. Sequence length grows → O(n²) cost grows → each successive decode step takes longer than the previous one.

On GPU, the cache makes almost no difference: 37.9 → 40.0 at xs (5%), 37.8 → 37.6 at long (essentially zero). The GPU's weight-loading bottleneck is the same whether it recomputes attention over 30 tokens or 400; the extra work costs little relative to the fixed per-layer parameter load.

### Latency stability: naive CPU latency degrades mid-generation

The latency data for CPU naive xs shows a characteristic quadratic growth pattern — early steps are fast, later ones are slow because the sequence is longer:

| Decode step | CPU naive lat | CPU cached lat | GPU naive lat | GPU cached lat |
|-------------|--------------|----------------|---------------|----------------|
| step 1 | 272 ms | ~84 ms | 27.6 ms | 38.5 ms |
| step 50 | ~850 ms | ~83 ms | ~26.4 ms | ~24.7 ms |
| step 99 (last) | ~1 400 ms | ~84 ms | ~26.4 ms | ~24.7 ms |

CPU naive latency grows 5× within a single generation run. CPU cached latency is flat (each step processes exactly 1 token). GPU latency is flat in both modes — the attention work at 30–130 tokens is negligible on GPU regardless of implementation.

The P99 numbers make this concrete. CPU naive xs: P99 = **1749 ms** vs mean = 777 ms. The tail is 2.3× the mean because the later decode steps are so much more expensive. CPU cached xs: P99 = **188 ms** vs mean = 84 ms — still some variance (system noise) but no systematic growth. GPU is tight in both modes: P99 within ~15% of mean.

### When the cache matters

| Condition | Cache benefit | Why |
|-----------|---------------|-----|
| CPU, any context length | Very large (9–46×) | Attention is the bottleneck; cache eliminates redundant O(n²) work |
| GPU, short context (< ~1000 tok) | Negligible (~0–5%) | Weight loading dominates; attention is cheap at these sizes |
| GPU, long context (thousands of tok) | Significant | Attention matrices eventually become large enough to rival weight loading |
| Latency stability | Always beneficial | Cache makes per-step cost constant; naive path grows monotonically |

The KV-cache is not primarily a GPU optimisation at inference scale. Its first beneficiary is CPU-class hardware — laptops, edge devices, anything that cannot absorb the quadratic attention cost with raw parallelism. On GPU it pays off at the sequence lengths typical of production workloads (multi-turn conversations, long documents), not at the short benchmark sequences we test here.

## Worth looking at in the code

The position offset logic in `Qwen3Model.forward()` is the linchpin of the whole implementation. It is two lines, but they do everything:

```python
past_len = kv_cache.seq_len if kv_cache is not None else 0
cos = self.cos[past_len : past_len + q_len]
sin = self.sin[past_len : past_len + q_len]
```

`past_len` is 0 on the prefill pass (empty cache) and `P` on every decode step. Slicing the pre-computed RoPE table to `[past_len, past_len + q_len)` ensures the new token(s) get the correct positional encoding without any changes to `apply_rope` itself.

The `KVCache.update()` method in `src/pumpference/model.py` is intentionally minimal: on the first call for a layer it stores the tensor as-is; on subsequent calls it `torch.cat`s along `dim=2` (the sequence dimension) and overwrites the stored entry. The returned tensor is the full accumulated K or V, ready to be attended to.

## What's next

Profiling to understand where the remaining ~26 ms/tok goes on GPU (and ~83 ms/tok on CPU). The GPU data suggests weight-loading bandwidth is the bottleneck at these sequence lengths — iterating over 1.2 GB of parameters on every token regardless of cache state. Quantisation (int8/int4 weights) would reduce that bandwidth and is the most direct path to higher throughput.
