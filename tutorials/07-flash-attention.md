# 7. Flash Attention: Tiling Away the N² Wall

## Where we are

The profiler showed that attention is not the bottleneck at short context lengths — weight loading dominates decode at 65–70% of time. But attention cost scales quadratically with sequence length while weight loading scales linearly. At 30 tokens the distinction is irrelevant; at 4K tokens the `N×N` score matrix consumes gigabytes of memory and starts dominating prefill time. Baseline prefill throughput at 218 tokens: 96 tok/s. At 372 tokens: 74 tok/s.

## What we're adding and why

**Tiled Flash Attention** in pure PyTorch: the same scaled dot-product attention result, computed without ever materialising the full `[batch, heads, N, N]` score matrix.

The driver is memory, not just speed. The score matrix for a 4 096-token context with 16 heads in float32:

```
16 heads × 4096 × 4096 × 4 bytes = 1 GB per layer
```

28 layers would need 28 GB just for score matrices, all allocated and discarded per forward pass. Flash attention replaces this with a constant-size tile buffer of 256 KB regardless of context length:

```
16 heads × 64 × 64 × 4 bytes = 262 KB  (block_size = 64)
```

## Key decisions

**Tiled outer loop over Q, inner loop over KV — not the other way around.**
Each Q block carries its own per-row accumulators (`acc_out`, `acc_lse`, `running_max`) that live across the inner KV loop. When the inner loop finishes, those rows of output are complete and written to the output tensor. This makes the output writes sequential and the accumulator lifetime bounded by one outer-loop iteration.

**Online softmax: three accumulators, one correction factor.**
Standard softmax requires knowing the row-max before computing any exponentials — you can't do it in a single pass. The online variant maintains a running maximum `running_max` and a running normaliser `acc_lse`. When tile `j` brings a new max `m_new > running_max`, both accumulators are rescaled by `exp(running_max - m_new)` before the new tile is added. This correction keeps the accumulated values consistent with the current max baseline without ever storing the full score row.

**Causal masking: two levels.**
At the tile level, the entire KV block is skipped if `k_start > q_end - 1` — every score in the tile would be `-inf` and contribute nothing. Since KV blocks are in ascending order, this is a `break`, not a `continue`. Inside a partial tile (where only some key positions are in the future), a per-element `k_pos > q_pos` boolean mask is applied to the score tile. Together these avoid redundant computation for the upper-triangle positions.

**V matmul in float32.**
Eager attention casts the softmax weights to bfloat16 before `weights @ V`. Flash accumulates `p @ V` in float32, then divides by `acc_lse` in float32 before the final cast. This is more precise than eager — but that precision means flash and eager can disagree on argmax when two logits differ by less than 1 ULP in bfloat16. Both implementations are correct; they represent the same mathematical value rounded differently. The correctness tests verify logit-value closeness (`|flash − eager| < 1.0` at every position), not token equality.

**Auto-fallback to eager during decode.**
With a KV cache, decode steps feed a single new token (`q_len = 1`). The score matrix is a single row — already O(n) memory, already no opportunity for tiling. Flash attention would add Python loop overhead with no benefit. The branch `if self.use_flash_attn and q_len > 1` routes decode steps to the existing eager path automatically. Flash is never explicitly turned off by the caller.

**`use_flash_attn` on `Qwen3Config`, propagated via `dataclasses.replace()`.**
A boolean flag on the config dataclass keeps the switch in one place and avoids mutating the module-level `QWEN3_0_6B_CONFIG` singleton (which is imported across multiple files). The CLI uses `replace(QWEN3_0_6B_CONFIG, use_flash_attn=True)` to produce a fresh config object.

## The tricky parts

| Issue | Symptom | Fix |
|-------|---------|-----|
| `running_max` initialised to 0 | First tile's correction is `exp(0 − new_max)` which is < 1 even when the tile is the first; accumulators shrink incorrectly | Initialise `running_max` to `-inf` so `exp(-inf − new_max) = 0`, zeroing out the (already-zero) accumulators cleanly on the first tile |
| `break` vs `continue` in causal skip | Using `continue` processes all KV blocks and wastes compute on tiles that contribute nothing | KV blocks are in ascending positional order; once `k_start > q_end - 1` no subsequent block can be valid, so `break` is correct |
| Float32 V matmul differs from bfloat16 eager | Token-equality generation test fails from step 2 onward even though logits are close | Flash (float32 V matmul) and eager (bfloat16 V matmul) are both correct; test logit-value closeness, not token identity |
| Per-tile `arange` creates tensors in the inner loop | Profiling shows tensor allocation overhead on every inner iteration | Acceptable for the educational implementation; production would pre-allocate position tensors outside the loop or use Triton |
| Ambiguous variable names `O` and `l` | Ruff flags `O` (looks like `0`) and `l` (looks like `1`) | Renamed to `acc_out` and `acc_lse` — also more descriptive |

## Results

Measured on CPU (Apple M3 Pro, bfloat16, 50 generated tokens, 1 warmup run):

| Preset | Prompt tokens | Eager prefill | Flash prefill | Speedup |
|--------|--------------|---------------|---------------|---------|
| medium | 218 | 2 264 ms / 96 tok/s | 1 619 ms / 135 tok/s | **1.4×** |
| long | 372 | 5 055 ms / 74 tok/s | 2 672 ms / 139 tok/s | **1.9×** |

Decode throughput is unchanged (both paths use eager when `q_len = 1`).

The prefill speedup is larger than expected from an implementation that adds Python loop overhead. The reason is the **causal skip**: eager computes the full `N×N` score matrix and then applies masking, while flash only processes lower-triangle tiles. At 372 tokens, roughly half the tiles are above the causal diagonal and are skipped entirely — halving the attention FLOPs before any loop overhead is counted.

### Memory: the real story at longer contexts

The benchmark peak-memory figure is dominated by model weights and the KV cache. The score matrix contribution is only visible once context length grows beyond what fits in L3 cache. The theoretical reduction is:

| Context length | Eager score matrix (per layer) | Flash tile buffer |
|---------------|-------------------------------|-------------------|
| 218 tokens | 3 MB | 256 KB |
| 372 tokens | 9 MB | 256 KB |
| 4 096 tokens | 1 GB | 256 KB |
| 32 768 tokens | 64 GB (impossible) | 256 KB |

Flash makes 32K-token contexts tractable. Eager makes them impossible.

## Worth looking at in the code

The online softmax update is the algorithmic core. Three lines in the inner loop:

```python
new_max    = torch.maximum(running_max, s.amax(dim=-1, keepdim=True))
correction = torch.exp(running_max - new_max)
p          = torch.exp(s - new_max)

acc_lse = correction * acc_lse + p.sum(dim=-1, keepdim=True)
acc_out = correction * acc_out + p @ v_block
running_max = new_max
```

The `correction` factor is what makes this equivalent to standard softmax despite processing one tile at a time. If `new_max > running_max`, then `correction < 1`: all previously accumulated exponentials were computed relative to a max that has since been superseded, so they need to shrink. If `new_max == running_max`, `correction = 1` and the old accumulators are preserved exactly. At the end of the inner loop, `acc_out / acc_lse` produces the correctly normalised weighted-V sum.

## What's next

Speculative decoding — where flash attention reduces prefill cost, speculative decoding attacks the other side: the number of serial decode steps.
