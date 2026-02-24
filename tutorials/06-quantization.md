# 6. Quantization: The Storage Win That Isn't a Speed Win (Yet)

## Where we are

The profiler from the previous chapter showed the bottleneck clearly: every decode step reads all ~1.1 GB of model weights from DRAM, and this linear projection work — `aten::mm` and `aten::bmm` together — accounts for 65–70% of decode time. The arithmetic intensity of a matrix–vector multiply is 1 FLOP/byte, roughly 24× below the hardware ridge point on Apple M3 Pro. Compute is not the constraint. Bytes are.

The obvious response: store weights in fewer bytes. Int8 cuts the weight size in half; int4 cuts it to a quarter. This chapter implements both from scratch and measures what actually happens.

## What we're adding and why

**Weight-only quantization** — store model weights as int8 or int4, but keep activations and arithmetic in bfloat16. The two schemes:

- **W8A16 (int8):** symmetric per-output-channel quantization. Each row of a weight matrix gets one scale factor: `scale = max(abs(row)) / 127`. Weights stored as int8 (1 byte each), scales stored as float32.
- **W4A16 (int4, group-wise):** symmetric quantization with one scale per group of 128 input elements. Every 128 consecutive elements in a row share one scale. Two int4 values are packed into a single byte. Scales stored as float32.

Both are implemented as drop-in `nn.Module` replacements (`Int8Linear`, `Int4Linear`) that are swapped in after weight loading. No changes to the model architecture, no changes to the generation loop. The `quantize_model()` function walks the entire module tree and replaces every `nn.Linear` in-place.

## Key decisions

**Weight-only, not full quantization.**
Activations during single-token decode are tiny: shape `[1, 1, emb_dim]`. Quantizing them would save almost nothing while adding precision-sensitive operations (activation ranges vary per input and are harder to calibrate). The dominant cost is weight loading; quantizing only weights addresses exactly that.

**Symmetric quantization with per-channel scale (int8) and per-group scale (int4).**
Symmetric means the quantization range is `[-max, max]` — no zero-point offset. This keeps dequantization to a single multiply: `w_fp16 = w_int8 * scale`. Per-channel scale (one per output row) is sufficient for int8 because a single row's dynamic range is narrow. Int4 has half the precision (7 levels per side vs 127), so finer granularity is needed: per-group scale (one per 128-element block) limits the error within each group. Per-channel int4 degrades quality noticeably; group_size=128 is the industry standard for a reason.

**Clamping to ±7 for int4, not ±8.**
Int4 ranges from -8 to +7 in two's complement, but using -8 means the minimum value has no positive mirror, which complicates symmetric dequantization. Clamping to ±7 (15 total levels) preserves symmetry at the cost of one unused code point.

**Pack two int4 values per byte.**
Each int4 occupies the lower or upper nibble of a uint8 byte. Unpacking uses bitwise ops: `even = (packed & 0x0F) - 8` (low nibble), `odd = ((packed >> 4) & 0x0F) - 8` (high nibble). This halves weight storage compared to int8.

**Dequantize-on-forward in pure PyTorch.**
The simplest possible forward pass: dequantize the weight to bfloat16, then call `F.linear`. No custom CUDA kernels, no custom CPU kernels. This is the reference implementation — correct, readable, and a fair baseline for measuring what production optimisations buy you.

**Apply quantization after weight loading, before moving to device.**
Loading bfloat16 weights first and quantizing afterward avoids needing a quantization-aware loading path. The full-precision weights are needed briefly to compute scales; then the `nn.Linear` modules are replaced in-place and the originals are garbage-collected.

## The tricky parts

| Issue | Symptom | Fix |
|-------|---------|-----|
| Weight tying (tok\_emb = out\_head) | After quantization, out\_head becomes Int8Linear with its own int8 copy of the weights. The tie is effectively broken. | Intentional: embedding stays bfloat16 for lookup; output projection uses the quantized copy. Documented explicitly. |
| Per-channel scale stored as float32 | Converting to bfloat16 in forward loses precision in the scale itself | Keep scale as float32 in the buffer; cast to `x.dtype` only at forward time (`.to(x.dtype)`) |
| Int4 packing sign convention | Shifting [-7,7] to [0,15] before packing into uint8 nibbles; forgetting the shift produces garbage on unpack | Encode: `w_shifted = (w_q + 8).to(torch.uint8)`; Decode: `(packed & 0x0F).to(torch.int8) - 8` |
| Int4 group alignment | `in_features` must be divisible by `group_size` | Assert in `quantize_per_group`; Qwen3's dimensions (1024, 3072, 2048) are all divisible by 128 |
| in_features vs out_features axis for group scale | Grouping along the wrong axis would scale across output channels, which is meaningless | Groups are along `dim=1` (input features) — each group of input weights shares a scale |
| Argmax agreement drops with short prompts | A 10-token prompt yields noisy statistics (each position is 10% of the result) | Use the 30-token xs preset for logit comparison; accept that a single differing position causes autoregressive divergence |

## The surprising result: dequantize-on-forward is slower, not faster

| Mode | Decode TPS | vs bfloat16 |
|------|-----------|-------------|
| bfloat16 (baseline) | 12.4 tok/s | — |
| int8 dequant-on-forward | 3.7 tok/s | **3.4× slower** |
| int4 dequant-on-forward | 1.8 tok/s | **6.9× slower** |

This is the opposite of the expected direction. The explanation is in the bandwidth math.

### Why dequantize-on-forward adds, not removes, bandwidth

Consider a single FFN linear layer with shape [out=1024, in=3072] in bfloat16. A forward pass touches:
- **Load weight:** 1024 × 3072 × 2 bytes = **6.3 MB** (from DRAM to cache/registers)
- **Matmul:** computes output from the loaded weight and the input vector

With int8 dequantize-on-forward, the forward pass instead does:
- **Load int8 weight:** 1024 × 3072 × 1 byte = **3.1 MB** — this is the saving
- **Load scale and dequantize to bfloat16:** creates a temporary `[1024, 3072]` bfloat16 tensor, writing **6.3 MB** back to memory
- **Matmul over the temporary bfloat16 tensor:** reads **6.3 MB** again

Total: 3.1 + 6.3 + 6.3 = **15.7 MB** of memory traffic, versus 6.3 MB for bfloat16. We tripled it.

The matmul still runs in bfloat16 — we haven't changed the compute at all. We only changed where the weight bytes come from and added an explicit dequantization step that materialises a full bfloat16 copy of the weight.

For int4, the packed weight is 1.6 MB, but the temporary bfloat16 tensor is still 6.3 MB. The overhead is even worse: 1.6 + 6.3 + 6.3 = 14.2 MB versus 6.3 MB.

Measured on a single [1024, 3072] fc1 layer:

| Mode | ms/call | vs bfloat16 |
|------|---------|------------|
| bfloat16 | 0.09 ms | — |
| int8 dequant-on-forward | 1.08 ms | 11.6× slower |
| int4 dequant-on-forward | 2.68 ms | 28.8× slower |

### What production frameworks do instead

The slowdown is entirely due to materialising the fp16 temporary tensor. A **fused kernel** avoids this by loading int8 bytes and feeding them directly into the multiply-accumulate pipeline, without ever writing a full fp16 weight matrix. This is what makes quantization actually fast:

- **llama.cpp / GGUF:** custom CPU kernels (NEON, AVX-512) that load packed int4 or int8 weights and compute the dot product inline, typically 1.5–2× faster than fp16 for single-token decode.
- **CUDA int8 GEMM (cuBLAS / cutlass):** the GPU reads int8 weights, performs the accumulation in int32, and produces fp16 output without any fp16 weight tensor in memory.
- **`torch.ao.quantization`** (PyTorch's built-in path): instruments the model, calibrates quantization ranges on sample data, and produces a model with fused int8 kernels under the hood. More setup than our approach but correct performance.

Our dequantize-on-forward is the *right reference implementation for understanding the math* — it shows exactly what quantization is doing and makes the correctness testable. It is not the right performance implementation.

### What the quantization does buy: storage reduction

The model parameter bytes (buffers + parameters) after quantization:

| Mode | Model bytes | vs bfloat16 |
|------|------------|------------|
| bfloat16 | 3,223 MB | — |
| int8 | 2,629 MB | 0.82× |
| int4 | 2,348 MB | 0.73× |

The reduction is real but modest because a large fraction of the model's buffers is not weights. The pre-allocated causal mask (`[context_length, context_length]` bool tensor, context_length = 40 960) is 1.6 GB on its own — larger than the entire weight set. The weight-only reduction from 3.2 GB to 2.6 GB (int8) is the model's linear layers getting smaller, but the fixed overhead of the causal mask dilutes the ratio.

### Quality

Argmax agreement of the quantized model vs bfloat16 baseline (30-token prompt, single forward pass):

| Mode | Argmax agreement |
|------|-----------------|
| int8 | 90% |
| int4 (group_size=128) | 80% |

Ten positions per token out of 30 disagree for int4. Whether this is acceptable depends on the application; for most chat/completion tasks it is fine because the top-1 probability mass is usually concentrated enough that small weight errors do not change the outcome. The 20% disagreement is larger than reported by frameworks that calibrate scales on representative data rather than using the round-to-nearest (RTN) absmax approach we use here.

## Worth looking at in the code

The arithmetic in `unpack_int4` is the only non-obvious part:

```python
even = (packed & 0x0F).to(torch.int8) - 8   # low  nibble: [1,15] → int8 [-7,7]
odd  = ((packed >> 4) & 0x0F).to(torch.int8) - 8  # high nibble
w_q = torch.empty(...)
w_q[:, 0::2] = even
w_q[:, 1::2] = odd
```

The packing contract is: `even index → low nibble, odd index → high nibble`. The +8 bias (encoding) and -8 de-bias (decoding) map the signed range [-7,7] to the unsigned nibble range [1,15]. Using 0 as the zero is intentional: it can't be produced by clamping to ±7, which means a packed byte of `0x00` is an error marker (not a valid all-zero weight).

## What's next

Two paths forward, each addressing a different aspect of what we learned:

**Fused kernels for real performance.** Implement a custom CPU GEMM that loads int8 weights and accumulates directly, bypassing the temporary fp16 tensor. This closes the gap between our reference implementation and what llama.cpp achieves. The kernel is ~50 lines of C++ with pybind11 or a simple CUDA kernel for GPU.

**Calibration-based quantization.** Our RTN (round-to-nearest) absmax scales are computed purely from the weight distribution, with no knowledge of typical input activations. Calibration-based methods (GPTQ, AWQ) minimise the output reconstruction error using a small calibration dataset and typically achieve higher argmax agreement — 95%+ for int4 — at the cost of a one-time offline calibration step.
