# 5. Profiling: Where Does the Time Go?

## Where we are

The KV-cache makes CPU decode linear in sequence length instead of quadratic: ~12 tok/s at any context length rather than 0.2–1.3 tok/s. On GPU, the cached path delivers ~38 tok/s. For short sequences those GPU numbers look flat — the cache makes essentially no difference, which means the real bottleneck is elsewhere. Before writing more optimisation code, we measure.

## What we're adding and why

A profiling harness: two complementary views of a single decode step.

- **Coarse timing** registers `forward_pre_hook` / `forward_hook` pairs on each transformer block and the surrounding ops (`tok_emb`, `final_norm`, `out_head`). This tells us which *module* consumes each millisecond without touching `model.py`.
- **Fine breakdown** wraps a handful of decode steps in `torch.profiler.profile` and prints the top operators by self CPU time. This tells us which *PyTorch kernel* inside those modules is responsible.

The two levels together give a complete picture: coarse timing names the guilty module, the operator breakdown names the operation.

## Key decisions

- **Forward hooks, not model surgery.** Wrapping individual `nn.Module.forward()` calls in `time.perf_counter()` is the least invasive approach — no changes to `model.py`, no risk of accidentally altering computation. The hooks attach by name, fire symmetrically (pre = start clock, post = stop), and are removed after the measurement loop.
- **Decode only, not prefill.** The profiling loop runs a short warmup prefill to fill the KV-cache, then measures single-token decode steps. Prefill and decode have different cost profiles (prefill is matrix–matrix; decode is matrix–vector), and the bottleneck we care about is decode.
- **`torch.cuda.synchronize()` before each clock read on GPU.** CUDA operations are asynchronous by default; `perf_counter` without a sync will measure dispatch latency (fast and meaningless) rather than kernel execution time. The `_sync()` helper calls `synchronize()` on CUDA devices and is a no-op on CPU.
- **`torch.profiler.profile` with `record_shapes=True` and `with_flops=True`.** Shape recording lets you cross-check that each matmul has the expected dimensions. FLOP counting lets you compute arithmetic intensity from first principles.

## Results

Running `make profile` on Apple M3 Pro CPU, 30-token prompt, 10 decode steps:

### Coarse timing — by module (mean over 10 decode steps)

| Module | ms/step | % |
|--------|---------|---|
| tok\_emb | 0.03 | 0.0% |
| block\_00 … block\_27 | 2.1–2.9 each | **92.1% total** |
| final\_norm | 0.05 | 0.1% |
| out\_head | 3.3 | **4.8%** |
| **TOTAL** | **~67** | 100% |

Every block costs almost exactly the same — no block is special, no block is a bottleneck in isolation. The model is entirely uniform. `out_head` is the single most expensive individual module: it projects a 1024-dimensional vector into 151 936 vocabulary logits, so it reads ~311 MB of weight on every token.

### Fine breakdown — top operators (torch.profiler, 5 decode steps)

| Operator | Self CPU % | What it is |
|----------|-----------|------------|
| `aten::mm` | 32.9% | Linear layer forward (weight matrix × activation vector) |
| `aten::bmm` | 32.8% | Batched attention scores (Q×Kᵀ) and weighted sum (attn×V) |
| `aten::copy_` | 5.5% | dtype casts (float32 upcast in RMSNorm + softmax, then back) |
| `aten::cat` | 2.8% | KV-cache concatenation — appending one new K or V slice |
| `aten::_softmax` | 2.0% | Attention softmax in float32 |
| everything else | 24.0% | Indexing, reshape, add, mul, pow, … |

The headline result: **`aten::mm` and `aten::bmm` together account for 65.7% of decode time.** Everything else combined is the minority.

There are 197 `mm` calls per decode step (28 blocks × 7 linear layers = 196, plus one for `out_head`) and 56 `bmm` calls (28 blocks × 2 — one for Q×Kᵀ, one for attn×V). At the measured throughput of ~12 tok/s, each `mm` averages ~116 µs and each `bmm` averages ~464 µs.

## The bandwidth argument

During a decode step, the input is a single 1024-dimensional token vector. Every linear layer computes a matrix–vector product: `y = Wx`. For a weight matrix of shape `[out, in]`, this touches `out × in` elements — the entire matrix — to produce a single output vector. There is no reuse.

The total parameter count (weights only, bfloat16):

| Component | Parameters | Bytes |
|-----------|-----------|-------|
| 28 blocks × 7 linear layers | ~404 M | 808 MB |
| tok\_emb / out\_head (tied) | 156 M | 312 MB |
| RMSNorm scales, KV-norm scales | ~0.3 M | 0.6 MB |
| **Total** | **~560 M** | **~1,121 MB** |

On every single decode step — every token generated — the model reads all ~1.1 GB of weights from memory. At ~12 tok/s on CPU that is **13 GB/s** of sustained weight-read bandwidth. At 38 tok/s on GPU it is **42 GB/s**.

This makes the system *memory-bandwidth-bound*. For a matrix–vector multiply the arithmetic intensity is:

```
intensity = FLOPs / bytes
          = (2 × out × in) / (2 × out × in)   [bf16, one multiply-add each]
          = 1 FLOP / byte
```

Modern hardware can sustain far more FLOPs per byte than 1. The Apple M3 Pro does ~3.6 TFLOPS of BF16 compute over ~150 GB/s of memory bandwidth — a hardware ridge point of ~24 FLOP/byte. We are 24× below the ridge. The GPU picture is worse: an A100 has a ridge around 156 FLOP/byte. We are 156× below it.

This means the CUDA cores (or M3 ANE) sit idle for the vast majority of each decode step, waiting for weights to arrive from DRAM. Adding more compute (bigger GPU, faster clock, fused kernels) does not help at all. Reducing the bytes per weight does.

### What changes with quantization

| Scheme | Bytes/weight | Intensity | Bandwidth at 38 tok/s |
|--------|-------------|-----------|----------------------|
| bfloat16 (current) | 2.0 | 1.0 FLOP/byte | 42 GB/s |
| int8 | 1.0 | 2.0 FLOP/byte | 21 GB/s |
| int4 | 0.5 | 4.0 FLOP/byte | 10.5 GB/s |

If bandwidth is the limit, int8 should roughly double throughput and int4 should quadruple it. In practice, dequantize-on-forward adds a small overhead (unpacking + scale multiply), but the direction is clear.

## Worth looking at in the code

The coarse profiler in `src/pumpference/profile.py` uses `register_forward_pre_hook` and `register_forward_hook` to instrument any `nn.Module` without touching its source:

```python
def make_hooks(n: str):
    start_ref: list[float] = []

    def pre_hook(module, args):
        _sync(device)
        start_ref.clear()
        start_ref.append(time.perf_counter())

    def post_hook(module, args, output):
        _sync(device)
        elapsed_ms = (time.perf_counter() - start_ref[0]) * 1000.0
        timings[n].append(elapsed_ms)

    return pre_hook, post_hook
```

`start_ref` is a one-element list rather than a plain float so the closure can mutate it — Python closures can rebind names inside a list but cannot rebind a plain scalar. The handles returned by `register_forward_pre_hook` and `register_forward_hook` are collected and removed after the timing loop, leaving the model in its original state.

## What's next

The profile identifies the fix: every decode step reads the full ~1.1 GB of model weights from DRAM, and there is nothing to be done about the *number* of reads — we need all the weights. But we can shrink *how many bytes* each weight occupies. That is quantization.
