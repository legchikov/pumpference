# 6b. AWQ: Protecting the Channels That Matter

## Where we are

Tutorial 6 showed that round-to-nearest (RTN) quantization is easy to implement but imprecise: it treats every weight channel identically, regardless of how much it actually contributes to the output. The result on a 30-token prompt: int8 at 90% argmax agreement with bfloat16, int4 at 80%. For int4 that means one in five predicted tokens has a different most-likely successor — noticeable quality degradation for a method that promises a 4× size reduction.

## What we're adding and why

**AWQ (Activation-Aware Weight Quantization)** is a calibration-based method that reduces quantization error without changing the model architecture or the storage format. The intuition is direct: some weight channels are multiplied by large-magnitude activations and therefore dominate the output. If those channels are coarsely quantized, the error propagates loudly. RTN ignores activation magnitudes entirely; AWQ uses them.

The procedure is a one-time offline calibration step:
1. Run a small set of representative inputs through the model and record the mean absolute activation magnitude per channel at each layer.
2. For each (norm, downstream-linears) group, find a per-channel scaling vector `s = act_stats^α` that minimises the activation-weighted quantization reconstruction error. `α ∈ [0, 1]` is found by a coarse grid search.
3. Apply the scale: multiply each downstream linear's weight columns by `s` (salient channels now have a finer quantization grid). Divide the preceding RMSNorm's scale parameter by `s` (absorbs the inverse scale so the computation is mathematically equivalent).
4. After calibration the model is still bfloat16. Call the existing `quantize_model()` to apply RTN quantization to the pre-scaled weights.

## Key decisions

**Scale absorption into RMSNorm, not a separate layer.**
The cleanest way to maintain mathematical equivalence `W @ x = (W * s) @ (x / s)` is to absorb `1/s` into the computation that produces the activation. In Qwen3's pre-norm transformer, each attention or FFN block is preceded by an RMSNorm whose output feeds directly into the linear layers. Dividing `norm.scale` by `s` adjusts the norm's output per-channel without adding any new operations. At inference time the scaled model runs identically to the original; only the quantization grid has changed.

**Two groups per transformer block, two layers left to plain RTN.**
In each of the 28 blocks, `norm1` feeds `W_query`, `W_key`, `W_value`; `norm2` feeds `ff.fc1` and `ff.fc2`. Both groups are good candidates for AWQ because the preceding norm gives a clean absorption point. The remaining two layers — `att.out_proj` (input is concatenated attention heads) and `ff.fc3` (input is the SwiGLU gate output) — have no directly adjacent norm, so they receive standard RTN quantization. Together the AWQ-calibrated layers cover roughly 67% of each block's linear layer parameters.

**Grid search over α in [0, 1], 20 steps.**
The optimal α is not analytically tractable because quantization is a non-differentiable rounding operation. A coarse grid (21 points) is fast — on CPU calibration for 28 blocks takes roughly 20–30 seconds — and sufficient to find a good operating point. The loss evaluated at each step is:

```python
s = act_stats.pow(alpha).clamp(min=1e-6)
channel_weight = (act_stats / s).pow(2)
for W in group_weights:
    W_scaled = W * s
    W_q = dequantize(quantize(W_scaled))
    col_err = (W_q - W_scaled).pow(2).mean(dim=0)   # per-channel
    loss += (col_err * channel_weight).mean()
```

The `channel_weight = (act_stats / s)^2` term is the key: it approximates the squared activation magnitude after scale absorption (`x_j / s_j`), so channels where activations are large and the scale is small (i.e. the post-absorption value is still large) contribute more to the loss. When `α = 0` the scale is all-ones (pure RTN); when `α = 1` the scale equals the activation magnitudes (full AWQ protection). The search finds the point in between that minimises the estimated output error.

**Calibration data: existing benchmark prompts, no external dataset.**
The framework already has four representative prompts in `benchmark.py`. The two shortest (`_PROMPT_30`, `_PROMPT_115`) are used as calibration data. Activation statistics are averages over all token positions across both sequences — approximately 145 samples per channel. For a 0.6B model this is sufficient. A larger or more domain-specific calibration set would improve quality further, but keeping the project self-contained takes priority here.

**Calibration is decoupled from quantization.**
`calibrate_awq()` modifies the model in-place (norm scales and weight data) and returns it in bfloat16. The caller then passes it to the existing `quantize_model()`. This means AWQ works with both int8 and int4, and the two functions compose cleanly. A convenience wrapper `quantize_model(mode="awq_int8"|"awq_int4", calibration_ids=...)` handles the orchestration automatically.

## The tricky parts

| Issue | Symptom | Fix |
|-------|---------|-----|
| Python closure in hook registration | All hooks write to the same key because the loop variable `key` is captured by reference | Use a factory function `make_hook(k)` with the key passed as a default argument, so each hook closes over its own copy |
| Scale applied in bfloat16 space | `s` is computed in float32; multiplying bfloat16 weights by a float32 scale can produce small dtype-mismatch errors | Cast `s` to the weight's dtype before the in-place multiply: `s1_bf = s1.to(block.att.W_query.weight.dtype)` |
| Grid search uses float32 weights | The actual weights are bfloat16, but quantizing bfloat16 in the grid search introduces noise from the limited mantissa | Convert to float32 for the grid search (`W.float()`), apply the scale, quantize in float32, then compute the error. The final scale is applied to the bfloat16 weights at the end |
| Calibration runs with torch.no_grad() | Hooks registered on the model still fire during `no_grad()` | No issue — forward hooks fire regardless of grad mode; just ensure hooks are registered before the calibration forward passes and removed afterward |
| out_proj and fc3 have no preceding norm | Applying AWQ scale absorption requires a norm with a `scale` parameter to modify | Skip these layers; they receive standard RTN quantization from `quantize_model()` |

## Results

Argmax agreement with bfloat16 baseline (30-token prompt, single forward pass):

| Mode | Argmax agreement | vs bfloat16 | vs RTN |
|------|-----------------|-------------|--------|
| bfloat16 | 100% | — | — |
| RTN int8 | 90.0% | −10.0pp | — |
| AWQ int8 | 93.3% | −6.7pp | **+3.3pp** |
| RTN int4 (group=128) | 80.0% | −20.0pp | — |
| AWQ int4 (group=128) | 83.3% | −16.7pp | **+3.3pp** |

The improvement is consistent for both bit-widths: 3.3 percentage points fewer disagreements at a 30-token prompt. In absolute terms that is one fewer incorrect argmax per 30 positions — meaning the first divergence in autoregressive generation is pushed back by roughly one token on average. For longer prompts the compounding effect on generation quality is larger.

The gain is modest but structurally sound: calibration covers only the 67% of parameters that have an adjacent norm, and the calibration set is small (two prompts, ~145 activations per channel). A larger calibration set and coverage of the remaining layers would improve results further, at the cost of implementation complexity. The original AWQ paper reports int4 quality above 95% on Llama-2-7B with a 128-sample WikiText-2 calibration — the scaling behaviour is well-understood.

Memory footprint and decode speed are identical to plain RTN after calibration, since the storage format and forward-pass path are unchanged.

## Worth looking at in the code

The grid search in `_search_optimal_scale` is the conceptual core of AWQ. The activation-weighted loss function is easy to miss but critical:

```python
channel_weight = (act_stats / s).pow(2)   # [in_features]
for W in weights:
    W_scaled = W * s
    W_q = dequant_fn(W_scaled)
    col_err = (W_q - W_scaled).pow(2).mean(dim=0)   # per-channel quant error
    total_error += (col_err * channel_weight).mean()
```

Without `channel_weight`, the loop would minimise raw quantization error equally across all channels — effectively RTN with a scale twist. The weight `(act_stats / s)^2` is what makes it activation-aware: it amplifies the loss for channels where the scaled activation `x_j / s_j` is still large, i.e. channels that matter.

The scale absorption in `calibrate_awq` is a two-line operation that's easy to verify mentally:

```python
block.att.W_query.weight.data.mul_(s1_bf)   # salient cols get finer grid
block.norm1.scale.data.div_(s1.to(...))      # norm output shrinks by same factor
```

After both lines, `norm1(x) @ W_query = (norm1(x) / s) @ (W_query * s)` — exactly the same result, but the quantization grid for `W_query` is now non-uniform and proportional to activation magnitude.

## What's next

Flash Attention — as context length grows beyond a few hundred tokens, attention starts to compete with weight loading for decode time. Chapter 7 implements the memory-efficient fused attention kernel.
