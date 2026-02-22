# Part 2: Sampling — Temperature, Top-k, and Nucleus Decoding

*Pumpference series — from zero to a working inference framework in plain PyTorch*

---

At the end of Tutorial 1, the model worked. You type a prompt, it generates text. Every matrix multiply is correct, every precision gotcha is handled, the logits match HuggingFace's to within floating-point noise. That felt like enough.

Then I started actually talking to it. And I noticed a problem.

Ask it to tell you something interesting and it gives you the most statistically average interesting thing. Ask it to write the opening of a story and it produces the most probable continuation of that genre of story across all training data. Every sentence is technically correct, grammatically clean, and completely predictable. It sounds like a committee's first draft. Safe. Generic. Like something that was generated rather than said.

That's what `argmax` does. The model produces a probability distribution over 151,936 possible next tokens — a rich, textured probability landscape shaped by everything it learned. `argmax` looks at all 151,936 possibilities and picks one. The remaining 151,935 might as well not exist.

This is optimal when there is one right answer. "The capital of France is ___" has one answer, and greedy will find it reliably. But most generation isn't like that. "Once upon a time" could continue a thousand different ways. Greedy will always tell the most average story.

The fix is to actually *sample* from the distribution. Not always grab the peak, but draw a token according to its probability. This is one line of code. Controlling *how* you sample takes three parameters. Getting the details right is more interesting than it sounds.

---

## Temperature

Temperature T rescales logits before softmax. Divide by T < 1 and the distribution sharpens — high-probability tokens become even more dominant. Divide by T > 1 and it flattens — probability spreads more evenly across candidates.

At T → 0, one token monopolizes the mass. You recover greedy. At T → ∞, every token is equally likely. At T = 1, you sample from the raw learned distribution.

The implementation is literally one line: `logits = logits / temperature`. That's it. The entire "temperature" concept, which you'll see referenced in every LLM paper and blog post, is a single division before softmax.

The practical range is roughly 0.3–1.5. Below 0.3, output becomes nearly deterministic (you're approaching greedy). Above 1.5, it becomes erratic (low-probability tokens show up regularly). For creative writing: 0.6–0.9. For factual work: 0.3–0.6. The classic interactive default: 0.7.

One implementation decision worth making explicit: **`temperature=0` is greedy argmax, not division by a small number.** Dividing logits by 1e-6 amplifies them so aggressively that floating-point noise can produce inconsistent results. And conceptually, `temperature=0` means maximum determinism, which is exactly what argmax gives you. Better to make it an explicit branch:

```python
if temperature == 0.0:
    return logits.argmax(dim=-1, keepdim=True)
```

This also preserves backward compatibility — all the benchmarks and tests from Tutorial 1 implicitly use greedy decoding. With `temperature=0` defaulting to argmax, nothing breaks.

---

## Top-k

Top-k is a hard vocabulary filter. Before sampling, remove all tokens except the k most probable ones. If k=50, only the top 50 candidates can be sampled. The rest get `-inf` (which becomes probability 0 after softmax).

```python
if top_k is not None:
    top_logits, _ = torch.topk(logits, top_k, dim=-1)
    threshold = top_logits[..., -1]
    logits = logits.masked_fill(logits < threshold, -torch.inf)
```

We use `torch.topk` rather than a full sort — we only need the k-th largest value as a threshold, not the sorted order. One subtlety: strict less-than (`<`) rather than less-than-or-equal. If multiple tokens share the k-th logit value, `<` keeps all of them. `<=` would incorrectly discard them all.

Think of top-k as a safety net more than a quality control. Even with well-calibrated temperature, some tokens in the long tail occasionally get enough probability to sneak through. Top-k prevents that by hard-cutting the vocabulary. When temperature is well set, top-k barely changes anything. When it's not, top-k prevents disasters.

---

## Top-p (Nucleus Sampling)

Top-k has a fundamental problem: the right number of candidates depends on the situation, and you're using the same k everywhere.

When the model is very confident — one token has probability 0.95 — a nucleus of size 1 or 2 is appropriate. You don't need 50 candidates; 49 of them are noise. When the model is genuinely uncertain — probability spread across hundreds of plausible continuations — you want a much larger nucleus to preserve that richness.

Top-p (nucleus sampling) adapts. Sort the vocabulary by probability descending, then keep the smallest prefix of tokens whose cumulative probability reaches p. Set p=0.9 and you're saying: "find the minimum set that accounts for 90% of the mass, sample from those." The nucleus shrinks when the model is confident, expands when it's not.

The implementation has one genuinely subtle step. You sort probabilities, compute cumulative sums, and want to remove all tokens where the cumsum *before* that token's contribution already exceeds p:

```python
sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
cumulative = sorted_probs.cumsum(dim=-1)
remove = (cumulative - sorted_probs) >= top_p
remove[..., 0] = False  # always keep the highest-probability token
sorted_probs[remove] = 0.0
probs.scatter_(-1, sorted_indices, sorted_probs)
```

The `cumulative - sorted_probs` is the shift that makes this correct. Here's why it matters.

Suppose p=0.9 and your sorted probabilities are `[0.5, 0.3, 0.15, 0.04, 0.01]`. The cumulative sums are `[0.5, 0.8, 0.95, 0.99, 1.0]`. A naive `cumulative >= 0.9` mask removes everything from index 2 onwards — including the token with probability 0.15, the one that *crossed* the threshold. But you want to keep it. It's part of the nucleus.

`cumulative - sorted_probs` gives you the cumsum *before* each position: `[0.0, 0.5, 0.8, 0.95, 0.99]`. Now `>= 0.9` masks only indices 3 and 4. The 0.15 token survives. That's the correct nucleus.

Three lines. Half this chapter is about these three lines. They're also where most implementations get it subtly wrong.

The `remove[..., 0] = False` handles the degenerate case where `top_p=0.0` (or the model's top token alone exceeds p) would otherwise mask everything, handing `multinomial` all-zero weights and a `RuntimeError`. The highest-probability token is always kept, no matter what.

---

## Composability

Temperature, top-k, and top-p compose in a fixed order and the order matters:

1. **Temperature**: scale logits — `logits = logits.float() / temperature`
2. **Top-k**: mask all but the k highest-logit tokens
3. **Top-p**: mask all but the cumulative-p nucleus
4. **Sample**: `torch.multinomial(F.softmax(logits, dim=-1), 1)`

Temperature first, because it changes the shape of the distribution before any masking decisions. Top-k before top-p, because top-p should run on the already-narrowed vocabulary (otherwise it can keep tokens that top-k would have excluded). Sample last.

The float32 cast at step 1 is non-negotiable. `torch.multinomial` can silently produce wrong samples from bfloat16 tensors on some backends — MPS in particular. The rule is the same as in attention: anything involving probability distributions gets float32.

---

## Key decisions

**Everything stays in `generate.py`.** The temptation when adding "the sampling module" is to create `sampling.py`. I resisted it. The file went from ~60 to ~110 lines. `sample_next_token` is 25 lines. Splitting that into a separate file would be modularization theater — the appearance of organization without the substance. If we later add beam search or speculative decoding, the split becomes obviously justified. Not yet.

**No resampling on rejection.** When top-k or top-p filter aggressively and very few tokens survive, we don't retry. We just sample from whatever's left. The `remove[..., 0] = False` guard ensures at least one token always has nonzero probability, so we never fail. Retrying adds complexity for a case that rarely matters in practice.

---

## The tricky parts

| Issue | Symptom | Fix |
|---|---|---|
| Top-p shift | Naive cumsum masks the crossing token, collapsing the nucleus prematurely | `remove = (cumulative - sorted_probs) >= top_p` — subtract before comparing |
| `top_p=0.0` edge case | All tokens masked → `multinomial` receives zero weights → `RuntimeError` | `remove[..., 0] = False` unconditionally |
| bfloat16 multinomial | Silently wrong samples on MPS/CPU | Cast to float32 before temperature scaling; never cast back |
| Top-p + top-k order | Top-p on full vocab can keep tokens top-k would exclude | Always top-k first, then top-p on the narrowed distribution |

---

## Results

Sampling adds no measurable overhead. `sample_next_token` takes microseconds. The forward pass takes ~1 second. The throughput numbers from Tutorial 1 are unchanged — sampling doesn't touch the model, only the post-processing of its final output.

What changes is output character. With the prompt "Tell me an interesting fact about the ocean":

| Settings | Output character |
|---|---|
| `temperature=0` (greedy) | Deterministic; polished, but tends to reuse the same sentence structures |
| `temperature=0.7, top_k=50` | Varied phrasing; coherent sentences you wouldn't have predicted |
| `temperature=1.0, top_p=0.9` | Noticeably creative; occasional word choice that surprises |
| `temperature=1.5` | Erratic; interesting tokens, sometimes incoherent assembly |

The practical default for interactive use: `temperature=0.7, top_p=0.9`. For benchmarks and reproducible tests: `temperature=0`. Greedy, always. Determinism is a feature when you're trying to measure something.

We also have 17 unit tests in `tests/test_sampling.py`, none of which require loading the model: greedy equivalence, seed reproducibility, top-k/top-p correctness, composability, output shapes. They run in milliseconds. When the implementation is 25 lines, thorough testing is essentially free.

---

## Worth looking at in the code

The full implementation is in [`src/pumpference/generate.py`](../src/pumpference/generate.py). Read `sample_next_token` — it's 25 lines and the entire chapter is in there.

The structure of the function is: early return for greedy → temperature scaling and float32 cast → optional top-k mask → optional top-p mask with the shift → softmax → multinomial. No hidden complexity. What you read is what happens.

---

## What's next

The model is correct and now more expressive. But here's what's been nagging at me since Tutorial 1: I wrote "about 1 tok/s" as a performance number. I got that by running a few generations, watching the clock, doing rough arithmetic in my head. That's not a measurement.

It doesn't tell me whether the slowness is in prefill or decode. It doesn't give me a latency distribution. It doesn't record which version of the code produced the number. And most importantly — before I build the KV-cache that will fix the O(n²) decode problem — I need a baseline I can actually trust.

**Tutorial 3**: Building a proper benchmark harness. Before we optimize, we measure.

---

*This is part of the Pumpference tutorial series. Source code: [github.com/legchikov/pumpference](https://github.com/legchikov/pumpference).*

*Found an error? Open an issue.*
