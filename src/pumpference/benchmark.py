"""
Benchmark harness for pumpference inference.

Measures prefill/decode TPS, TTFT, peak memory, and per-token latency.

Usage:
    uv run python -m pumpference.benchmark                    # default: xs (~30 tok)
    uv run python -m pumpference.benchmark --preset short     # ~115 tokens
    uv run python -m pumpference.benchmark --preset medium    # ~218 tokens
    uv run python -m pumpference.benchmark --preset long      # ~373 tokens
    uv run python -m pumpference.benchmark --preset 373       # numeric aliases still work
    uv run python -m pumpference.benchmark --prompt "Hello"   # custom prompt
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, quantiles

import torch

from .model import QWEN3_0_6B_CONFIG, KVCache, Qwen3Model, download_and_load_weights
from .quantize import quantize_model
from .tokenizer import download_tokenizer


# ---------------------------------------------------------------------------
# Preset prompts — four lengths that reveal different cost regimes
# ---------------------------------------------------------------------------
#
# ~30 tokens
_PROMPT_30 = (
    "Explain what a transformer neural network is and why the attention "
    "mechanism is its central component. Focus on the intuition behind "
    "computing weighted sums over value vectors."
)

# ~115 tokens
_PROMPT_115 = (
    "You are a helpful AI assistant. The user has asked you to explain "
    "the difference between encoder-only models like BERT and decoder-only "
    "models like GPT. Encoder-only models process the full input sequence "
    "bidirectionally: every token attends to every other token. This makes "
    "them excellent for tasks that require understanding the whole context, "
    "such as sentence classification or named entity recognition. "
    "Decoder-only models, on the other hand, use a causal mask so that "
    "each token can only attend to previous tokens. This makes them suited "
    "for text generation. Please give a concise summary of these trade-offs."
)

# ~218 tokens
_PROMPT_218 = (
    "You are a senior machine learning engineer writing internal documentation. "
    "A junior engineer has asked you to explain how rotary positional embeddings "
    "work and why they have largely replaced learned absolute position embeddings "
    "in modern large language models. "
    "Learned absolute embeddings assign a fixed vector to each position index. "
    "They are simple but do not generalise well beyond the maximum sequence "
    "length seen during training — the model has never seen position 1025 if "
    "it was only trained up to 1024. "
    "Rotary embeddings (RoPE) instead rotate the query and key vectors in the "
    "attention computation by an angle proportional to the token position. "
    "The rotation is applied in the frequency domain using pairs of dimensions, "
    "and the dot product between a rotated query at position m and a rotated key "
    "at position n depends only on their relative offset m-n. This relative "
    "encoding generalises naturally to unseen sequence lengths. "
    "Additionally, RoPE requires no extra parameters — the rotation angles are "
    "computed deterministically from the position and a base frequency theta. "
    "Explain all of this clearly and add a short example of how the rotation "
    "is applied to a single dimension pair."
)

# ~373 tokens
_PROMPT_373 = (
    "You are an expert in large language model inference optimisation. "
    "A team of engineers is preparing to deploy a 7-billion-parameter "
    "decoder-only transformer model in a production serving system. "
    "They have asked you to prepare a technical report covering the most "
    "important optimisations they should consider, in order of expected impact. "
    "The first and most important optimisation is the KV-cache. Without it, "
    "the model must re-compute the key and value tensors for every token in "
    "the context on every generation step. With a prompt of length P and a "
    "generation of length G, the total compute is proportional to "
    "P squared plus (P+1) squared up to (P+G) squared, which is clearly "
    "quadratic in the context length. "
    "A KV-cache stores the key and value tensors computed during the prefill "
    "pass and reuses them on subsequent steps. Each decode step then only "
    "computes new K and V tensors for the single new token, making decode "
    "cost O(n) per step instead of O(n squared). This is the single largest "
    "speedup available and should always be implemented first. "
    "The second important optimisation is quantisation. Storing model weights "
    "in 4-bit or 8-bit integers rather than bfloat16 reduces memory bandwidth "
    "requirements proportionally. On memory-bandwidth-bound hardware, which "
    "most inference workloads are, this translates directly into higher "
    "tokens-per-second throughput. The trade-off is a small degradation in "
    "output quality, which is usually acceptable for chat or completion tasks. "
    "The third optimisation is continuous batching, also known as iteration-level "
    "scheduling. A naive serving system processes one request at a time. A "
    "continuous batching system fills idle compute with tokens from other "
    "in-flight requests, dramatically increasing GPU utilisation. "
    "Please elaborate on each of these three techniques, describe when each "
    "one should be prioritised, and suggest how to measure the improvement "
    "each one provides in a production environment."
)

# Keys are approximate token counts, verified against the Qwen3 tokenizer.
PRESETS: dict[int, str] = {
    30: _PROMPT_30,
    115: _PROMPT_115,
    218: _PROMPT_218,
    373: _PROMPT_373,
}

# Human-readable aliases for the four presets.
PRESET_ALIASES: dict[str, int] = {
    "xs": 30,
    "short": 115,
    "medium": 218,
    "long": 373,
}

# Short prompts used as AWQ calibration data.
_CALIBRATION_PROMPTS = [_PROMPT_30, _PROMPT_115]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    timestamp: str
    git_commit: str
    device: str
    dtype: str
    quantization: str
    model: str
    prompt_tokens: int
    generated_tokens: int
    prefill_ms: float
    decode_total_ms: float
    ttft_ms: float
    prefill_tps: float
    decode_tps: float
    peak_memory_mb: float
    flash_attn: bool = False
    decode_step_latencies_ms: list[float] = field(default_factory=list)
    per_token_latency_ms: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {}
    pcts = quantiles(latencies_ms, n=100) if len(latencies_ms) >= 2 else latencies_ms
    return {
        "mean": round(mean(latencies_ms), 2),
        "p50": round(pcts[49] if len(pcts) > 49 else latencies_ms[0], 2),
        "p90": round(pcts[89] if len(pcts) > 89 else latencies_ms[-1], 2),
        "p99": round(pcts[98] if len(pcts) > 98 else latencies_ms[-1], 2),
    }


# ---------------------------------------------------------------------------
# Instrumented generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def timed_generate(
    model: Qwen3Model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
    device: torch.device,
    quantization: str = "none",
    flash_attn: bool = False,
) -> tuple[torch.Tensor, BenchmarkResult]:
    """
    Run greedy generation with KV-cache while measuring prefill, decode, and memory.

    Two-phase structure mirrors the cached generate() loop so the timings are
    meaningful: prefill covers the full-prompt forward pass that fills the
    cache, and decode measures the per-token cost when only one token is fed.
    """
    model.eval()
    prompt_len = input_ids.shape[1]
    cache = KVCache()

    # --- Memory tracking setup ---
    use_cuda_mem = device.type == "cuda"
    if use_cuda_mem:
        torch.cuda.reset_peak_memory_stats(device)

    # --- Prefill (full prompt, fills the cache) ---
    _sync(device)
    t_prefill_start = time.perf_counter()

    logits = model(input_ids, kv_cache=cache)
    next_token = logits[:, -1].argmax(dim=-1, keepdim=True)

    _sync(device)
    t_prefill_end = time.perf_counter()

    prefill_ms = (t_prefill_end - t_prefill_start) * 1000.0
    generated = [next_token]
    hit_eos = eos_token_id is not None and next_token.item() == eos_token_id

    # --- Decode (single token per step, cache grows by one each step) ---
    decode_step_latencies: list[float] = []

    if not hit_eos:
        for _ in range(max_new_tokens - 1):
            _sync(device)
            t_step_start = time.perf_counter()

            logits = model(next_token, kv_cache=cache)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)

            _sync(device)
            t_step_end = time.perf_counter()

            decode_step_latencies.append((t_step_end - t_step_start) * 1000.0)
            generated.append(next_token)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    tokens = torch.cat([input_ids, *generated], dim=1)

    # --- Memory measurement ---
    if use_cuda_mem:
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        # ru_maxrss: peak RSS in bytes on macOS, kilobytes on Linux
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            peak_memory_mb = rss / (1024 * 1024)
        else:
            peak_memory_mb = rss / 1024

    # --- Compute metrics ---
    generated_tokens = tokens.shape[1] - prompt_len
    decode_total_ms = sum(decode_step_latencies)
    decode_tokens = max(generated_tokens - 1, 0)

    result = BenchmarkResult(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        git_commit=_git_commit(),
        device=str(device),
        dtype=str(QWEN3_0_6B_CONFIG.dtype).replace("torch.", ""),
        quantization=quantization,
        model=QWEN3_0_6B_CONFIG.repo_id.split("/")[-1],
        prompt_tokens=prompt_len,
        generated_tokens=generated_tokens,
        prefill_ms=round(prefill_ms, 2),
        decode_total_ms=round(decode_total_ms, 2),
        ttft_ms=round(prefill_ms, 2),
        prefill_tps=round(prompt_len / (prefill_ms / 1000.0), 1) if prefill_ms > 0 else 0.0,
        decode_tps=round(decode_tokens / (decode_total_ms / 1000.0), 1) if decode_total_ms > 0 else 0.0,
        peak_memory_mb=round(peak_memory_mb, 1),
        flash_attn=flash_attn,
        decode_step_latencies_ms=[round(x, 2) for x in decode_step_latencies],
        per_token_latency_ms=_latency_stats(decode_step_latencies),
    )

    return tokens, result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

W = 51  # table width

def format_results(result: BenchmarkResult) -> str:
    lat = result.per_token_latency_ms
    quant_label = result.quantization if result.quantization != "none" else "none (bfloat16)"
    flash_label = "on" if result.flash_attn else "off"
    lines = [
        "═" * W,
        f"  pumpference benchmark — {result.model}",
        "═" * W,
        f"  Device:             {result.device} ({result.dtype})",
        f"  Quantization:       {quant_label}",
        f"  Flash attention:    {flash_label}",
        f"  Prompt tokens:      {result.prompt_tokens}",
        f"  Generated tokens:   {result.generated_tokens}",
        "─" * W,
        "  Prefill",
        f"    Time:             {result.prefill_ms:.1f} ms",
        f"    Tokens/sec:       {result.prefill_tps:.1f}",
        "  Decode",
        f"    Time:             {result.decode_total_ms:.1f} ms",
        f"    Tokens/sec:       {result.decode_tps:.1f}",
        "  Latency (decode)",
        f"    Mean:             {lat.get('mean', 0):.1f} ms/tok",
        f"    P50:              {lat.get('p50', 0):.1f} ms/tok",
        f"    P99:              {lat.get('p99', 0):.1f} ms/tok",
        f"  TTFT:               {result.ttft_ms:.1f} ms",
        f"  Peak memory:        {result.peak_memory_mb:.1f} MB",
        "─" * W,
    ]
    return "\n".join(lines)


def _save_json(result: BenchmarkResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = result.device.replace(":", "_")
    tokens = f"{result.prompt_tokens}tok"
    path = output_dir / f"{ts}_{device}_{tokens}.json"
    path.write_text(json.dumps(asdict(result), indent=2) + "\n")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_preset(value: str) -> str:
    """Accept a named alias (xs/short/medium/long) or a numeric token count."""
    if value in PRESET_ALIASES:
        return value
    try:
        key = int(value)
    except ValueError:
        valid = ", ".join(PRESET_ALIASES) + ", " + ", ".join(str(k) for k in PRESETS)
        raise argparse.ArgumentTypeError(
            f"invalid preset {value!r}. Valid values: {valid}"
        )
    if key not in PRESETS:
        raise argparse.ArgumentTypeError(
            f"numeric preset {key} not found. Valid counts: {list(PRESETS)}"
        )
    return str(key)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark pumpference inference (prefill/decode TPS, TTFT, memory)",
    )

    _alias_help = " | ".join(
        f"{name}≈{tok}tok" for name, tok in PRESET_ALIASES.items()
    )
    _num_help = ", ".join(str(k) for k in PRESETS)

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--prompt",
        type=str,
        help="Custom text prompt",
    )
    group.add_argument(
        "--preset",
        type=_resolve_preset,
        default="xs",
        metavar="PRESET",
        help=(
            f"Named alias ({_alias_help}) or token count ({_num_help}). "
            "Default: xs"
        ),
    )

    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs before timing")
    parser.add_argument("--output-dir", type=str, default="benchmarks")
    parser.add_argument(
        "--quantize",
        choices=["none", "int8", "int4", "awq_int8", "awq_int4"],
        default="none",
        help=(
            "Weight-only quantization scheme. "
            "int8/int4: plain RTN (no calibration). "
            "awq_int8/awq_int4: AWQ calibration-based (better quality, slower setup)."
        ),
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        default=False,
        help="Use tiled Flash Attention during prefill (O(n) memory vs O(n²)).",
    )
    args = parser.parse_args()

    # Resolve prompt — named alias → token count → prompt text
    if args.prompt:
        prompt = args.prompt
    else:
        raw = args.preset
        if raw in PRESET_ALIASES:
            preset_key = PRESET_ALIASES[raw]
        else:
            preset_key = int(raw)
        prompt = PRESETS[preset_key]

    # --- Device ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # --- Load model + tokenizer ---
    print(f"Loading model on {device} …")
    cfg = replace(QWEN3_0_6B_CONFIG, use_flash_attn=args.flash_attn)
    model = Qwen3Model(cfg)
    download_and_load_weights(model, repo_id=cfg.repo_id)
    tokenizer = download_tokenizer(repo_id=QWEN3_0_6B_CONFIG.repo_id)
    if args.quantize != "none":
        print(f"Quantizing weights ({args.quantize}) …")
        if args.quantize.startswith("awq"):
            print("  Running AWQ calibration (collecting activation statistics) …")
            cal_ids = [
                torch.tensor([tokenizer.encode(p)], device=device)
                for p in _CALIBRATION_PROMPTS
            ]
            quantize_model(model, mode=args.quantize, calibration_ids=cal_ids)
        else:
            quantize_model(model, mode=args.quantize)
    model.to(device)
    model.eval()
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)], device=device,
    )

    # --- Warmup ---
    for i in range(args.warmup):
        print(f"  warmup {i + 1}/{args.warmup} …")
        _ = model(input_ids)
        _sync(device)

    # --- Benchmark ---
    print("Running benchmark …")
    _, result = timed_generate(
        model,
        input_ids,
        max_new_tokens=args.max_tokens,
        eos_token_id=tokenizer.eos_token_id,
        device=device,
        quantization=args.quantize,
        flash_attn=args.flash_attn,
    )

    # --- Output ---
    print()
    print(format_results(result))
    json_path = _save_json(result, Path(args.output_dir))
    print(f"  Results saved to {json_path}")
    print("═" * W)


if __name__ == "__main__":
    main()
