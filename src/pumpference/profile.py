"""
Decode-step profiler for pumpference.

Two complementary views of where per-token time goes:

  1. Coarse — manual timing per transformer block using forward hooks
     and torch.cuda.synchronize() on GPU.  Shows which layers dominate.

  2. Fine — torch.profiler trace over a handful of decode steps.
     Shows the top PyTorch operators (aten::mm, aten::addmm, etc.)
     and their contribution as percentages.

Usage:
    uv run python -m pumpference.profile [--preset xs] [--device auto] [--trace out.json]
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch

from .benchmark import PRESET_ALIASES, PRESETS, _resolve_preset
from .model import QWEN3_0_6B_CONFIG, KVCache, Qwen3Model, download_and_load_weights
from .tokenizer import download_tokenizer


# ---------------------------------------------------------------------------
# Hook-based coarse profiler
# ---------------------------------------------------------------------------

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _coarse_profile(
    model: Qwen3Model,
    input_ids: torch.Tensor,
    device: torch.device,
    n_decode_steps: int = 10,
) -> dict[str, float]:
    """
    Time each named submodule of the model over *n_decode_steps* decode steps.

    Returns a dict {module_name -> mean_ms} for every module that has
    a registered hook.  Only the decode phase is measured (single-token
    forward); the prefill warmup is excluded from timing.
    """
    # --- Prefill to populate the KV-cache ----------------------------------
    cache = KVCache()
    with torch.no_grad():
        _ = model(input_ids, kv_cache=cache)

    # --- Register forward hooks on the modules we care about ---------------
    # We time: each TransformerBlock and the four surrounding ops.
    names_to_module: dict[str, torch.nn.Module] = {
        "tok_emb": model.tok_emb,
        **{f"block_{i:02d}": block for i, block in enumerate(model.trf_blocks)},
        "final_norm": model.final_norm,
        "out_head": model.out_head,
    }

    timings: dict[str, list[float]] = defaultdict(list)
    handles = []

    for name, mod in names_to_module.items():
        def make_hooks(n: str):
            start_ref: list[float] = []

            def pre_hook(module, args):  # noqa: ARG001
                _sync(device)
                start_ref.clear()
                start_ref.append(time.perf_counter())

            def post_hook(module, args, output):  # noqa: ARG001
                _sync(device)
                elapsed_ms = (time.perf_counter() - start_ref[0]) * 1000.0
                timings[n].append(elapsed_ms)

            return pre_hook, post_hook

        pre, post = make_hooks(name)
        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))

    # --- Decode steps -------------------------------------------------------
    last_token = input_ids[:, -1:]
    with torch.no_grad():
        for _ in range(n_decode_steps):
            logits = model(last_token, kv_cache=cache)
            last_token = logits[:, -1].argmax(dim=-1, keepdim=True)

    # --- Cleanup ------------------------------------------------------------
    for h in handles:
        h.remove()

    return {name: sum(ts) / len(ts) for name, ts in timings.items() if ts}


def _print_coarse_table(timings: dict[str, float]) -> None:
    total = sum(timings.values())

    rows: list[tuple[str, float, float]] = []
    for name, ms in timings.items():
        rows.append((name, ms, 100.0 * ms / total))

    W = 54
    print()
    print("═" * W)
    print("  Decode step — coarse timing (mean over 10 steps)")
    print("═" * W)
    print(f"  {'Module':<22}  {'ms/step':>8}  {'%':>7}")
    print("─" * W)
    for name, ms, pct in rows:
        print(f"  {name:<22}  {ms:>8.3f}  {pct:>6.1f}%")
    print("─" * W)
    print(f"  {'TOTAL':<22}  {total:>8.3f}  {'100.0%':>7}")
    print("═" * W)
    print()


# ---------------------------------------------------------------------------
# torch.profiler fine-grained breakdown
# ---------------------------------------------------------------------------

def _fine_profile(
    model: Qwen3Model,
    input_ids: torch.Tensor,
    device: torch.device,
    n_decode_steps: int = 5,
    trace_path: Path | None = None,
) -> None:
    """
    Run torch.profiler over *n_decode_steps* decode steps and print the
    top-20 operators by self CPU time (and self CUDA time if applicable).
    Optionally exports a Chrome-compatible trace to *trace_path*.
    """
    cache = KVCache()
    with torch.no_grad():
        _ = model(input_ids, kv_cache=cache)

    last_token = input_ids[:, -1:]

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_flops=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(n_decode_steps):
                logits = model(last_token, kv_cache=cache)
                last_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                prof.step()

    # Export trace for Chrome://tracing if requested
    if trace_path is not None:
        prof.export_chrome_trace(str(trace_path))
        print(f"  Chrome trace saved → {trace_path}")

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print()
    print("═" * 54)
    print("  Decode step — top operators (torch.profiler)")
    print("═" * 54)
    print(prof.key_averages().table(
        sort_by=sort_key,
        row_limit=20,
    ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile pumpference decode step (coarse + torch.profiler)"
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prompt", type=str)
    group.add_argument("--preset", type=_resolve_preset, default="xs")

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--trace",
        type=str,
        default=None,
        metavar="PATH",
        help="Export Chrome trace to this path (e.g. profile.json)",
    )
    args = parser.parse_args()

    # --- Resolve prompt -----------------------------------------------------
    if args.prompt:
        prompt = args.prompt
    else:
        raw = args.preset
        preset_key = PRESET_ALIASES[raw] if raw in PRESET_ALIASES else int(raw)
        prompt = PRESETS[preset_key]

    # --- Device -------------------------------------------------------------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # --- Load model ---------------------------------------------------------
    print("Loading model …")
    model = Qwen3Model(QWEN3_0_6B_CONFIG)
    download_and_load_weights(model, repo_id=QWEN3_0_6B_CONFIG.repo_id)
    model.to(device)
    model.eval()

    tokenizer = download_tokenizer(repo_id=QWEN3_0_6B_CONFIG.repo_id)
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    print(f"Prompt: {input_ids.shape[1]} tokens")

    # --- Warmup -------------------------------------------------------------
    print("Warming up …")
    with torch.no_grad():
        _ = model(input_ids)
    _sync(device)

    # --- Coarse profiling ---------------------------------------------------
    print("Coarse profiling (10 decode steps) …")
    timings = _coarse_profile(model, input_ids, device, n_decode_steps=10)
    _print_coarse_table(timings)

    # --- Fine profiling (torch.profiler) ------------------------------------
    print("Fine profiling (5 decode steps) …")
    trace_path = Path(args.trace) if args.trace else None
    _fine_profile(model, input_ids, device, n_decode_steps=5, trace_path=trace_path)


if __name__ == "__main__":
    main()
