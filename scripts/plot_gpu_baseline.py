"""Generate the GPU baseline benchmark plot for Tutorial 3."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).parent.parent

CPU_FILES = [
    ROOT / "benchmarks" / "20260217_182537_cpu_30tok.json",
    ROOT / "benchmarks" / "20260217_182635_cpu_115tok.json",
    ROOT / "benchmarks" / "20260217_182812_cpu_218tok.json",
    ROOT / "benchmarks" / "20260217_183034_cpu_372tok.json",
]

GPU_FILES = [
    ROOT / "benchmarks" / "20260222_191517_cuda_30tok.json",
    ROOT / "benchmarks" / "20260222_191633_cuda_115tok.json",
    ROOT / "benchmarks" / "20260222_191646_cuda_218tok.json",
    ROOT / "benchmarks" / "20260222_191659_cuda_372tok.json",
]

PRESET_LABELS = ["xs\n(30)", "short\n(115)", "medium\n(218)", "long\n(372)"]
X = np.arange(4)
WIDTH = 0.38

GREY = "#555555"
CPU_COLOR = "#4C72B0"
GPU_COLOR = "#DD8452"


def load(files: list[Path]) -> list[dict]:
    results = []
    for f in files:
        if f.exists():
            results.append(json.loads(f.read_text()))
        else:
            results.append(None)
    return results


def main() -> None:
    gpu = load(GPU_FILES)

    prompt_tok = [r["prompt_tokens"] for r in gpu]
    prefill_tps = [r["prefill_tps"] for r in gpu]
    decode_tps = [r["decode_tps"] for r in gpu]
    ttft_ms = [r["ttft_ms"] for r in gpu]
    lat_mean = [r["per_token_latency_ms"]["mean"] for r in gpu]
    lat_p50 = [r["per_token_latency_ms"]["p50"] for r in gpu]
    lat_p99 = [r["per_token_latency_ms"]["p99"] for r in gpu]
    peak_mb = [r["peak_memory_mb"] for r in gpu]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor("#f9f9f9")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cccccc")
        ax.tick_params(colors=GREY, labelsize=9)
        ax.set_xticks(range(4))
        ax.set_xticklabels(PRESET_LABELS, fontsize=8.5, color=GREY)

    # --- Panel 1: Prefill + Decode TPS ---
    ax = axes[0]
    bars1 = ax.bar(X - WIDTH / 2, prefill_tps, WIDTH, label="Prefill TPS", color=GPU_COLOR, alpha=0.85)
    bars2 = ax.bar(X + WIDTH / 2, decode_tps, WIDTH, label="Decode TPS", color=CPU_COLOR, alpha=0.85)
    ax.set_title("Throughput (tok/s)", fontsize=10, color=GREY, pad=8)
    ax.set_ylabel("tok/s", fontsize=9, color=GREY)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(fontsize=8, framealpha=0.5)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 60, f"{h:,.0f}",
                ha="center", va="bottom", fontsize=7, color=GREY)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 60, f"{h:.1f}",
                ha="center", va="bottom", fontsize=7, color=GREY)

    # --- Panel 2: Decode latency (mean, P50, P99) ---
    ax = axes[1]
    ax.plot(range(4), lat_p50, "o-", color=CPU_COLOR, linewidth=1.8, markersize=5, label="P50")
    ax.plot(range(4), lat_mean, "s--", color=GPU_COLOR, linewidth=1.5, markersize=5, label="Mean")
    ax.plot(range(4), lat_p99, "^:", color="#c44e52", linewidth=1.5, markersize=5, label="P99")
    ax.set_title("Decode latency (ms/tok)", fontsize=10, color=GREY, pad=8)
    ax.set_ylabel("ms / token", fontsize=9, color=GREY)
    ax.legend(fontsize=8, framealpha=0.5)

    # --- Panel 3: TTFT ---
    ax = axes[2]
    ax.bar(range(4), ttft_ms, color=GPU_COLOR, alpha=0.85, width=0.55)
    ax.set_title("Time-to-first-token (ms)", fontsize=10, color=GREY, pad=8)
    ax.set_ylabel("ms", fontsize=9, color=GREY)
    for i, v in enumerate(ttft_ms):
        ax.text(i, v + 0.3, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5, color=GREY)

    fig.suptitle(
        "GPU baseline — NVIDIA A100-SXM4-80GB, CUDA, bfloat16, 100 generated tokens",
        fontsize=10, color=GREY, y=1.01,
    )
    fig.tight_layout()

    out = ROOT / "tutorials" / "assets" / "gpu-baseline-benchmark.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
