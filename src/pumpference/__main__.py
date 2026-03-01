"""
Run Qwen3-0.6B inference from the command line.

Usage:
    uv run python -m pumpference --prompt "Explain transformers"
"""

import argparse
from dataclasses import replace

import torch

from .benchmark import _PROMPT_115, _PROMPT_30
from .generate import generate, speculative_generate
from .model import (
    QWEN3_0_6B_CONFIG,
    QWEN3_1_7B_CONFIG,
    Qwen3Model,
    download_and_load_weights,
)
from .quantize import quantize_model
from .tokenizer import download_tokenizer

_AWQ_CALIBRATION_PROMPTS = [_PROMPT_30, _PROMPT_115]


def main() -> None:
    # --- CLI args ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="Run Qwen3-0.6B inference")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Give me a short introduction to large language models.",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to run inference on (default: auto-detect)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy argmax, default: 0.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k filtering: keep only the k most likely tokens (0 = disabled, default: 0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling threshold: cumulative probability cutoff (1.0 = disabled, default: 1.0)",
    )
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
    parser.add_argument(
        "--speculative",
        action="store_true",
        default=False,
        help=(
            "Enable speculative decoding: use Qwen3-0.6B as the draft model "
            "and Qwen3-1.7B as the target model.  Reduces the number of serial "
            "target-model forward passes.  Requires ~5 GB of additional memory "
            "for the target model (~4 GB) on first run."
        ),
    )
    parser.add_argument(
        "--draft-k",
        type=int,
        default=5,
        metavar="K",
        help="Number of draft tokens proposed per speculative round (default: 5).",
    )
    args = parser.parse_args()

    # --- Device -----------------------------------------------------------
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Tokenizer --------------------------------------------------------
    tokenizer = download_tokenizer(repo_id=QWEN3_0_6B_CONFIG.repo_id)

    # --- Model(s) ---------------------------------------------------------
    if args.speculative:
        print("Speculative decoding mode — loading draft (Qwen3-0.6B) …")
        draft_cfg = replace(QWEN3_0_6B_CONFIG, use_flash_attn=args.flash_attn)
        draft_model = Qwen3Model(draft_cfg)
        download_and_load_weights(draft_model, repo_id=draft_cfg.repo_id)
        draft_model.to(device)
        draft_model.eval()

        print("Loading target (Qwen3-1.7B) …")
        target_cfg = replace(QWEN3_1_7B_CONFIG, use_flash_attn=args.flash_attn)
        model = Qwen3Model(target_cfg)
        download_and_load_weights(model, repo_id=target_cfg.repo_id)
        model.to(device)
        model.eval()

        draft_params = sum(p.numel() for p in draft_model.parameters())
        target_params = sum(p.numel() for p in model.parameters())
        print(f"Draft parameters:  {draft_params:,}")
        print(f"Target parameters: {target_params:,}")
    else:
        print("Loading model …")
        cfg = replace(QWEN3_0_6B_CONFIG, use_flash_attn=args.flash_attn)
        model = Qwen3Model(cfg)
        download_and_load_weights(model, repo_id=cfg.repo_id)
        if args.quantize != "none":
            print(f"Quantizing weights ({args.quantize}) …")
            if args.quantize.startswith("awq"):
                print("  Running AWQ calibration (collecting activation statistics) …")
                cal_ids = [
                    torch.tensor([tokenizer.encode(p)])
                    for p in _AWQ_CALIBRATION_PROMPTS
                ]
                quantize_model(model, mode=args.quantize, calibration_ids=cal_ids)
            else:
                quantize_model(model, mode=args.quantize)
        model.to(device)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")

    # --- Generate ---------------------------------------------------------
    print(f"\nPrompt: {args.prompt}\n")

    input_ids = tokenizer.encode(args.prompt)
    input_tensor = torch.tensor([input_ids], device=device)

    if args.speculative:
        output_ids, stats = speculative_generate(
            target_model=model,
            draft_model=draft_model,
            input_ids=input_tensor,
            max_new_tokens=args.max_tokens,
            num_speculative_tokens=args.draft_k,
            eos_token_id=tokenizer.eos_token_id,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        new_ids = output_ids[0, len(input_ids):].tolist()
        print(tokenizer.decode(new_ids))
        print(
            f"\n[Speculative stats]  rounds={stats.num_rounds}  "
            f"draft_tokens={stats.total_draft_tokens}  "
            f"accepted={stats.total_accepted_tokens}  "
            f"acceptance_rate={stats.acceptance_rate:.1%}  "
            f"tokens/round={stats.tokens_per_round:.1f}"
        )
    else:
        output_ids = generate(
            model,
            input_tensor,
            max_new_tokens=args.max_tokens,
            eos_token_id=tokenizer.eos_token_id,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        new_ids = output_ids[0, len(input_ids):].tolist()
        print(tokenizer.decode(new_ids))


if __name__ == "__main__":
    main()
