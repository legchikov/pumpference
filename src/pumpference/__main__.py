"""
Run Qwen3-0.6B inference from the command line.

Usage:
    uv run python -m pumpference --prompt "Explain transformers"
"""

import argparse

import torch

from .generate import generate
from .model import QWEN3_0_6B_CONFIG, Qwen3Model, download_and_load_weights
from .quantize import quantize_model
from .tokenizer import download_tokenizer


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
        choices=["none", "int8", "int4"],
        default="none",
        help="Weight-only quantization: none (default), int8, or int4",
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

    # --- Model ------------------------------------------------------------
    print("Loading model …")
    model = Qwen3Model(QWEN3_0_6B_CONFIG)
    download_and_load_weights(model, repo_id=QWEN3_0_6B_CONFIG.repo_id)
    if args.quantize != "none":
        print(f"Quantizing weights ({args.quantize}) …")
        quantize_model(model, mode=args.quantize)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # --- Tokenizer --------------------------------------------------------
    tokenizer = download_tokenizer(repo_id=QWEN3_0_6B_CONFIG.repo_id)

    # --- Generate ---------------------------------------------------------
    print(f"\nPrompt: {args.prompt}\n")

    input_ids = tokenizer.encode(args.prompt)
    input_tensor = torch.tensor([input_ids], device=device)

    output_ids = generate(
        model,
        input_tensor,
        max_new_tokens=args.max_tokens,
        eos_token_id=tokenizer.eos_token_id,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    new_ids = output_ids[0, len(input_ids) :].tolist()
    print(tokenizer.decode(new_ids))


if __name__ == "__main__":
    main()
