"""
Compare our Qwen3Model greedy generation against HuggingFace transformers.

Both models load the same Qwen3-0.6B weights, run in eval mode with a fixed
seed, and use pure greedy decoding (argmax).  
The generated token sequences must be identical.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from pumpference.generate import generate
from pumpference.model import QWEN3_0_6B_CONFIG, Qwen3Model, download_and_load_weights
from pumpference.tokenizer import download_tokenizer

REPO_ID = "Qwen/Qwen3-0.6B" # https://huggingface.co/Qwen/Qwen3-0.6B
DEVICE = "cpu"
MAX_NEW_TOKENS = 20
PROMPT = "Give me a short introduction to large language models."


# ------------------------------------------------------------------
# Fixtures — models and tokenizer are loaded once per test module
# ------------------------------------------------------------------

@pytest.fixture(scope="module")
def our_model() -> Qwen3Model:
    """Instantiate and load our from-scratch Qwen3 model."""
    model = Qwen3Model(QWEN3_0_6B_CONFIG)
    download_and_load_weights(model, repo_id=REPO_ID)
    model.to(DEVICE)
    model.eval()
    return model


@pytest.fixture(scope="module")
def hf_model() -> AutoModelForCausalLM:
    """Load the official HuggingFace Qwen3-0.6B model."""
    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID,
        dtype=torch.bfloat16,
        # Use "eager" (explicit PyTorch ops) instead of default "sdpa" (fused kernel).
        # Our implementation uses manual matmul/softmax, so we need HF to do the same
        # for numerical consistency.
        attn_implementation="eager",
    )
    model.to(DEVICE)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    return download_tokenizer(repo_id=REPO_ID)


@pytest.fixture(scope="module")
def input_ids(tokenizer) -> torch.Tensor:
    return torch.tensor([tokenizer.encode(PROMPT)], device=DEVICE)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

@torch.no_grad()
def _hf_greedy_generate(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Greedy generation using HuggingFace model — manual loop, no generate()."""
    tokens = input_ids
    for _ in range(max_new_tokens):
        logits = model(tokens).logits        # [1, seq_len, vocab]
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
    return tokens


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

@torch.no_grad()
def test_logits_argmax_matches_single_forward(our_model, hf_model, input_ids) -> None:
    """The argmax of logits at every position must be identical."""
    our_logits = our_model(input_ids)
    hf_logits = hf_model(input_ids).logits

    # Comparing argmax (not exact logit values) is the right approach for validating a bfloat16 re-implementation. 
    assert torch.equal(our_logits.argmax(dim=-1), hf_logits.argmax(dim=-1)), (
        "Argmax mismatch between our model and HF"
    )

    # Logit values differ slightly due to bfloat16 precision (different
    # scaling order, SDPA vs manual attention), but should be close.
    diff = (our_logits.float() - hf_logits.float()).abs()
    assert diff.max().item() < 1.0, f"Max logit diff too large: {diff.max().item()}"


@torch.no_grad()
def test_greedy_generation_matches_hf(our_model, hf_model, input_ids) -> None:
    """Greedy-decoded tokens must be identical between our model and HF."""

    # --- Our generation ---------------------------------------------------
    our_output = generate(
        our_model,
        input_ids.clone(),
        max_new_tokens=MAX_NEW_TOKENS,
    )

    # --- HF generation (manual greedy loop) -------------------------------
    hf_output = _hf_greedy_generate(
        hf_model,
        input_ids.clone(),
        max_new_tokens=MAX_NEW_TOKENS,
    )

    our_tokens = our_output[0].tolist()
    hf_tokens = hf_output[0].tolist()

    assert our_tokens == hf_tokens, (
        f"Token mismatch!\n"
        f"  Ours: {our_tokens}\n"
        f"  HF:   {hf_tokens}"
    )
