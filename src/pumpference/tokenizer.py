"""
Minimal tokenizer for Qwen3

Loads `tokenizer.json` from the HuggingFace repo and handles the special tokens
"""

import re
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


# Special tokens from https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/tokenizer_config.json
# plus <think>/<\think> for reasoning mode (newer Qwen variants)
_SPECIAL_TOKENS = (
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<think>",
    "</think>",
)

# Regex that splits text at known special token boundaries only.
# Built from _SPECIAL_TOKENS so the regex and the lookup table always agree.
_SPLIT_RE = re.compile("(" + "|".join(re.escape(t) for t in _SPECIAL_TOKENS) + ")")


class Qwen3Tokenizer:
    """Encode / decode text using the Qwen3 tokenizer."""

    def __init__(self, tokenizer_json_path: str | Path) -> None:
        self._tok = Tokenizer.from_file(str(tokenizer_json_path))

        # Build a mapping from special-token string -> token id.
        self._special_to_id: dict[str, int] = {}
        for token_str in _SPECIAL_TOKENS:
            token_id = self._tok.token_to_id(token_str)
            if token_id is not None:
                self._special_to_id[token_str] = token_id

        self.eos_token_id: int = self._special_to_id["<|im_end|>"]

    def __repr__(self) -> str:
        return f"Qwen3Tokenizer(vocab_size={self._tok.get_vocab_size()}, eos={self.eos_token_id})"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Tokenize *text*, correctly handling special tokens."""
        ids: list[int] = []
        for part in filter(None, _SPLIT_RE.split(text)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to a string."""
        return self._tok.decode(ids, skip_special_tokens=False)


def download_tokenizer(repo_id: str = "Qwen/Qwen3-0.6B") -> Qwen3Tokenizer:
    """Download tokenizer.json from HuggingFace Hub and return a Qwen3Tokenizer."""
    local_dir = Path(repo_id).parts[-1]
    hf_hub_download(repo_id=repo_id, filename="tokenizer.json", local_dir=local_dir)
    return Qwen3Tokenizer(Path(local_dir) / "tokenizer.json")
