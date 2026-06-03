"""
Tokenizer wrapper using tiktoken (OpenAI's fast BPE tokenizer).

Uses pre-trained BPE models for efficient tokenization without needing
to train from scratch. Focus on the transformer model, not tokenization.

Reference: https://github.com/openai/tiktoken
"""

import json
from pathlib import Path
from typing import List
import tiktoken


class BPETokenizer:
    """
    BPE tokenizer using tiktoken.

    Uses OpenAI's gpt2 encoding (50,257 tokens) for efficient training.
    Following nanoGPT's approach of using pre-trained BPE tokenization.
    """

    def __init__(self, encoding_name: str = "gpt2"):
        """
        Initialize tokenizer.

        Args:
            encoding_name: tiktoken encoding name (default: gpt2 for GPT-2 compatible)
        """
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name
        self.vocab_size = self.encoding.n_vocab

    def train(self, text: str, vocab_size: int, verbose: bool = True) -> None:
        """
        No-op - using pre-trained tiktoken model.

        Args:
            text: Training text (ignored, using pre-trained model)
            vocab_size: Target vocab size (ignored, using pre-trained model)
            verbose: Whether to print training info
        """
        if verbose:
            print(f"Using pre-trained tiktoken model: {self.encoding_name}")
            print(f"Vocabulary size: {self.vocab_size:,}")
            print("No training needed - using OpenAI's pre-trained BPE merges")

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        return self.encoding.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text
        """
        return self.encoding.decode(token_ids)

    def save(self, path: str) -> None:
        """
        Save tokenizer config.

        Args:
            path: Path to save config
        """
        path_obj = Path(path)
        data = {
            "encoding_name": self.encoding_name,
            "vocab_size": self.vocab_size,
        }
        path_obj.write_text(json.dumps(data, indent=2))

    def load(self, path: str) -> None:
        """
        Load tokenizer config.

        Args:
            path: Path to load config from
        """
        path_obj = Path(path)
        data = json.loads(path_obj.read_text())
        self.encoding = tiktoken.get_encoding(data["encoding_name"])
        self.encoding_name = data["encoding_name"]
        self.vocab_size = data["vocab_size"]


# Quick test
if __name__ == "__main__":
    tokenizer = BPETokenizer()

    # Test encode/decode
    test_text = "Hello, world! This is a test of the tokenizer."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"Original: '{test_text}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    print(f"Roundtrip OK: {test_text == decoded}")
    print(f"Vocab size: {tokenizer.vocab_size:,}")
