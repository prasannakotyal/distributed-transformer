"""
Key-Value Cache for efficient autoregressive generation.

During generation, we compute attention incrementally:
- At step t, we only need to compute Q for the new token
- K and V from previous steps can be cached and reused

This reduces generation from O(T^2) to O(T) per token.
"""

import torch
from typing import List, Tuple, Optional


class KVCache:
    """
    Cache for storing key-value pairs during autoregressive generation.

    Usage:
        cache = KVCache(num_layers, num_heads)

        # During generation loop:
        for token in tokens:
            logits, cache_update = model(token, kv_cache=cache.get())
            cache.update(cache_update)
    """

    def __init__(self, num_layers: int, num_heads: int):
        """
        Initialize empty cache.

        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        # cache[layer][head] = (key_tensor, value_tensor)
        self.cache: List[Optional[List[Tuple[torch.Tensor, torch.Tensor]]]] = [
            None for _ in range(num_layers)
        ]

    def get(self, layer: int) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Get cached KV pairs for a specific layer."""
        return self.cache[layer]

    def update(
        self,
        layer: int,
        new_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """
        Update cache with new key-value pairs.

        Args:
            layer: Layer index
            new_kv: List of (key, value) tuples, one per head
        """
        self.cache[layer] = new_kv

    def clear(self) -> None:
        """Clear all cached values."""
        self.cache = [None for _ in range(self.num_layers)]

    @property
    def seq_len(self) -> int:
        """Current cached sequence length."""
        if self.cache[0] is None:
            return 0
        # Get length from first head of first layer
        return self.cache[0][0][0].shape[1]

    def __repr__(self) -> str:
        return f"KVCache(num_layers={self.num_layers}, num_heads={self.num_heads}, seq_len={self.seq_len})"
