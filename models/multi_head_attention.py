"""
Multi-Head Self-Attention.

Implements multi-head attention from "Attention Is All You Need":
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

Each head operates on a different learned linear projection.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from .self_attention import SelfAttention


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention.

    Combines multiple attention heads, each attending to different
    representation subspaces.

    Args:
        num_heads: Number of attention heads
        embedding_dim: Input/output embedding dimension
        context_length: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_heads: int,
        embedding_dim: int,
        context_length: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0, (
            "embedding_dim must be divisible by num_heads"
        )

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Create attention heads
        self.heads = nn.ModuleList(
            [
                SelfAttention(embedding_dim, self.head_dim, context_length, dropout)
                for _ in range(num_heads)
            ]
        )

        # Output projection: W^O in the paper
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_attention: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    ]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, embedding_dim)
            kv_cache: Optional list of (key, value) caches for each head
            return_attention: Whether to return attention weights

        Returns:
            output: Multi-head attention output (batch, seq_len, embedding_dim)
            attn_weights: Concatenated attention weights if return_attention=True
            new_kv_cache: Updated list of (key, value) caches
        """
        head_outputs = []
        head_attentions = []
        new_kv_cache = []

        for i, head in enumerate(self.heads):
            head_cache = kv_cache[i] if kv_cache is not None else None
            out, attn, new_cache = head(
                x, kv_cache=head_cache, return_attention=return_attention
            )
            head_outputs.append(out)
            if attn is not None:
                head_attentions.append(attn)
            new_kv_cache.append(new_cache)

        # Concatenate heads: (B, T, num_heads * head_dim) = (B, T, embedding_dim)
        concat = torch.cat(head_outputs, dim=-1)

        # Output projection
        output = self.dropout(self.output_proj(concat))

        # Average attention weights across heads for visualization
        attn_weights = None
        if return_attention and head_attentions:
            attn_weights = torch.stack(head_attentions, dim=1)  # (B, num_heads, T, T)

        return output, attn_weights, new_kv_cache
