"""
Single-Head Self-Attention.

Implements scaled dot-product attention from "Attention Is All You Need":
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

With causal masking for autoregressive language modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SelfAttention(nn.Module):
    """
    Single head of causal self-attention.

    Args:
        embedding_dim: Input embedding dimension
        head_dim: Dimension of this attention head
        context_length: Maximum sequence length (for causal mask)
        dropout: Dropout probability
    """

    def __init__(
        self,
        embedding_dim: int,
        head_dim: int,
        context_length: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.head_dim = head_dim

        # Q, K, V projections
        self.query = nn.Linear(embedding_dim, head_dim, bias=False)
        self.key = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, head_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Causal mask: lower triangular matrix
        # Registered as buffer (not a parameter, but saved with model)
        mask = torch.tril(torch.ones(context_length, context_length))
        self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attention: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, embedding_dim)
            kv_cache: Optional cached (key, value) from previous steps
            return_attention: Whether to return attention weights

        Returns:
            output: Attention output (batch, seq_len, head_dim)
            attn_weights: Attention weights if return_attention=True
            new_kv_cache: Updated (key, value) cache
        """
        B, T, C = x.shape

        # Compute Q, K, V
        q = self.query(x)  # (B, T, head_dim)
        k = self.key(x)  # (B, T, head_dim)
        v = self.value(x)  # (B, T, head_dim)

        # Handle KV cache for efficient generation
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)

        new_kv_cache = (k, v)
        T_kv = k.shape[1]  # May differ from T if using cache

        # Scaled dot-product attention
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        scale = self.head_dim**-0.5
        attn_scores = (q @ k.transpose(-2, -1)) * scale  # (B, T, T_kv)

        # Apply causal mask (only attend to past positions)
        # For cached generation, we only mask based on query positions
        if kv_cache is None:
            attn_scores = attn_scores.masked_fill(
                self.mask[:T, :T_kv] == 0, float("-inf")
            )
        # With cache, no masking needed (generating one token at a time)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        output = attn_weights @ v  # (B, T, head_dim)

        if return_attention:
            return output, attn_weights, new_kv_cache
        return output, None, new_kv_cache
