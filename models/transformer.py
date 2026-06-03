"""
Full Transformer Language Model.

GPT-style decoder-only transformer implementing "Attention Is All You Need".

Architecture:
    Input -> Token Embedding + Positional Encoding
          -> N x (Multi-Head Attention -> Add & Norm -> FFN -> Add & Norm)
          -> Linear -> Softmax -> Output

Uses Pre-LayerNorm (more stable than original Post-LayerNorm).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, List

from .multi_head_attention import MultiHeadAttention
from .kv_cache import KVCache


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention Is All You Need".

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, embedding_dim: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, : x.size(1), :]


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    FFN(x) = GELU(xW_1 + b_1)W_2 + b_2

    Standard uses 4x expansion factor.
    """

    def __init__(self, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = 4 * embedding_dim
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Single transformer block.

    Pre-LayerNorm architecture (more stable training):
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attn = MultiHeadAttention(
            num_heads, embedding_dim, context_length, dropout
        )
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ffn = FeedForward(embedding_dim, dropout)

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

        Returns:
            output: Block output
            attn_weights: Attention weights if return_attention=True
            new_kv_cache: Updated KV cache
        """
        # Pre-norm attention
        attn_out, attn_weights, new_kv_cache = self.attn(
            self.ln1(x), kv_cache=kv_cache, return_attention=return_attention
        )
        x = x + attn_out

        # Pre-norm FFN
        x = x + self.ffn(self.ln2(x))

        return x, attn_weights, new_kv_cache


class Transformer(nn.Module):
    """
    GPT-style decoder-only Transformer.

    Args:
        vocab_size: Vocabulary size
        embedding_dim: Model dimension (d_model in paper)
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        context_length: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        context_length: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.context_length = context_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.activation_checkpointing = False

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Sinusoidal positional encoding (from original paper)
        self.pos_encoding = SinusoidalPositionalEncoding(embedding_dim, context_length)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_dim, num_heads, context_length, dropout)
                for _ in range(num_layers)
            ]
        )

        # Final layer norm and output projection
        self.ln_final = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

        # Weight tying (optional, reduces parameters)
        self.token_embedding.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def set_activation_checkpointing(self, enabled: bool) -> None:
        self.activation_checkpointing = enabled

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with small values for stable training."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        return_attention: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[KVCache],
    ]:
        """
        Forward pass.

        Args:
            idx: Input token indices (batch, seq_len)
            targets: Target token indices for loss computation
            kv_cache: Optional KV cache for efficient generation
            return_attention: Whether to return attention weights

        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided
            attentions: List of attention weights per layer if return_attention=True
            kv_cache: Updated KV cache
        """
        B, T = idx.shape

        # Token embeddings + positional encoding
        x = self.token_embedding(idx)  # (B, T, embedding_dim)

        # For cached generation, offset positional encoding
        if kv_cache is not None and kv_cache.seq_len > 0:
            pos_offset = kv_cache.seq_len
            # Only add pos encoding for new positions
            x = x + self.pos_encoding.pe[:, pos_offset : pos_offset + T, :]
        else:
            x = self.pos_encoding(x)

        # Transformer blocks
        attentions = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache.get(i) if kv_cache is not None else None
            use_checkpoint = (
                self.activation_checkpointing
                and self.training
                and kv_cache is None
                and not return_attention
            )
            if use_checkpoint:
                x = checkpoint(self._checkpoint_block, block, x, use_reentrant=False)
                attn_weights = None
                new_cache = None
            else:
                x, attn_weights, new_cache = block(
                    x, kv_cache=layer_cache, return_attention=return_attention
                )
            if kv_cache is not None:
                kv_cache.update(i, new_cache)
            if return_attention and attn_weights is not None:
                attentions.append(attn_weights)

        # Final layer norm and projection to vocab
        x = self.ln_final(x)
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss, attentions if return_attention else None, kv_cache

    @staticmethod
    def _checkpoint_block(block: TransformerBlock, x: torch.Tensor) -> torch.Tensor:
        output, _, _ = block(x)
        return output

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.

        Args:
            idx: Starting token indices (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_k: If set, only sample from top k tokens
            use_cache: Whether to use KV cache for efficiency

        Returns:
            Generated token indices (batch, seq_len + max_new_tokens)
        """
        # Initialize KV cache
        kv_cache = KVCache(self.num_layers, self.num_heads) if use_cache else None

        # Process initial context
        if use_cache and idx.size(1) > 1:
            logits, _, _, kv_cache = self.forward(idx[:, :-1], kv_cache=kv_cache)
            idx_current = idx[:, -1:]
        else:
            idx_current = idx

        for _ in range(max_new_tokens):
            # Crop to context length if needed
            if not use_cache and idx_current.size(1) > self.context_length:
                idx_current = idx_current[:, -self.context_length :]

            # Forward pass
            logits, _, _, kv_cache = self.forward(idx_current, kv_cache=kv_cache)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat([idx, idx_next], dim=1)
            idx_current = idx_next

        return idx

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Quick test
if __name__ == "__main__":
    model = Transformer(vocab_size=1000, embedding_dim=128, num_layers=2, num_heads=4)
    print(f"Parameters: {model.count_parameters():,}")

    # Test forward
    x = torch.randint(0, 1000, (2, 16))
    logits, loss, _, _ = model(x, targets=x)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Test generation
    generated = model.generate(x[:, :4], max_new_tokens=10)
    print(f"Generated shape: {generated.shape}")
