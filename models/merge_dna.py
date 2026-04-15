"""
MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization.
Hierarchical autoencoder with local-window attention and full attention stages.
Token merging and source matrix tracking to be added separately.
"""

import torch
import torch.nn as nn
from local_attention import LocalAttention


class LocalAttentionBlock(nn.Module):
    """Transformer block with local (windowed) self-attention via local-attention lib."""

    def __init__(self, dim, num_heads, dim_feedforward, window_size, drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn = LocalAttention(
            dim=head_dim,
            window_size=window_size,
            causal=False,
            look_backward=1,
            look_forward=1,
            autopad=True,
        )
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(drop)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        B, N, C = x.shape
        h = self.num_heads
        head_dim = C // h

        # Self-attention (pre-norm)
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).reshape(B, N, 3, h, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, h, N, head_dim)

        attn_out = self.attn(q, k, v)  # (B, h, N, head_dim)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = self.attn_drop(self.proj(attn_out))
        x = x + attn_out

        # MLP (pre-norm)
        x = x + self.mlp(self.norm2(x))
        return x


class FullAttentionBlock(nn.Module):
    """Standard transformer block with full self-attention."""

    def __init__(self, dim, num_heads, dim_feedforward, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MergeDNA(nn.Module):
    """
    MergeDNA hierarchical autoencoder.

    Architecture:
        Local Encoder -> Latent Encoder -> Latent Decoder -> Local Decoder

    Args:
        vocab_size: DNA alphabet size (4 for A/T/C/G).
        dim: Embedding dimension (shared across all components).
        local_depth: Number of blocks in Local Encoder / Local Decoder.
        local_num_heads: Attention heads for local attention blocks.
        local_dim_ff: Feedforward dim for local attention blocks.
        local_window_size: Window size for local attention.
        latent_depth: Number of blocks in Latent Encoder / Latent Decoder.
        latent_num_heads: Attention heads for full attention blocks.
        latent_dim_ff: Feedforward dim for full attention blocks.
        drop: Dropout rate.
    """

    def __init__(
        self,
        vocab_size=4,
        dim=1024,
        local_depth=4,
        local_num_heads=8,
        local_dim_ff=2048,
        local_window_size=16,
        latent_depth=20,
        latent_num_heads=8,
        latent_dim_ff=2048,
        drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size

        # Input embedding: nucleotide -> dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_drop = nn.Dropout(drop)

        # Local Encoder: local-window attention blocks
        self.local_encoder = nn.ModuleList([
            LocalAttentionBlock(dim, local_num_heads, local_dim_ff, local_window_size, drop)
            for _ in range(local_depth)
        ])

        # Latent Encoder: full attention blocks
        self.latent_encoder = nn.ModuleList([
            FullAttentionBlock(dim, latent_num_heads, latent_dim_ff, drop)
            for _ in range(latent_depth)
        ])

        # Latent Decoder: full attention blocks (same params as latent encoder)
        self.latent_decoder = nn.ModuleList([
            FullAttentionBlock(dim, latent_num_heads, latent_dim_ff, drop)
            for _ in range(latent_depth)
        ])

        # Local Decoder: local-window attention blocks (same params as local encoder)
        self.local_decoder = nn.ModuleList([
            LocalAttentionBlock(dim, local_num_heads, local_dim_ff, local_window_size, drop)
            for _ in range(local_depth)
        ])

        # Output projection: dim -> vocab_size
        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def encode_local(self, x):
        """Local Encoder: local-window attention blocks."""
        for blk in self.local_encoder:
            x = blk(x)
        return x

    def encode_latent(self, x):
        """Latent Encoder: full attention blocks."""
        for blk in self.latent_encoder:
            x = blk(x)
        return x

    def decode_latent(self, x):
        """Latent Decoder: full attention blocks."""
        for blk in self.latent_decoder:
            x = blk(x)
        return x

    def decode_local(self, x):
        """Local Decoder: local-window attention blocks."""
        for blk in self.local_decoder:
            x = blk(x)
        return x

    def forward(self, x):
        """
        Full autoencoder forward pass.

        Args:
            x: Input token IDs [B, N] with values in {0,1,2,3}.

        Returns:
            logits: [B, N, vocab_size] reconstruction logits.
        """
        x = self.embed(x)
        x = self.pos_drop(x)

        x = self.encode_local(x)
        x = self.encode_latent(x)
        x = self.decode_latent(x)
        x = self.decode_local(x)

        x = self.norm_out(x)
        x = self.head(x)
        return x
