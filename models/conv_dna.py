"""
ConvDNA: Variant that replaces token merging with strided/dilated convolutions.
Key insight: strided convolutions learn to compress tokens like ToMe,
but with learnable parameters instead of similarity-based matching.

Architecture:
    Embed -> [Local Attn + Strided Conv] x N -> Latent Attn -> [Dilated Conv + Local Attn] x N -> Head
"""

import torch
import torch.nn as nn
from local_attention import LocalAttention


class LocalAttentionBlock(nn.Module):
    """Transformer block with local (windowed) self-attention."""

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

        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).reshape(B, N, 3, h, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_out = self.attn(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = self.attn_drop(self.proj(attn_out))
        x = x + attn_out

        x = x + self.mlp(self.norm2(x))
        return x


class StridedDownsample(nn.Module):
    """Strided conv to halve sequence length. (B, N, C) -> (B, N//2, C)"""

    def __init__(self, dim, kernel_size=4, stride=2):
        super().__init__()
        padding = (kernel_size - stride) // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, N, C) -> conv needs (B, C, N)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x)
        return x


class DilatedUpsample(nn.Module):
    """Upsample via interpolation + dilated conv. (B, N, C) -> (B, N*2, C)"""

    def __init__(self, dim, kernel_size=3, dilation=2):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, N, C) -> interpolate needs (B, C, N)
        x = x.transpose(1, 2)
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x).transpose(1, 2)
        x = self.norm(x)
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


class ConvDNA(nn.Module):
    """
    ConvDNA: hierarchical autoencoder with strided/dilated convolutions
    replacing token merging.

    Encoder: Local Attn -> Strided Conv (halve) per block
    Latent:  Full Attn blocks
    Decoder: Dilated Conv (double) -> Local Attn per block

    U-Net skip connections between encoder and decoder stages.

    Args:
        vocab_size: DNA alphabet size (4).
        dim: Embedding dimension.
        local_depth: Number of encoder/decoder stages (each halves/doubles length).
        local_num_heads: Heads for local attention.
        local_dim_ff: FF dim for local attention.
        local_window_size: Window size for local attention.
        latent_depth: Number of full attention blocks.
        latent_num_heads: Heads for full attention.
        latent_dim_ff: FF dim for full attention.
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
        self.local_depth = local_depth

        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_drop = nn.Dropout(drop)

        # Encoder: local attn + strided downsample per stage
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for _ in range(local_depth):
            self.encoder_blocks.append(
                LocalAttentionBlock(dim, local_num_heads, local_dim_ff, local_window_size, drop)
            )
            self.downsamples.append(StridedDownsample(dim))

        # Latent: full attention
        self.latent_blocks = nn.ModuleList([
            FullAttentionBlock(dim, latent_num_heads, latent_dim_ff, drop)
            for _ in range(latent_depth)
        ])

        # Decoder: dilated upsample + local attn per stage
        # Skip connections double the channels, project back with a linear
        self.upsamples = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for _ in range(local_depth):
            self.upsamples.append(DilatedUpsample(dim))
            self.skip_projs.append(nn.Linear(dim * 2, dim))
            self.decoder_blocks.append(
                LocalAttentionBlock(dim, local_num_heads, local_dim_ff, local_window_size, drop)
            )

        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        """
        Args:
            x: [B, N] integer token IDs.
        Returns:
            logits: [B, N, vocab_size].
        """
        x = self.embed(x)
        x = self.pos_drop(x)

        # Encoder with skip connections
        skips = []
        for attn, down in zip(self.encoder_blocks, self.downsamples):
            x = attn(x)
            skips.append(x)
            x = down(x)

        # Latent
        for blk in self.latent_blocks:
            x = blk(x)

        # Decoder with skip connections (reverse order)
        for up, proj, attn, skip in zip(
            self.upsamples, self.skip_projs, self.decoder_blocks, reversed(skips)
        ):
            x = up(x)
            # Match lengths in case of rounding
            min_len = min(x.shape[1], skip.shape[1])
            x = torch.cat([x[:, :min_len], skip[:, :min_len]], dim=-1)
            x = proj(x)
            x = attn(x)

        x = self.norm_out(x)
        x = self.head(x)
        return x
