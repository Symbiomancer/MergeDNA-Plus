"""
DiffusionDNA: ConvDNA U-Net with timestep conditioning for diffusion denoising.
Same strided/dilated conv architecture, but conditioned on diffusion timestep t.
"""

import math
import torch
import torch.nn as nn
from local_attention import LocalAttention


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class LocalAttentionBlock(nn.Module):
    """Local attention block with timestep conditioning via scale/shift."""

    def __init__(self, dim, num_heads, dim_feedforward, window_size, time_dim, drop=0.0):
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

        # Time conditioning: project time_dim -> scale + shift for each norm
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 2),
        )

    def forward(self, x, t_emb):
        B, N, C = x.shape
        h = self.num_heads
        head_dim = C // h

        # Time conditioning (scale, shift)
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=-1)
        scale = scale.unsqueeze(1)  # [B, 1, C]
        shift = shift.unsqueeze(1)

        # Self-attention (pre-norm + time condition)
        x_norm = self.norm1(x) * (1 + scale) + shift
        qkv = self.qkv(x_norm).reshape(B, N, 3, h, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_out = self.attn(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = self.attn_drop(self.proj(attn_out))
        x = x + attn_out

        # MLP (pre-norm)
        x = x + self.mlp(self.norm2(x))
        return x


class FullAttentionBlock(nn.Module):
    """Full attention block with timestep conditioning."""

    def __init__(self, dim, num_heads, dim_feedforward, time_dim, drop=0.0):
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
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 2),
        )

    def forward(self, x, t_emb):
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)

        x_norm = self.norm1(x) * (1 + scale) + shift
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class StridedDownsample(nn.Module):
    def __init__(self, dim, kernel_size=4, stride=2):
        super().__init__()
        padding = (kernel_size - stride) // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x)
        return x


class DilatedUpsample(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=2):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x).transpose(1, 2)
        x = self.norm(x)
        return x


class DiffusionDNA(nn.Module):
    """
    ConvDNA U-Net with timestep conditioning for DDPM training.
    Predicts noise epsilon given noisy embedded sequence + timestep.

    Args:
        vocab_size: DNA alphabet size (4).
        dim: Embedding dimension.
        local_depth: Number of encoder/decoder stages.
        local_num_heads, local_dim_ff, local_window_size: Local attn params.
        latent_depth, latent_num_heads, latent_dim_ff: Full attn params.
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

        # Timestep embedding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input: embedded sequence (continuous) -> project to dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.input_proj = nn.Linear(dim, dim)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for _ in range(local_depth):
            self.encoder_blocks.append(
                LocalAttentionBlock(dim, local_num_heads, local_dim_ff, local_window_size, time_dim, drop)
            )
            self.downsamples.append(StridedDownsample(dim))

        # Bottleneck
        self.latent_blocks = nn.ModuleList([
            FullAttentionBlock(dim, latent_num_heads, latent_dim_ff, time_dim, drop)
            for _ in range(latent_depth)
        ])

        # Decoder
        self.upsamples = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for _ in range(local_depth):
            self.upsamples.append(DilatedUpsample(dim))
            self.skip_projs.append(nn.Linear(dim * 2, dim))
            self.decoder_blocks.append(
                LocalAttentionBlock(dim, local_num_heads, local_dim_ff, local_window_size, time_dim, drop)
            )

        self.norm_out = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)  # predict noise in embedding space

    def forward(self, x_noisy, t):
        """
        Args:
            x_noisy: Noisy embeddings [B, N, dim].
            t: Timesteps [B].
        Returns:
            Predicted noise [B, N, dim].
        """
        t_emb = self.time_mlp(t)
        x = self.input_proj(x_noisy)

        # Encoder
        skips = []
        for attn, down in zip(self.encoder_blocks, self.downsamples):
            x = attn(x, t_emb)
            skips.append(x)
            x = down(x)

        # Bottleneck
        for blk in self.latent_blocks:
            x = blk(x, t_emb)

        # Decoder
        for up, proj, attn, skip in zip(
            self.upsamples, self.skip_projs, self.decoder_blocks, reversed(skips)
        ):
            x = up(x)
            min_len = min(x.shape[1], skip.shape[1])
            x = torch.cat([x[:, :min_len], skip[:, :min_len]], dim=-1)
            x = proj(x)
            x = attn(x, t_emb)

        x = self.norm_out(x)
        return self.out_proj(x)
