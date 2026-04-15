"""
MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization.
Hierarchical autoencoder with local-window attention, full attention,
and token merging with source matrix tracking.
"""

import math
import torch
import torch.nn as nn
from local_attention import LocalAttention


def bipartite_soft_matching(k: torch.Tensor, r: int):
    """
    ToMe bipartite soft matching (Bolya et al., ICLR 2023, Appendix D).
    Adapted for MergeDNA: no CLS token protection, returns merge fn + index info
    for source matrix updates.

    Args:
        k: Keys [B, T, C] for similarity.
        r: Number of tokens to merge.

    Returns:
        merge: Function that merges a [B, T, C] tensor -> [B, T-r, C].
        src_idx: Indices of merged tokens from set A [B, r, 1].
        dst_idx: Indices of their targets in set B [B, r, 1].
        unm_idx: Indices of unmerged tokens from set A [B, T_a - r, 1].
    """
    if r <= 0:
        return (lambda x: x), None, None, None

    r = min(r, k.shape[1] // 2)

    k = k / k.norm(dim=-1, keepdim=True)
    a, b = k[..., ::2, :], k[..., 1::2, :]
    scores = a @ b.transpose(-1, -2)

    node_max, node_idx = scores.max(dim=-1)
    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

    unm_idx = edge_idx[..., r:, :]
    src_idx = edge_idx[..., :r, :]
    dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    unm_idx = unm_idx.sort(dim=-2)[0]

    def merge(x: torch.Tensor) -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_add(-2, dst_idx.expand(n, r, c), src)
        return torch.cat([unm, dst], dim=-2)

    return merge, src_idx, dst_idx, unm_idx


def update_source_matrix(S, src_idx, dst_idx, unm_idx, r, num_a):
    """
    Update source matrix S after a merge step.

    Args:
        S: Current source matrix [B, T_current, N_original].
        src_idx: [B, r, 1] indices of merged tokens in set A.
        dst_idx: [B, r, 1] indices of their targets in set B.
        unm_idx: [B, T_a - r, 1] indices of unmerged tokens in set A.
        r: Number of merged tokens.
        num_a: Number of tokens in set A before merge.

    Returns:
        Updated S [B, T_current - r, N_original].
    """
    B, T, N = S.shape

    S_a, S_b = S[:, ::2, :], S[:, 1::2, :]

    # Unmerged rows from A
    S_unm = S_a.gather(1, unm_idx.expand(B, num_a - r, N))

    # Merged rows from A get added to their dst rows in B
    S_src = S_a.gather(1, src_idx.expand(B, r, N))
    S_b = S_b.scatter_add(1, dst_idx.expand(B, r, N), S_src)

    return torch.cat([S_unm, S_b], dim=1)


def unmerge(z, S):
    """
    Token unmerging: scatter merged representations back to original positions.
    Z_bar = S^T @ Z

    Args:
        z: Merged tokens [B, L, D].
        S: Source matrix [B, L, N] (binary, rows sum to indicate source positions).

    Returns:
        Unmerged tokens [B, N, D].
    """
    # S^T: [B, N, L] @ z: [B, L, D] -> [B, N, D]
    return torch.bmm(S.transpose(1, 2).float(), z)


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

    def get_keys(self, x):
        """Extract keys for ToMe similarity."""
        x_norm = self.norm1(x)
        B, N, C = x_norm.shape
        h = self.num_heads
        head_dim = C // h
        qkv = self.qkv(x_norm).reshape(B, N, 3, h, head_dim)
        k = qkv[:, :, 1].mean(dim=2)  # average across heads
        return k


class LocalEncoderBlock(nn.Module):
    """Local attention + token merging."""

    def __init__(self, dim, num_heads, dim_feedforward, window_size, r, drop=0.0):
        super().__init__()
        self.attn_block = LocalAttentionBlock(dim, num_heads, dim_feedforward, window_size, drop)
        self.r = r

    def forward(self, x, S):
        x = self.attn_block(x)

        keys = self.attn_block.get_keys(x)
        merge, src_idx, dst_idx, unm_idx = bipartite_soft_matching(keys, self.r)
        x = merge(x)

        if src_idx is not None:
            num_a = keys.shape[1] // 2 + keys.shape[1] % 2
            S = update_source_matrix(S, src_idx, dst_idx, unm_idx, self.r, num_a)

        return x, S


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
    MergeDNA hierarchical autoencoder with token merging.

    Architecture:
        Local Encoder (local attn + ToMe) -> Latent Encoder (full attn)
        -> Latent Decoder (full attn) -> Unmerge (S^T) -> Local Decoder (local attn)

    Args:
        vocab_size: DNA alphabet size (4 for A/T/C/G).
        dim: Embedding dimension.
        local_depth: Number of local encoder/decoder blocks.
        local_num_heads: Attention heads for local blocks.
        local_dim_ff: Feedforward dim for local blocks.
        local_window_size: Window size for local attention.
        latent_depth: Number of latent encoder/decoder blocks.
        latent_num_heads: Attention heads for latent blocks.
        latent_dim_ff: Feedforward dim for latent blocks.
        merge_r: Tokens to merge per local encoder block.
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
        merge_r=4,
        drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_drop = nn.Dropout(drop)

        # Local Encoder: local attention + token merging per block
        self.local_encoder = nn.ModuleList([
            LocalEncoderBlock(dim, local_num_heads, local_dim_ff, local_window_size, merge_r, drop)
            for _ in range(local_depth)
        ])

        # Latent Encoder: full attention
        self.latent_encoder = nn.ModuleList([
            FullAttentionBlock(dim, latent_num_heads, latent_dim_ff, drop)
            for _ in range(latent_depth)
        ])

        # Latent Decoder: full attention
        self.latent_decoder = nn.ModuleList([
            FullAttentionBlock(dim, latent_num_heads, latent_dim_ff, drop)
            for _ in range(latent_depth)
        ])

        # Local Decoder: local attention (no merging)
        self.local_decoder = nn.ModuleList([
            LocalAttentionBlock(dim, local_num_heads, local_dim_ff, local_window_size, drop)
            for _ in range(local_depth)
        ])

        self.norm_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def encode_local(self, x):
        """Local Encoder with token merging. Returns compressed tokens + source matrix."""
        B, N, _ = x.shape
        S = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1)

        for blk in self.local_encoder:
            x, S = blk(x, S)
        return x, S

    def encode_latent(self, x):
        for blk in self.latent_encoder:
            x = blk(x)
        return x

    def decode_latent(self, x):
        for blk in self.latent_decoder:
            x = blk(x)
        return x

    def decode_local(self, x):
        for blk in self.local_decoder:
            x = blk(x)
        return x

    def forward(self, x):
        """
        Args:
            x: Input token IDs [B, N] with values in {0,1,2,3}.
        Returns:
            logits: [B, N, vocab_size] reconstruction logits.
        """
        x = self.embed(x)
        x = self.pos_drop(x)

        # Encode: compress with token merging
        x, S = self.encode_local(x)
        x = self.encode_latent(x)

        # Decode: decompress with unmerging
        x = self.decode_latent(x)
        x = unmerge(x, S)
        x = self.decode_local(x)

        x = self.norm_out(x)
        x = self.head(x)
        return x
