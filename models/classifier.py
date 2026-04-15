"""
Classification head on top of a frozen pre-trained encoder.
Supports MergeDNA, ConvDNA, and DiffusionDNA encoders.
"""

import torch
import torch.nn as nn


class DNAClassifier(nn.Module):
    """
    Frozen encoder + trainable classification head.

    Args:
        encoder: Pre-trained model (MergeDNA, ConvDNA, or DiffusionDNA).
        dim: Encoder embedding dimension.
        num_classes: Number of output classes.
        pool: Pooling strategy ("mean" or "cls").
    """

    def __init__(self, encoder, dim, num_classes, pool="mean"):
        super().__init__()
        self.encoder = encoder
        self.pool = pool
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_classes),
        )

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def get_latents(self, x):
        """Run input through encoder, return latent representations."""
        with torch.no_grad():
            emb = self.encoder.embed(x)

            if hasattr(self.encoder, 'encode_local'):
                # MergeDNA
                emb = self.encoder.pos_drop(emb)
                emb, _ = self.encoder.encode_local(emb)
                emb = self.encoder.encode_latent(emb)
            elif hasattr(self.encoder, 'input_proj'):
                # DiffusionDNA
                t_emb = self.encoder.time_mlp(torch.zeros(x.shape[0], device=x.device))
                emb = self.encoder.input_proj(emb)
                for attn, down in zip(self.encoder.encoder_blocks, self.encoder.downsamples):
                    emb = attn(emb, t_emb)
                    emb = down(emb)
                for blk in self.encoder.latent_blocks:
                    emb = blk(emb, t_emb)
            else:
                # ConvDNA
                emb = self.encoder.pos_drop(emb)
                for attn, down in zip(self.encoder.encoder_blocks, self.encoder.downsamples):
                    emb = attn(emb)
                    emb = down(emb)
                for blk in self.encoder.latent_blocks:
                    emb = blk(emb)

        return emb

    def forward(self, x):
        """
        Args:
            x: [B, N] integer token IDs.
        Returns:
            logits: [B, num_classes].
        """
        latents = self.get_latents(x)

        # Pool over sequence dimension
        if self.pool == "mean":
            pooled = latents.mean(dim=1)
        else:
            pooled = latents[:, 0]  # first token

        return self.head(pooled)
