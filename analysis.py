"""
Post-training latent analysis: extract latents, PCA, and save 3D scatter plots.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader


def extract_latents(model, dataloader, device):
    """Run sequences through encoder, return mean-pooled latents and labels."""
    model.eval()
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            seqs = batch[0].to(device)
            labels = batch[1] if len(batch) > 1 else None

            x = model.embed(seqs)

            if hasattr(model, 'encode_local'):
                # MergeDNA
                x = model.pos_drop(x)
                x, _ = model.encode_local(x)
                x = model.encode_latent(x)
            elif hasattr(model, 'input_proj'):
                # DiffusionDNA — run encoder at t=0 (no noise)
                t_emb = model.time_mlp(torch.zeros(seqs.shape[0], device=device))
                x = model.input_proj(x)
                for attn, down in zip(model.encoder_blocks, model.downsamples):
                    x = attn(x, t_emb)
                    x = down(x)
                for blk in model.latent_blocks:
                    x = blk(x, t_emb)
            else:
                # ConvDNA
                x = model.pos_drop(x)
                for attn, down in zip(model.encoder_blocks, model.downsamples):
                    x = attn(x)
                    x = down(x)
                for blk in model.latent_blocks:
                    x = blk(x)

            latent = x.mean(dim=1)
            all_latents.append(latent.cpu().numpy())
            if labels is not None:
                all_labels.append(labels.numpy())

    latents = np.concatenate(all_latents, axis=0)
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros(len(latents))
    return latents, labels


def _param_text(params):
    """Format param dict as multi-line string for plot annotation."""
    if not params:
        return ""
    lines = [f"{k}: {v}" for k, v in params.items()]
    return "\n".join(lines)


def plot_pca_3d(latents, labels, output_path, title="Latent PCA", params=None):
    """PCA project to 3D and save scatter plot."""
    pca = PCA(n_components=3)
    projected = pca.fit_transform(latents)
    explained = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(labels)
    for lab in unique_labels:
        mask = labels == lab
        ax.scatter(
            projected[mask, 0], projected[mask, 1], projected[mask, 2],
            s=10, alpha=0.6, label=f"Class {int(lab)}"
        )

    ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
    ax.set_zlabel(f"PC3 ({explained[2]:.1%})")
    ax.set_title(title)
    ax.legend(markerscale=3)

    if params:
        fig.text(0.02, 0.02, _param_text(params), fontsize=7, fontfamily='monospace',
                 verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA plot: {output_path}")


def plot_pca_2d(latents, labels, output_path, title="Latent PCA (2D)", params=None):
    """PCA project to 2D and save scatter plot."""
    pca = PCA(n_components=2)
    projected = pca.fit_transform(latents)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 6))

    unique_labels = np.unique(labels)
    for lab in unique_labels:
        mask = labels == lab
        ax.scatter(
            projected[mask, 0], projected[mask, 1],
            s=10, alpha=0.6, label=f"Class {int(lab)}"
        )

    ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
    ax.set_title(title)
    ax.legend(markerscale=3)

    if params:
        fig.text(0.02, 0.02, _param_text(params), fontsize=7, fontfamily='monospace',
                 verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved 2D PCA plot: {output_path}")


def run_analysis(model, dataloader, device, model_name="model", dataset_name="data",
                 output_dir="outputs/images", params=None):
    """Full analysis pipeline: extract latents, PCA, save plots."""
    os.makedirs(output_dir, exist_ok=True)

    latents, labels = extract_latents(model, dataloader, device)
    print(f"Extracted {latents.shape[0]} latents, dim={latents.shape[1]}")

    title = f"{model_name} - {dataset_name}"
    plot_pca_3d(latents, labels,
                os.path.join(output_dir, f"{model_name}_{dataset_name}_pca3d.png"),
                title=f"{title} (3D)", params=params)
    plot_pca_2d(latents, labels,
                os.path.join(output_dir, f"{model_name}_{dataset_name}_pca2d.png"),
                title=f"{title} (2D)", params=params)
