"""
Extract latent representations, run PCA, and save for MATLAB visualization.
Outputs a .mat file with PCA-projected latents and labels.
"""

import argparse
import torch
import numpy as np
from sklearn.decomposition import PCA
from scipy.io import savemat
from torch.utils.data import DataLoader

from models import MergeDNA, ConvDNA
from data_loader import load_dataset_by_name


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to saved model .pt")
    p.add_argument("--dataset", type=str, default="synthetic")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--output", type=str, default="latents.mat")
    p.add_argument("--n_components", type=int, default=3)

    # Model args (must match checkpoint)
    p.add_argument("--model", choices=["merge", "conv"], default="merge")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--local_depth", type=int, default=2)
    p.add_argument("--local_num_heads", type=int, default=4)
    p.add_argument("--local_dim_ff", type=int, default=256)
    p.add_argument("--local_window_size", type=int, default=16)
    p.add_argument("--latent_depth", type=int, default=4)
    p.add_argument("--latent_num_heads", type=int, default=4)
    p.add_argument("--latent_dim_ff", type=int, default=256)
    p.add_argument("--merge_r", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_model(args):
    common = dict(
        vocab_size=4,
        dim=args.dim,
        local_depth=args.local_depth,
        local_num_heads=args.local_num_heads,
        local_dim_ff=args.local_dim_ff,
        local_window_size=args.local_window_size,
        latent_depth=args.latent_depth,
        latent_num_heads=args.latent_num_heads,
        latent_dim_ff=args.latent_dim_ff,
    )
    if args.model == "conv":
        return ConvDNA(**common)
    else:
        return MergeDNA(**common, merge_r=args.merge_r)


def extract_latents(model, dataloader, device):
    """Run sequences through local + latent encoder, return mean-pooled latents."""
    model.eval()
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            seqs = batch[0].to(device)
            labels = batch[1] if len(batch) > 1 else None

            x = model.embed(seqs)
            x = model.pos_drop(x)

            if hasattr(model, 'encode_local'):
                # MergeDNA
                x, _ = model.encode_local(x)
                x = model.encode_latent(x)
            else:
                # ConvDNA — run encoder blocks + downsamples
                for attn, down in zip(model.encoder_blocks, model.downsamples):
                    x = attn(x)
                    x = down(x)
                for blk in model.latent_blocks:
                    x = blk(x)

            # Mean pool over sequence dim -> [B, dim]
            latent = x.mean(dim=1)
            all_latents.append(latent.cpu().numpy())

            if labels is not None:
                all_labels.append(labels.numpy())

    latents = np.concatenate(all_latents, axis=0)
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros(len(latents))
    return latents, labels


def main():
    args = get_args()

    _, test_set = load_dataset_by_name(args.dataset, args.data_dir, args.max_len)
    train_set, _ = load_dataset_by_name(args.dataset, args.data_dir, args.max_len)
    dataset = train_set if args.split == "train" else test_set
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    print(f"Loaded checkpoint: {args.checkpoint}")

    latents, labels = extract_latents(model, loader, args.device)
    print(f"Extracted latents: {latents.shape}")

    # PCA
    pca = PCA(n_components=args.n_components)
    projected = pca.fit_transform(latents)
    explained = pca.explained_variance_ratio_
    print(f"PCA variance explained: {explained}")

    # Save for MATLAB
    savemat(args.output, {
        "latents_pca": projected,
        "labels": labels,
        "explained_variance": explained,
    })
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
