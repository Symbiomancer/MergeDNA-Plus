"""
Train DiffusionDNA with DDPM on DNA sequences.
Diffusion operates in embedding space: embed discrete tokens,
add noise, predict noise with the U-Net.
"""

import argparse
import json
import math
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.diffusion_dna import DiffusionDNA
from data_loader import load_dataset_by_name
from analysis import run_analysis


# --- DDPM noise schedule ---

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule from Nichol & Dhariwal."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps)


class DDPM:
    """Simple DDPM helper for forward diffusion and loss."""

    def __init__(self, timesteps=1000, schedule="cosine", device="cpu"):
        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)

        self.timesteps = timesteps
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: add noise to x_0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]

        return sqrt_alpha * x_0 + sqrt_one_minus * noise, noise


def get_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--dataset", type=str, default="synthetic")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)

    # Model
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--local_depth", type=int, default=2)
    p.add_argument("--local_num_heads", type=int, default=4)
    p.add_argument("--local_dim_ff", type=int, default=256)
    p.add_argument("--local_window_size", type=int, default=16)
    p.add_argument("--latent_depth", type=int, default=4)
    p.add_argument("--latent_num_heads", type=int, default=4)
    p.add_argument("--latent_dim_ff", type=int, default=256)

    # Diffusion
    p.add_argument("--diffusion_steps", type=int, default=1000)
    p.add_argument("--schedule", choices=["cosine", "linear"], default="cosine")

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = get_args()

    # Load data
    train_set, test_set = load_dataset_by_name(args.dataset, args.data_dir, args.max_len)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    print(f"Dataset: {args.dataset} | Train: {len(train_set)} | Test: {len(test_set)}")

    # Model
    model = DiffusionDNA(
        vocab_size=4,
        dim=args.dim,
        local_depth=args.local_depth,
        local_num_heads=args.local_num_heads,
        local_dim_ff=args.local_dim_ff,
        local_window_size=args.local_window_size,
        latent_depth=args.latent_depth,
        latent_num_heads=args.latent_num_heads,
        latent_dim_ff=args.latent_dim_ff,
    ).to(args.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"DIFFUSION | dim={args.dim} | local_depth={args.local_depth} | latent_depth={args.latent_depth}")
    print(f"Parameters: {num_params:,} | Steps: {args.diffusion_steps} | Schedule: {args.schedule}")

    ddpm = DDPM(args.diffusion_steps, args.schedule, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    total_train_time = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, total = 0.0, 0
        epoch_start = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs}", leave=False)

        for batch in pbar:
            seqs = batch[0].to(args.device)

            # Embed discrete tokens to continuous space
            with torch.no_grad():
                x_0 = model.embed(seqs)  # [B, N, dim]

            # Sample random timesteps
            t = torch.randint(0, args.diffusion_steps, (seqs.shape[0],), device=args.device)

            # Forward diffusion: add noise
            x_noisy, noise = ddpm.q_sample(x_0, t)

            # Predict noise
            noise_pred = model(x_noisy, t)

            # MSE loss on noise prediction
            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_total = seqs.shape[0]
            train_loss += loss.item() * batch_total
            total += batch_total
            pbar.set_postfix(loss=f"{train_loss/total:.6f}")

        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        scheduler.step()

        # Eval loss
        model.eval()
        test_loss, test_total = 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                seqs = batch[0].to(args.device)
                x_0 = model.embed(seqs)
                t = torch.randint(0, args.diffusion_steps, (seqs.shape[0],), device=args.device)
                x_noisy, noise = ddpm.q_sample(x_0, t)
                noise_pred = model(x_noisy, t)
                loss = nn.functional.mse_loss(noise_pred, noise)
                test_loss += loss.item() * seqs.shape[0]
                test_total += seqs.shape[0]

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss/total:.6f} | "
              f"Test Loss: {test_loss/test_total:.6f} | "
              f"Epoch Time: {epoch_time:.2f}s")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss / total, 6),
            "test_loss": round(test_loss / test_total, 6),
            "epoch_time": round(epoch_time, 2),
        })

    # Save checkpoint
    ckpt_path = "checkpoint_diffusion.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved checkpoint: {ckpt_path}")
    print(f"Total train time: {total_train_time:.2f}s | Avg epoch: {total_train_time/args.epochs:.2f}s")

    # Save results
    os.makedirs("outputs/results", exist_ok=True)
    dataset_tag = args.dataset.replace(":", "_")
    results = {
        "model": "diffusion",
        "dataset": args.dataset,
        "config": {k: v for k, v in vars(args).items() if k != "device"},
        "num_params": num_params,
        "total_train_time": round(total_train_time, 2),
        "history": history,
    }
    results_path = f"outputs/results/diffusion_{dataset_tag}_pretrain.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {results_path}")

    # Post-training analysis (use encoder path for latents)
    train_params = {
        "model": "diffusion",
        "dim": args.dim,
        "local_depth": args.local_depth,
        "local_dim_ff": args.local_dim_ff,
        "local_window": args.local_window_size,
        "latent_depth": args.latent_depth,
        "latent_dim_ff": args.latent_dim_ff,
        "diff_steps": args.diffusion_steps,
        "schedule": args.schedule,
        "epochs": args.epochs,
        "lr": args.lr,
        "params": f"{num_params:,}",
        "test_loss": f"{test_loss/test_total:.6f}",
    }
    run_analysis(model, test_loader, args.device,
                 model_name="diffusion", dataset_name=dataset_tag, params=train_params)


if __name__ == "__main__":
    main()
