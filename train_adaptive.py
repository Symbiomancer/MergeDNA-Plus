"""
Adaptive pre-training for MergeDNA (Section 3.4).
Three forward passes per iteration:
  1. L_MTR:   Full autoencoder reconstruction (all params)
  2. L_latent_MTR: Latent path with global ToMe, local encoder frozen
  3. L_AMTM:  Adaptive masked token modeling (importance-weighted masking)

L_total = L_MTR + 0.25 * L_latent_MTR + L_AMTM
"""

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.merge_dna import MergeDNA, unmerge
from data_loader import load_dataset_by_name
from analysis import run_analysis


def compute_amtm_mask(S_latent, S_local, K, device):
    """
    Compute importance-weighted mask for AMTM (Section 3.4).

    Args:
        S_latent: [B, K_latent, L] source matrix from global ToMe.
        S_local: [B, L, N] source matrix from local encoder.
        K: Number of tokens to mask.
        device: torch device.

    Returns:
        mask_N: [B, N] binary mask (1 = masked, 0 = visible).
    """
    B, K_lat, L = S_latent.shape

    # Group size for each latent token: how many local tokens it absorbed
    g = S_latent.sum(dim=-1)  # [B, K_lat]

    # Importance weight: inversely proportional to group size
    w = 1.0 / g.clamp(min=1)  # [B, K_lat]

    # Spread weights back to local tokens: each local token j in group i gets w_i/g_i
    # P_L[j] = sum over i of S_latent[i,j] * w[i] / g[i]
    w_per_group = w / g.clamp(min=1)  # [B, K_lat]
    P_L = torch.bmm(S_latent.transpose(1, 2).float(), w_per_group.unsqueeze(-1)).squeeze(-1)  # [B, L]

    # Normalize to probability distribution
    P_L = P_L / P_L.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # Sample K local tokens to mask
    K = min(K, L)
    mask_indices = torch.multinomial(P_L, K, replacement=False)  # [B, K]

    # Build local mask
    mask_L = torch.zeros(B, L, device=device)
    mask_L.scatter_(1, mask_indices, 1.0)

    # Map to input space via source matrix: if a merged token is masked, all bases are masked
    # mask_N = S_local^T @ mask_L
    mask_N = torch.bmm(S_local.transpose(1, 2).float(), mask_L.unsqueeze(-1)).squeeze(-1)  # [B, N]
    mask_N = (mask_N > 0).float()

    return mask_N


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="multispecies")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)

    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--local_depth", type=int, default=2)
    p.add_argument("--local_num_heads", type=int, default=4)
    p.add_argument("--local_dim_ff", type=int, default=256)
    p.add_argument("--local_window_size", type=int, default=16)
    p.add_argument("--latent_depth", type=int, default=4)
    p.add_argument("--latent_num_heads", type=int, default=4)
    p.add_argument("--latent_dim_ff", type=int, default=256)
    p.add_argument("--merge_r", type=int, default=4)
    p.add_argument("--latent_merge_r", type=int, default=8,
                   help="Tokens to merge globally in latent encoder for L_latent_MTR")

    p.add_argument("--lambda_latent", type=float, default=0.25)
    p.add_argument("--amtm_k", type=int, default=32,
                   help="Number of tokens to mask in AMTM")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = get_args()

    train_set, test_set = load_dataset_by_name(args.dataset, args.data_dir, args.max_len)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    print(f"Dataset: {args.dataset} | Train: {len(train_set)} | Test: {len(test_set)}")

    model = MergeDNA(
        vocab_size=4,
        dim=args.dim,
        local_depth=args.local_depth,
        local_num_heads=args.local_num_heads,
        local_dim_ff=args.local_dim_ff,
        local_window_size=args.local_window_size,
        latent_depth=args.latent_depth,
        latent_num_heads=args.latent_num_heads,
        latent_dim_ff=args.latent_dim_ff,
        merge_r=args.merge_r,
    ).to(args.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"ADAPTIVE | dim={args.dim} | local_depth={args.local_depth} | "
          f"latent_depth={args.latent_depth} | merge_r={args.merge_r} | "
          f"latent_merge_r={args.latent_merge_r} | lambda={args.lambda_latent}")
    print(f"Parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    total_train_time = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_mtr, epoch_latent, epoch_amtm, total = 0.0, 0.0, 0.0, 0
        epoch_start = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs}", leave=False)

        for batch in pbar:
            seqs = batch[0].to(args.device)
            B, N = seqs.shape

            # === Forward Pass 1: L_MTR (full autoencoder, compression ratio sampling) ===
            logits_mtr = model(seqs, sample_compression=True)
            loss_mtr = criterion(logits_mtr.view(-1, 4), seqs.view(-1))

            # === Forward Pass 2: L_latent_MTR (local encoder frozen) ===
            with torch.no_grad():
                x_emb = model.embed(seqs)
                x_emb = model.pos_drop(x_emb)
                z_L, S_local = model.encode_local(x_emb, sample_compression=True)

            # Latent encoder with global ToMe (gradients flow here)
            z_K, S_latent = model.encode_latent_with_tome(z_L, S_local, args.latent_merge_r)

            # Decode
            z_hat_L = model.decode_latent(z_K)
            z_hat_L = unmerge(z_hat_L, S_latent)  # K -> L
            z_bar_N = unmerge(z_hat_L, S_local)    # L -> N
            z_bar_N = model.decode_local(z_bar_N)
            logits_latent = model.head(model.norm_out(z_bar_N))
            loss_latent = criterion(logits_latent.view(-1, 4), seqs.view(-1))

            # === Forward Pass 3: L_AMTM (importance-weighted masking) ===
            # Compute mask from S_latent
            mask_N = compute_amtm_mask(S_latent, S_local, args.amtm_k, args.device)

            # Mask the input embeddings (zero out masked positions)
            x_masked = model.embed(seqs) * (1 - mask_N.unsqueeze(-1))
            x_masked = model.pos_drop(x_masked)

            # Full pipeline without latent token merging
            z_L_m, S_local_m = model.encode_local(x_masked, sample_compression=True)
            z_L_m = model.encode_latent(z_L_m)
            z_hat_m = model.decode_latent(z_L_m)
            z_bar_m = unmerge(z_hat_m, S_local_m)
            z_bar_m = model.decode_local(z_bar_m)
            logits_amtm = model.head(model.norm_out(z_bar_m))

            # Loss only on masked positions
            mask_flat = mask_N.view(-1).bool()
            if mask_flat.any():
                loss_amtm = criterion(logits_amtm.view(-1, 4)[mask_flat],
                                      seqs.view(-1)[mask_flat])
            else:
                loss_amtm = torch.tensor(0.0, device=args.device)

            # === Total loss ===
            loss_total = loss_mtr + args.lambda_latent * loss_latent + loss_amtm

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch_n = B * N
            epoch_mtr += loss_mtr.item() * batch_n
            epoch_latent += loss_latent.item() * batch_n
            epoch_amtm += loss_amtm.item() * batch_n
            total += batch_n
            pbar.set_postfix(
                mtr=f"{epoch_mtr/total:.4f}",
                lat=f"{epoch_latent/total:.4f}",
                amtm=f"{epoch_amtm/total:.4f}",
            )

        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        scheduler.step()

        # Eval (simple reconstruction)
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                seqs = batch[0].to(args.device)
                logits = model(seqs)
                test_correct += (logits.argmax(-1) == seqs).sum().item()
                test_total += seqs.numel()
        test_acc = 100.0 * test_correct / test_total

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"MTR: {epoch_mtr/total:.4f} | "
              f"Latent: {epoch_latent/total:.4f} | "
              f"AMTM: {epoch_amtm/total:.4f} | "
              f"Test Recon Acc: {test_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s")

    ckpt_path = "checkpoint_adaptive.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved checkpoint: {ckpt_path}")
    print(f"Total train time: {total_train_time:.2f}s | Avg epoch: {total_train_time/args.epochs:.2f}s")

    # Analysis
    dataset_tag = args.dataset.replace(":", "_")
    train_params = {
        "model": "adaptive_merge",
        "dim": args.dim,
        "local_depth": args.local_depth,
        "latent_depth": args.latent_depth,
        "merge_r": args.merge_r,
        "latent_merge_r": args.latent_merge_r,
        "lambda": args.lambda_latent,
        "amtm_k": args.amtm_k,
        "epochs": args.epochs,
        "params": f"{num_params:,}",
        "test_recon_acc": f"{test_acc:.2f}%",
    }
    run_analysis(model, test_loader, args.device,
                 model_name="adaptive", dataset_name=dataset_tag, params=train_params)


if __name__ == "__main__":
    main()
