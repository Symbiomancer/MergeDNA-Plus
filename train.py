"""Train MergeDNA autoencoder on DNA sequences."""

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models import MergeDNA


def get_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--data_dir", type=str, default="data")
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

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = get_args()

    # Load data
    train_data = torch.load(f"{args.data_dir}/train.pt")
    test_data = torch.load(f"{args.data_dir}/test.pt")

    train_loader = DataLoader(TensorDataset(train_data), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=args.batch_size, shuffle=False)

    # Model
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
    ).to(args.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"MergeDNA | dim={args.dim} | local_depth={args.local_depth} | latent_depth={args.latent_depth}")
    print(f"Parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    total_train_time = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        epoch_start = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs}", leave=False)
        for (seqs,) in pbar:
            seqs = seqs.to(args.device)
            logits = model(seqs)  # [B, N, 4]

            loss = criterion(logits.view(-1, 4), seqs.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_total = seqs.numel()
            train_loss += loss.item() * batch_total
            correct += (logits.argmax(-1) == seqs).sum().item()
            total += batch_total
            pbar.set_postfix(loss=f"{train_loss/total:.4f}", acc=f"{100.*correct/total:.1f}%")

        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        scheduler.step()
        train_acc = 100.0 * correct / total

        # Eval
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for (seqs,) in test_loader:
                seqs = seqs.to(args.device)
                logits = model(seqs)
                test_correct += (logits.argmax(-1) == seqs).sum().item()
                test_total += seqs.numel()
        test_acc = 100.0 * test_correct / test_total

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss/total:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | "
              f"Epoch Time: {epoch_time:.2f}s")

    print(f"\nTotal train time: {total_train_time:.2f}s | Avg epoch: {total_train_time/args.epochs:.2f}s")


if __name__ == "__main__":
    main()
