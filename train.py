"""Train MergeDNA autoencoder on DNA sequences."""

import argparse
import json
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import MergeDNA, ConvDNA
from data_loader import load_dataset_by_name
from analysis import run_analysis


def get_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--dataset", type=str, default="synthetic",
                   help="'synthetic', 'nt:<task>', or 'genomic:<task>'")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)

    # Model
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
        model = ConvDNA(**common).to(args.device)
    else:
        model = MergeDNA(**common, merge_r=args.merge_r).to(args.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model.upper()} | dim={args.dim} | local_depth={args.local_depth} | latent_depth={args.latent_depth}" +
          (f" | merge_r={args.merge_r}" if args.model == "merge" else ""))
    print(f"Parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    total_train_time = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        epoch_start = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs}", leave=False)
        for batch in pbar:
            seqs = batch[0].to(args.device)
            logits = model(seqs)  # [B, N, 4]

            loss = criterion(logits.view(-1, 4), seqs.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_total = seqs.numel()
            train_loss += loss.item() * batch_total
            correct += (logits.argmax(-1) == seqs).sum().item()
            total += batch_total
            pbar.set_postfix(loss=f"{train_loss/total:.4f}", recon_acc=f"{100.*correct/total:.1f}%")

        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        scheduler.step()
        train_acc = 100.0 * correct / total

        # Eval
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
              f"Train Loss: {train_loss/total:.4f} | Train Recon Acc: {train_acc:.2f}% | "
              f"Test Recon Acc: {test_acc:.2f}% | "
              f"Epoch Time: {epoch_time:.2f}s")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss / total, 6),
            "train_recon_acc": round(train_acc, 2),
            "test_recon_acc": round(test_acc, 2),
            "epoch_time": round(epoch_time, 2),
        })

    # Save checkpoint
    ckpt_path = f"checkpoint_{args.model}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved checkpoint: {ckpt_path}")
    print(f"Total train time: {total_train_time:.2f}s | Avg epoch: {total_train_time/args.epochs:.2f}s")

    # Save results
    os.makedirs("outputs/results", exist_ok=True)
    dataset_tag = args.dataset.replace(":", "_")
    results = {
        "model": args.model,
        "dataset": args.dataset,
        "config": {k: v for k, v in vars(args).items() if k != "device"},
        "num_params": num_params,
        "total_train_time": round(total_train_time, 2),
        "history": history,
    }
    results_path = f"outputs/results/{args.model}_{dataset_tag}_pretrain.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {results_path}")

    # Post-training analysis
    train_params = {
        "model": args.model,
        "dim": args.dim,
        "local_depth": args.local_depth,
        "local_dim_ff": args.local_dim_ff,
        "local_window": args.local_window_size,
        "latent_depth": args.latent_depth,
        "latent_dim_ff": args.latent_dim_ff,
        "epochs": args.epochs,
        "lr": args.lr,
        "params": f"{num_params:,}",
        "test_recon_acc": f"{test_acc:.2f}%",
    }
    if args.model == "merge":
        train_params["merge_r"] = args.merge_r
    run_analysis(model, test_loader, args.device,
                 model_name=args.model, dataset_name=dataset_tag, params=train_params)


if __name__ == "__main__":
    main()
