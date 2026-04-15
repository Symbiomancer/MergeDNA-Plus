"""
Fine-tune a classification head on a frozen pre-trained encoder.

Usage:
  1. Pre-train:  python train.py --model merge --dataset multispecies --epochs 20
  2. Fine-tune:  python train_classifier.py --checkpoint checkpoint_merge.pt \
                   --encoder merge --dataset nt:enhancers --epochs 10
"""

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import MergeDNA, ConvDNA
from models.diffusion_dna import DiffusionDNA
from models.classifier import DNAClassifier
from data_loader import load_dataset_by_name
from analysis import run_analysis


def get_args():
    p = argparse.ArgumentParser()
    # Pre-trained encoder
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--encoder", choices=["merge", "conv", "diffusion"], required=True)

    # Data
    p.add_argument("--dataset", type=str, required=True,
                   help="Downstream task: 'nt:<task>' or 'genomic:<task>'")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_classes", type=int, default=2)

    # Model (must match pre-trained checkpoint)
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
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_encoder(args):
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
    if args.encoder == "conv":
        return ConvDNA(**common)
    elif args.encoder == "diffusion":
        return DiffusionDNA(**common)
    else:
        return MergeDNA(**common, merge_r=args.merge_r)


def main():
    args = get_args()

    # Load downstream dataset
    train_set, test_set = load_dataset_by_name(args.dataset, args.data_dir, args.max_len)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    print(f"Dataset: {args.dataset} | Train: {len(train_set)} | Test: {len(test_set)}")
    print(f"Num classes: {args.num_classes}")

    # Load pre-trained encoder
    encoder = build_encoder(args)
    state = torch.load(args.checkpoint, map_location=args.device)
    encoder.load_state_dict(state)
    print(f"Loaded encoder from {args.checkpoint}")

    # Build classifier
    model = DNAClassifier(encoder, args.dim, args.num_classes).to(args.device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Classifier | Trainable: {trainable:,} / {total:,} total")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    total_train_time = 0.0
    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, correct, total_samples = 0.0, 0, 0
        epoch_start = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs}", leave=False)

        for seqs, labels in pbar:
            seqs, labels = seqs.to(args.device), labels.to(args.device)
            logits = model(seqs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * seqs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total_samples += seqs.size(0)
            pbar.set_postfix(
                loss=f"{train_loss/total_samples:.4f}",
                acc=f"{100.*correct/total_samples:.1f}%"
            )

        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        scheduler.step()
        train_acc = 100.0 * correct / total_samples

        # Eval
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for seqs, labels in test_loader:
                seqs, labels = seqs.to(args.device), labels.to(args.device)
                logits = model(seqs)
                test_correct += (logits.argmax(1) == labels).sum().item()
                test_total += seqs.size(0)
        test_acc = 100.0 * test_correct / test_total
        best_test_acc = max(best_test_acc, test_acc)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss/total_samples:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | "
              f"Epoch Time: {epoch_time:.2f}s")

    print(f"\nBest Test Acc: {best_test_acc:.2f}%")
    print(f"Total train time: {total_train_time:.2f}s | Avg epoch: {total_train_time/args.epochs:.2f}s")

    # Save classifier
    ckpt_path = f"checkpoint_classifier_{args.encoder}.pt"
    torch.save(model.head.state_dict(), ckpt_path)
    print(f"Saved classifier head: {ckpt_path}")

    # Analysis
    dataset_tag = args.dataset.replace(":", "_")
    train_params = {
        "encoder": args.encoder,
        "dim": args.dim,
        "local_depth": args.local_depth,
        "latent_depth": args.latent_depth,
        "num_classes": args.num_classes,
        "epochs": args.epochs,
        "lr": args.lr,
        "trainable_params": f"{trainable:,}",
        "best_test_acc": f"{best_test_acc:.2f}%",
    }
    if args.encoder == "merge":
        train_params["merge_r"] = args.merge_r
    run_analysis(model.encoder, test_loader, args.device,
                 model_name=f"classifier_{args.encoder}", dataset_name=dataset_tag,
                 params=train_params)


if __name__ == "__main__":
    main()
