"""
Generate synthetic DNA sequences with learnable structure.
Sums random sine/cosine waves to create a 1D signal, then discretizes
the y-axis into 4 bins mapped to nucleotides (0=A, 1=T, 2=C, 3=G).
"""

import argparse
import os
import torch
import numpy as np


def generate_signal(num_seqs, seq_len, num_waves_range=(2, 6)):
    """Create structured 1D signals from summed sine/cosine waves."""
    signals = np.zeros((num_seqs, seq_len))
    t = np.linspace(0, 2 * np.pi, seq_len)

    for i in range(num_seqs):
        num_waves = np.random.randint(*num_waves_range)
        for _ in range(num_waves):
            freq = np.random.uniform(0.5, 10.0)
            amp = np.random.uniform(0.2, 1.0)
            phase = np.random.uniform(0, 2 * np.pi)
            if np.random.rand() > 0.5:
                signals[i] += amp * np.sin(freq * t + phase)
            else:
                signals[i] += amp * np.cos(freq * t + phase)

    return signals


def discretize(signals):
    """Map continuous signal to 4 bins (0-3) using quartiles per sequence."""
    seqs = np.zeros_like(signals, dtype=np.int64)
    for i in range(signals.shape[0]):
        s = signals[i]
        q25, q50, q75 = np.percentile(s, [25, 50, 75])
        seqs[i][s < q25] = 0
        seqs[i][(s >= q25) & (s < q50)] = 1
        seqs[i][(s >= q50) & (s < q75)] = 2
        seqs[i][s >= q75] = 3
    return seqs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_seqs", type=int, default=1000)
    p.add_argument("--test_seqs", type=int, default=200)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--out_dir", type=str, default="data")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_signals = generate_signal(args.train_seqs, args.seq_len)
    test_signals = generate_signal(args.test_seqs, args.seq_len)

    train = torch.tensor(discretize(train_signals))
    test = torch.tensor(discretize(test_signals))

    torch.save(train, os.path.join(args.out_dir, "train.pt"))
    torch.save(test, os.path.join(args.out_dir, "test.pt"))

    print(f"Saved train: {train.shape} -> {args.out_dir}/train.pt")
    print(f"Saved test:  {test.shape} -> {args.out_dir}/test.pt")


if __name__ == "__main__":
    main()


