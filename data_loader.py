"""
Load DNA datasets for MergeDNA training/evaluation.

Supports:
  - "synthetic": generated sine/cosine sequences (generate_data.py)
  - "nt:<task>": Nucleotide Transformer benchmark (18 tasks)
      e.g. "nt:enhancers", "nt:H3K4me3", "nt:splice_sites_donors"
  - "genomic:<task>": Genomic Benchmarks (8 tasks)
      e.g. "genomic:human_enhancers_cohn", "genomic:demo_coding_vs_intergenomic_seqs"
"""

import torch
from torch.utils.data import Dataset

# Nucleotide to int mapping
NUC_MAP = {"A": 0, "T": 1, "C": 2, "G": 3, "N": 0}  # N -> A as fallback


def encode_sequence(seq, max_len=None):
    """Convert DNA string to integer tensor."""
    encoded = [NUC_MAP.get(c, 0) for c in seq.upper()]
    if max_len is not None:
        encoded = encoded[:max_len]
        encoded += [0] * (max_len - len(encoded))  # pad with A
    return torch.tensor(encoded, dtype=torch.long)


class DNADataset(Dataset):
    """Wraps encoded sequences and optional labels."""

    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return (self.sequences[idx],)


def load_synthetic(data_dir="data"):
    """Load pre-generated synthetic data."""
    train = torch.load(f"{data_dir}/train.pt")
    test = torch.load(f"{data_dir}/test.pt")
    return DNADataset(train), DNADataset(test)


def load_hf_dataset(dataset_name, task, max_len=256):
    """Load a HuggingFace dataset with sequence + label columns."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name)

    def process_split(split_data, task_name):
        # Filter by task if the dataset has a 'task' column
        if "task" in split_data.column_names and task_name:
            split_data = split_data.filter(lambda x: x["task"] == task_name)
        seqs = [encode_sequence(ex["sequence"], max_len) for ex in split_data]
        labels = torch.tensor([ex["label"] for ex in split_data], dtype=torch.long)
        return DNADataset(torch.stack(seqs), labels)

    train = process_split(ds["train"], task)
    test = process_split(ds["test"], task)
    return train, test


def load_dataset_by_name(name, data_dir="data", max_len=256):
    """
    Load dataset by config string.

    Args:
        name: One of:
            "synthetic" - load from data_dir
            "nt:<task>" - Nucleotide Transformer benchmark
            "genomic:<task>" - Genomic Benchmarks
        data_dir: Directory for synthetic data.
        max_len: Max sequence length (truncate/pad).

    Returns:
        (train_dataset, test_dataset)
    """
    if name == "synthetic":
        return load_synthetic(data_dir)
    elif name.startswith("nt:"):
        task = name[3:]
        return load_hf_dataset(
            "InstaDeepAI/nucleotide_transformer_downstream_tasks", task, max_len
        )
    elif name.startswith("genomic:"):
        task = name[8:]
        return load_hf_dataset(
            "InstaDeepAI/genomic_benchmarks", task, max_len
        )
    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'synthetic', 'nt:<task>', or 'genomic:<task>'")
