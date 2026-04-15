# MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging

**Paper:** https://arxiv.org/pdf/2511.14806
**Authors:** Siyuan Li, Kai Yu, Anna Wang, Zicheng Liu, Chang Yu, Jingbo Zhou, Qirong Yang, Yucheng Guo, Xiaoming Zhang, Stan Z. Li
**Venue:** AAAI 2026

## Problem

Genomic DNA modeling faces challenges from heterogeneous information density (only ~2% is coding sequence), no inherent word boundaries, and extreme sequence lengths. Fixed tokenization wastes capacity on repetitive regions and under-represents dense regions.

## Solution

MergeDNA is a hierarchical autoencoder Transformer that uses ToMe-style token merging to create a learnable tokenizer. It dynamically segments DNA sequences, allocating finer granularity to information-dense regions (coding sequences) and coarser granularity to repetitive regions.

## Architecture (4 components)

1. **Local Encoder** (4 blocks, 51M params) — local-window self-attention + ToMe merging per layer. Compresses N tokens to L tokens (~N/4). Outputs a source matrix S tracking which bases merged into which tokens.

2. **Latent Encoder** (20 blocks, 253M params) — standard full-attention Transformer on the L compressed tokens. During pre-training, does an additional ToMe pass to select K salient tokens (~N/8).

3. **Latent Decoder** (4 blocks, 51M params, pre-training only) — decodes back to L tokens.

4. **Local Decoder** (2 blocks, 25M params, pre-training only) — unmerges using S^T then refines back to N base-level predictions.

## Three Pre-training Losses

1. **L_MTR** — full autoencoder reconstruction (all params updated)
2. **lambda * L_MTR** (lambda=0.25) — latent path only, Local Encoder frozen, forces latent model to recover from aggressive merging
3. **L_AMTM** — adaptive masked token modeling, masks high-information tokens preferentially (inversely proportional to merge group size)

## Key Differences from ToMe

- **Local-window** merging (window=16) instead of global
- **Grouping embedding** for similarity (not attention keys)
- **Source matrix S** tracking merges across layers for unmerging
- **Compression ratio sampling** — L sampled from Gaussian each iteration
- 380M params total, trained on multi-species genomes, SOTA on 3 major DNA benchmarks

## Detailed Configuration

| Component    | Embedding Dim | Blocks | Block Type  | Params |
|-------------|---------------|--------|-------------|--------|
| Local Enc.  | 1024          | 4      | Local-Attn  | 51M    |
| Latent Enc. | 1024          | 20     | Attention   | 253M   |
| Latent Dec. | 1024          | 4      | Attention   | 51M    |
| Local Dec.  | 1024          | 2      | Local-Attn  | 25M    |

## Training Hyperparameters

| Setting              | Value             |
|---------------------|-------------------|
| Optimizer           | AdamW             |
| Betas               | (0.9, 0.95)       |
| Training iterations | 100,000           |
| Weight decay        | 1e-8              |
| Base learning rate  | 1e-4              |
| Batch size          | 256               |
| LR scheduler        | Cosine Annealing  |
| Warmup iterations   | 10,000            |
| Gradient clipping   | 1.0               |
| Local window size   | 16                |
| Max sequence length | 4096              |

## Pre-training Algorithm (per iteration)

```
Input: DNA sequence X of length N
Sample target compression L ~ Gaussian(N/2, variance) clipped to [0.4N, 0.6N]
Compute per-layer merge counts r_l

# Forward Pass 1: Full Autoencoder (MTR)
Z_L, S = LocalEncoder(X)                    # Stack of LocalAttn + ToMeMerge layers
Z'_L = LatentEncoder(Z_L)                   # Full attention Transformer
Z_hat_L = LatentDecoder(Z'_L)               # Symmetric to LatentEncoder
Z_bar_N = Unmerge(Z_hat_L, S)               # S^T @ Z_hat_L
X_hat = LocalDecoder(Z_bar_N)               # Stack of local attention
L_MTR = CrossEntropy(X_hat, X)

# Forward Pass 2: Latent MTR (Local Encoder frozen)
Z_L, S = LocalEncoder(X).detach()           # No gradient to phi
Z'_K, S' = LatentEncoder_with_ToMe(Z_L, S)  # Global ToMe selects K tokens
Z_hat_L = LatentDecoder(Z'_K)
Z_L_reconstructed = Unmerge(Z_hat_L, S')
Z_bar_N = Unmerge(Z_L_reconstructed, S)
X_hat = LocalDecoder(Z_bar_N)
L_latent_MTR = CrossEntropy(X_hat, X)

# Forward Pass 3: Adaptive Masked Token Modeling
Compute importance P_L from S' (inversely proportional to merge group size)
Sample K positions from P_L -> M_L
M_N = Unmerge(M_L, S)                       # Map mask to input space
X_masked = X * M_N
X_hat = FullPipeline(X_masked)              # No latent token merging
L_AMTM = CrossEntropy(X_hat[masked_positions], X[masked_positions])

# Total loss
L_total = L_MTR + 0.25 * L_latent_MTR + L_AMTM
Backpropagate and update theta
```

## Key Results

| Benchmark          | MergeDNA | Best Prior | Prior Model   |
|-------------------|----------|-----------|---------------|
| Genomic Bench (8) | 90.87    | 90.71     | GENERator 1.3B|
| NT Bench (18)     | 78.39    | 78.14     | MxDNA 100M    |
| GUE Bench (24)    | 77.11    | 76.42     | HybriDNA 7B   |

## Ablation (Genomic Benchmark avg accuracy)

| Configuration                                    | Accuracy | Delta  |
|-------------------------------------------------|----------|--------|
| Byte-level, 24 layers, naive MTM                | 89.30    | —      |
| + Local Encoder (4 merge blocks)                | 89.69    | +0.39  |
| + L_MTR + L^0_MTR                               | 90.33    | +1.03  |
| + L_MTR + 0.25*L^0_MTR + L_AMTM (full)         | 90.87    | +1.57  |
