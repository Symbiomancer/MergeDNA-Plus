# MergeDNA+

Implementation of [MergeDNA](https://arxiv.org/abs/2511.14806) with additional compression variants. There are three variants: (1) The original version based on Token Merging (see below), and traditional and local attention transformer layers, (2) A U-Net convolutional attentional network used as the backbone of diffusion but without the diffusion training process, (3) the U-Net combined with denoising diffusion learning.

## Install

```bash
pip install torch local-attention tqdm numpy scikit-learn matplotlib 'datasets==2.21.0' biopython
```

## Models

### 1. Merge (ToMe)

Token merging via [bipartite soft matching](https://arxiv.org/abs/2210.09461). Local attention encoder merges r tokens per block using cosine similarity on keys. Same as in paper,. a source matrix S tracks merges for unmerging in decoder.  The decoder is only used for pretraining; for the classification task the decoder is unhooked and a classifier appended. 

```bash
python train.py --model merge --dataset nt:enhancers --merge_r 4
```

Or set `"model": "merge"` in `config.json` and run `./run.sh`.

### 2. Conv (U-Net)

Strided convolutions replace token merging via ToME. This creates a "learnable" compression instead of similarity-based matching. Contains dilated convolutions for upsampling and skip connections. Uses the U-Net and diffusion code from https://github.com/lucidrains/denoising-diffusion-pytorch.

```mermaid
graph TD
    subgraph ENCODER
        A["DNA Input [B, N]"] --> B["Embedding [B, N, D]"]
        B --> C["Local Attn"]
        C --> D["Strided Conv — N → N/2"]
        D --> E["Local Attn"]
        E --> F["Strided Conv — N/2 → N/4"]
    end

    subgraph BOTTLENECK
        F --> G["Full Self-Attention × L — [B, N/4, D]"]
    end

    subgraph DECODER
        G --> H["Dilated Conv — N/4 → N/2"]
        H --> I["Concat + Proj"]
        I --> J["Local Attn"]
        J --> K["Dilated Conv — N/2 → N"]
        K --> L["Concat + Proj"]
        L --> M["Local Attn"]
        M --> N["Head → [B, N, 4]"]
    end

    C -. "skip" .-> L
    E -. "skip" .-> I

    style A fill:#4ECDC4,stroke:#333
    style B fill:#4ECDC4,stroke:#333
    style C fill:#45B7D1,stroke:#333
    style D fill:#F7DC6F,stroke:#333
    style E fill:#45B7D1,stroke:#333
    style F fill:#F7DC6F,stroke:#333
    style G fill:#E74C3C,stroke:#333,color:#fff
    style H fill:#F39C12,stroke:#333
    style I fill:#D5D8DC,stroke:#333
    style J fill:#45B7D1,stroke:#333
    style K fill:#F39C12,stroke:#333
    style L fill:#D5D8DC,stroke:#333
    style M fill:#45B7D1,stroke:#333
    style N fill:#2ECC71,stroke:#333
```

```bash
python train.py --model conv --dataset nt:enhancers
```

Or set `"model": "conv"` in `config.json` and run `./run.sh`.

### 3. Diffusion (DDPM + Conv U-Net)

The same Conv U-Net but trained with DDPM denoising. Embeds discrete tokens to continuous space, adds Gaussian noise at timestep t, and predicts noise. Can use cosine or linear schedule.

```bash
python train_diffusion.py --dataset nt:enhancers --diffusion_steps 1000 --schedule cosine
```

Diffusion uses its own training script (`train_diffusion.py`); set dataset/model params in `config.json` and run directly.

## Pre-train + Classification Pipeline

Following the MergeDNA paper: pre-train the autoencoder on [Multi-Species Genomes](https://huggingface.co/datasets/InstaDeepAI/multi_species_genomes), freeze the encoder, then fine-tune a classification head on a downstream task.

```bash
# Step 1: Pre-train on multi-species genomes
python train.py --model merge --dataset multispecies --epochs 20

# Step 2: Fine-tune classifier (encoder frozen, only head trains)
python train_classifier.py \
  --checkpoint checkpoint_merge.pt \
  --encoder merge \
  --dataset nt:enhancers \
  --num_classes 2 \
  --epochs 10
```

Run all three variants automatically:

```bash
./run_experiment.sh
```

## Encoding

Nucleotides (A/T/C/G) are discrete categories with no ordinal relationship. Thus, uses embedding + cross-entropy loss, not integer MSE.

## Datasets

```json
"dataset": "synthetic"                     // sine/cosine waves discretized to 4 bins
"dataset": "multispecies"                  // Multi-Species Genomes (pre-training)
"dataset": "nt:enhancers"                 // Nucleotide Transformer benchmark (18 tasks)
"dataset": "genomic:human_enhancers_cohn" // Genomic Benchmarks (8 tasks)
```

Generate synthetic: `python generate_data.py --train_seqs 1000 --seq_len 256`

## Config

All params are located in `config.json`, CLI overrides: `./run.sh --model conv --dim 128`. Make adjustments there and then use the ./run.sh script. 

## Results

![PCA 3D — ToMe Merge on nt:enhancers](outputs/images/merge_nt_enhancers_pca3d.png)

Output of running PCA on the latents produced by the ToMe merge method on the nt:enhancers dataset (i.e., pretrained on this set). We can see the network clusters the classes without labels fairly well. Important to note this is Linear PCA as well.

## Adaptive Pre-training (Section 3.4)

Full MergeDNA pre-training with all three losses and compression ratio sampling:

```bash
python train_adaptive.py --dataset multispecies --merge_r 4 --latent_merge_r 8 --amtm_k 32
```

**Compression ratio sampling** (Section 3.3): each forward pass samples target length L ~ Gaussian(N/2) clipped to [0.4N, 0.6N], preventing overfitting to a fixed compression rate.

Three forward passes per iteration:
1. **L_MTR** — full autoencoder reconstruction, compression ratio sampled from Gaussian
2. **L_latent_MTR** (x0.25) — global ToMe in latent encoder, local encoder frozen
3. **L_AMTM** — importance-weighted masking, loss only on high-information tokens
