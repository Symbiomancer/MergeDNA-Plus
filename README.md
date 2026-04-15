# MergeDNA+

Implementation of [MergeDNA](https://arxiv.org/abs/2511.14806) with additional compression variants. Hierarchical autoencoder for genomic DNA.

## Install

```bash
pip install torch local-attention tqdm numpy scikit-learn matplotlib datasets
```

## Models

### 1. Merge (ToMe)

Token merging via [bipartite soft matching](https://arxiv.org/abs/2210.09461). Local attention encoder merges r tokens per block using cosine similarity on keys. Source matrix S tracks merges for unmerging in decoder.

```bash
python train.py --model merge --dataset nt:enhancers --merge_r 4
```

### 2. Conv (U-Net)

Strided convolutions replace token merging. Learnable compression instead of similarity-based matching. Dilated convolutions for upsampling. U-Net skip connections.

```bash
python train.py --model conv --dataset nt:enhancers
```

### 3. Diffusion (DDPM + Conv U-Net)

Same Conv U-Net but trained with DDPM denoising. Embeds discrete tokens to continuous space, adds Gaussian noise at timestep t, predicts noise. Cosine or linear schedule.

```bash
python train_diffusion.py --dataset nt:enhancers --diffusion_steps 1000 --schedule cosine
```

## Encoding

Nucleotides (A/T/C/G) are discrete categories with no ordinal relationship. Embedding + cross-entropy loss, not integer MSE.

## Datasets

```json
"dataset": "synthetic"                  // sine/cosine waves discretized to 4 bins
"dataset": "nt:enhancers"               // Nucleotide Transformer benchmark (18 tasks)
"dataset": "genomic:human_enhancers_cohn"  // Genomic Benchmarks (8 tasks)
```

Generate synthetic: `python generate_data.py --train_seqs 1000 --seq_len 256`

## Config

All params in `config.json`, CLI overrides: `./run.sh --model conv --dim 128`

## Results

![PCA 3D — ToMe Merge on nt:enhancers](outputs/images/merge_nt_enhancers_pca3d.png)

Output of running the ToMe merge method on the nt:enhancers dataset (i.e., pretrained on this set), we can see the network clusters the classes without labels fairly well. Important to note this is Linear PCA as well.

## TODO

- Adaptive pre-training objectives (L_MTR, L_AMTM)
- Compression ratio sampling
- Classification fine-tuning head
