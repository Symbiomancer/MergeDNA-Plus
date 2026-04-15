# MergeDNA+

Implementation of [MergeDNA](https://arxiv.org/abs/2511.14806) — hierarchical autoencoder for genomic DNA with learnable tokenization via token merging.

## Install

```bash
pip install torch torchvision local-attention tqdm numpy
```

## Architecture

`models/merge_dna.py` — four-stage autoencoder:

1. **Local Encoder** — local-window attention ([lucidrains/local-attention](https://github.com/lucidrains/local-attention)), compresses tokens
2. **Latent Encoder** — full self-attention for long-range context
3. **Latent Decoder** — mirrors latent encoder
4. **Local Decoder** — mirrors local encoder, reconstructs base-level output

Local and latent stages share parameter configs (`local_*` and `latent_*` args) between their encoder/decoder pairs.

## Encoding

Nucleotides are discrete categories with no ordinal relationship. Using integer labels with MSE would imply A(0) is "closer" to T(1) than G(3). Instead, we use embedding + cross-entropy loss, which treats all four bases as equally distinct.

## Synthetic Data

`generate_data.py` creates structured test sequences by summing random sine/cosine waves and discretizing into 4 bins via quartiles. This gives learnable local patterns, unlike random sequences which cap at 25% accuracy.

```bash
python generate_data.py --train_seqs 1000 --test_seqs 200 --seq_len 256
```

## Train

```bash
python train.py --dim 64 --local_depth 2 --latent_depth 4 --epochs 20
```

## TODO

- Token merging (ToMe) integration with source matrix tracking
- Adaptive pre-training objectives (L_MTR, L_AMTM)
- Compression ratio sampling
