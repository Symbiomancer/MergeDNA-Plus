#!/bin/bash
# Full experiment: pre-train on multispecies, then fine-tune classifier on downstream task.
# Runs all three model variants: merge, conv, diffusion.

source ~/diffusion/.venv/bin/activate
set -e

# --- Shared params (edit these) ---
PRETRAIN_DATASET="multispecies"
FINETUNE_DATASET="nt:enhancers"
NUM_CLASSES=2
DIM=64
LOCAL_DEPTH=2
LOCAL_NUM_HEADS=4
LOCAL_DIM_FF=256
LOCAL_WINDOW_SIZE=16
LATENT_DEPTH=4
LATENT_NUM_HEADS=4
LATENT_DIM_FF=256
MERGE_R=4
PRETRAIN_EPOCHS=20
FINETUNE_EPOCHS=10
PRETRAIN_LR=1e-3
FINETUNE_LR=1e-4
WEIGHT_DECAY=1e-4
MAX_LEN=256
BATCH_SIZE=32
DIFFUSION_STEPS=1000

SHARED_ARGS="--dim $DIM --local_depth $LOCAL_DEPTH --local_num_heads $LOCAL_NUM_HEADS \
  --local_dim_ff $LOCAL_DIM_FF --local_window_size $LOCAL_WINDOW_SIZE \
  --latent_depth $LATENT_DEPTH --latent_num_heads $LATENT_NUM_HEADS \
  --latent_dim_ff $LATENT_DIM_FF --max_len $MAX_LEN --batch_size $BATCH_SIZE"

CLASSIFIER_ARGS="--dataset $FINETUNE_DATASET --num_classes $NUM_CLASSES \
  --epochs $FINETUNE_EPOCHS --lr $FINETUNE_LR --weight_decay $WEIGHT_DECAY $SHARED_ARGS"

echo "========================================"
echo "  EXPERIMENT: MergeDNA+ Comparison"
echo "========================================"

# --- 1. MERGE ---
echo ""
echo "========== [1/3] MERGE: Pre-train =========="
python train.py --model merge --dataset $PRETRAIN_DATASET \
  --epochs $PRETRAIN_EPOCHS --lr $PRETRAIN_LR --weight_decay $WEIGHT_DECAY \
  --merge_r $MERGE_R $SHARED_ARGS

echo ""
echo "========== [1/3] MERGE: Fine-tune classifier =========="
python train_classifier.py --checkpoint checkpoint_merge.pt \
  --encoder merge --merge_r $MERGE_R $CLASSIFIER_ARGS

# --- 2. CONV ---
echo ""
echo "========== [2/3] CONV: Pre-train =========="
python train.py --model conv --dataset $PRETRAIN_DATASET \
  --epochs $PRETRAIN_EPOCHS --lr $PRETRAIN_LR --weight_decay $WEIGHT_DECAY \
  $SHARED_ARGS

echo ""
echo "========== [2/3] CONV: Fine-tune classifier =========="
python train_classifier.py --checkpoint checkpoint_conv.pt \
  --encoder conv $CLASSIFIER_ARGS

# --- 3. DIFFUSION ---
echo ""
echo "========== [3/3] DIFFUSION: Pre-train =========="
python train_diffusion.py --dataset $PRETRAIN_DATASET \
  --epochs $PRETRAIN_EPOCHS --lr $PRETRAIN_LR --weight_decay $WEIGHT_DECAY \
  --diffusion_steps $DIFFUSION_STEPS $SHARED_ARGS

echo ""
echo "========== [3/3] DIFFUSION: Fine-tune classifier =========="
python train_classifier.py --checkpoint checkpoint_diffusion.pt \
  --encoder diffusion $CLASSIFIER_ARGS

echo ""
echo "========================================"
echo "  DONE. Results in outputs/images/"
echo "========================================"
