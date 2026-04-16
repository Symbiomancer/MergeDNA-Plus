#!/bin/bash
# Full experiment: pre-train on multispecies, then fine-tune classifier on downstream task.
# Runs all three model variants: merge, conv, diffusion.
# Reads all params from config_experiment.json.

source ~/diffusion/.venv/bin/activate
set -e

CONFIG=${CONFIG:-config_experiment.json}

# Read params from config
read_cfg() { python -c "import json; print(json.load(open('$CONFIG'))['$1'])"; }

PRETRAIN_DATASET=$(read_cfg pretrain_dataset)
FINETUNE_DATASET=$(read_cfg finetune_dataset)
NUM_CLASSES=$(read_cfg num_classes)
DIM=$(read_cfg dim)
LOCAL_DEPTH=$(read_cfg local_depth)
LOCAL_NUM_HEADS=$(read_cfg local_num_heads)
LOCAL_DIM_FF=$(read_cfg local_dim_ff)
LOCAL_WINDOW_SIZE=$(read_cfg local_window_size)
LATENT_DEPTH=$(read_cfg latent_depth)
LATENT_NUM_HEADS=$(read_cfg latent_num_heads)
LATENT_DIM_FF=$(read_cfg latent_dim_ff)
MERGE_R=$(read_cfg merge_r)
MAX_LEN=$(read_cfg max_len)
BATCH_SIZE=$(read_cfg batch_size)
PRETRAIN_EPOCHS=$(read_cfg pretrain_epochs)
PRETRAIN_LR=$(read_cfg pretrain_lr)
FINETUNE_EPOCHS=$(read_cfg finetune_epochs)
FINETUNE_LR=$(read_cfg finetune_lr)
WEIGHT_DECAY=$(read_cfg weight_decay)
DIFFUSION_STEPS=$(read_cfg diffusion_steps)

SHARED_ARGS="--dim $DIM --local_depth $LOCAL_DEPTH --local_num_heads $LOCAL_NUM_HEADS \
  --local_dim_ff $LOCAL_DIM_FF --local_window_size $LOCAL_WINDOW_SIZE \
  --latent_depth $LATENT_DEPTH --latent_num_heads $LATENT_NUM_HEADS \
  --latent_dim_ff $LATENT_DIM_FF --max_len $MAX_LEN --batch_size $BATCH_SIZE"

CLASSIFIER_ARGS="--dataset $FINETUNE_DATASET --num_classes $NUM_CLASSES \
  --epochs $FINETUNE_EPOCHS --lr $FINETUNE_LR --weight_decay $WEIGHT_DECAY $SHARED_ARGS"

echo "========================================"
echo "  EXPERIMENT: MergeDNA+ Comparison"
echo "  Config: $CONFIG"
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
