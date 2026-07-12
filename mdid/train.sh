#!/bin/bash
# Train PsyIntent on MDID (7 intent categories; feature-based, no image backbone).
# MDID has no held-out test set; this uses fold 0 (the "Val_0 set").
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RELEASE_DIR="$SCRIPT_DIR/.."
SRC_DIR="$RELEASE_DIR/src"
cd "$SRC_DIR"

dataname='MDID'
num_class=7
epochs=30
batch_size=64
seed=666
gac=10.0
arch=R101-448
backbone=resnet101
img_size=224
lr=1e-4
workers=4

output_dir="$RELEASE_DIR/output/mdid/seed${seed}_gac${gac}"
mkdir -p "$output_dir"

echo "============================================"
echo "Training MDID (fold 0)"
echo "  num_class=$num_class, epochs=$epochs, batch_size=$batch_size"
echo "  gac=$gac, seed=$seed"
echo "  output=$output_dir"
echo "============================================"

CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 torchrun \
    --nproc_per_node=2 --master_port=2320 \
    train.py \
    --dataname "$dataname" \
    --output "$output_dir" \
    --batch-size $batch_size \
    --num_class $num_class \
    --epochs $epochs \
    --seed $seed \
    --arch $arch \
    --backbone $backbone \
    --img_size_hight $img_size \
    --img_size_weight $img_size \
    --lr $lr \
    --gac $gac \
    --workers $workers \
    --hidden_dim 512 \
    --duppos_mode repeat

echo "Training complete. Best model saved to $output_dir/model_best.pth.tar"
