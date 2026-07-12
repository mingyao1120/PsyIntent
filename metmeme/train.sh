#!/bin/bash
# Train PsyIntent on MET-Meme English (5 intent categories).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RELEASE_DIR="$SCRIPT_DIR/.."
SRC_DIR="$RELEASE_DIR/src"
cd "$SRC_DIR"

dataname='METMEME'
num_class=5
epochs=30
batch_size=48
seed=666
gac=5.0
arch=R101-448
backbone=resnet101
img_size=224
lr=1e-4
workers=4

output_dir="$RELEASE_DIR/output/metmeme/seed${seed}_gac${gac}"
mkdir -p "$output_dir"

echo "============================================"
echo "Training MET-Meme"
echo "  num_class=$num_class, epochs=$epochs, batch_size=$batch_size"
echo "  gac=$gac, seed=$seed, arch=$arch"
echo "  output=$output_dir"
echo "============================================"

CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 torchrun \
    --nproc_per_node=2 --master_port=2319 \
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
    --duppos_mode repeat

echo "Training complete. Best model saved to $output_dir/model_best.pth.tar"
