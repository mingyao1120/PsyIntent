#!/bin/bash
# Evaluate PsyIntent on MDID (single GPU, feature-based, fold 0).
# Usage: bash PsyIntent/mdid/test.sh [checkpoint]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RELEASE_DIR="$SCRIPT_DIR/.."
SRC_DIR="$RELEASE_DIR/src"
cd "$SRC_DIR"

dataname='MDID'
num_class=7
batch_size=64
arch=R101-448
backbone=resnet101
img_size=224
seed=666

resume="${1:-$RELEASE_DIR/output/mdid/seed666_gac10.0/model_best.pth.tar}"
output_dir="$RELEASE_DIR/output/mdid/test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$output_dir"

echo "============================================"
echo "Testing MDID (Single GPU, feature-based, fold 0)"
echo "  resume=$resume"
echo "  output=$output_dir"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python test_mdid.py \
    --dataname "$dataname" \
    --resume "$resume" \
    --output "$output_dir" \
    --batch-size $batch_size \
    --num_class $num_class \
    --arch $arch \
    --backbone $backbone \
    --img_size_hight $img_size \
    --img_size_weight $img_size \
    --hidden_dim 512 \
    --seed $seed
