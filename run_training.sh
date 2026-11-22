#!/usr/bin/env bash
# E3: two identical runs that differ only in the batch sampler, one per GPU. Run inside tmux from code/.
#   bash run_training.sh            # defaults: DINOv2-B, 10 epochs
#   EPOCHS=3 bash run_training.sh   # quicker
set -e
cd "$(dirname "$0")"
source ../env.sh
EP=${EPOCHS:-10}
BB=${BACKBONE:-vit_base_patch14_dinov2.lvd142m}
mkdir -p ../logs
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --name arc_random --sampler random --backbone $BB --epochs $EP \
  > ../logs/train_arc_random.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --name arc_coat --sampler coat --backbone $BB --epochs $EP \
  > ../logs/train_arc_coat.log 2>&1 &
wait
R=../checkpoints
python scripts/03_embed.py --models ckpt:$R/arc_random/best.pt ckpt:$R/arc_coat/best.pt
python scripts/05_evaluate.py --tag trained --models petface megadesc dinov2 ckpt:$R/arc_random/best.pt ckpt:$R/arc_coat/best.pt
python scripts/06_stranger_enrollment.py --tag trained --models petface megadesc dinov2 ckpt:$R/arc_random/best.pt ckpt:$R/arc_coat/best.pt
python scripts/07_gradcam.py --models ckpt:$R/arc_random/best.pt ckpt:$R/arc_coat/best.pt petface
