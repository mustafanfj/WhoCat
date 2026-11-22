#!/usr/bin/env bash
# Round 2 of training. Run inside tmux from code/:  bash run_training2.sh
#   GPU 1: arc_random (E3 control: identical to arc_coat except random batches), then the Pi student (E7)
#   GPU 0: arc_coat_door (coat-aware + door/night augmentation); gradient checkpointing so it fits
#          next to the other users' jobs on GPU 0
# Then embeds every trained model, re-runs E1/E2/E5 and E4 Grad-CAM.
cd "$(dirname "$0")"
source ../env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EP=${EPOCHS:-10}
BB=vit_base_patch14_dinov2.lvd142m
R=../checkpoints
mkdir -p ../logs
( CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --name arc_random --sampler random --backbone $BB --epochs $EP \
    > ../logs/train_arc_random.log 2>&1
  CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --name student_coat_door --sampler coat --door-aug 0.5 \
    --backbone mobilenetv3_large_100 --lr-bb 1e-3 --epochs $EP > ../logs/train_student.log 2>&1 ) &
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --name arc_coat_door --sampler coat --door-aug 0.5 --grad-ckpt \
    --backbone $BB --epochs $EP > ../logs/train_arc_coat_door.log 2>&1 &
wait
M=""
for r in arc_random arc_coat arc_coat_door student_coat_door; do
  [ -f $R/$r/best.pt ] && M="$M ckpt:$R/$r/best.pt" || echo "!! $r has no best.pt (see logs/train_$r.log)"
done
python scripts/03_embed.py --models $M
python scripts/05_evaluate.py --tag trained --models petface megadesc dinov2 $M
python scripts/06_stranger_enrollment.py --tag trained --models petface megadesc dinov2 $M
python scripts/07_gradcam.py --models petface $M
python scripts/08_compare.py
echo ALL DONE
