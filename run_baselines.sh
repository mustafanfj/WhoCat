#!/usr/bin/env bash
# E1 + E2 + E5 for the frozen baselines. Run inside tmux from code/:  bash run_baselines.sh
set -e
cd "$(dirname "$0")"
source ../env.sh
mkdir -p ../logs
{
python scripts/01_index.py
python scripts/02_coat_labels.py
python scripts/04_protocols.py
CUDA_VISIBLE_DEVICES=0 python scripts/03_embed.py --models petface dinov2 &
CUDA_VISIBLE_DEVICES=1 python scripts/03_embed.py --models megadesc &
wait
python scripts/05_evaluate.py --models petface megadesc dinov2
python scripts/06_stranger_enrollment.py --models petface megadesc dinov2
} 2>&1 | tee ../logs/baselines_$(date +%Y%m%d_%H%M).log
