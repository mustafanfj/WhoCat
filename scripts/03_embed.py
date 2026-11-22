"""Cache embeddings of all val+test images for each model and probe variant.

  python scripts/03_embed.py --models petface megadesc dinov2 --variants clean door gray
  python scripts/03_embed.py --models ckpt:../checkpoints/arc_coat/best.pt
Variants: clean, door (P4 corruptions), gray (night-IR proxy), small (resolution only).
"""
import _boot  # noqa: F401
import argparse
import gc

import torch

from whocat.embed import cache_file, get_eval_embeddings
from whocat.models import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["petface", "megadesc", "dinov2"])
    ap.add_argument("--variants", nargs="+", default=["clean", "door", "gray"])
    ap.add_argument("--batch", type=int, default=0, help="0 = 64 for megadesc/miewid, 256 otherwise")
    args = ap.parse_args()
    for name in args.models:
        todo = [v for v in args.variants if not cache_file(name, v).exists()]
        if not todo:
            print(name, "already cached")
            continue
        try:
            model = build(name)
        except Exception as e:  # optional baselines (e.g. miewid) may fail to load
            print(f"!! could not load {name}: {e}")
            continue
        for v in todo:
            bs = args.batch or (64 if name in ("megadesc", "miewid") else 256)
            e = get_eval_embeddings(name, v, model=model, batch=bs)
            print(name, v, e.shape)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
