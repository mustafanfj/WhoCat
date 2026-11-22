"""E9: score fusion of two complementary models (training-free, CPU, cached embeddings).

s = w * s_A + (1 - w) * s_B, with w chosen on val look-alike pairs (P3, TAR@FAR=1%) and then frozen.
Reports test TAR@FAR=1% on P1/P2/P3 for clean/door/gray probes, and a paired bootstrap of fusion minus A.

  python scripts/11_fusion.py --a ckpt:../checkpoints/arc_coat_door/best.pt --b petface ckpt:../checkpoints/student_coat_door/best.pt
Outputs: outputs/results/e9_fusion.csv, summary_fusion.md
"""
import _boot  # noqa: F401
import argparse

import numpy as np
import pandas as pd

from whocat import metrics as M, paths, protocols
from whocat.embed import cache_file
from whocat.models import cache_name


def scores(model, t, v):
    e, ep = np.load(cache_file(model, "clean")), np.load(cache_file(model, v))
    return protocols.score(t, e, ep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="main model")
    ap.add_argument("--b", nargs="+", required=True, help="partner models, fused with A one at a time")
    args = ap.parse_args()
    rows, md = [], ["# E9 score fusion\n", "w = weight of the main model, tuned on val P3 (TAR@FAR=1%).\n"]
    tv = protocols.load("P3", "val")
    sa_val = scores(args.a, tv, "clean")
    for b in args.b:
        if not cache_file(b, "clean").exists():
            print("no cache for", b); continue
        sb_val = scores(b, tv, "clean")
        grid = np.round(np.arange(0, 1.01, 0.1), 2)
        val_tar = [M.tar_at_far(w * sa_val + (1 - w) * sb_val, tv["label"], 0.01)[0] for w in grid]
        w = float(grid[int(np.argmax(val_tar))])
        print(b, "val P3 TAR by w:", dict(zip(grid.tolist(), np.round(100 * np.array(val_tar), 1).tolist())), "-> w =", w)
        for v in ("clean", "door", "gray"):
            if not (cache_file(args.a, v).exists() and cache_file(b, v).exists()):
                continue
            for p in ("P1", "P2", "P3"):
                t = protocols.load(p, "test")
                sa, sb = scores(args.a, t, v), scores(b, t, v)
                sf = w * sa + (1 - w) * sb
                y = t["label"]
                d, lo, hi = M.paired_diff(M.tar_fn(sf, y, 0.01), M.tar_fn(sa, y, 0.01), t["owner"])
                rows.append(dict(a=cache_name(args.a), b=cache_name(b), w=w, probe=v, protocol=p,
                                 tar_a=100 * M.tar_at_far(sa, y, 0.01)[0], tar_b=100 * M.tar_at_far(sb, y, 0.01)[0],
                                 tar_fused=100 * M.tar_at_far(sf, y, 0.01)[0],
                                 gain=100 * d, lo=100 * lo, hi=100 * hi, significant=bool(lo > 0 or hi < 0)))
                print({k: (round(x, 2) if isinstance(x, float) else x) for k, x in rows[-1].items()})
    df = pd.DataFrame(rows)
    df.to_csv(paths.RESULTS / "e9_fusion.csv", index=False)
    md += [df.round(2).to_markdown(index=False)]
    (paths.RESULTS / "summary_fusion.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
