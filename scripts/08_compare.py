"""Hypothesis tests: paired cluster bootstrap of TAR@FAR=1% differences on the same test trials.

  H1  arc_coat      vs arc_random  (only the batch sampler differs)      P1, P2, P3, clean probes
  aug arc_coat_door vs arc_coat    (door/night augmentation added)        P1, P3 on clean, door, gray
  ref arc_coat      vs petface     (our best ViT vs the strongest baseline)
A difference is significant when its 95% CI excludes 0.
"""
import _boot  # noqa: F401

import numpy as np
import pandas as pd

from whocat import metrics as M, paths, protocols
from whocat.embed import cache_file

R = paths.CKPT
TESTS = [
    ("H1 coat vs random", f"ckpt:{R}/arc_coat/best.pt", f"ckpt:{R}/arc_random/best.pt", ["P1", "P2", "P3"], ["clean"]),
    ("door aug", f"ckpt:{R}/arc_coat_door/best.pt", f"ckpt:{R}/arc_coat/best.pt", ["P1", "P3"], ["clean", "door", "gray"]),
    ("ours vs PetFace", f"ckpt:{R}/arc_coat/best.pt", "petface", ["P1", "P2", "P3"], ["clean"]),
    ("ours+door vs PetFace", f"ckpt:{R}/arc_coat_door/best.pt", "petface", ["P1", "P2", "P3"], ["clean", "door", "gray"]),
    # round 3 (skipped until the runs exist)
    ("aug on random batches", f"ckpt:{R}/arc_random_door/best.pt", f"ckpt:{R}/arc_random/best.pt", ["P1", "P2", "P3"], ["clean", "door", "gray"]),
    ("H1 with aug: coat vs random", f"ckpt:{R}/arc_coat_door/best.pt", f"ckpt:{R}/arc_random_door/best.pt", ["P1", "P2", "P3"], ["clean", "door", "gray"]),
    ("random+aug vs PetFace", f"ckpt:{R}/arc_random_door/best.pt", "petface", ["P1", "P2", "P3"], ["clean", "door", "gray"]),
    ("background swap (E12b)", f"ckpt:{R}/arc_coat_door_bg/best.pt", f"ckpt:{R}/arc_coat_door/best.pt", ["P1", "P2", "P3"], ["clean", "door", "gray"]),
    ("small: coat vs random (aug)", f"ckpt:{R}/student_coat_door/best.pt", f"ckpt:{R}/student_random_door/best.pt", ["P1", "P3"], ["clean", "door", "gray"]),
]


def main():
    rows = []
    for name, a, b, protos, variants in TESTS:
        if not all(cache_file(m, "clean").exists() for m in (a, b)):
            print("skip", name, "(missing embeddings)")
            continue
        for p in protos:
            t = protocols.load(p, "test")
            for v in variants:
                if not all(cache_file(m, v).exists() for m in (a, b)):
                    continue
                sa = protocols.score(t, np.load(cache_file(a, "clean")), np.load(cache_file(a, v)))
                sb = protocols.score(t, np.load(cache_file(b, "clean")), np.load(cache_file(b, v)))
                d, lo, hi = M.paired_diff(M.tar_fn(sa, t["label"], 0.01), M.tar_fn(sb, t["label"], 0.01), t["owner"])
                rows.append(dict(test=name, protocol=p, probe=v, diff_pts=100 * d, lo=100 * lo, hi=100 * hi,
                                 significant=bool(lo > 0 or hi < 0)))
                print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(paths.RESULTS / "e3_hypothesis_tests.csv", index=False)
    if len(df):
        with open(paths.RESULTS / "summary_trained.md", "a") as f:
            f.write("\n## Paired differences in TAR@FAR=1% (points, 95% CI)\n\n"
                    + df.round(2).to_markdown(index=False) + "\n")


if __name__ == "__main__":
    main()
