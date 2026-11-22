"""Use PetFace's own colour annotation as the coat label (98% coverage), keeping CLIP's as coat_clip.

coat = Color1 (+ Color2 when it is known), e.g. 'tabby_brown', 'black', 'calico'.
Fine labels with fewer than --min-cats training cats fall back to Color1 alone.
Writes outputs/results/coat_petface_vs_clip.csv (agreement table).
"""
import _boot  # noqa: F401
import argparse

import numpy as np
import pandas as pd

from whocat import paths
from whocat.data import load_ids


def clean(s):
    return s.fillna("unknown").astype(str).str.strip().str.lower().str.replace(" ", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cats", type=int, default=300)
    args = ap.parse_args()
    ids = load_ids()
    if "coat_clip" not in ids:
        ids["coat_clip"] = ids["coat"]
    c1, c2 = clean(ids.color1), clean(ids.color2)
    fine = np.where(c2 != "unknown", c1 + "_" + c2, c1)
    counts = pd.Series(fine[ids.split == "train"]).value_counts()
    rare = ~pd.Series(fine).isin(counts[counts >= args.min_cats].index).to_numpy()
    coat = np.where(rare, c1, fine)
    coat = np.where(c1 == "unknown", "unknown", coat)
    ids["coat"] = coat
    ids["coat_conf"] = np.where(coat == "unknown", 0.0, 1.0)
    ids.to_csv(paths.INDEX / "ids.csv.gz", index=False)
    pd.crosstab(ids.coat, ids.coat_clip).to_csv(paths.RESULTS / "coat_petface_vs_clip.csv")
    tab = ids.groupby("split").coat.value_counts().unstack(0).fillna(0).astype(int)
    print(tab.sort_values("train", ascending=False).to_string())


if __name__ == "__main__":
    main()
