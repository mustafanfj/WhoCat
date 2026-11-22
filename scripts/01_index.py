"""Index PetFace cats: official splits, all images on disk for val/test cats, colour labels.

Outputs (outputs/index/):
  images.csv.gz       path, id, split           (train from train.csv; val/test scanned on disk)
  eval_images.csv.gz  val+test images in the fixed order used by every embedding cache
  ids.csv.gz          one row per identity with n_imgs and PetFace colour annotation
  index_report.txt    counts, images-per-cat histogram, colour-label coverage
"""
import _boot  # noqa: F401
import os
from collections import Counter

import pandas as pd

from whocat import paths


def fix_image_root():
    """Make sure images/cat/<id>/ exists, whatever folder layout the tarball extracted to."""
    if (paths.IMAGES / "cat").is_dir():
        return
    for dirpath, dirnames, _ in os.walk(paths.IMAGES):
        depth = len(os.path.relpath(dirpath, paths.IMAGES).split(os.sep))
        if "000000" in dirnames:
            os.symlink(dirpath, paths.IMAGES / "cat")
            print("linked", paths.IMAGES / "cat", "->", dirpath)
            return
        if depth > 3:
            dirnames[:] = []
    raise SystemExit(f"Could not find cat/000000 under {paths.IMAGES}; is the tarball extracted?")


def read_list(name):
    with open(paths.SPLIT / name) as f:
        return [l.strip() for l in f if l.strip()]


def scan_ids(ids):
    rows = []
    for i in ids:
        d = paths.IMAGES / "cat" / i
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                rows.append((f"cat/{i}/{fn}", i))
    return rows


def main():
    fix_image_root()
    tr = pd.read_csv(paths.SPLIT / "train.csv")
    tr["id"] = tr["filename"].str.split("/").str[1]
    train = pd.DataFrame({"path": tr["filename"], "id": tr["id"], "split": "train"})

    parts = [train]
    listed = {}
    for split, fn in (("val", "val.txt"), ("test", "test.txt")):
        files = read_list(fn)
        listed[split] = set(files)
        ids = sorted({f.split("/")[1] for f in files})
        df = pd.DataFrame(scan_ids(ids), columns=["path", "id"])
        df["split"] = split
        parts.append(df)
    images = pd.concat(parts, ignore_index=True)
    images.to_csv(paths.INDEX / "images.csv.gz", index=False)

    ev = images[images.split.isin(["val", "test"])].reset_index(drop=True)
    ev.to_csv(paths.INDEX / "eval_images.csv.gz", index=False)

    ids = images.groupby(["id", "split"]).size().rename("n_imgs").reset_index()
    ann = pd.read_csv(paths.ANNOT, dtype={"Name": str})
    ann["Name"] = ann["Name"].str.zfill(6)
    ann = ann.rename(columns={"Name": "id", "Breed": "breed", "Color1": "color1",
                              "Color2": "color2", "Gender": "sex"})
    ids = ids.merge(ann, on="id", how="left")
    ids.to_csv(paths.INDEX / "ids.csv.gz", index=False)

    lines = []
    for split in ("train", "val", "test"):
        s = ids[ids.split == split]
        known = (s.color1.notna() & (s.color1 != "Unknown")).mean()
        hist = Counter(s.n_imgs.clip(upper=8))
        lines.append(f"{split}: {len(s)} cats, {s.n_imgs.sum()} images, "
                     f"imgs/cat {dict(sorted(hist.items()))}, PetFace colour known {known:.1%}")
        if split != "train":
            extra = len(images[images.split == split]) - len(listed[split])
            lines.append(f"   {split}: {extra} images on disk beyond the official list")
    top = ids[ids.color1.notna() & (ids.color1 != "Unknown")].color1.value_counts().head(15)
    lines.append("PetFace colour1 values: " + ", ".join(f"{k}={v}" for k, v in top.items()))
    rep = "\n".join(lines)
    print(rep)
    (paths.INDEX / "index_report.txt").write_text(rep + "\n")


if __name__ == "__main__":
    main()
