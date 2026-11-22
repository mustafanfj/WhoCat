"""Zero-shot coat labels with CLIP for every cat, plus perceptual hashes for duplicate removal.

PetFace's own colour annotation is mostly 'Unknown', so coat labels come from CLIP
(a model that is never evaluated), and agreement with PetFace labels is reported where known.

Outputs:
  outputs/index/ids.csv.gz         + coat, coat_conf columns
  outputs/index/eval_dhash.npy     64-bit dHash per eval image
  outputs/cache/emb_clip_clean.npy CLIP embeddings of eval images (also used to mine P3)
  outputs/results/coat_vs_petface.csv
"""
import _boot  # noqa: F401
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from PIL import Image

from whocat import paths
from whocat.data import l2n, load_eval_images, load_ids, load_images
from whocat.embed import device, embed_paths, get_eval_embeddings
from whocat.models import build

COATS = {
    "black": "solid black cat",
    "white": "solid white cat",
    "orange": "orange ginger tabby cat",
    "cream": "cream coloured cat",
    "gray": "solid grey blue cat",
    "brown_tabby": "brown tabby cat",
    "silver_tabby": "silver grey tabby cat",
    "tuxedo": "black and white tuxedo cat",
    "orange_white": "orange and white cat",
    "gray_white": "grey and white cat",
    "calico": "calico cat",
    "tortie": "tortoiseshell cat",
    "colorpoint": "siamese colourpoint cat",
}
TEMPLATES = ["a close-up photo of the face of a {}.", "a photo of a {}.", "a {}."]


def dhash(rel):
    img = Image.open(paths.IMAGES / rel).convert("L").resize((9, 8), Image.BILINEAR)
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-conf", type=float, default=0.4)
    ap.add_argument("--train-imgs-per-cat", type=int, default=2)
    args = ap.parse_args()

    dev = device()
    clip = build("clip").to(dev).eval()
    names = list(COATS)
    txt = torch.stack([clip.text([t.format(COATS[n]) for t in TEMPLATES], dev).mean(0) for n in names])
    txt = torch.nn.functional.normalize(txt, dim=-1).cpu().numpy()

    # eval cats: mean over all their images
    ev = load_eval_images()
    e_eval = get_eval_embeddings("clip", "clean", model=clip)
    # train cats: first k images each
    imgs = load_images()
    tr = imgs[imgs.split == "train"].groupby("id").head(args.train_imgs_per_cat).reset_index(drop=True)
    e_tr = embed_paths(clip, tr.path.tolist(), desc="clip/train")

    def per_id(df, e):
        g = pd.DataFrame(e).groupby(df.id.values).mean()
        return g.index.values, l2n(g.values)

    ids = load_ids()
    rows = []
    for df, e in ((ev, e_eval), (tr, e_tr)):
        cid, ce = per_id(df, e)
        logits = 100.0 * ce @ txt.T
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        rows.append(pd.DataFrame({"id": cid, "coat": np.array(names)[p.argmax(1)], "coat_conf": p.max(1)}))
    lab = pd.concat(rows)
    ids = ids.drop(columns=[c for c in ("coat", "coat_conf") if c in ids]).merge(lab, on="id", how="left")
    ids.loc[ids.coat_conf < args.min_conf, "coat"] = "unknown"
    ids.to_csv(paths.INDEX / "ids.csv.gz", index=False)

    known = ids[ids.color1.notna() & (ids.color1 != "Unknown")]
    if len(known):
        pd.crosstab(known.color1, known.coat).to_csv(paths.RESULTS / "coat_vs_petface.csv")
    print(ids.groupby("split").coat.value_counts(normalize=True).unstack().round(3).T)

    with Pool(min(32, __import__("os").cpu_count() or 4)) as pool:
        h = pool.map(dhash, ev.path.tolist(), chunksize=256)
    np.save(paths.INDEX / "eval_dhash.npy", np.array(h, dtype=np.uint64))
    print("saved coat labels and dHash for", len(ev), "eval images")


if __name__ == "__main__":
    main()
