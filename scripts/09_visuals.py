"""Visual results for the report and the interactive page (no GPU needed; uses cached embeddings).

Outputs outputs/visuals/:
  lookalike_false_accepts.png  look-alike strangers the PetFace model lets in (P3, threshold tuned on P1)
  owner_false_rejects.png      owner photo pairs the model rejects
  degradations.png             clean vs door-camera vs night-style vs low-resolution probes
  coat_gallery.png             one face per coat with the share of same-coat strangers let in
  web/                         small JPEG thumbnails + data.json for the interactive page
PetFace images may not be redistributed: use a handful in figures only, never the dataset itself.
"""
import _boot  # noqa: F401
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from whocat import paths, protocols  # noqa: E402
from whocat.data import load_eval_images, load_ids, stable_seed  # noqa: E402
from whocat.degrade import degrade  # noqa: E402
from whocat.embed import cache_file  # noqa: E402

V = paths.OUT / "visuals"
W = V / "web"
V.mkdir(exist_ok=True)
W.mkdir(exist_ok=True)


def img(rel, size=224):
    return Image.open(paths.IMAGES / rel).convert("RGB").resize((size, size), Image.BICUBIC)


def thumb(rel, name, variant="clean"):
    im = img(rel)
    if variant != "clean":
        im = degrade(im, variant, seed=stable_seed(rel + variant))
    im.resize((128, 128), Image.BICUBIC).save(W / name, quality=85)
    return name


def pair_grid(pairs, ev, coat, title, out, label):
    n = len(pairs)
    fig, ax = plt.subplots(2, n, figsize=(1.6 * n, 3.6))
    for j, (a, b, s) in enumerate(pairs):
        for r, idx in enumerate((a, b)):
            ax[r, j].imshow(img(ev.path[idx]))
            ax[r, j].set_xticks([]), ax[r, j].set_yticks([])
            ax[r, j].set_xlabel(coat.get(ev.id[idx], "?").replace("_", " "), fontsize=7)
        ax[0, j].set_title(f"{label} {s:.2f}", fontsize=8)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="petface")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    ev, ids = load_eval_images(), load_ids()
    coat = ids.set_index("id").coat.to_dict()
    e = np.load(cache_file(args.model, "clean"))
    leak = pd.read_csv(paths.RESULTS / "e1_leak.csv")
    thr = float(leak[(leak.model == args.model)].threshold.iloc[0])
    rng = np.random.default_rng(7)
    web = {"model": args.model, "threshold": thr, "lookalikes": [], "owners": [], "degradations": [], "coats": []}

    # look-alike strangers let in: sample among accepted P3 impostors (not just the top, which may be duplicates)
    t = protocols.load("P3", "test")
    s = protocols.score(t, e, e)
    imp = np.flatnonzero((t["label"] == 0) & (s > thr))
    pick = rng.choice(imp, min(args.n, len(imp)), replace=False)
    pick = pick[np.argsort(-s[pick])]
    pairs = [(t["enroll"][i, 0], t["probe"][i], s[i]) for i in pick]
    pair_grid(pairs, ev, coat, f"Different cats the {args.model} model accepts as the same cat "
              f"(P3 look-alikes, threshold {thr:.2f} tuned for 1% on random pairs)",
              V / "lookalike_false_accepts.png", "sim")
    for k, (a, b, sc) in enumerate(pairs):
        web["lookalikes"].append(dict(a=thumb(ev.path[a], f"la_{k}_a.jpg"), b=thumb(ev.path[b], f"la_{k}_b.jpg"),
                                      score=round(float(sc), 3), coat_a=coat.get(ev.id[a]), coat_b=coat.get(ev.id[b])))

    # owners rejected
    gen = np.flatnonzero((t["label"] == 1) & (s <= thr))
    pick = rng.choice(gen, min(args.n, len(gen)), replace=False)
    pairs = [(t["enroll"][i, 0], t["probe"][i], s[i]) for i in pick]
    pair_grid(pairs, ev, coat, "Same cat, rejected (owner locked out)", V / "owner_false_rejects.png", "sim")
    for k, (a, b, sc) in enumerate(pairs):
        web["owners"].append(dict(a=thumb(ev.path[a], f"ow_{k}_a.jpg"), b=thumb(ev.path[b], f"ow_{k}_b.jpg"),
                                  score=round(float(sc), 3)))

    # degradations
    test_imgs = ev[ev.split == "test"].path.to_numpy()
    sel = rng.choice(test_imgs, 4, replace=False)
    variants = ["clean", "door", "gray", "small"]
    names = {"clean": "clean", "door": "door camera (P4)", "gray": "night-style", "small": "48-64 px"}
    fig, ax = plt.subplots(len(sel), 4, figsize=(6.4, 1.6 * len(sel)))
    for r, rel in enumerate(sel):
        row = {}
        for c, v in enumerate(variants):
            im = img(rel)
            if v != "clean":
                im = degrade(im, v, seed=stable_seed(rel + v))
            ax[r, c].imshow(im)
            ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(names[v], fontsize=8)
            row[v] = thumb(rel, f"dg_{r}_{v}.jpg", v)
        web["degradations"].append(row)
    plt.tight_layout()
    plt.savefig(V / "degradations.png", dpi=200)
    plt.close()

    # coat gallery with per-coat false-accept rate
    e2 = pd.read_csv(paths.RESULTS / "e2_coat.csv")
    e2 = e2[(e2.model == args.model) & (e2.probe == "clean")].sort_values("far", ascending=False)
    tid = ids[(ids.split == "test")]
    fig, ax = plt.subplots(1, len(e2), figsize=(1.5 * len(e2), 2.1))
    for c, (_, r) in enumerate(e2.iterrows()):
        cat = tid[tid.coat == r.coat].id.iloc[rng.integers((tid.coat == r.coat).sum())]
        rel = ev[ev.id == cat].path.iloc[0]
        ax[c].imshow(img(rel))
        ax[c].axis("off")
        ax[c].set_title(f"{r.coat.replace('_', ' ')}\n{100 * r.far:.1f}% let in", fontsize=7)
        web["coats"].append(dict(coat=r.coat, far=round(100 * float(r.far), 2), img=thumb(rel, f"coat_{c}.jpg")))
    plt.tight_layout()
    plt.savefig(V / "coat_gallery.png", dpi=200)
    plt.close()

    (W / "data.json").write_text(json.dumps(web, indent=1))
    print("wrote", sorted(p.name for p in V.iterdir()))


if __name__ == "__main__":
    main()
