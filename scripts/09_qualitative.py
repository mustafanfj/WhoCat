"""Qualitative figures for the report and the results page (no GPU needed).

Uses the cached PetFace-baseline embeddings and the test protocols.
Outputs (outputs/figures/):
  q_lookalikes_accepted.png  look-alike strangers the PetFace model wrongly lets in (P3, val-tuned threshold)
  q_owner_matches.png        owner photo pairs: accepted vs. wrongly rejected
  q_degradations.png         clean / door-camera / night-gray versions of the same faces
  q_coats.png                one example face per coat, sorted by how often same-coat strangers get in
  web/                       small JPEG thumbnails + web.json with scores, for the interactive page
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

from whocat import metrics as M, paths, protocols  # noqa: E402
from whocat.data import load_eval_images, load_ids, stable_seed  # noqa: E402
from whocat.degrade import degrade  # noqa: E402
from whocat.embed import cache_file  # noqa: E402

rng = np.random.default_rng(7)
WEB = paths.FIGS / "web"


def img(rel, variant="clean"):
    im = Image.open(paths.IMAGES / rel).convert("RGB").resize((224, 224))
    return im if variant == "clean" else degrade(im, variant, seed=stable_seed(rel + variant))


def thumb(im, name, size=160):
    WEB.mkdir(parents=True, exist_ok=True)
    im.resize((size, size), Image.BICUBIC).save(WEB / name, quality=85)
    return name


def pair_grid(pairs, ev, coat_of, title, fname, web_key, web, thr):
    n = len(pairs)
    fig, axes = plt.subplots(2, n, figsize=(1.9 * n, 4.4))
    for j, (a, b, s, lab) in enumerate(pairs):
        for r, idx in enumerate((a, b)):
            ax = axes[r, j] if n > 1 else axes[r]
            ax.imshow(img(ev.path[idx]))
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_xlabel(coat_of(ev.id[idx]), fontsize=7) if r == 1 else None
        top = axes[0, j] if n > 1 else axes[0]
        ok = (s > thr) == bool(lab)
        top.set_title(f"score {s:.2f}\n{'ACCEPT' if s > thr else 'REJECT'}", fontsize=8,
                      color="tab:green" if ok else "tab:red")
        web.setdefault(web_key, []).append(dict(
            a=thumb(img(ev.path[a]), f"{web_key}_{j}_a.jpg"), b=thumb(img(ev.path[b]), f"{web_key}_{j}_b.jpg"),
            score=round(float(s), 3), same_cat=bool(lab), accepted=bool(s > thr),
            coat_a=coat_of(ev.id[a]), coat_b=coat_of(ev.id[b])))
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(paths.FIGS / fname, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="petface")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    ev, ids = load_eval_images(), load_ids()
    coat = ids.set_index("id").coat.to_dict()
    coat_of = lambda i: str(coat.get(i, "unknown")).replace("_", " ")  # noqa: E731
    e = np.load(cache_file(args.model, "clean"))
    tv = protocols.load("P1", "val")
    _, thr = M.tar_at_far(protocols.score(tv, e, e), tv["label"], 0.01)
    web = {"model": args.model, "threshold": round(float(thr), 3)}

    t = protocols.load("P3", "test")
    s = protocols.score(t, e, e)
    a, b, y = t["enroll"][:, 0], t["probe"], t["label"]
    fa = np.flatnonzero((y == 0) & (s > thr))
    pick = rng.choice(fa, min(args.n, len(fa)), replace=False)
    pair_grid([(a[i], b[i], s[i], y[i]) for i in pick], ev, coat_of,
              f"Look-alike strangers wrongly let in ({args.model}, threshold {thr:.2f} = 1% FAR on random cats)",
              "q_lookalikes_accepted.png", "lookalikes", web, thr)
    rej = np.flatnonzero((y == 0) & (s <= thr))
    pick = rng.choice(rej, min(args.n // 2, len(rej)), replace=False)
    gen_ok = np.flatnonzero((y == 1) & (s > thr))
    gen_bad = np.flatnonzero((y == 1) & (s <= thr))
    pairs = [(a[i], b[i], s[i], 1) for i in rng.choice(gen_ok, args.n // 2, replace=False)] + \
            [(a[i], b[i], s[i], 1) for i in rng.choice(gen_bad, args.n // 2, replace=False)]
    pair_grid(pairs, ev, coat_of, "Same cat: correctly accepted (left) and wrongly rejected (right)",
              "q_owner_matches.png", "owners", web, thr)
    pair_grid([(a[i], b[i], s[i], 0) for i in pick], ev, coat_of, "Look-alike strangers correctly rejected",
              "q_lookalikes_rejected.png", "rejected", web, thr)

    # degradations
    test = ev[ev.split == "test"]
    rows = test.sample(4, random_state=3).path.tolist()
    fig, axes = plt.subplots(len(rows), 3, figsize=(5.4, 1.9 * len(rows)))
    for r, rel in enumerate(rows):
        for c, v in enumerate(("clean", "door", "gray")):
            im = img(rel, v)
            axes[r, c].imshow(im), axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title({"clean": "clean", "door": "door camera (P4)", "gray": "night / IR proxy"}[v], fontsize=9)
            web.setdefault("degradations", []).append(dict(row=r, variant=v, src=thumb(im, f"deg_{r}_{v}.jpg")))
    plt.tight_layout()
    plt.savefig(paths.FIGS / "q_degradations.png", dpi=200)
    plt.close()

    # coats, hardest first (by the model's same-coat false-accept rate)
    far = {}
    f = paths.RESULTS / "e2_coat.csv"
    if f.exists():
        c = pd.read_csv(f)
        c = c[(c.model == args.model) & (c.probe == "clean")]
        far = dict(zip(c.coat, c.far))
    coats = [k for k in sorted(far, key=far.get, reverse=True)] or sorted(set(ids.coat) - {"unknown"})
    test_ids = ids[(ids.split == "test")]
    fig, axes = plt.subplots(1, len(coats), figsize=(1.7 * len(coats), 2.3))
    for j, k in enumerate(coats):
        cid = test_ids[test_ids.coat == k].id.sample(1, random_state=j).iloc[0]
        rel = ev[ev.id == cid].path.iloc[0]
        im = img(rel)
        ax = axes[j] if len(coats) > 1 else axes
        ax.imshow(im), ax.axis("off")
        lab = k.replace("_", " ") + (f"\n{100 * far[k]:.1f}% let in" if k in far else "")
        ax.set_title(lab, fontsize=8)
        web.setdefault("coats", []).append(dict(coat=k, far=far.get(k), src=thumb(im, f"coat_{k}.jpg")))
    plt.tight_layout()
    plt.savefig(paths.FIGS / "q_coats.png", dpi=200)
    plt.close()

    (WEB / "web.json").write_text(json.dumps(web, indent=1))
    print("saved figures to", paths.FIGS)


if __name__ == "__main__":
    main()
