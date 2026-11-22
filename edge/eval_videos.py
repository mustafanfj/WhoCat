"""Step 2 for the campus videos (E6): does the door recognise real campus cats?

Each cat is enrolled from its FIRST clip (alphabetical order; the 10 sharpest face crops are averaged).
Every other clip is a visit. A visit is scored against every enrolled cat; the visit score is the mean of
its 5 best frames (frame voting). We report, per model:
  owner_accept      share of visits where the cat is let in through its own door
  stranger_admit    share of (visit, other cat's door) pairs where a different cat is let in
  same_coat_admit   the same, only for pairs of cats you marked as similar (similar.txt, optional)
  rank1             share of visits whose best-matching enrolled cat is the right one
for three door settings: random-tuned threshold, look-alike-tuned threshold, and look-alike threshold
plus neighbour memory (all other enrolled campus cats are remembered neighbours).
Also the same numbers with a single frame per visit instead of voting.

  python eval_videos.py                      # uses faces/ from extract_faces.py
  python eval_videos.py --models small big   # big = ViT, slower on CPU
Optional similar.txt: one pair of similar-looking cats per line, e.g.  ginger1,ginger2
Writes results/e6_campus.csv, results/e6_scores.csv, results/e6_summary.md
"""
import argparse
import csv
import os

import cv2
import numpy as np

from whocat_edge import HERE, Embedder, load_config, sharpness, vote, decide


def load_faces(root):
    data = {}
    for cat in sorted(os.listdir(root)):
        cdir = os.path.join(root, cat)
        if not os.path.isdir(cdir):
            continue
        for clip in sorted(os.listdir(cdir)):
            p = os.path.join(cdir, clip)
            if os.path.isdir(p):
                fs = sorted(f for f in os.listdir(p) if f.endswith(".jpg"))
                if fs:
                    data.setdefault(cat, []).append(
                        (clip, [cv2.cvtColor(cv2.imread(os.path.join(p, f)), cv2.COLOR_BGR2RGB) for f in fs]))
    return data


def embed_all(emb, crops, bs=16):
    return np.concatenate([emb(crops[i:i + bs]) for i in range(0, len(crops), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", default=os.path.join(HERE, "faces"))
    ap.add_argument("--models", nargs="+", default=["small"])
    ap.add_argument("--variant", default="fp32", choices=["fp32", "int8"])
    ap.add_argument("--enroll-k", type=int, default=10)
    ap.add_argument("--vote-k", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config()
    data = load_faces(args.faces)
    cats = [c for c in data if len(data[c]) >= 2]
    print("%d cats with >= 2 clips: %s" % (len(cats), ", ".join(cats)))
    if len(cats) < 2:
        raise SystemExit("need at least 2 cats with 2 clips each")
    similar = set()
    sp = os.path.join(HERE, "similar.txt")
    if os.path.exists(sp):
        for line in open(sp):
            a = [s.strip() for s in line.split(",") if s.strip()]
            if len(a) == 2:
                similar |= {(a[0], a[1]), (a[1], a[0])}
    out = os.path.join(HERE, "results")
    if not os.path.isdir(out):
        os.makedirs(out)
    rows, score_rows = [], []
    for key in args.models:
        emb = Embedder(cfg, key, args.variant)
        T, visits = {}, []
        for c in cats:
            clip, crops = data[c][0]
            order = np.argsort([-sharpness(x) for x in crops])[:args.enroll_k]
            T[c] = embed_all(emb, [crops[i] for i in order]).mean(0)
            T[c] /= np.linalg.norm(T[c])
            for clip, crops in data[c][1:]:
                visits.append((c, clip, embed_all(emb, crops)))
        names = list(T)
        M = np.stack([T[c] for c in names])
        for mode in ("vote", "single"):
            for setting in ("random", "lookalike", "lookalike+memory"):
                thr = emb.thr_random if setting == "random" else emb.thr_lookalike
                mem = setting.endswith("memory")
                own_ok, strange, same, rank = [], [], [], []
                for c, clip, E in visits:
                    S = E @ M.T                                   # frames x cats
                    if mode == "vote":
                        s = np.array([vote(S[:, j], args.vote_k) for j in range(len(names))])
                    else:
                        s = S[len(S) // 2]                        # the middle frame only
                    rank.append(names[int(np.argmax(s))] == c)
                    for j, door in enumerate(names):
                        others = [s[k] for k in range(len(names)) if k != j]
                        ok = decide(s[j], others, thr, emb.margin if mem else None)
                        if door == c:
                            own_ok.append(ok)
                        else:
                            strange.append(ok)
                            if (c, door) in similar:
                                same.append(ok)
                        if mode == "vote" and setting == "random":
                            score_rows.append(dict(model=key, visit_cat=c, clip=clip, door=door, score=round(float(s[j]), 4)))
                rows.append(dict(model=key, variant=args.variant, frames=mode, door=setting, visits=len(visits),
                                 owner_accept=100 * np.mean(own_ok), stranger_admit=100 * np.mean(strange),
                                 same_coat_admit=(100 * np.mean(same) if same else float("nan")),
                                 rank1=100 * np.mean(rank)))
                print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in rows[-1].items()})
    for name, rs in (("e6_campus.csv", rows), ("e6_scores.csv", score_rows)):
        with open(os.path.join(out, name), "w") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0]))
            w.writeheader()
            w.writerows(rs)
    with open(os.path.join(out, "e6_summary.md"), "w") as f:
        f.write("# E6 campus cats (%d cats)\n\n| model | frames | door | visits | owner accept | stranger admit | "
                "similar-cat admit | rank-1 |\n|---|---|---|---|---|---|---|---|\n" % len(cats))
        for r in rows:
            f.write("| %s | %s | %s | %d | %.1f | %.2f | %s | %.1f |\n" % (
                r["model"], r["frames"], r["door"], r["visits"], r["owner_accept"], r["stranger_admit"],
                "-" if r["same_coat_admit"] != r["same_coat_admit"] else "%.1f" % r["same_coat_admit"], r["rank1"]))
    print(open(os.path.join(out, "e6_summary.md")).read())


if __name__ == "__main__":
    main()
