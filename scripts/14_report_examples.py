"""Picture examples for the report, chosen automatically from the results (CPU only, cached embeddings).

Each figure comes as a final version (a few clear rows) and a numbered candidates sheet to choose from.
Accept / reject marks use each model's own threshold tuned on random validation pairs (1% FAR), as in E1.

  fig_lookalike_fixed     owner | owner's other photo | look-alike: PetFace lets the look-alike in, ours keeps it out
  fig_lookalike_failures  look-alikes our best model still scores above the owner (honest failure cases)
  fig_shortcut            the owner's photo with only lighting / night-style / background changed:
                          coat-aware (no aug) breaks on lighting; + aug holds; + background swap holds
  fig_memory              neighbour memory: a look-alike neighbour above the threshold, stopped by the stored photo
  fig_variants            one face as the models see it in every test (clean, door, night, lighting, masked)

  python scripts/14_report_examples.py
Outputs: outputs/report_figs/*.png and *.pdf. PetFace images: a few examples in figures only, no redistribution.
"""
import _boot  # noqa: F401
import argparse

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

OUT = paths.OUT / "report_figs"
OUT.mkdir(exist_ok=True)
R = "ckpt:../checkpoints"
M = dict(petface="petface", coat=f"{R}/arc_coat/best.pt", best=f"{R}/arc_coat_door/best.pt",
         bg=f"{R}/arc_coat_door_bg/best.pt")
NAME = dict(petface="PetFace model", coat="Coat-aware, no aug", best="Ours (coat-aware + aug)",
            bg="Ours + background swap")
GREEN, RED = "#1a7f37", "#c62828"
VNAME = {"clean": "original", "light": "lighting changed", "gray": "night-style", "masked": "background removed",
         "door": "door camera"}


def load_img(rel, variant="clean"):
    im = Image.open(paths.IMAGES / rel).convert("RGB").resize((224, 224), Image.BICUBIC)
    return im if variant == "clean" else degrade(im, variant, seed=stable_seed(rel + variant))


def emb(key, variant="clean"):
    f = cache_file(M[key], variant)
    return np.load(f) if f.exists() else None


def thresholds():
    lk = pd.read_csv(paths.RESULTS / "e1_leak_trained.csv")
    out = {}
    for k, m in M.items():
        name = m.split("/")[-2] + "_best" if m.startswith("ckpt:") else m
        row = lk[(lk.model == name) & (lk.protocol == "P1") & (lk.probe == "clean")]
        if len(row):
            out[k] = float(row.threshold.iloc[0])
    return out


def mark(ax, s, t, txt, y):
    ok = s > t
    ax.text(0.5, y, f"{txt}: {s:.2f} {'in' if ok else 'out'}", transform=ax.transAxes, ha="center", va="top",
            fontsize=7, color=GREEN if ok else RED)


def grid(rows, cols, out, colnames, title=None, cell=1.55, text_h=0.5):
    fig, ax = plt.subplots(rows, cols, figsize=(cell * cols, (cell + text_h) * rows), squeeze=False)
    for a in ax.ravel():
        a.set_xticks([]), a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    for c, n in enumerate(colnames):
        ax[0, c].set_title(n, fontsize=8)
    if title:
        fig.suptitle(title, fontsize=9)
    return fig, ax


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=220)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def pick_diverse(idx, key, coat_of, n):
    """Best-first, at most one per coat on the first pass, then fill."""
    out, seen = [], set()
    for i in idx:
        c = coat_of(i)
        if c not in seen:
            out.append(i); seen.add(c)
        if len(out) == n:
            return out
    for i in idx:
        if i not in out:
            out.append(i)
        if len(out) == n:
            break
    return out


def triplet_fig(rows, ev, coat, sc, thr, name, title, keys, n_label=False):
    if not rows:
        print("no cases for", name); return
    fig, ax = grid(len(rows), 3, None, ["owner (enrolled)", "owner, another day", "look-alike neighbour"], title)
    for r, (g, i) in enumerate(rows):
        idx = [rows_t["enroll"][g, 0], rows_t["probe"][g], rows_t["probe"][i]]
        for c, k in enumerate(idx):
            ax[r, c].imshow(load_img(ev.path[k]))
        ax[r, 0].set_ylabel(("#%d  " % r if n_label else "") + coat.get(ev.id[idx[0]], "?").replace("_", " "), fontsize=7)
        for j, key in enumerate(keys):
            mark(ax[r, 1], sc[key][g], thr[key], NAME[key].split(" (")[0], -0.04 - 0.13 * j)
            mark(ax[r, 2], sc[key][i], thr[key], NAME[key].split(" (")[0], -0.04 - 0.13 * j)
    save(fig, name)


rows_t = None


def main():
    global rows_t
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="rows in the final figures")
    ap.add_argument("--cand", type=int, default=16, help="rows in the candidate sheets")
    args = ap.parse_args()
    ev, ids = load_eval_images(), load_ids()
    coat = ids.set_index("id").coat.to_dict()
    thr = thresholds()
    print("thresholds (P1 val, 1% FAR):", thr)

    t = protocols.load("P3", "test")
    rows_t = t
    E = {k: emb(k) for k in M}
    sc = {k: protocols.score(t, E[k], E[k]) for k in M if E[k] is not None}
    gen = {o: g for g, o in zip(np.flatnonzero(t["label"] == 1), t["owner"][t["label"] == 1])}
    imp = np.flatnonzero((t["label"] == 0) & np.isin(t["owner"], list(gen)))
    g_of = np.array([gen[o] for o in t["owner"][imp]])
    coat_of = lambda i: coat.get(ev.id[t["probe"][i]], "?")  # noqa: E731

    # 1. look-alikes PetFace lets in and ranks above the owner; ours keeps out and lets the owner in
    pf, us = sc["petface"], sc["best"]
    ok = ((pf[imp] > thr["petface"]) & (pf[g_of] > thr["petface"]) & (us[imp] < thr["best"]) &
          (us[g_of] > thr["best"] + 0.1))      # both models accept the owner pair: no label-noise owners
    cand = imp[ok][np.argsort(-((us[g_of] - us[imp]) - (pf[g_of] - pf[imp]))[ok])]
    print("fixed look-alike cases:", len(cand))
    for n, nm, lab in ((args.n, "fig_lookalike_fixed", False), (args.cand, "fig_lookalike_fixed_candidates", True)):
        sel = pick_diverse(cand, None, coat_of, n)
        triplet_fig([(gen[t["owner"][i]], i) for i in sel], ev, coat, sc, thr, nm,
                    "Look-alike neighbours: PetFace's model lets them in; ours keeps them out and still admits the owner",
                    ["petface", "best"], lab)

    # 2a. suspected label errors: the owner's two photos disagree for both models, while the "look-alike"
    #     matches the enrolled photo for both models (the same cat filed under two IDs)
    lab_err = imp[(us[g_of] < thr["best"]) & (pf[g_of] < thr["petface"]) &
                  (us[imp] > thr["best"] + 0.3) & (pf[imp] > thr["petface"] + 0.15)]
    lab_err = lab_err[np.argsort(-us[lab_err])]
    print("suspected label errors:", len(lab_err))
    for n, nm, lab in ((3, "fig_label_errors", False), (args.cand // 2, "fig_label_errors_candidates", True)):
        sel = pick_diverse(lab_err, None, coat_of, n)
        triplet_fig([(gen[t["owner"][i]], i) for i in sel], ev, coat, sc, thr, nm,
                    "Dataset label errors: the 'owner, another day' photo is a different cat, the 'look-alike' is the same cat",
                    ["petface", "best"], lab)

    # 2b. real failures: both models accept the owner pair, PetFace keeps the look-alike out, ours lets it in
    bad = imp[(us[g_of] > thr["best"]) & (pf[g_of] > thr["petface"]) & (us[imp] > thr["best"]) &
              (pf[imp] < thr["petface"]) & (us[imp] < 0.85)]
    bad = bad[np.argsort(-(us[bad] - thr["best"]))]
    print("real failure cases:", len(bad))
    for n, nm, lab in ((args.n, "fig_lookalike_failures", False), (args.cand, "fig_lookalike_failures_candidates", True)):
        sel = pick_diverse(bad, None, coat_of, n)
        triplet_fig([(gen[t["owner"][i]], i) for i in sel], ev, coat, sc, thr, nm,
                    "Look-alikes our best model still lets in", ["petface", "best"], lab)

    # 2c. how many look-alike pairs look like the same cat to two very different models?
    hi_pf = np.median(pf[g_of]); hi_us = np.median(us[g_of])
    dup = (pf[imp] > hi_pf) & (us[imp] > hi_us)
    from whocat import metrics as Mx
    keep = np.ones(len(t["label"]), bool); keep[imp[dup]] = False
    lines = [f"look-alike pairs scored above the median same-cat score by BOTH PetFace and ours: "
             f"{dup.sum()} of {len(imp)} ({100 * dup.mean():.2f}%) -- suspected same cat under two IDs"]
    for k in ("petface", "best"):
        a0 = 100 * Mx.tar_at_far(sc[k], t["label"], 0.01)[0]
        a1 = 100 * Mx.tar_at_far(sc[k][keep], t["label"][keep], 0.01)[0]
        lines.append(f"{NAME[k]}: look-alike TAR@FAR=1% {a0:.1f} -> {a1:.1f} without them")
    (OUT / "label_noise_estimate.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # 3. shortcut: owner pairs with only one thing changed
    keys = [k for k in ("coat", "best", "bg") if k in sc]
    V = ["clean", "light", "gray", "masked"]
    S = {}
    for k in keys:
        for v in V:
            ee = E[k] if v != "masked" else emb(k, "masked")
            ep = emb(k, v)
            if ee is None or ep is None:
                continue
            S[(k, v)] = protocols.score(t, ee, ep)
    g_all = np.array(sorted(gen.values()))
    if all((k, v) in S for k in keys for v in V):
        okc = ((S[("coat", "clean")][g_all] > thr["coat"]) & (S[("coat", "light")][g_all] < thr["coat"]) &
               (S[("coat", "gray")][g_all] < thr["coat"]) &
               np.all([S[(k, v)][g_all] > thr[k] for k in ("best", "bg") for v in V], axis=0))
        cand = g_all[okc]
        cand = cand[np.argsort(-(S[("coat", "clean")][cand] - S[("coat", "light")][cand]))]
        print("shortcut cases:", len(cand))
        for n, nm, lab in ((3, "fig_shortcut", False), (args.cand // 2, "fig_shortcut_candidates", True)):
            sel = pick_diverse(cand, None, lambda g: coat.get(ev.id[t["probe"][g]], "?"), n)
            if not sel:
                print("no cases for", nm); continue
            fig, ax = grid(len(sel), 5, None, ["enrolled photo"] + [VNAME[v] for v in V],
                           "Same owner, one thing changed in the visit photo", text_h=0.62)
            for r, g in enumerate(sel):
                a, b = t["enroll"][g, 0], t["probe"][g]
                ax[r, 0].imshow(load_img(ev.path[a]))
                ax[r, 0].set_ylabel(("#%d  " % r if lab else "") + coat.get(ev.id[a], "?").replace("_", " "), fontsize=7)
                for c, v in enumerate(V):
                    ax[r, c + 1].imshow(load_img(ev.path[b], v))
                    for j, k in enumerate(keys):
                        mark(ax[r, c + 1], S[(k, v)][g], thr[k], ["no aug", "+ aug", "+ bg swap"][j], -0.04 - 0.13 * j)
            save(fig, nm)
    else:
        print("shortcut figure skipped: missing light/masked embeddings")

    # 4. neighbour memory (E5 rule, best model): neighbour above threshold, stopped by its stored photo
    e5 = pd.read_csv(paths.RESULTS / "e5_stranger_enrollment_trained.csv")
    m = float(e5[(e5.model == "arc_coat_door_best") & (e5.probe == "clean")].margin.iloc[0])
    nn = pd.read_csv(paths.PAIRS / "test_neighbours.csv.gz", dtype=str)
    tst = ev[ev.split == "test"]
    by_id = {k: v.index.to_numpy() for k, v in tst.groupby("id")}
    e = E["best"]
    rows = []
    for i, neigh in zip(nn.id, nn.neighbours):
        if len(by_id.get(i, [])) < 2:
            continue
        o0, o1 = by_id[i][:2]
        T = e[o0]
        own = float(T @ e[o1])
        if own <= thr["best"]:
            continue
        for j in neigh.split()[:3]:
            if j == i or len(by_id.get(j, [])) < 2:
                continue
            s0, s1 = by_id[j][:2]
            sv, sn = float(T @ e[s1]), float(e[s0] @ e[s1])
            if (sv > thr["best"] and sv - sn <= m and own - float(e[s0] @ e[o1]) > m and sn < 0.85
                    and own > thr["best"] + 0.1 and float(E["petface"][o0] @ E["petface"][o1]) > thr["petface"]):
                rows.append((o0, o1, s0, s1, own, sv, sn, sn - sv))
        if len(rows) > 400:
            break
    rows.sort(key=lambda r: -(r[5] - thr["best"]))      # neighbours furthest above the threshold first
    print("memory cases:", len(rows))
    for n, nm, lab in ((3, "fig_memory", False), (args.cand // 2, "fig_memory_candidates", True)):
        sel, seen = [], set()
        for r in rows:
            c = coat.get(ev.id[r[0]], "?")
            if c not in seen:
                sel.append(r); seen.add(c)
            if len(sel) == n:
                break
        if not sel:
            print("no cases for", nm); continue
        fig, ax = grid(len(sel), 4, None, ["owner (enrolled)", "neighbour photo\nthe door remembers",
                                            "neighbour visits", "owner visits"],
                       "Neighbour memory: the visitor matches a remembered neighbour better than the owner", text_h=0.62)
        for r, (o0, o1, s0, s1, own, sv, sn, _) in enumerate(sel):
            for c, k in enumerate((o0, s0, s1, o1)):
                ax[r, c].imshow(load_img(ev.path[k]))
            ax[r, 0].set_ylabel(("#%d  " % r if lab else "") + coat.get(ev.id[o0], "?").replace("_", " "), fontsize=7)
            ax[r, 2].text(0.5, -0.04, f"vs owner {sv:.2f}: above threshold", transform=ax[r, 2].transAxes,
                          ha="center", va="top", fontsize=7, color=RED)
            ax[r, 2].text(0.5, -0.17, f"vs remembered {sn:.2f}: stopped", transform=ax[r, 2].transAxes,
                          ha="center", va="top", fontsize=7, color=GREEN)
            ax[r, 3].text(0.5, -0.04, f"vs owner {own:.2f}: let in", transform=ax[r, 3].transAxes,
                          ha="center", va="top", fontsize=7, color=GREEN)
        save(fig, nm)

    # 5. the test photo versions
    rng = np.random.default_rng(3)
    rels = rng.choice(tst.path.to_numpy(), 3, replace=False)
    V2 = ["clean", "door", "gray", "light", "masked"]
    fig, ax = grid(len(rels), len(V2), None, [VNAME[v] for v in V2], cell=1.4, text_h=0.1)
    for r, rel in enumerate(rels):
        for c, v in enumerate(V2):
            ax[r, c].imshow(load_img(rel, v))
    save(fig, "fig_variants")


if __name__ == "__main__":
    main()
