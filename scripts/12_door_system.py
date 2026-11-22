"""E10: the whole door, end to end. Which decisions keep look-alikes out while still letting the owner in?

Simulated homes on validation cats (owners need >= 4 photos so 1- and 3-photo enrolment use the SAME visits):
  owner     photos 1..3 = enrolment (k=1 uses photo 1 only), photos 4.. = owner visits
  strangers the owner's 3 nearest look-alikes (CLIP) from the same half; photo 1 of each is remembered
            by the door (as if from its reject log), the rest are visits
Tuning on even-ID cats (valA homes), reporting on odd-ID cats (valB homes): no cat is used for both.

Ladder (each step adds one decision):
  A  baseline model, 1 photo, threshold tuned on random cats          (what a product team ships)
  B  our model,      1 photo, threshold tuned on random cats
  C  our model,      1 photo, threshold tuned on look-alikes (1% of look-alike visits admitted on valA)
  D  C + neighbour memory (admit only if owner score beats every remembered neighbour by margin m)
  E  D with 3 enrolment photos (threshold and margin re-tuned for 3 photos)
  F  the baseline model with the full system (E's decisions)

  python scripts/12_door_system.py --ours ckpt:../checkpoints/arc_coat_door/best.pt --base petface
Outputs: outputs/results/e10_door_system.csv, e10_curve.csv, summary_door.md, outputs/figures/e10_curve.png
"""
import _boot  # noqa: F401
import argparse

import numpy as np
import pandas as pd

from whocat import metrics as M, paths, protocols
from whocat.data import l2n, load_eval_images
from whocat.embed import cache_file
from whocat.models import cache_name

K_MAX, N_STR = 3, 3


def build_homes(ev, half, seed):
    rng = np.random.default_rng(seed)
    sub = ev[(ev.split == "val") & ev.id.isin(half)]
    by_id = {k: v.index.to_numpy() for k, v in sub.groupby("id")}
    nn = pd.read_csv(paths.PAIRS / "val_neighbours.csv.gz", dtype=str)
    out = []
    for i, neigh in zip(nn.id, nn.neighbours):
        if i not in half or len(by_id.get(i, [])) < K_MAX + 1:
            continue
        s = [j for j in neigh.split() if j in half and j != i and len(by_id.get(j, [])) >= 2][:N_STR]
        if len(s) < N_STR:
            continue
        own = rng.permutation(by_id[i])
        st = [rng.permutation(by_id[j]) for j in s]
        out.append(dict(enr=own[:K_MAX], own_v=own[K_MAX:], str_t=np.array([x[0] for x in st]),
                        str_v=np.concatenate([x[1:] for x in st])))
    return out


def visit_scores(H, e_c, e_p, k):
    """Per visit: owner score, best remembered-neighbour score, kind (0 owner, 1 look-alike), home index."""
    so, ss, kind, home = [], [], [], []
    for h_i, h in enumerate(H):
        T = l2n(e_c[h["enr"][:k]].mean(0))
        Ts = e_c[h["str_t"]]
        for kd, idx in ((0, h["own_v"]), (1, h["str_v"])):
            P = e_p[idx]
            so.append(P @ T); ss.append((P @ Ts.T).max(1))
            kind.append(np.full(len(idx), kd)); home.append(np.full(len(idx), h_i))
    return tuple(np.concatenate(a) for a in (so, ss, kind, home))


def decide(S, t, m=None):
    so, ss, _, _ = S
    return (so > t) if m is None else (so > t) & (so - ss > m)


def rates(S, adm, w=None):
    _, _, kind, home = S
    o, s = kind == 0, kind == 1
    ww = np.ones(len(adm)) if w is None else w
    own = (adm[o] * ww[o]).sum() / ww[o].sum()
    look = (adm[s] * ww[s]).sum() / ww[s].sum()
    return own, look


def breached(S, adm):
    _, _, kind, home = S
    s = kind == 1
    return pd.Series(adm[s]).groupby(home[s]).any().mean()


def tune(Sa, t_rand, k_lookalike=True):
    """Look-alike threshold (1% of valA look-alike visits) and the largest margin costing <= 1 pt owner acceptance."""
    so, ss, kind, _ = Sa
    t = M.threshold_at_far(so[kind == 1], 0.01) if k_lookalike else t_rand
    base = rates(Sa, decide(Sa, t))[0]
    m_best = None
    for m in np.round(np.arange(-0.3, 0.31, 0.02), 2):
        if base - rates(Sa, decide(Sa, t, m))[0] <= 0.01:
            m_best = float(m)
    return float(t), m_best


def random_threshold(model, ev, half):
    e = np.load(cache_file(model, "clean"))
    tv = protocols.load("P1", "val")
    codes = ev.id.unique()
    ok = np.array([c in half for c in codes])
    keep = ok[tv["owner"]] & ok[tv["other"]]
    tv = {k: v[keep] for k, v in tv.items()}
    return M.tar_at_far(protocols.score(tv, e, e), tv["label"], 0.01)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--base", default="petface")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    ev = load_eval_images()
    val_ids = ev[ev.split == "val"].id.unique()
    A = {i for i in val_ids if int(i) % 2 == 0}
    B = {i for i in val_ids if int(i) % 2 == 1}
    Ha, Hb = build_homes(ev, A, args.seed), build_homes(ev, B, args.seed + 1)
    print(f"{len(Ha)} tuning homes (valA), {len(Hb)} report homes (valB)")

    cfg = {}
    for role, model in (("base", args.base), ("ours", args.ours)):
        ec = np.load(cache_file(model, "clean"))
        t_rand = random_threshold(model, ev, A)
        for k in (1, 3):
            Sa = visit_scores(Ha, ec, ec, k)
            t_look, m = tune(Sa, t_rand)
            cfg[(role, k)] = dict(model=model, t_rand=float(t_rand), t_look=t_look, m=m)
            print(role, k, cfg[(role, k)])

    ladder = [("A", "base", 1, "rand", False), ("B", "ours", 1, "rand", False), ("C", "ours", 1, "look", False),
              ("D", "ours", 1, "look", True), ("E", "ours", 3, "look", True), ("F", "base", 3, "look", True)]
    names = {"A": "Baseline model, 1 photo, random-tuned threshold", "B": "Our model, 1 photo, random-tuned threshold",
             "C": "+ threshold tuned on look-alikes", "D": "+ neighbour memory", "E": "+ 3 enrolment photos",
             "F": "Baseline model with the full system"}
    rows, curve = [], []
    rng = np.random.default_rng(0)
    for v in ("clean", "door"):
        S = {}
        for role in ("base", "ours"):
            model = cfg[(role, 1)]["model"]
            if not cache_file(model, v).exists():
                continue
            ec, ep = np.load(cache_file(model, "clean")), np.load(cache_file(model, v))
            for k in (1, 3):
                S[(role, k)] = visit_scores(Hb, ec, ep, k)
        for step, role, k, tk, mem in ladder:
            if (role, k) not in S:
                continue
            c, s = cfg[(role, k)], S[(role, k)]
            t = c["t_rand"] if tk == "rand" else c["t_look"]
            adm = decide(s, t, c["m"] if mem else None)
            own, look = rates(s, adm)
            boots = [rates(s, adm, M._weights(s[3], rng)) for _ in range(200)]
            lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
            rows.append(dict(step=step, system=names[step], model=cache_name(c["model"]), photos=k, probe=v,
                             threshold=round(t, 4), margin=c["m"] if mem else None,
                             owner_accept=100 * own, owner_lo=100 * lo[0], owner_hi=100 * hi[0],
                             lookalike_admitted=100 * look, look_lo=100 * lo[1], look_hi=100 * hi[1],
                             homes_breached=100 * breached(s, adm)))
            rows[-1] = {kk: (float(x) if isinstance(x, np.floating) else x) for kk, x in rows[-1].items()}
            print({kk: (round(x, 2) if isinstance(x, float) else x) for kk, x in rows[-1].items()})
        # operating curves (report homes): owner acceptance vs look-alike admission as the threshold sweeps
        for lab, role, k, mem in (("our model, 1 photo", "ours", 1, False), ("+ memory + 3 photos", "ours", 3, True),
                                  ("baseline, 1 photo", "base", 1, False)):
            if (role, k) not in S:
                continue
            s, c = S[(role, k)], cfg[(role, k)]
            for t in np.quantile(s[0][s[2] == 1], np.linspace(0.80, 0.9995, 60)):
                own, look = rates(s, decide(s, t, c["m"] if mem else None))
                curve.append(dict(curve=lab, probe=v, t=t, owner_accept=100 * own, lookalike_admitted=100 * look))

    df, cdf = pd.DataFrame(rows), pd.DataFrame(curve)
    df.to_csv(paths.RESULTS / "e10_door_system.csv", index=False)
    cdf.to_csv(paths.RESULTS / "e10_curve.csv", index=False)
    cols = ["step", "system", "model", "photos", "probe", "owner_accept", "owner_lo", "owner_hi",
            "lookalike_admitted", "look_lo", "look_hi", "homes_breached"]
    md = ["# E10 the whole door (valB report homes; tuned on valA)\n",
          f"{len(Hb)} report homes, {N_STR} look-alike neighbours each. 95% CI from a home-level bootstrap.\n",
          df[cols].round(2).to_markdown(index=False)]
    (paths.RESULTS / "summary_door.md").write_text("\n".join(md))
    print("\n".join(md))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for a, v in zip(ax, ("clean", "door")):
            for lab, g in cdf[cdf.probe == v].groupby("curve"):
                a.plot(g.lookalike_admitted, g.owner_accept, label=lab)
            a.set_xscale("log"); a.set_xlabel("look-alike visits admitted (%)"); a.set_title(f"{v} probes")
            a.axvline(1, ls="--", c="gray", lw=0.8)
        ax[0].set_ylabel("owner visits accepted (%)"); ax[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(paths.FIGS / "e10_curve.png", dpi=150)
    except Exception as ex:  # noqa: BLE001
        print("plot skipped:", ex)


if __name__ == "__main__":
    main()
