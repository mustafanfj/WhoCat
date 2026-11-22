"""E8: cohort score normalisation (adaptive S-norm) against the look-alike leak.

Each raw cosine score s(T, p) is rescaled by how similar the enrolment template T and the probe p
are to a cohort of other cats:
    s_norm = 0.5 * ((s - mu_T) / sd_T + (s - mu_p) / sd_p)
where mu/sd are the mean/std of the top-K cohort similarities. Cats that resemble many cats
(plain coats, look-alikes) get their scores discounted; distinctive cats do not.

Splits (no leakage): cohort = val cats with even ID ('valA'); thresholds tuned on val cats with odd ID
('valB'); results reported on test cats. Needs only cached embeddings (no GPU).

  python scripts/10_score_norm.py --models petface ckpt:../checkpoints/arc_coat_door/best.pt ...
Outputs: outputs/results/e8_score_norm.csv, e8_score_norm_coats.csv, summary_norm.md
"""
import _boot  # noqa: F401
import argparse

import numpy as np
import pandas as pd

from whocat import metrics as M, paths, protocols
from whocat.data import l2n, load_eval_images, load_ids
from whocat.embed import cache_file
from whocat.models import cache_name


def topk_stats(X, C, k, chunk=8192):
    """Mean and std of the k largest similarities of each row of X to the cohort C."""
    mu, sd = np.zeros(len(X), np.float32), np.zeros(len(X), np.float32)
    for s in range(0, len(X), chunk):
        S = X[s:s + chunk] @ C.T
        top = -np.partition(-S, k, axis=1)[:, :k]
        mu[s:s + chunk], sd[s:s + chunk] = top.mean(1), top.std(1) + 1e-6
    return mu, sd


def templates(t, e):
    E = t["enroll"]
    mask = (E >= 0)[..., None]
    return l2n((e[np.clip(E, 0, None)] * mask).sum(1))


def subset(t, keep):
    return {k: v[keep] for k, v in t.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=100, help="top-K cohort size for the statistics")
    args = ap.parse_args()

    ev, ids = load_eval_images(), load_ids()
    codes = ev.id.unique()
    code_of = {c: i for i, c in enumerate(codes)}
    val_ids = ev[ev.split == "val"].id.unique()
    valA = {i for i in val_ids if int(i) % 2 == 0}
    valB_codes = np.array(sorted(code_of[i] for i in val_ids if int(i) % 2 == 1))
    coat = ids.set_index("id").coat.to_dict()
    owner_coat = np.array([coat.get(c, "unknown") for c in codes])

    rows, coat_rows = [], []
    for m in args.models:
        if not cache_file(m, "clean").exists():
            print("no cache for", m); continue
        name = cache_name(m)
        e = np.load(cache_file(m, "clean"))
        cohort = l2n(pd.DataFrame(e[ev.id.isin(valA).to_numpy()]).groupby(ev.id[ev.id.isin(valA)].to_numpy()).mean().to_numpy())
        img_mu, img_sd = {}, {}

        def probe_stats(v):
            if v not in img_mu:
                ep = np.load(cache_file(m, v))
                img_mu[v], img_sd[v] = topk_stats(ep, cohort, args.k)
            return img_mu[v], img_sd[v]

        def scores(t, v):
            ep = np.load(cache_file(m, v))
            T = templates(t, e)
            raw = (T * ep[t["probe"]]).sum(1)
            if (t["enroll"][:, 1:] < 0).all():          # single-photo templates: reuse per-image stats
                mc, sc = probe_stats("clean")
                mt, st = mc[t["enroll"][:, 0]], sc[t["enroll"][:, 0]]
            else:
                mt, st = topk_stats(T, cohort, args.k)
            mp, sp = probe_stats(v)
            mp, sp = mp[t["probe"]], sp[t["probe"]]
            return raw, 0.5 * ((raw - mt) / st + (raw - mp) / sp)

        # thresholds on valB (owner and stranger both valB cats)
        thr = {}
        for p in ("P1", "P3"):
            tv = protocols.load(p, "val")
            keep = np.isin(tv["owner"], valB_codes) & np.isin(tv["other"], valB_codes)
            tv = subset(tv, keep)
            raw, nrm = scores(tv, "clean")
            for kind, s in (("raw", raw), ("norm", nrm)):
                thr[(kind, p)] = M.tar_at_far(s, tv["label"], 0.01)[1]

        for v in ("clean", "door", "gray"):
            if not cache_file(m, v).exists():
                continue
            res = {}
            for p in ("P1", "P2", "P3"):
                t = protocols.load(p, "test")
                raw, nrm = scores(t, v)
                y = t["label"]
                for kind, s in (("raw", raw), ("norm", nrm)):
                    r = res.setdefault(kind, dict(model=name, probe=v, scoring=kind))
                    r[f"tar1_{p}"] = 100 * M.tar_at_far(s, y, 0.01)[0]
                    r[f"leak_{p}"] = 100 * M.rate_above(s[y == 0], thr[(kind, "P1")])
                    if p == "P1":
                        r["owner_accept_randomthr"] = 100 * M.rate_above(s[y == 1], thr[(kind, "P1")])
                        r["owner_accept_hardthr"] = 100 * M.rate_above(s[y == 1], thr[(kind, "P3")])
                    if p == "P3":
                        r["leak_P3_hardthr"] = 100 * M.rate_above(s[y == 0], thr[(kind, "P3")])
                    if p == "P2" and v == "clean":
                        oc = owner_coat[t["owner"]]
                        for c in sorted(set(oc) - {"unknown"}):
                            sel = (y == 0) & (oc == c)
                            if sel.sum() >= 50:
                                coat_rows.append(dict(model=name, scoring=kind, coat=c,
                                                      far=100 * M.rate_above(s[sel], thr[(kind, "P1")])))
            for r in res.values():
                rows.append(r)
                print({k: (round(x, 2) if isinstance(x, float) else x) for k, x in r.items()})

    df = pd.DataFrame(rows)
    df.to_csv(paths.RESULTS / "e8_score_norm.csv", index=False)
    cdf = pd.DataFrame(coat_rows)
    cdf.to_csv(paths.RESULTS / "e8_score_norm_coats.csv", index=False)
    md = ["# E8 cohort score normalisation\n",
          "Threshold tuned on random val pairs (as a product team would), unless noted.\n"]
    cols = ["model", "probe", "scoring", "tar1_P1", "tar1_P3", "leak_P2", "leak_P3", "owner_accept_randomthr",
            "leak_P3_hardthr", "owner_accept_hardthr"]
    md += [df[cols].round(2).to_markdown(index=False), ""]
    if len(cdf):
        piv = cdf.pivot_table(index="coat", columns=["model", "scoring"], values="far").round(2)
        md += ["## Same-coat strangers let in, by coat (%)\n", piv.to_markdown(), ""]
    (paths.RESULTS / "summary_norm.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
