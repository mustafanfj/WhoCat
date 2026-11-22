"""E5: stranger-aware enrolment. The door also stores templates of the neighbour cats it keeps rejecting.

Simulated homes (one per test cat with >= 2 photos):
  owner     enrolled with 1 photo; its other photos are owner visits
  strangers the owner's 3 nearest look-alikes (CLIP neighbours) with >= 2 photos; one photo of each
            is stored as a 'known stranger' template (e.g. from the door's REJECT log), the rest are visits
Rules:
  baseline        admit if s_owner > t
  stranger-aware  admit if s_owner > t  and  s_owner - max_j s_stranger_j > m
t is the P1-val threshold at FAR=1%; m is tuned on val homes (largest m costing <= 1 pt owner FRR).
"""
import _boot  # noqa: F401
import argparse

import numpy as np
import pandas as pd

from whocat import metrics as M, paths, protocols
from whocat.data import load_eval_images
from whocat.embed import cache_file
from whocat.models import cache_name

rng = np.random.default_rng(1)


def homes(ev, split, n_str):
    sub = ev[ev.split == split]
    by_id = {k: v.index.to_numpy() for k, v in sub.groupby("id")}
    nn = pd.read_csv(paths.PAIRS / f"{split}_neighbours.csv.gz", dtype=str)
    out = []
    for i, neigh in zip(nn.id, nn.neighbours):
        if len(by_id.get(i, [])) < 2:
            continue
        s = [j for j in neigh.split() if len(by_id.get(j, [])) >= 2][:n_str]
        if len(s) < n_str:
            continue
        own = rng.permutation(by_id[i])
        st = [rng.permutation(by_id[j]) for j in s]
        out.append(dict(owner_t=own[0], owner_v=own[1:], str_t=[x[0] for x in st],
                        str_v=np.concatenate([x[1:] for x in st])))
    return out


def scores(H, e_c, e_p):
    """Owner score and best known-stranger score for every visit (computed once per model/variant)."""
    so, ss, kind, home = [], [], [], []
    for h_i, h in enumerate(H):
        T_o, T_s = e_c[h["owner_t"]], e_c[h["str_t"]]
        for k, idx in ((0, h["owner_v"]), (1, h["str_v"])):
            P = e_p[idx]
            so.append(P @ T_o)
            ss.append((P @ T_s.T).max(1))
            kind.append(np.full(len(idx), k))
            home.append(np.full(len(idx), h_i))
    return tuple(np.concatenate(a) for a in (so, ss, kind, home))


def run(S, t, m):
    so, ss, kind, home = S
    base = so > t
    aware = base & (so - ss > m)
    return pd.DataFrame(dict(home=home, kind=np.where(kind == 0, "owner", "stranger"), base=base, aware=aware))


def summarise(df):
    o, s = df[df.kind == "owner"], df[df.kind == "stranger"]
    breach = s.groupby("home")[["base", "aware"]].any().mean()
    return dict(owner_frr_base=1 - o.base.mean(), owner_frr_aware=1 - o.aware.mean(),
                stranger_far_base=s.base.mean(), stranger_far_aware=s.aware.mean(),
                homes_breached_base=breach.base, homes_breached_aware=breach.aware)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["petface", "megadesc", "dinov2"])
    ap.add_argument("--strangers", type=int, default=3)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    ev = load_eval_images()
    Hv, Ht = homes(ev, "val", args.strangers), homes(ev, "test", args.strangers)
    print(f"{len(Hv)} val homes, {len(Ht)} test homes")
    rows = []
    for m_name in args.models:
        if not cache_file(m_name, "clean").exists():
            continue
        e_c = np.load(cache_file(m_name, "clean"))
        tv = protocols.load("P1", "val")
        _, t = M.tar_at_far(protocols.score(tv, e_c, e_c), tv["label"], 0.01)
        # tune m on val
        Sv = scores(Hv, e_c, e_c)
        base_frr = summarise(run(Sv, t, -9))["owner_frr_base"]
        best = -1.0
        for m in np.round(np.arange(-0.2, 0.31, 0.02), 2):
            if summarise(run(Sv, t, m))["owner_frr_aware"] - base_frr <= 0.01:
                best = float(m) + 0.0
        for v in ("clean", "door"):
            if not cache_file(m_name, v).exists():
                continue
            e_p = np.load(cache_file(m_name, v))
            df = run(scores(Ht, e_c, e_p), t, best)
            r = dict(model=cache_name(m_name), probe=v, threshold=t, margin=best, **summarise(df))
            s = df[df.kind == "stranger"].reset_index(drop=True)
            homes_ = s.home.to_numpy()
            d, lo, hi = M.paired_diff(lambda w: M.rate_above(s.aware.to_numpy().astype(float), 0.5, w),
                                      lambda w: M.rate_above(s.base.to_numpy().astype(float), 0.5, w), homes_)
            r.update(stranger_far_diff=d, diff_lo=lo, diff_hi=hi)
            rows.append(r)
            print(r)
    tag = f"_{args.tag}" if args.tag else ""
    df = pd.DataFrame(rows)
    df.to_csv(paths.RESULTS / f"e5_stranger_enrollment{tag}.csv", index=False)
    if len(df):
        cols = ["model", "probe", "margin", "owner_frr_base", "owner_frr_aware", "stranger_far_base",
                "stranger_far_aware", "homes_breached_base", "homes_breached_aware"]
        t = df[cols].copy()
        for c in cols[3:]:
            t[c] = (t[c] * 100).round(2)
        with open(paths.RESULTS / f"summary{tag}.md", "a") as f:
            f.write("\n## E5 stranger-aware enrolment (%, recurring look-alike neighbours)\n\n" + t.to_markdown(index=False) + "\n")
        print(t.to_markdown(index=False))


if __name__ == "__main__":
    main()
