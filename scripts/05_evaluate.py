"""E1 + E2 (+P4, P5): evaluate every cached model on every protocol.

  python scripts/05_evaluate.py --models petface megadesc dinov2 [ckpt:...]

Outputs (outputs/results/):
  e1_verification.csv  TAR@FAR=1%/0.1%, EER, AUC with 95% CIs, per model x protocol x probe variant
  e1_leak.csv          look-alike false-accept rate on P2/P3 at the threshold tuned on P1 (val, FAR=1%)
  e2_coat.csv          per-coat false-accept rate at that threshold (clean and night-gray probes)
  e1_faceid.csv        P5: false-reject rate at FAR=1% for 1/3/5 enrolment photos
  summary.md           the main tables in Markdown
  figures/det_*.png    DET curves
"""
import _boot  # noqa: F401
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from whocat import metrics as M, paths, protocols  # noqa: E402
from whocat.data import load_eval_images, load_ids  # noqa: E402
from whocat.embed import cache_file  # noqa: E402
from whocat.models import cache_name  # noqa: E402

PROTOS = ["P1", "P2", "P3"]
VARIANTS = ["clean", "door", "gray"]


def emb(model, variant):
    f = cache_file(model, variant)
    return np.load(f) if f.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["petface", "megadesc", "dinov2"])
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--tag", default="", help="suffix for output files")
    args = ap.parse_args()

    ev, ids = load_eval_images(), load_ids()
    id_of_code = np.array(ev.id.unique())
    coat = ids.set_index("id").coat.to_dict()

    rows, leak, coat_rows, fid = [], [], [], []
    det = {p: {} for p in PROTOS}
    for m in args.models:
        e_clean = emb(m, "clean")
        if e_clean is None:
            print("no cache for", m, "- run 03_embed.py first")
            continue
        name = cache_name(m)
        # threshold tuned on easy random pairs (val), as a practitioner would
        tv = protocols.load("P1", "val")
        s_val = protocols.score(tv, e_clean, e_clean)
        _, thr = M.tar_at_far(s_val, tv["label"], 0.01)

        if protocols.exists("P0", "test"):
            t0 = protocols.load("P0", "test")
            s0 = protocols.score(t0, e_clean, e_clean)
            rows.append(dict(model=name, protocol="P0", probe="clean", auc=M.auc(np.clip(s0, 0, None), t0["label"])))

        for p in PROTOS:
            t = protocols.load(p, "test")
            for v in VARIANTS:
                e_probe = emb(m, v)
                if e_probe is None:
                    continue
                s, y, o = protocols.score(t, e_clean, e_probe), t["label"], t["owner"]
                tar1, _ = M.tar_at_far(s, y, 0.01)
                tar01, _ = M.tar_at_far(s, y, 0.001)
                lo, hi = M.bootstrap(M.tar_fn(s, y, 0.01), o, args.boot)
                r = dict(model=name, protocol=p, probe=v, tar_far1=tar1, tar_far1_lo=lo, tar_far1_hi=hi,
                         tar_far01=tar01, eer=M.eer(s, y), n_gen=int(y.sum()), n_imp=int((1 - y).sum()))
                if p == "P1" and v == "clean":
                    r["auc"] = M.auc(s, y)
                rows.append(r)
                # E1 leak: look-alikes accepted at the P1-val threshold
                imp = y == 0
                far = M.rate_above(s[imp], thr)
                flo, fhi = M.bootstrap(lambda w: M.rate_above(s[imp], thr, w), o[imp], args.boot)
                frr = 1 - M.rate_above(s[~imp], thr)
                leak.append(dict(model=name, protocol=p, probe=v, threshold=thr, far=far, far_lo=flo,
                                 far_hi=fhi, frr=frr))
                if v == "clean":
                    det[p][name] = M.det_points(s, y)
                # E2: per coat (P2 = same-coat impostors, so FAR per coat = how often a same-coat stranger passes)
                if p == "P2" and v in ("clean", "gray"):
                    owner_coat = np.array([coat.get(id_of_code[c], "unknown") for c in o])
                    for c in sorted(set(owner_coat) - {"unknown"}):
                        sel = imp & (owner_coat == c)
                        selg = (~imp) & (owner_coat == c)
                        if sel.sum() < 50:
                            continue
                        coat_rows.append(dict(model=name, probe=v, coat=c, n_cats=len(np.unique(o[sel])),
                                              far=M.rate_above(s[sel], thr),
                                              frr=1 - M.rate_above(s[selg], thr) if selg.any() else np.nan))
        # P5 Face ID
        for k in (1, 3, 5):
            nm = f"P5k{k}"
            tune, rep = ("valA", "valB") if k == 3 else ("val", "test")
            if not (protocols.exists(nm, tune) and protocols.exists(nm, rep)):
                continue
            tv5, t5 = protocols.load(nm, tune), protocols.load(nm, rep)
            if t5["label"].sum() < 50:
                continue
            for v in VARIANTS:
                e_probe = emb(m, v)
                if e_probe is None:
                    continue
                sv = protocols.score(tv5, e_clean, emb(m, v))
                _, t_k = M.tar_at_far(sv, tv5["label"], 0.01)          # threshold from val owners
                s, y = protocols.score(t5, e_clean, e_probe), t5["label"]
                fid.append(dict(model=name, k=k, cats=rep, probe=v, n_owners=len(np.unique(t5["owner"])),
                                frr=1 - M.rate_above(s[y == 1], t_k), far=M.rate_above(s[y == 0], t_k),
                                tar_far1=M.tar_at_far(s, y, 0.01)[0]))

    tag = f"_{args.tag}" if args.tag else ""
    R = paths.RESULTS
    df, lk, cr, fd = map(pd.DataFrame, (rows, leak, coat_rows, fid))
    df.to_csv(R / f"e1_verification{tag}.csv", index=False)
    lk.to_csv(R / f"e1_leak{tag}.csv", index=False)
    cr.to_csv(R / f"e2_coat{tag}.csv", index=False)
    fd.to_csv(R / f"e1_faceid{tag}.csv", index=False)

    md = ["# WhoCat results\n"]
    if len(df):
        main_t = df[df.protocol.isin(PROTOS)].pivot_table(index="model", columns=["protocol", "probe"],
                                                           values="tar_far1")
        md += ["## TAR @ FAR=1% (test cats)\n", (main_t * 100).round(1).to_markdown(), ""]
        p0 = df[df.protocol == "P0"]
        if len(p0):
            md += ["## PetFace official verification AUC (sanity)\n", (p0.set_index("model").auc * 100).round(2).to_markdown(), ""]
    if len(lk):
        t = lk.pivot_table(index="model", columns=["protocol", "probe"], values="far")
        md += ["## Strangers let in at the P1-tuned threshold (target 1%)\n", (t * 100).round(2).to_markdown(), ""]
    if len(cr):
        t = cr.pivot_table(index="coat", columns=["model", "probe"], values="far")
        md += ["## Same-coat strangers let in, by coat (%)\n", (t * 100).round(2).to_markdown(), ""]
    if len(fd):
        t = fd.pivot_table(index="model", columns=["k", "probe"], values="frr")
        md += ["## Face ID: owner rejected at FAR=1% (%)\n", (t * 100).round(2).to_markdown(), ""]
    (R / f"summary{tag}.md").write_text("\n".join(md))
    print("\n".join(md))

    for p, curves in det.items():
        if not curves:
            continue
        plt.figure(figsize=(4.5, 4))
        for n, (fpr, fnr) in curves.items():
            plt.loglog(np.clip(fpr, 1e-5, 1), np.clip(fnr, 1e-5, 1), label=n)
        plt.xlabel("False accept rate (stranger let in)")
        plt.ylabel("False reject rate (owner locked out)")
        plt.title(f"DET, {p}")
        plt.grid(alpha=0.3, which="both")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(paths.FIGS / f"det_{p}{tag}.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
