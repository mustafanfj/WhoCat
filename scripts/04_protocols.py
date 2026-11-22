"""Build the CatFace-Hard protocols on val (for thresholds) and test (for reporting).

  P0 official  PetFace's own verification.csv pairs (test only; sanity check vs. their paper)
  P1 random    impostor = any other cat
  P2 samecoat  impostor = another cat with the same (confident) CLIP coat label
  P3 lookalike impostor = one of the cat's 10 nearest other cats in CLIP space (CLIP is never evaluated)
  P5 faceid    k = 1/3/5 enrolment photos; probes = owner's other photos, random strangers, look-alikes
P4 (door quality) = P1-P3 with the probe scored on degraded images; it needs no extra pairs.
Near-duplicate impostors (same photo re-posted under two IDs) are removed by dHash and CLIP similarity.
"""
import _boot  # noqa: F401
import argparse
import itertools

import numpy as np
import pandas as pd

from whocat import paths, protocols
from whocat.data import l2n, load_eval_images, load_ids

rng = np.random.default_rng(0)


def popcount64(x):
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)


class Pool:
    def __init__(self, ev, ids, split, clip, dh, dup_cos):
        sub = ev[ev.split == split]
        self.by_id = {k: v.index.to_numpy() for k, v in sub.groupby("id")}
        self.ids = np.array(sorted(self.by_id))
        self.code = {k: i for i, k in enumerate(ev.id.unique())}
        coat = ids.set_index("id").coat.reindex(self.ids).fillna("unknown").to_numpy()
        self.coat = dict(zip(self.ids, coat))
        self.clip, self.dh, self.dup_cos = clip, dh, dup_cos
        self.dropped = 0

    def pick(self, i, k=1):
        return rng.choice(self.by_id[i], size=k, replace=False)

    def is_dup(self, a, b):
        same = popcount64(np.array([self.dh[a] ^ self.dh[b]], dtype=np.uint64))[0] <= 6
        return same or float(self.clip[a] @ self.clip[b]) >= self.dup_cos

    def trials(self):
        return dict(enroll=[], probe=[], label=[], owner=[], other=[])


def add(t, P, enroll, probe, label, owner, other):
    e = np.full(protocols.KMAX, -1)
    e[: len(enroll)] = enroll
    t["enroll"].append(e), t["probe"].append(probe), t["label"].append(label)
    t["owner"].append(P.code[owner]), t["other"].append(P.code[other])


def finish(t):
    out = {k: np.asarray(v, dtype=np.int64) for k, v in t.items()}
    out["enroll"] = out["enroll"].reshape(-1, protocols.KMAX)
    return out


def neighbours(P, k):
    """For each cat, the k nearest other cats by mean CLIP embedding (same split)."""
    M = l2n(np.stack([P.clip[P.by_id[i]].mean(0) for i in P.ids]))
    nn = np.zeros((len(P.ids), k), int)
    for s in range(0, len(M), 4096):
        S = M[s:s + 4096] @ M.T
        S[np.arange(len(S)), np.arange(s, s + len(S))] = -9      # exclude the cat itself
        part = np.argpartition(-S, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(S, part, 1), axis=1)
        nn[s:s + len(S)] = np.take_along_axis(part, order, 1)     # nearest first
    return {P.ids[i]: P.ids[nn[i]] for i in range(len(P.ids))}


def genuine(P, t, owners, g):
    for i in owners:
        imgs = P.by_id[i]
        if len(imgs) < 2:
            continue
        combos = list(itertools.combinations(imgs, 2))
        for j in rng.permutation(len(combos))[:g]:
            a, b = combos[j]
            add(t, P, [a], b, 1, i, i)


def impostor(P, t, owner, other):
    a, b = P.pick(owner)[0], P.pick(other)[0]
    if P.is_dup(a, b):
        P.dropped += 1
        return
    add(t, P, [a], b, 0, owner, other)


def build_pairs(P, nn, n_imp, g):
    out = {}
    owners = [i for i in P.ids if len(P.by_id[i]) >= 2]
    # P1
    t = P.trials()
    genuine(P, t, owners, g)
    for i in owners:
        for o in [o for o in rng.choice(P.ids, n_imp + 1, replace=False) if o != i][:n_imp]:
            impostor(P, t, i, o)
    out["P1"] = finish(t)
    # P2
    t = P.trials()
    coat_groups = pd.Series(P.coat).groupby(lambda x: P.coat[x]).groups
    own2 = [i for i in owners if P.coat[i] != "unknown"]
    genuine(P, t, own2, g)
    for i in own2:
        grp = np.array(coat_groups[P.coat[i]])
        for o in [o for o in rng.choice(grp, min(n_imp + 1, len(grp)), replace=False) if o != i][:n_imp]:
            impostor(P, t, i, o)
    out["P2"] = finish(t)
    # P3
    t = P.trials()
    genuine(P, t, owners, g)
    for i in owners:
        for o in nn[i]:
            impostor(P, t, i, o)
    out["P3"] = finish(t)
    return out


def build_faceid(P, nn, n_rand, n_look, ks=(1, 3, 5)):
    out = {}
    for k in ks:
        t = P.trials()
        owners = [i for i in P.ids if len(P.by_id[i]) >= k + 1]
        for i in owners:
            perm = rng.permutation(P.by_id[i])
            enroll, rest = perm[:k], perm[k:]
            for p in rest:
                add(t, P, enroll, p, 1, i, i)
            strangers = [o for o in rng.choice(P.ids, n_rand + 1, replace=False) if o != i][:n_rand]
            for o in list(strangers) + list(nn[i][:n_look]):
                add(t, P, enroll, P.pick(o)[0], 0, i, o)
        out[f"P5k{k}"] = finish(t)
        print(f"  P5 k={k}: {len(owners)} owners qualify")
    return out


def official(ev):
    df = pd.read_csv(paths.SPLIT / "verification.csv")
    pos = {p: i for i, p in enumerate(ev.path)}
    keep = df.filename1.isin(pos) & df.filename2.isin(pos)
    df = df[keep]
    ids = ev.id.unique()
    code = {k: i for i, k in enumerate(ids)}
    a = df.filename1.map(pos).to_numpy()
    e = np.full((len(df), protocols.KMAX), -1)
    e[:, 0] = a
    return dict(enroll=e, probe=df.filename2.map(pos).to_numpy(), label=df.label.to_numpy(),
                owner=df.filename1.str.split("/").str[1].map(code).to_numpy(),
                other=df.filename2.str.split("/").str[1].map(code).to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impostors", type=int, default=10, help="impostor pairs per cat for P1-P3")
    ap.add_argument("--genuine", type=int, default=3, help="max genuine pairs per cat")
    ap.add_argument("--neighbours", type=int, default=10)
    args = ap.parse_args()

    ev, ids = load_eval_images(), load_ids()
    clip = np.load(paths.CACHE / "emb_clip_clean.npy")
    dh = np.load(paths.INDEX / "eval_dhash.npy")

    # duplicate threshold: 99th percentile of CLIP similarity between photos of the same val cat
    val = ev[ev.split == "val"]
    sims = [float(clip[g.index[0]] @ clip[g.index[1]]) for _, g in val.groupby("id") if len(g) >= 2]
    dup_cos = float(np.quantile(sims, 0.99))
    print(f"duplicate threshold (CLIP cos) = {dup_cos:.3f}")

    report = [f"dup_cos={dup_cos:.4f}"]
    for split in ("val", "test"):
        print(split)
        P = Pool(ev, ids, split, clip, dh, dup_cos)
        nn = neighbours(P, args.neighbours)
        sets = build_pairs(P, nn, args.impostors, args.genuine)
        sets.update(build_faceid(P, nn, n_rand=20, n_look=10))
        for name, t in sets.items():
            protocols.save(name, split, **t)
            msg = f"{split} {name}: {int(t['label'].sum())} genuine, {int((1 - t['label']).sum())} impostor"
            print(" ", msg)
            report.append(msg)
        report.append(f"{split}: {P.dropped} near-duplicate impostor pairs removed")
        pd.DataFrame({"id": list(nn), "neighbours": [" ".join(v) for v in nn.values()]}) \
            .to_csv(paths.PAIRS / f"{split}_neighbours.csv.gz", index=False)
    protocols.save("P0", "test", **official(ev))

    # Test cats have only 2 photos each, so multi-photo enrolment (P5 k=3) is evaluated on a held-out
    # half of the val cats ('valB'); its threshold comes from the other half ('valA').
    ev2 = ev.copy()
    half = ev2.id.map(lambda i: "valA" if int(i) % 2 == 0 else "valB")
    ev2.loc[ev2.split == "val", "split"] = half[ev2.split == "val"]
    for split in ("valA", "valB"):
        P = Pool(ev2, ids, split, clip, dh, dup_cos)
        sets = build_faceid(P, neighbours(P, args.neighbours), n_rand=20, n_look=10, ks=(3,))
        protocols.save("P5k3", split, **sets["P5k3"])
        msg = f"{split} P5k3: {int(sets['P5k3']['label'].sum())} genuine, {int((1 - sets['P5k3']['label']).sum())} impostor"
        print(" ", msg)
        report.append(msg)
    (paths.PAIRS / "protocols_report.txt").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
