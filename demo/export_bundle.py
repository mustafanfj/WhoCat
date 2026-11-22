"""Export a self-contained demo bundle: sample homes, their face crops and cached embeddings.

The demo page (demo/web/index.html) runs entirely in the browser from this bundle, so it can be
hosted as a static page. Homes are the E10 report homes (odd-ID validation cats), never used for
tuning; thresholds and margins are tuned on the even-ID homes exactly as in scripts/12_door_system.py.

  python demo/export_bundle.py --ours ckpt:../checkpoints/arc_coat_door/best.pt --homes 12
Needs the embedding caches written by scripts/03_embed.py (clean and door variants) for both models.
Writes demo/web/bundle/bundle.json and demo/web/bundle/img/*.jpg (PetFace crops: keep the bundle
private or share it only for research use, as the PetFace licence requires).
"""
import argparse
import base64
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import _boot  # noqa: E402,F401
from whocat import paths  # noqa: E402
from whocat.data import l2n, load_eval_images, stable_seed  # noqa: E402
from whocat.degrade import degrade  # noqa: E402
from whocat.embed import cache_file  # noqa: E402

spec = importlib.util.spec_from_file_location("door", HERE.parent / "scripts" / "12_door_system.py")
door = importlib.util.module_from_spec(spec)
spec.loader.exec_module(door)


def b64(v):
    return base64.b64encode(np.asarray(v, dtype=np.float32).tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help="our model, e.g. ckpt:../checkpoints/arc_coat_door/best.pt")
    ap.add_argument("--base", default="petface")
    ap.add_argument("--homes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(HERE / "web" / "bundle"))
    args = ap.parse_args()

    ev = load_eval_images()
    val_ids = ev[ev.split == "val"].id.unique()
    A = {i for i in val_ids if int(i) % 2 == 0}
    B = {i for i in val_ids if int(i) % 2 == 1}
    Ha = door.build_homes(ev, A, args.seed)
    Hb = door.build_homes(ev, B, args.seed + 1)

    models = {"ours": args.ours, "petface": args.base}
    emb, cfg = {}, {}
    for key, name in models.items():
        ec = np.load(cache_file(name, "clean"))
        ed = np.load(cache_file(name, "door"))
        emb[key] = {"clean": l2n(ec), "door": l2n(ed)}
        t_rand = float(door.random_threshold(name, ev, A))
        cfg[key] = {"t_random": round(t_rand, 4), "t_lookalike": {}, "margin": {}}
        for k in (1, 2, 3):
            t_look, m = door.tune(door.visit_scores(Ha, ec, ec, k), t_rand)
            cfg[key]["t_lookalike"][str(k)] = round(t_look, 4)
            cfg[key]["margin"][str(k)] = m
        print(key, cfg[key])

    # pick homes whose owner looks most like its neighbours under the baseline (the interesting ones)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(Hb))[: args.homes * 4]
    def hardness(h):
        e = emb["petface"]["clean"]
        T = l2n(e[h["enr"]].mean(0))
        return float((e[h["str_v"]] @ T).max())
    picked = sorted((Hb[i] for i in order), key=hardness, reverse=True)[: args.homes]

    out = Path(args.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "img_door").mkdir(parents=True, exist_ok=True)
    rows = {}

    def image(idx):
        idx = int(idx)
        if idx not in rows:
            r = ev.iloc[idx]
            name = f"{r.id}_{idx}.jpg"
            face = Image.open(paths.IMAGES / r.path).convert("RGB").resize((224, 224), Image.BICUBIC)
            face.save(out / "img" / name, quality=86)
            # the exact door-camera copy the cached door embedding was computed from (same seed as data.ImageList)
            door_img = degrade(face, "door", seed=stable_seed(r.path + "door")).convert("RGB")
            door_img.resize((224, 224), Image.NEAREST).save(out / "img_door" / name, quality=86)
            rows[idx] = {"src": f"img/{name}", "cat": str(r.id),
                         "emb": {k: {v: b64(emb[k][v][idx]) for v in ("clean", "door")} for k in models}}
        return rows[idx]["src"]

    homes = []
    for n, h in enumerate(picked):
        owner = str(ev.iloc[int(h["enr"][0])].id)
        neigh_ids = [str(ev.iloc[int(i)].id) for i in h["str_t"]]
        homes.append({
            "name": f"Home {n + 1:02d}",
            "owner": owner,
            "enrol": [image(i) for i in h["enr"]],
            "owner_visits": [image(i) for i in h["own_v"]],
            "neighbours": [{"cat": c, "stored": image(t),
                            "visits": [image(v) for v in h["str_v"] if str(ev.iloc[int(v)].id) == c]}
                           for c, t in zip(neigh_ids, h["str_t"])],
        })
    bundle = {"version": 1, "door_images": "exact", "dim": int(next(iter(emb["ours"]["clean"])).shape[0]),
              "models": {"ours": {"label": "WhoCat", "detail": "DINOv2-B, fine-tuned", **cfg["ours"]},
                         "petface": {"label": "PetFace", "detail": "ResNet-50, released weights", **cfg["petface"]}},
              "images": {r["src"]: {"cat": r["cat"], "emb": r["emb"]} for r in rows.values()},
              "homes": homes}
    (out / "bundle.json").write_text(json.dumps(bundle))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"wrote {len(homes)} homes, {len(rows)} images, {size:.1f} MB to {out}")


if __name__ == "__main__":
    main()
