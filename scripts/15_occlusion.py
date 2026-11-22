"""E4b: where does each model look? Occlusion maps (faithful alternative to Grad-CAM for ViTs).

For an owner pair (enrolled photo A, visit photo A2) we cover one 32 px square of A2 at a time with a blurred
copy of itself and record how much the similarity sim(A, A2) drops. Big drop = the match depends on that region.
The same is done for the owner's nearest look-alike B: which regions make B look like the owner.
No gradients are involved, so it works the same way for the ResNet, the ViTs and the MobileNet.

Summary over --n test cats: share of the total drop that falls in
  inner face (eyes, nose, whisker pads, mouth), upper head (forehead, ears), rest of the head oval, outside the oval.

  python scripts/15_occlusion.py --models petface ckpt:../checkpoints/arc_coat/best.pt \
      ckpt:../checkpoints/arc_coat_door/best.pt ckpt:../checkpoints/arc_coat_door_bg/best.pt
Outputs: outputs/results/e4_occlusion.csv, outputs/report_figs/occlusion_examples.npz, outputs/figures/e4_occlusion.png
"""
import _boot  # noqa: F401
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.transforms.functional import gaussian_blur  # noqa: E402

from whocat import paths  # noqa: E402
from whocat.data import load_eval_images  # noqa: E402
from whocat.degrade import oval_mask  # noqa: E402
from whocat.embed import device  # noqa: E402
from whocat.models import build, cache_name  # noqa: E402

PATCH, STRIDE = 32, 16          # in the 224 px frame


def regions():
    x0, y0, x1, y1 = paths.INNER_FACE_BOX
    inner = np.zeros((224, 224), bool); inner[y0:y1, x0:x1] = True
    oval = oval_mask(224)[..., 0] > 0.5
    upper = oval & ~inner; upper[y0:, :] = False            # forehead and ears: inside the oval, above the inner face
    rest = oval & ~inner & ~upper
    return dict(inner_face=inner, upper_head=upper, rest_of_head=rest, outside_oval=~oval)


@torch.no_grad()
def embed(model, x, dev, bs=256):
    out = []
    for i in range(0, len(x), bs):
        out.append(F.normalize(model(x[i:i + bs].to(dev)), dim=-1))
    return torch.cat(out)


@torch.no_grad()
def occlusion_map(model, ref, x, dev):
    """ref: (D,) unit embedding of the other photo. x: (3,H,W) input tensor of the photo to explain."""
    H = x.shape[-1]
    p, s = round(PATCH * H / 224), round(STRIDE * H / 224)
    blur = gaussian_blur(x[None], kernel_size=2 * (H // 14) + 1, sigma=H / 20)[0]
    pos = [(i, j) for i in range(0, H - p + 1, s) for j in range(0, H - p + 1, s)]
    batch = x[None].repeat(len(pos) + 1, 1, 1, 1)
    for k, (i, j) in enumerate(pos):
        batch[k + 1, :, i:i + p, j:j + p] = blur[:, i:i + p, j:j + p]
    sims = (embed(model, batch, dev) @ ref).cpu().numpy()
    drop = np.clip(sims[0] - sims[1:], 0, None)
    acc, cnt = np.zeros((H, H)), np.zeros((H, H))
    for d, (i, j) in zip(drop, pos):
        acc[i:i + p, j:j + p] += d; cnt[i:i + p, j:j + p] += 1
    m = acc / np.maximum(cnt, 1)
    m = np.array(Image.fromarray(m.astype(np.float32)).resize((224, 224), Image.BILINEAR))
    return m, float(sims[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=200, help="test cats for the region summary")
    ap.add_argument("--show", type=int, default=24, help="cases saved for figures")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    dev = device()
    ev = load_eval_images()
    test = ev[ev.split == "test"]
    by_id = {k: v.path.tolist() for k, v in test.groupby("id")}
    nn = pd.read_csv(paths.PAIRS / "test_neighbours.csv.gz", dtype=str)
    cases = [(by_id[i][0], by_id[i][1], by_id[n.split()[0]][0]) for i, n in zip(nn.id, nn.neighbours)
             if len(by_id.get(i, [])) >= 2 and len(by_id.get(n.split()[0], [])) >= 1]
    rng = np.random.default_rng(args.seed)
    cases = [cases[j] for j in rng.permutation(len(cases))[: max(args.n, args.show)]]
    R = regions()
    rows, ex = [], {"images": [], "paths": []}
    for a, a2, b in cases[: args.show]:
        ex["images"].append(np.stack([np.asarray(Image.open(paths.IMAGES / p).convert("RGB").resize((224, 224)))
                                      for p in (a, a2, b)]))
        ex["paths"].append(f"{a}|{a2}|{b}")
    for name in args.models:
        try:
            model = build(name).to(dev).eval()
        except Exception as e:  # noqa: BLE001
            print(f"!! skipping {name}: {e}"); continue
        tf = model.transform
        key = cache_name(name)
        acc = {k: {r: [] for r in R} for k in ("owner", "lookalike")}
        maps_o, maps_l, sim_o, sim_l = [], [], [], []
        for c, (a, a2, b) in enumerate(cases):
            xa, xa2, xb = (tf(Image.open(paths.IMAGES / p).convert("RGB").resize((224, 224))) for p in (a, a2, b))
            ref = embed(model, xa[None], dev)[0]
            mo, so = occlusion_map(model, ref, xa2, dev)
            ml, sl = occlusion_map(model, ref, xb, dev)
            for k, m in (("owner", mo), ("lookalike", ml)):
                tot = m.sum() + 1e-8
                for r, msk in R.items():
                    acc[k][r].append(m[msk].sum() / tot)
            if c < args.show:
                maps_o.append(mo); maps_l.append(ml); sim_o.append(so); sim_l.append(sl)
            if c % 50 == 0:
                print(key, c, "/", len(cases))
        ex[f"{key}__owner"] = np.stack(maps_o).astype(np.float16)
        ex[f"{key}__lookalike"] = np.stack(maps_l).astype(np.float16)
        ex[f"{key}__sim_owner"] = np.array(sim_o); ex[f"{key}__sim_lookalike"] = np.array(sim_l)
        for k in acc:
            row = dict(model=key, pair=k, n=len(cases))
            row.update({r: 100 * float(np.mean(v)) for r, v in acc[k].items()})
            row.update({f"area_{r}": 100 * float(msk.mean()) for r, msk in R.items()})
            rows.append(row); print(row)
        del model; torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(paths.RESULTS / "e4_occlusion.csv", index=False)
    out = paths.OUT / "report_figs"; out.mkdir(exist_ok=True)
    np.savez_compressed(out / "occlusion_examples.npz", **{k: (np.array(v) if isinstance(v, list) else v)
                                                           for k, v in ex.items()})
    names = [cache_name(m) for m in args.models if f"{cache_name(m)}__owner" in ex]
    show = min(6, args.show)
    fig, axes = plt.subplots(show, 1 + len(names), figsize=(2 * (1 + len(names)), 2 * show), squeeze=False)
    for r in range(show):
        axes[r, 0].imshow(np.concatenate([ex["images"][r][0], ex["images"][r][1]], 1))
        for c, n in enumerate(names):
            axes[r, c + 1].imshow(ex["images"][r][1])
            m = ex[f"{n}__owner"][r].astype(np.float32)
            axes[r, c + 1].imshow(m / (m.max() + 1e-8), cmap="inferno", alpha=0.55)
            if r == 0:
                axes[r, c + 1].set_title(n, fontsize=7)
    for a_ in axes.flat:
        a_.axis("off")
    plt.tight_layout(); plt.savefig(paths.FIGS / "e4_occlusion.png", dpi=150)
    print(df.round(1).to_markdown(index=False))


if __name__ == "__main__":
    main()
