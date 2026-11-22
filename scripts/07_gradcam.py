"""E4: where does each model look? Similarity Grad-CAM on cat faces.

For each test cat with >= 2 photos we explain sim(photo_1, photo_2) (what identifies the cat) and,
for its nearest look-alike, sim(photo_1, look-alike) (what makes the two cats look the same).
Score = share of CAM mass inside the inner face (eyes, nose, whisker pads, mouth) of the aligned
PetFace template; higher means the model relies on facial markings rather than overall fur.

  python scripts/07_gradcam.py --models ckpt:../checkpoints/arc_random/best.pt ckpt:../checkpoints/arc_coat/best.pt petface
"""
import _boot  # noqa: F401
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pytorch_grad_cam import GradCAM  # noqa: E402

from whocat import paths  # noqa: E402
from whocat.data import load_eval_images  # noqa: E402
from whocat.embed import device  # noqa: E402
from whocat.models import build, cache_name, PetFaceR50, FaceNet  # noqa: E402

rng = np.random.default_rng(3)


class SimilarityToConceptTarget:
    """Grad-CAM target: cosine similarity between the image embedding and a reference embedding."""

    def __init__(self, ref):
        self.ref = ref

    def __call__(self, out):
        return torch.nn.functional.cosine_similarity(out, self.ref, dim=0)


def target_layer(model):
    if isinstance(model, PetFaceR50):
        return model.m.layer4[-1], None
    if isinstance(model, FaceNet):
        bb = model.backbone
        if hasattr(bb, "patch_embed") and hasattr(bb, "blocks"):     # ViT (DINOv2)
            n_pre = getattr(bb, "num_prefix_tokens", 1)

            def reshape(t):
                t = t[:, n_pre:, :]
                s = int(round(t.shape[1] ** 0.5))
                return t.reshape(t.shape[0], s, s, t.shape[2]).permute(0, 3, 1, 2)
            return bb.blocks[-1].norm1, reshape
        if hasattr(bb, "layer4"):                                     # ResNet
            return bb.layer4[-1], None
        if hasattr(bb, "blocks"):                                     # MobileNetV3 / EfficientNet
            return bb.blocks[-1], None
    raise ValueError("Grad-CAM supported for petface and our checkpoints")


def load(rel, tf):
    img = Image.open(paths.IMAGES / rel).convert("RGB").resize((224, 224))
    return img, tf(img)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()
    dev = device()
    ev = load_eval_images()
    test = ev[ev.split == "test"]
    by_id = {k: v.path.tolist() for k, v in test.groupby("id")}
    nn = pd.read_csv(paths.PAIRS / "test_neighbours.csv.gz", dtype=str)
    cases = []
    for i, neigh in zip(nn.id, nn.neighbours):
        if len(by_id.get(i, [])) >= 2:
            cases.append((by_id[i][0], by_id[i][1], by_id[neigh.split()[0]][0]))
    cases = [cases[j] for j in rng.permutation(len(cases))[: args.n]]

    x0, y0, x1, y1 = paths.INNER_FACE_BOX
    rows, examples = [], {}
    for name in args.models:
        try:
            model = build(name).to(dev).eval()
        except Exception as e:
            print(f"!! skipping {name}: {e}")
            continue
        layer, reshape = target_layer(model)
        cam = GradCAM(model=model, target_layers=[layer], reshape_transform=reshape)
        share = {"self": [], "lookalike": []}
        ex = []
        for a, a2, b in cases:
            ia, xa = load(a, model.transform)
            with torch.no_grad():
                ref = {k: torch.nn.functional.normalize(model(load(p, model.transform)[1].to(dev)), dim=-1)[0]
                       for k, p in (("self", a2), ("lookalike", b))}
            maps = {}
            for k, r in ref.items():
                g = cam(input_tensor=xa.to(dev), targets=[SimilarityToConceptTarget(r)])[0]
                g = np.array(Image.fromarray(g.astype(np.float32)).resize((224, 224), Image.BILINEAR))
                share[k].append(g[y0:y1, x0:x1].sum() / max(g.sum(), 1e-8))
                maps[k] = g
            if len(ex) < args.show:
                ex.append((ia, maps, Image.open(paths.IMAGES / b).convert("RGB").resize((224, 224))))
        box_area = (x1 - x0) * (y1 - y0) / 224 ** 2
        rows.append(dict(model=cache_name(name), inner_share_self=np.mean(share["self"]),
                         inner_share_lookalike=np.mean(share["lookalike"]), uniform_baseline=box_area,
                         n=len(cases)))
        examples[cache_name(name)] = ex
        print(rows[-1])
        del model, cam
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(paths.RESULTS / "e4_gradcam.csv", index=False)

    names = list(examples)
    if not names:
        return
    fig, axes = plt.subplots(args.show, 1 + len(names), figsize=(2.2 * (1 + len(names)), 2.2 * args.show))
    for r in range(args.show):
        ia, _, b = examples[names[0]][r]
        axes[r, 0].imshow(np.concatenate([np.asarray(ia), np.asarray(b)], 1))
        axes[r, 0].set_title("cat | look-alike" if r == 0 else "", fontsize=8)
        for c, n in enumerate(names):
            ia, maps, _ = examples[n][r]
            axes[r, c + 1].imshow(ia)
            axes[r, c + 1].imshow(maps["self"], cmap="jet", alpha=0.45)
            if r == 0:
                axes[r, c + 1].set_title(n, fontsize=8)
    for a in axes.flat:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(paths.FIGS / "e4_gradcam_examples.png", dpi=200)
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
