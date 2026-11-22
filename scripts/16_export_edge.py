"""Build the edge package for the Raspberry Pi 4 / Jetson Nano (run on the training machine).

  python scripts/16_export_edge.py
1. exports the small model (MobileNetV3) and the big model (DINOv2-B) to ONNX (L2-normalised 512-d output)
2. makes 8-bit (INT8) versions, calibrated on 400 validation faces
3. checks accuracy of every file against PyTorch on a test subset (random strangers and look-alikes)
4. calibrates the face crop of the OpenCV cat-face detector to PetFace's aligned framing
5. writes thresholds (random-tuned and look-alike-tuned, 1% FAR on validation) and the memory margin
Output: ../edge_package/ and ../whocat_edge_package.zip (give the zip to whoever runs the devices).
"""
import os
os.environ.setdefault("TIMM_FUSED_ATTN", "0")   # plain attention ops: exportable at ONNX opset 13 (Jetson TensorRT 8.2)
import _boot  # noqa: F401
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from whocat import metrics as M, paths, protocols
from whocat.data import load_eval_images
from whocat.embed import cache_file
from whocat.models import build, cache_name

R = "ckpt:" + str(paths.CKPT)
MODELS = {"small": R + "/student_coat_door/best.pt", "big": R + "/arc_coat_door/best.pt"}
EDGE_SRC = Path(__file__).resolve().parents[1] / "edge"


class Wrapped(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return F.normalize(self.m(x), dim=-1)


def to_input(img, size, mean, std):
    a = np.asarray(img.convert("RGB").resize((size, size), Image.BICUBIC), np.float32) / 255.0
    return ((a - mean) / std).transpose(2, 0, 1).astype(np.float32)


def export(name, ckpt, out, size):
    net = Wrapped(build(ckpt)).eval()
    p = out / f"whocat_{name}_fp32.onnx"
    torch.onnx.export(net, torch.zeros(1, 3, size, size), str(p), input_names=["image"], output_names=["embedding"],
                      dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=13,
                      do_constant_folding=True, dynamo=False)
    return net.eval(), p


def quantize(fp32, int8, calib):
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"image": x[None]} for x in calib])

        def get_next(self):
            return next(self.it, None)
    pre = str(fp32).replace(".onnx", "_pre.onnx")
    try:
        quant_pre_process(str(fp32), pre)
    except Exception as e:  # noqa: BLE001
        print("  pre-processing skipped:", e); pre = str(fp32)
    quantize_static(pre, str(int8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    if pre != str(fp32):
        Path(pre).unlink(missing_ok=True)


def ort_embed(path, X, bs=64):
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.concatenate([s.run(None, {"image": X[i:i + bs]})[0] for i in range(0, len(X), bs)])


def calibrate_detector(ev, cascade, n=3000, seed=0):
    cc = cv2.CascadeClassifier(str(cascade))
    v = ev[ev.split == "val"].path
    rel = v.sample(min(n, len(v)), random_state=seed).tolist()
    hits = []
    for r in rel:
        g = cv2.imread(str(paths.IMAGES / r), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        g = cv2.equalizeHist(cv2.resize(g, (224, 224)))
        pad = cv2.copyMakeBorder(g, 56, 56, 56, 56, cv2.BORDER_REPLICATE)
        b = cc.detectMultiScale(pad, scaleFactor=1.05, minNeighbors=3, minSize=(60, 60))
        if len(b):
            x, y, w, h = max(b, key=lambda t: t[2] * t[3])
            hits.append((x - 56 + w / 2, y - 56 + h / 2, w))
    h = np.array(hits)
    cx, cy, w = np.median(h, 0)
    return dict(cascade=cascade.name, crop_dx=float((112 - cx) / w), crop_dy=float((112 - cy) / w),
                crop_scale=float(224 / w), min_face_px=40, scale_factor=1.05, min_neighbors=3, calibration_images=len(rel),
                detection_rate_on_petface=float(len(h) / len(rel)), box_width_px_iqr=np.percentile(h[:, 2], [25, 75]).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=1500, help="test owners for the accuracy check")
    ap.add_argument("--no-int8-big", action="store_true")
    args = ap.parse_args()
    out = paths.ROOT / "edge_package"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(EDGE_SRC, out, ignore=shutil.ignore_patterns("__pycache__", "faces", "results", "gallery"))
    mdir = out / "models"
    ev = load_eval_images()
    rng = np.random.default_rng(0)
    cfg = {"models": {}, "note": "thresholds tuned on PetFace validation at 1% FAR; scores are cosine similarities"}

    det = calibrate_detector(ev, mdir / "haarcascade_frontalcatface_extended.xml")
    cfg["detector"] = det
    print("detector:", det)

    # test subset for the accuracy check
    owners = rng.choice(ev[ev.split == "test"].id.unique(), args.subset, replace=False)
    codes = {c: i for i, c in enumerate(ev.id.unique())}
    own_codes = np.array([codes[o] for o in owners])
    trials = {}
    for p in ("P1", "P3"):
        t = protocols.load(p, "test")
        keep = np.isin(t["owner"], own_codes)
        trials[p] = {k: v[keep] for k, v in t.items()}
    need = np.unique(np.concatenate([np.concatenate([t["enroll"][:, 0], t["probe"]]) for t in trials.values()]))
    e5 = pd.read_csv(paths.RESULTS / "e5_stranger_enrollment_trained.csv")
    lk = pd.read_csv(paths.RESULTS / "e1_leak_trained.csv")
    rows = []
    for name, ckpt in MODELS.items():
        key = cache_name(ckpt)
        net0 = build(ckpt).eval()
        size = getattr(net0, "img_size", 224)
        mean, std = (np.array(v, np.float32) for v in net0.norm)
        net, f32 = export(name, ckpt, mdir, size)
        print("exported", f32)
        X = np.stack([to_input(Image.open(paths.IMAGES / ev.path[i]), size, mean, std) for i in need])
        with torch.no_grad():
            e_t = np.concatenate([net(torch.from_numpy(X[i:i + 64])).numpy() for i in range(0, len(X), 64)])
        files = {"fp32": f32.name}
        embs = {"pytorch": e_t, "fp32": ort_embed(f32, X)}
        if not (name == "big" and args.no_int8_big):
            vv = ev[ev.split == "val"]
            val = vv.sample(min(400, len(vv)), random_state=1)
            calib = [to_input(Image.open(paths.IMAGES / r), size, mean, std) for r in val.path]
            i8 = mdir / f"whocat_{name}_int8.onnx"
            quantize(f32, i8, calib)
            files["int8"] = i8.name
            embs["int8"] = ort_embed(i8, X)
            print("quantized", i8)
        pos = {g: k for k, g in enumerate(need)}
        full = np.zeros((len(ev), e_t.shape[1]), np.float32)
        for v, E in embs.items():
            full[need] = E
            r = dict(model=name, file=v, max_abs_diff_vs_pytorch=float(np.abs(E - e_t).max()),
                     size_mb=(Path(mdir / files[v]).stat().st_size / 1e6) if v in files else None)
            for p, t in trials.items():
                s = protocols.score(t, full, full)
                r[f"tar1_{p}"] = 100 * M.tar_at_far(s, t["label"], 0.01)[0]
            rows.append(r)
            print(r)
        # thresholds from cached validation embeddings (same model, PyTorch)
        ec = np.load(cache_file(ckpt, "clean"))
        tv = protocols.load("P3", "val")
        thr_l = float(M.tar_at_far(protocols.score(tv, ec, ec), tv["label"], 0.01)[1])
        thr_r = float(lk[(lk.model == key) & (lk.protocol == "P1") & (lk.probe == "clean")].threshold.iloc[0])
        mg = e5[(e5.model == key) & (e5.probe == "clean")]
        cfg["models"][name] = dict(checkpoint=key, img_size=size, mean=mean.tolist(), std=std.tolist(), files=files,
                                   threshold_random=thr_r, threshold_lookalike=thr_l,
                                   memory_margin=float(mg.margin.iloc[0]) if len(mg) else 0.0)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    df = pd.DataFrame(rows)
    df.to_csv(paths.RESULTS / "e7_onnx_accuracy.csv", index=False)
    (out / "ACCURACY_ON_PETFACE.md").write_text(
        f"# Accuracy of the exported files ({args.subset} test cats, owners accepted at 1% FAR)\n\n"
        + df.round(3).to_markdown(index=False) + "\n")
    print(df.round(3).to_markdown(index=False))
    zp = shutil.make_archive(str(paths.ROOT / "whocat_edge_package"), "zip", out.parent, out.name)
    print("package:", zp)


if __name__ == "__main__":
    main()
