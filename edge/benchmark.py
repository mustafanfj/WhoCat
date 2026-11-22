"""Speed test on the device (Raspberry Pi 4 or Jetson Nano).

  python benchmark.py                          # synthetic frames
  python benchmark.py --video some_clip.mp4    # real frames (more realistic detection time)
Measures, per model file: face detection, crop, embedding (one face), and the full frame time.
Writes results/benchmark_<device>.json and prints a table.
"""
import argparse
import json
import os
import platform
import socket
import time

import cv2
import numpy as np

from whocat_edge import HERE, Detector, Embedder, load_config


def device_name():
    for p in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        if os.path.exists(p):
            return open(p).read().strip("\x00\n ")
    return platform.node() or socket.gethostname()


def frames_from(video, n):
    out = []
    if video:
        cap = cv2.VideoCapture(video)
        while len(out) < n:
            ok, f = cap.read()
            if not ok:
                break
            out.append(f)
    if not out:
        rng = np.random.RandomState(0)           # old numpy on Jetson Nano has no default_rng
        out = [rng.randint(0, 255, (480, 640, 3)).astype(np.uint8) for _ in range(n)]
    return out


def ms(xs):
    xs = np.array(xs)
    return dict(mean=float(xs.mean()), median=float(np.median(xs)), p90=float(np.percentile(xs, 90)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--n", type=int, default=100, help="timed runs per item")
    ap.add_argument("--models", nargs="+", default=None, help="e.g. small:fp32 small:int8 big:fp32")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    det = Detector(cfg)
    frames = frames_from(args.video, args.n)
    res = dict(device=device_name(), machine=platform.machine(), python=platform.python_version(), results=[])
    import onnxruntime as ort
    res["onnxruntime"] = ort.__version__
    res["available_providers"] = ort.get_available_providers()

    tdet, boxes = [], []
    for f in frames:
        s = time.perf_counter()
        b = det.boxes(f)
        tdet.append(1000 * (time.perf_counter() - s))
        boxes.append(b[0] if b else (f.shape[1] // 4, f.shape[0] // 4, f.shape[1] // 2, f.shape[1] // 2))
    tcrop, crops = [], []
    for f, b in zip(frames, boxes):
        s = time.perf_counter()
        crops.append(det.crop(f, b))
        tcrop.append(1000 * (time.perf_counter() - s))
    res["detection_ms"], res["crop_ms"] = ms(tdet), ms(tcrop)
    res["frames_with_face"] = int(sum(1 for f in frames if det.boxes(f))) if args.video else None
    print("%s | onnxruntime %s | providers %s" % (res["device"], res["onnxruntime"], res["available_providers"]))
    print("face detection: %.1f ms (median)   crop: %.2f ms" % (res["detection_ms"]["median"], res["crop_ms"]["median"]))

    todo = args.models or ["%s:%s" % (k, v) for k, m in cfg["models"].items() for v in m["files"]]
    for item in todo:
        key, variant = item.split(":")
        if variant not in cfg["models"][key]["files"]:
            continue
        try:
            emb = Embedder(cfg, key, variant, prefer_gpu=not args.cpu_only, threads=args.threads)
        except Exception as e:  # noqa: BLE001
            print("!! %s: %s" % (item, e))
            continue
        for c in crops[:10]:                                       # warm-up (TensorRT builds its engine here)
            emb([c])
        t = []
        for c in crops[:args.n]:
            s = time.perf_counter()
            emb([c])
            t.append(1000 * (time.perf_counter() - s))
        e = ms(t)
        total = res["detection_ms"]["median"] + res["crop_ms"]["median"] + e["median"]
        r = dict(model=key, variant=variant, provider=emb.provider, embed_ms=e, frame_ms_median=total,
                 fps=1000.0 / total, size_mb=os.path.getsize(os.path.join(HERE, "models", cfg["models"][key]["files"][variant])) / 1e6)
        res["results"].append(r)
        print("%-6s %-5s %-28s embed %7.1f ms  | whole frame %7.1f ms = %5.1f fps | %.0f MB" % (
            key, variant, emb.provider, e["median"], total, r["fps"], r["size_mb"]))
    out = os.path.join(HERE, "results")
    if not os.path.isdir(out):
        os.makedirs(out)
    name = "benchmark_%s.json" % "".join(ch if ch.isalnum() else "_" for ch in res["device"])[:40]
    json.dump(res, open(os.path.join(out, name), "w"), indent=2)
    print("saved results/%s" % name)


if __name__ == "__main__":
    main()
