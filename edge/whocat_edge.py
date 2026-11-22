"""WhoCat edge runtime: cat-face detection, cropping, embedding and the door decision.

Needs only numpy, opencv and onnxruntime (no PyTorch). Works with Python 3.6+ (Jetson Nano JetPack 4.6)
and Raspberry Pi OS 64-bit.
"""
import json
import os
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config():
    with open(os.path.join(HERE, "config.json")) as f:
        return json.load(f)


def l2n(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class Detector:
    """OpenCV Haar cat-face detector + a crop calibrated to PetFace's aligned framing."""

    def __init__(self, cfg, work_side=480):
        d = cfg["detector"]
        self.cc = cv2.CascadeClassifier(os.path.join(HERE, "models", d["cascade"]))
        if self.cc.empty():
            raise RuntimeError("could not load the cat-face detector file in models/")
        self.dx, self.dy, self.scale = d["crop_dx"], d["crop_dy"], d["crop_scale"]
        self.work_side = work_side          # frames are shrunk to this longest side before detection (speed)
        self.min_face = d.get("min_face_px", 40)
        self.scale_factor = d.get("scale_factor", 1.05)     # smaller = finds more faces, slower
        self.min_neighbors = d.get("min_neighbors", 3)      # smaller = finds more faces, more false boxes

    def boxes(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        f = self.work_side / float(max(h, w))
        small = cv2.resize(frame_bgr, (int(w * f), int(h * f))) if f < 1 else frame_bgr
        f = min(f, 1.0)
        g = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        r = self.cc.detectMultiScale(g, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
                                     minSize=(int(self.min_face * f), int(self.min_face * f)))
        out = [(int(x / f), int(y / f), int(bw / f), int(bh / f)) for (x, y, bw, bh) in r]
        return sorted(out, key=lambda b: -b[2] * b[3])

    def crop(self, frame_bgr, box, size=224):
        x, y, w, h = box
        cx, cy = x + w / 2.0 + self.dx * w, y + h / 2.0 + self.dy * w
        half = self.scale * w / 2.0
        x0, y0, x1, y1 = int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))
        H, W = frame_bgr.shape[:2]
        pad = max(0, -x0, -y0, x1 - W, y1 - H)
        if pad:
            frame_bgr = cv2.copyMakeBorder(frame_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
        c = frame_bgr[y0:y1, x0:x1]
        c = cv2.resize(c, (size, size), interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)


def sharpness(rgb):
    return float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def providers(prefer_gpu=True):
    import onnxruntime as ort
    avail = ort.get_available_providers()
    order = (["TensorrtExecutionProvider", "CUDAExecutionProvider"] if prefer_gpu else []) + ["CPUExecutionProvider"]
    return [p for p in order if p in avail]


class Embedder:
    """One ONNX face model. Input: list of 224x224 RGB uint8 crops. Output: unit-length embeddings."""

    def __init__(self, cfg, key, variant="fp32", prefer_gpu=True, threads=None):
        import onnxruntime as ort
        m = cfg["models"][key]
        self.key, self.variant = key, variant
        self.size = m["img_size"]
        self.mean = np.array(m["mean"], np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(m["std"], np.float32).reshape(1, 3, 1, 1)
        self.thr_random = m["threshold_random"]          # 1% of random strangers get in
        self.thr_lookalike = m["threshold_lookalike"]    # 1% of look-alikes get in (recommended for the door)
        self.margin = m["memory_margin"]
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        path = os.path.join(HERE, "models", m["files"][variant])
        prov = providers(prefer_gpu and variant != "int8")
        self.sess = ort.InferenceSession(path, so, providers=prov)
        self.inp = self.sess.get_inputs()[0].name
        self.provider = self.sess.get_providers()[0]

    def __call__(self, crops):
        x = np.stack([cv2.resize(c, (self.size, self.size), interpolation=cv2.INTER_CUBIC) if c.shape[0] != self.size
                      else c for c in crops]).astype(np.float32) / 255.0
        x = (x.transpose(0, 3, 1, 2) - self.mean) / self.std
        return l2n(self.sess.run(None, {self.inp: x.astype(np.float32)})[0])


def gallery_dir(key):
    d = os.path.join(HERE, "gallery", key)
    if not os.path.isdir(d):
        os.makedirs(d)
    return d


def load_gallery(key):
    d = gallery_dir(key)
    return {f[:-4]: np.load(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".npy")}


def decide(score_owner, scores_neighbours, thr, margin=None):
    """WhoCat door rule. Without memory: admit if the owner score passes the threshold.
    With memory: also require the owner score to beat every remembered neighbour by the margin."""
    if score_owner <= thr:
        return False
    if margin is None or len(scores_neighbours) == 0:
        return True
    return score_owner - max(scores_neighbours) > margin


def vote(frame_scores, k=5):
    """Visit-level score from several frames: mean of the k best frame scores."""
    s = sorted(frame_scores, reverse=True)[:k]
    return float(np.mean(s)) if s else -1.0


class Timer:
    def __init__(self):
        self.t = {}

    def add(self, name, dt):
        self.t.setdefault(name, []).append(dt * 1000.0)

    def __call__(self, name):
        timer = self

        class _C:
            def __enter__(self):
                self.s = time.perf_counter()

            def __exit__(self, *a):
                timer.add(name, time.perf_counter() - self.s)
        return _C()
