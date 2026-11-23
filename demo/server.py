"""Live demo server: the demo page plus an /api/embed endpoint that runs the real models on your photos.

  python demo/get_weights.py     # once: joins our trained model from weights/ into checkpoints/
  python demo/server.py          # then open http://localhost:8123
Optional: PetFace's released cat model in ../checkpoints/petface/cat.pt adds the PetFace door, and the
sample-home bundle in demo/web/bundle adds the guided tour (both need PetFace's research licence). The browser only lets the page
use the camera on localhost or https, so open it as localhost, not by the machine's IP address.

Uploaded photos go through the OpenCV cat-face detector and the crop calibrated to PetFace's aligned
framing; if no face is found, the centre square is used. Nothing is stored on disk.
"""
import argparse
import base64
import importlib.util
import io
import json
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))
import _boot  # noqa: E402,F401
from whocat.degrade import degrade  # noqa: E402
from whocat.models import build  # noqa: E402

CASCADE = ROOT / "edge" / "models" / "haarcascade_frontalcatface_extended.xml"
# Detector crop calibrated on 600 PetFace validation images (scripts/16_export_edge.py), so that a
# detected face is framed like PetFace's aligned crops. --calibrate re-runs it (needs the dataset).
CALIBRATED_CROP = dict(crop_dx=0.01, crop_dy=-0.0525, crop_scale=1.12, min_face_px=40, scale_factor=1.05, min_neighbors=3)


def load_crop_calibration(recalibrate=False):
    if not recalibrate:
        return CALIBRATED_CROP
    try:
        spec = importlib.util.spec_from_file_location("exp", ROOT / "scripts" / "16_export_edge.py")
        exp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp)
        from whocat.data import load_eval_images
        return {**CALIBRATED_CROP, **exp.calibrate_detector(load_eval_images(), CASCADE, n=600)}
    except Exception as ex:  # noqa: BLE001
        print("re-calibration skipped, using the stored crop:", ex)
        return CALIBRATED_CROP


class Embedder:
    def __init__(self, names, device, recalibrate=False):
        self.device = device
        self.models = {}
        for key, name in names.items():
            m = build(name).eval().to(device)
            self.models[key] = m
            print("loaded", key, name)
        self.cc = cv2.CascadeClassifier(str(CASCADE))
        self.crop_cfg = load_crop_calibration(recalibrate)
        self.lock = threading.Lock()   # one request on the GPU at a time

    def face(self, img):
        rgb = np.asarray(img.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        f = 640 / max(h, w)
        small = cv2.resize(bgr, (int(w * f), int(h * f))) if f < 1 else bgr
        f = min(f, 1.0)
        g = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        c = self.crop_cfg
        mp = int(c.get("min_face_px", 40))
        boxes = self.cc.detectMultiScale(g, scaleFactor=c.get("scale_factor", 1.05), minNeighbors=int(c.get("min_neighbors", 3)), minSize=(mp, mp))
        if len(boxes) == 0:
            s = min(h, w)
            y0, x0 = (h - s) // 2, (w - s) // 2
            return Image.fromarray(rgb[y0:y0 + s, x0:x0 + s]).resize((224, 224), Image.BICUBIC), False
        x, y, bw, bh = max(boxes, key=lambda b: b[2] * b[3]) / f
        cx, cy = x + bw / 2 + c["crop_dx"] * bw, y + bh / 2 + c["crop_dy"] * bw
        half = c["crop_scale"] * bw / 2
        x0, y0, x1, y1 = (int(round(v)) for v in (cx - half, cy - half, cx + half, cy + half))
        pad = max(0, -x0, -y0, x1 - w, y1 - h)
        if pad:
            bgr = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
        crop = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        return Image.fromarray(crop).resize((224, 224), Image.BICUBIC), True

    @torch.no_grad()
    def embed(self, face):
        with self.lock:
            return self._embed(face)

    def _embed(self, face):
        variants = {"clean": face, "door": degrade(face, "door", seed=0)}
        out = {}
        for key, m in self.models.items():
            out[key] = {}
            for v, im in variants.items():
                e = m(m.transform(im).unsqueeze(0).to(self.device)).float().cpu().numpy()[0]
                e = e / max(np.linalg.norm(e), 1e-12)
                out[key][v] = base64.b64encode(e.astype(np.float32).tobytes()).decode()
        return out


def data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def make_handler(emb):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(HERE / "web"), **kw)

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/health":
                return self._json(200, {"ok": True, "models": list(emb.models)})
            return super().do_GET()

        def do_POST(self):
            if self.path != "/api/embed":
                return self._json(404, {"error": "unknown endpoint"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n > 15_000_000:
                    return self._json(413, {"error": "That photo is larger than 15 MB."})
                req = json.loads(self.rfile.read(n))
                raw = base64.b64decode(req["image"].split(",", 1)[-1])
                img = Image.open(io.BytesIO(raw))
                face, found = emb.face(img)
                door = degrade(face, "door", seed=0).convert("RGB")   # the same copy embed() scores
                return self._json(200, {"crop": data_url(face), "door_crop": data_url(door), "face_found": found, "emb": emb.embed(face)})
            except Exception as ex:  # noqa: BLE001
                return self._json(400, {"error": f"Could not read that photo ({type(ex).__name__})."})

        def log_message(self, fmt, *a):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    from whocat import paths
    ap.add_argument("--ours", default=f"ckpt:{paths.CKPT / 'arc_coat_door' / 'best.pt'}")
    ap.add_argument("--base", default="petface")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--calibrate", action="store_true", help="re-fit the detector crop on PetFace (needs the dataset)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ours = Path(args.ours[5:]) if args.ours.startswith("ckpt:") else None
    if ours is not None and not (ours.exists() and (ours.parent / "config.json").exists()):
        sys.exit(f"Our trained model is missing ({ours}).\nDownload it with:  python demo/get_weights.py")
    names = {"ours": args.ours}
    if args.base == "petface" and not paths.PETFACE_WEIGHTS.exists():
        print(f"Note: PetFace's model not found at {paths.PETFACE_WEIGHTS}; the PetFace door is switched off (optional, see README).")
    else:
        names["petface"] = args.base
    if not (HERE / "web" / "bundle" / "bundle.json").exists():
        print("Note: no sample homes in demo/web/bundle; the page opens on 'Your own cat' (sample homes are optional, see README).")
    emb = Embedder(names, dev, args.calibrate)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(emb))
    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"WhoCat demo on http://{shown}:{args.port}  (device: {dev}). Ctrl+C to stop.")
    srv.serve_forever()


if __name__ == "__main__":
    main()
