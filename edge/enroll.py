"""Enroll a cat for the live door demo from a short video (or a folder of photos).

  python enroll.py --name luna --source luna.mp4 [--model small]
Saves gallery/<model>/luna.npy (the cat's face template) and gallery/<model>/luna_faces.jpg (check it).
"""
import argparse
import os

import cv2
import numpy as np

from whocat_edge import Detector, Embedder, gallery_dir, load_config, sharpness
from extract_faces import sheet


def frames(source, every):
    if os.path.isdir(source):
        for f in sorted(os.listdir(source)):
            im = cv2.imread(os.path.join(source, f))
            if im is not None:
                yield im
        return
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        i += 1
        if i % every == 0:
            yield f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", default="small")
    ap.add_argument("--variant", default="fp32")
    ap.add_argument("--k", type=int, default=10, help="sharpest face crops to average")
    ap.add_argument("--every", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config()
    det, emb = Detector(cfg), Embedder(cfg, args.model, args.variant)
    crops = []
    for f in frames(args.source, args.every):
        b = det.boxes(f)
        if b:
            crops.append(det.crop(f, b[0]))
        if len(crops) >= 150:
            break
    if not crops:
        raise SystemExit("no cat face found in %s" % args.source)
    crops = [crops[i] for i in np.argsort([-sharpness(c) for c in crops])[:args.k]]
    t = emb(crops).mean(0)
    t /= np.linalg.norm(t)
    d = gallery_dir(args.model)
    np.save(os.path.join(d, args.name + ".npy"), t)
    sheet(crops, os.path.join(d, args.name + "_faces.jpg"))
    print("enrolled %s from %d face crops -> gallery/%s/%s.npy" % (args.name, len(crops), args.model, args.name))


if __name__ == "__main__":
    main()
