"""Live WhoCat door: camera (or video) in, OPEN / STAY CLOSED out.

  python door.py --owner luna --source 0                # USB camera 0
  python door.py --owner luna --source clip.mp4 --show
  python door.py --owner luna --source picam            # Raspberry Pi camera module (needs picamera2)
Every other cat in the gallery is treated as a remembered neighbour (neighbour memory), unless --no-memory.
The door decides once it has seen --window face frames: mean of the 5 best owner scores vs the threshold.
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np

from whocat_edge import Detector, Embedder, decide, load_config, load_gallery, vote


def camera(source):
    if source == "picam":
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
        cam.start()
        while True:
            yield cam.capture_array()
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    while True:
        ok, f = cap.read()
        if not ok:
            return
        yield f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--source", default="0")
    ap.add_argument("--model", default="small")
    ap.add_argument("--variant", default="fp32")
    ap.add_argument("--threshold", default="lookalike", choices=["lookalike", "random"])
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    det, emb = Detector(cfg), Embedder(cfg, args.model, args.variant)
    gal = load_gallery(args.model)
    if args.owner not in gal:
        raise SystemExit("enroll %s first (enroll.py)" % args.owner)
    owner = gal.pop(args.owner)
    neigh = np.stack(list(gal.values())) if gal else np.zeros((0, owner.shape[0]))
    thr = emb.thr_lookalike if args.threshold == "lookalike" else emb.thr_random
    margin = None if args.no_memory else emb.margin
    so, sn = deque(maxlen=args.window), deque(maxlen=args.window)
    t0, n = time.perf_counter(), 0
    for f in camera(args.source):
        n += 1
        b = det.boxes(f)
        label, col = "no cat face", (180, 180, 180)
        if b:
            e = emb([det.crop(f, b[0])])[0]
            so.append(float(e @ owner))
            sn.append(neigh @ e if len(neigh) else np.zeros(0))
            if len(so) >= args.window:
                s_owner = vote(list(so))
                s_nb = [vote([x[j] for x in sn]) for j in range(len(neigh))]
                ok = decide(s_owner, s_nb, thr, margin)
                label, col = ("OPEN (%.2f)" % s_owner, (0, 170, 0)) if ok else ("STAY CLOSED (%.2f)" % s_owner, (0, 0, 220))
            else:
                label = "looking... %d/%d" % (len(so), args.window)
        fps = n / (time.perf_counter() - t0)
        print("\r%-28s %5.1f fps" % (label, fps), end="", flush=True)
        if args.show:
            for (x, y, w, h) in b[:1]:
                cv2.rectangle(f, (x, y), (x + w, y + h), col, 2)
            cv2.putText(f, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            cv2.imshow("WhoCat", f)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    print()


if __name__ == "__main__":
    main()
