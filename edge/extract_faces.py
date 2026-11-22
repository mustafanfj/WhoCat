"""Step 1 for the campus videos: find the cat's face in every few frames and save 224x224 crops.

  python extract_faces.py --videos /path/to/videos          # videos/<cat_name>/<clip>.mp4 (or .mov)
Writes faces/<cat_name>/<clip>/*.jpg, faces/summary.csv and faces/<cat_name>/<clip>_sheet.jpg (check these!).
"""
import argparse
import csv
import os

import cv2
import numpy as np

from whocat_edge import HERE, Detector, load_config, sharpness

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def sheet(crops, path, n=24):
    crops = crops[:n]
    if not crops:
        return
    cols = 8
    rows = (len(crops) + cols - 1) // cols
    canvas = np.full((rows * 112, cols * 112, 3), 255, np.uint8)
    for i, c in enumerate(crops):
        r, k = divmod(i, cols)
        canvas[r * 112:(r + 1) * 112, k * 112:(k + 1) * 112] = cv2.resize(c, (112, 112))
    cv2.imwrite(path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder with one sub-folder per cat")
    ap.add_argument("--out", default=os.path.join(HERE, "faces"))
    ap.add_argument("--every", type=int, default=3, help="look at every N-th frame")
    ap.add_argument("--max-faces", type=int, default=120, help="per clip")
    ap.add_argument("--min-sharp", type=float, default=20.0, help="drop very blurry crops (Laplacian variance)")
    args = ap.parse_args()
    det = Detector(load_config())
    rows = []
    for cat in sorted(os.listdir(args.videos)):
        cdir = os.path.join(args.videos, cat)
        if not os.path.isdir(cdir):
            continue
        for clip in sorted(os.listdir(cdir)):
            if not clip.lower().endswith(VIDEO_EXT):
                continue
            name = os.path.splitext(clip)[0]
            odir = os.path.join(args.out, cat, name)
            if not os.path.isdir(odir):
                os.makedirs(odir)
            cap = cv2.VideoCapture(os.path.join(cdir, clip))
            i = n_frames = n_faces = 0
            kept = []
            while len(kept) < args.max_faces:
                ok, frame = cap.read()
                if not ok:
                    break
                i += 1
                if i % args.every:
                    continue
                n_frames += 1
                b = det.boxes(frame)
                if not b:
                    continue
                n_faces += 1
                crop = det.crop(frame, b[0])
                if sharpness(crop) < args.min_sharp:
                    continue
                cv2.imwrite(os.path.join(odir, "f%06d.jpg" % i), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                kept.append(crop)
            cap.release()
            sheet(kept, os.path.join(args.out, cat, name + "_sheet.jpg"))
            rows.append(dict(cat=cat, clip=name, frames_checked=n_frames, frames_with_face=n_faces, crops_kept=len(kept)))
            print("%-20s %-20s frames %5d  face found %5d  kept %4d" % (cat, name, n_frames, n_faces, len(kept)))
    with open(os.path.join(args.out, "summary.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nOpen the *_sheet.jpg files and delete any crop that is not the right cat's face before the next step.")


if __name__ == "__main__":
    main()
