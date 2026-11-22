"""E12a: which shortcut does each model use? Change one thing in the photo, keep the cat.

  background removed : enrolment AND probe photos with everything outside the head oval greyed out
  lighting changed   : probe photo brighter/darker and warmer/cooler (colour and sharpness kept)
  night-style        : probe photo grey + dark (existing 'gray' variant, for reference)
Each is compared with clean photos on the same trials (paired cluster bootstrap, 95% CI).
A model that leans on background drops on the first; one that leans on lighting drops on the second.

  python scripts/03_embed.py --models <models> --variants masked light
  python scripts/13_shortcut_tests.py --models <models>
Outputs: outputs/results/e12_shortcuts.csv, summary_shortcuts.md
"""
import _boot  # noqa: F401
import argparse

import numpy as np
import pandas as pd

from whocat import metrics as M, paths, protocols
from whocat.embed import cache_file
from whocat.models import cache_name

TESTS = [("background removed", "masked", "masked"), ("lighting changed", "clean", "light"),
         ("night-style", "clean", "gray")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    rows = []
    for m in args.models:
        if not cache_file(m, "clean").exists():
            print("no cache for", m); continue
        clean = np.load(cache_file(m, "clean"))
        for p in ("P1", "P3"):
            t = protocols.load(p, "test")
            y = t["label"]
            s0 = protocols.score(t, clean, clean)
            for name, ve, vp in TESTS:
                if not (cache_file(m, ve).exists() and cache_file(m, vp).exists()):
                    continue
                s1 = protocols.score(t, np.load(cache_file(m, ve)), np.load(cache_file(m, vp)))
                d, lo, hi = M.paired_diff(M.tar_fn(s1, y, 0.01), M.tar_fn(s0, y, 0.01), t["owner"])
                rows.append(dict(model=cache_name(m), protocol=p, test=name,
                                 tar_clean=100 * M.tar_at_far(s0, y, 0.01)[0], tar_changed=100 * M.tar_at_far(s1, y, 0.01)[0],
                                 change=100 * d, lo=100 * lo, hi=100 * hi, significant=bool(lo > 0 or hi < 0)))
                print({k: (round(x, 2) if isinstance(x, float) else x) for k, x in rows[-1].items()})
    df = pd.DataFrame(rows)
    df.to_csv(paths.RESULTS / "e12_shortcuts.csv", index=False)
    md = ["# E12a shortcut tests (TAR@FAR=1%, test cats; change = changed minus clean, points)\n",
          df.round(2).to_markdown(index=False)]
    (paths.RESULTS / "summary_shortcuts.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
