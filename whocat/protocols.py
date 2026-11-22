"""Verification protocols as (enrolment set, probe) trials over the eval-image order.

Every protocol is stored the same way, so one scorer handles all of them:
  enroll  (n, 5) int   eval-image indices of the enrolment photos (-1 = unused); clean images
  probe   (n,)   int   eval-image index of the probe (may be scored on a degraded variant)
  label   (n,)   int   1 = same cat, 0 = different cat
  owner   (n,)   int   identity code of the enrolled cat (bootstrap cluster)
  other   (n,)   int   identity code of the probe cat
P1-P3 use one enrolment photo (plain pair verification); P5 uses 1, 3 or 5.
"""
import numpy as np

from . import paths

KMAX = 5


def save(name, split, **arrs):
    np.savez_compressed(paths.PAIRS / f"{split}_{name}.npz", **arrs)


def load(name, split):
    z = np.load(paths.PAIRS / f"{split}_{name}.npz")
    return {k: z[k] for k in z.files}


def exists(name, split):
    return (paths.PAIRS / f"{split}_{name}.npz").exists()


def score(trials, emb_enroll, emb_probe):
    """Cosine between the mean enrolment template and the probe."""
    E = trials["enroll"]
    mask = (E >= 0)[..., None]
    T = (emb_enroll[np.clip(E, 0, None)] * mask).sum(1)
    T /= np.clip(np.linalg.norm(T, axis=1, keepdims=True), 1e-12, None)
    return (T * emb_probe[trials["probe"]]).sum(1)
