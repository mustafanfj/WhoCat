"""Door-camera degradations.

Test-time (P4) ranges are deliberately disjoint from the training-augmentation ranges,
so a model trained with augmentation cannot simply memorise the test corruptions.
"""
import io
import random

import cv2
import numpy as np
from PIL import Image

TEST = dict(size=(48, 64), blur=(5, 9), gamma=(1.5, 2.2), gain=(0.55, 0.75), jpeg=(20, 40), noise=10.0)
TRAIN = dict(size=(72, 128), blur=(3, 3), gamma=(1.0, 1.4), gain=(0.8, 1.0), jpeg=(50, 90), noise=5.0)


def _shrink(img, s):
    w, h = img.size
    return img.resize((s, s), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def _motion_blur(img, k, angle):
    kern = np.zeros((k, k), np.float32)
    kern[k // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kern = cv2.warpAffine(kern, rot, (k, k))
    kern /= max(kern.sum(), 1e-6)
    return Image.fromarray(cv2.filter2D(np.asarray(img), -1, kern))


def _darken(img, gamma, gain):
    a = np.asarray(img).astype(np.float32) / 255.0
    a = np.clip(gain * a ** gamma, 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8))


def _jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(q))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gray_ir(img, rng, sigma):
    """Night-IR proxy: grayscale plus sensor noise."""
    g = np.asarray(img.convert("L")).astype(np.float32)
    g = np.clip(g + rng.normal(0, sigma, g.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([g] * 3, -1))


def door(img, rng, r, p_each=1.0, p_gray=0.5):
    """Random combination of door-camera corruptions with ranges r (TEST or TRAIN)."""
    if rng.random() < p_each:
        img = _shrink(img, rng.randint(*r["size"]))
    if rng.random() < p_each:
        k = rng.randrange(r["blur"][0], r["blur"][1] + 1, 2)
        img = _motion_blur(img, k, rng.uniform(0, 180))
    if rng.random() < p_each:
        img = _darken(img, rng.uniform(*r["gamma"]), rng.uniform(*r["gain"]))
    if rng.random() < p_gray:
        img = gray_ir(img, np.random.default_rng(rng.randint(0, 2**31)), r["noise"])
    if rng.random() < p_each:
        img = _jpeg(img, rng.randint(*r["jpeg"]))
    return img


# Head oval in the aligned 224 px PetFace frame (from the keypoint template): keeps eyes, nose, cheeks,
# forehead and most of the ears; outside it is mostly background in the corners.
OVAL = dict(cx=112, cy=122, rx=104, ry=104, feather=4)
_MASK = None


def oval_mask(size=224):
    """Float mask in [0, 1], 1 inside the head oval, soft 4 px edge."""
    global _MASK
    if _MASK is None or _MASK.shape[0] != size:
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) * (224.0 / size)
        d = np.sqrt(((xx - OVAL["cx"]) / OVAL["rx"]) ** 2 + ((yy - OVAL["cy"]) / OVAL["ry"]) ** 2)
        _MASK = np.clip((1.0 - d) * OVAL["rx"] / OVAL["feather"] + 0.5, 0, 1)[..., None]
    return _MASK


def paste_on(img, bg):
    """The head from img on top of background bg (both PIL RGB, same size)."""
    m = oval_mask(img.size[0])
    out = np.asarray(img, np.float32) * m + np.asarray(bg, np.float32) * (1 - m)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def relight(img, rng):
    """Lighting only: brighter or darker, warmer or cooler; colour and sharpness kept."""
    a = np.asarray(img, np.float32) / 255.0
    gamma = rng.uniform(1.4, 2.0) if rng.random() < 0.5 else rng.uniform(0.55, 0.75)
    a = a ** gamma * rng.uniform(0.75, 1.1)
    t = rng.uniform(0.12, 0.2) * (1 if rng.random() < 0.5 else -1)       # white balance shift
    a = a * np.array([1 + t, 1.0, 1 - t], np.float32)
    return Image.fromarray((a.clip(0, 1) * 255).astype(np.uint8))


def degrade(img, variant, seed=0):
    rng = random.Random(seed)
    if variant == "masked":     # E12a: background outside the head oval greyed out
        return paste_on(img, Image.new("RGB", img.size, (128, 128, 128)))
    if variant == "light":      # E12a: lighting / white balance only
        return relight(img, rng)
    if variant == "gray":       # night-style probe only
        return gray_ir(img, np.random.default_rng(seed), TEST["noise"])
    if variant == "door":       # P4: each corruption with p=0.6 (usually 2-4 at once), half in grayscale
        return door(img, rng, TEST, p_each=0.6, p_gray=0.5)
    if variant == "small":      # just the resolution drop
        return _shrink(img, rng.randint(*TEST["size"]))
    raise ValueError(variant)


class TrainDoorAug:
    """Training-time door augmentation (milder, disjoint ranges), applied with probability p."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        return door(img, random.Random(random.getrandbits(32)), TRAIN, p_each=0.5, p_gray=0.25)


class BgSwap:
    """E12b training augmentation: with probability p, keep the head oval and replace the background
    with another random training photo's background (or, half the time, flat grey / noise)."""

    def __init__(self, rel_paths, root, p=0.5):
        self.paths, self.root, self.p = list(rel_paths), root, p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        if random.random() < 0.5:
            other = Image.open(self.root / random.choice(self.paths)).convert("RGB")
            bg = other.resize(img.size, Image.BICUBIC) if other.size != img.size else other
        elif random.random() < 0.5:
            bg = Image.new("RGB", img.size, tuple(random.randint(0, 255) for _ in range(3)))
        else:
            bg = Image.fromarray(np.random.randint(0, 256, (img.size[1], img.size[0], 3), np.uint8))
        return paste_on(img, bg)
