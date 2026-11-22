"""Index tables and image datasets."""
import hashlib

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import paths
from .degrade import degrade


def load_images():
    """All indexed images: columns path, id, split (train/val/test)."""
    return pd.read_csv(paths.INDEX / "images.csv.gz", dtype={"id": str})


def load_ids():
    """One row per identity: id, split, n_imgs, coat, coat_conf, petface colour/breed."""
    return pd.read_csv(paths.INDEX / "ids.csv.gz", dtype={"id": str})


def load_eval_images():
    """Val + test images in the fixed order used by every embedding cache."""
    return pd.read_csv(paths.INDEX / "eval_images.csv.gz", dtype={"id": str})


def stable_seed(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


class ImageList(Dataset):
    """Loads aligned 224x224 faces, optionally degrades them, then applies a model transform."""

    def __init__(self, rel_paths, transform, variant="clean", root=None):
        self.rel_paths = list(rel_paths)
        self.transform = transform
        self.variant = variant
        self.root = root or paths.IMAGES

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, i):
        rel = self.rel_paths[i]
        img = Image.open(self.root / rel).convert("RGB")
        if img.size != (224, 224):
            img = img.resize((224, 224), Image.BICUBIC)
        if self.variant != "clean":
            img = degrade(img, self.variant, seed=stable_seed(rel + self.variant))
        return self.transform(img), i


def l2n(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def collate_skip_none(batch):
    return torch.utils.data.default_collate([b for b in batch if b is not None])
