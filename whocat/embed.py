"""Batch embedding with caching."""
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import paths
from .data import ImageList, load_eval_images
from .models import build, cache_name


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def n_workers():
    return int(os.environ.get("WHOCAT_WORKERS", min(16, os.cpu_count() or 4)))


@torch.no_grad()
def embed_paths(model, rel_paths, variant="clean", batch=256, desc=""):
    dev = device()
    model = model.to(dev).eval()
    dl = DataLoader(ImageList(rel_paths, model.transform, variant), batch_size=batch,
                    num_workers=n_workers(), pin_memory=dev == "cuda")
    out = None
    amp = torch.autocast("cuda", dtype=torch.float16) if dev == "cuda" else torch.autocast("cpu", enabled=False)
    for x, idx in tqdm(dl, desc=desc or variant, mininterval=10):
        with amp:
            e = model(x.to(dev, non_blocking=True))
        e = F.normalize(e.float(), dim=-1).cpu().numpy()
        if out is None:
            out = np.zeros((len(rel_paths), e.shape[1]), np.float32)
        out[idx.numpy()] = e
    return out


def cache_file(model_name, variant):
    return paths.CACHE / f"emb_{cache_name(model_name)}_{variant}.npy"


def get_eval_embeddings(model_name, variant="clean", model=None, batch=256):
    """Embeddings of all val+test images (fixed order of eval_images.csv.gz), cached on disk."""
    f = cache_file(model_name, variant)
    if f.exists():
        return np.load(f)
    model = model or build(model_name)
    ev = load_eval_images()
    e = embed_paths(model, ev.path.tolist(), variant, batch, desc=f"{cache_name(model_name)}/{variant}")
    np.save(f, e)
    return e
