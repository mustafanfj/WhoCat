"""Project paths. PROJECT root = $WHOCAT_ROOT; else this repository folder when it has its own
checkpoints/ (a demo copy); else the parent of the code/ folder when that holds data/ or checkpoints/
(the training layout); else this repository folder (a fresh clone)."""
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_PARENT = _REPO.parent
ROOT = Path(os.environ.get("WHOCAT_ROOT") or (
    _REPO if (_REPO / "checkpoints").is_dir()
    else _PARENT if (_PARENT / "checkpoints").is_dir() or (_PARENT / "data").is_dir() else _REPO))
DATA = ROOT / "data"
PETFACE = DATA / "PetFace"
IMAGES = PETFACE / "images"            # contains cat/<id>/<nn>.png
SPLIT = PETFACE / "split" / "cat"
ANNOT = PETFACE / "annotations" / "cat.csv"
OUT = ROOT / "outputs"
INDEX = OUT / "index"                  # image / identity tables
CACHE = OUT / "cache"                  # cached embeddings
PAIRS = OUT / "pairs"                  # protocol pair lists
RESULTS = OUT / "results"
FIGS = OUT / "figures"
CKPT = ROOT / "checkpoints"
PETFACE_WEIGHTS = CKPT / "petface" / "cat.pt"

# Aligned-face landmarks of PetFace's cat template (224x224): eyes, nose, mouth corners.
CAT_KEYPOINTS = [(56.0, 114.8), (168.0, 114.6), (111.2, 156.4), (76.2, 183.5), (147.3, 183.5)]
# Inner face = eyes-nose-mouth box with a margin; used for the Grad-CAM inner-face score.
INNER_FACE_BOX = (40, 95, 184, 200)    # x0, y0, x1, y1

for p in (OUT, INDEX, CACHE, PAIRS, RESULTS, FIGS):
    p.mkdir(parents=True, exist_ok=True)
