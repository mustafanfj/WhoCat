"""Set up the model weights for the demo (no training, no download needed).

  python demo/get_weights.py

GitHub does not accept files over 100 MB, so the weights are stored in this repository as pieces
under weights/. This script joins them, checks them against their SHA-256, and puts them where
demo/server.py looks:
  weights/arc_coat_door/best.pt.part*  ->  checkpoints/arc_coat_door/best.pt   (our WhoCat model)
  weights/petface/cat.pt.part*         ->  checkpoints/petface/cat.pt          (PetFace's model, if present)
Uses only the Python standard library.
"""
import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from whocat import paths  # noqa: E402

WEIGHTS = REPO / "weights"
SHA256 = {
    "arc_coat_door/best.pt": "8fb0189c49b41d300ba78030953fc2756fd9c5d95b6dc4c6f2cca17eb95750d6",
    "petface/cat.pt": "f167eb84add4f8d4f39115290da02700a501a5f6ba2b64489b5306f6f7ef92a6",
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def assemble(rel):
    out = paths.CKPT / rel
    parts = sorted((WEIGHTS / rel).parent.glob(Path(rel).name + ".part*"))
    if out.exists() and sha256(out) == SHA256[rel]:
        print(f"already there: {out}")
        return True
    if not parts:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"joining {len(parts)} pieces -> {out}")
    with open(out, "wb") as f:
        for p in parts:
            with open(p, "rb") as src:
                shutil.copyfileobj(src, f, 1 << 22)
    if sha256(out) != SHA256[rel]:
        out.unlink()
        sys.exit(f"{rel} is corrupted (checksum mismatch). Download the repository again: "
                 "git clone, or Code > Download ZIP (a partial download breaks the pieces).")
    return True


def main():
    if not assemble("arc_coat_door/best.pt"):
        sys.exit("weights/arc_coat_door is missing. Download the full repository again.")
    shutil.copy(WEIGHTS / "arc_coat_door" / "config.json", paths.CKPT / "arc_coat_door" / "config.json")
    print(f"Our WhoCat model is ready in {paths.CKPT / 'arc_coat_door'}")
    if assemble("petface/cat.pt"):
        print(f"PetFace's model is ready in {paths.CKPT / 'petface'}: the demo compares both doors.")
    else:
        print("PetFace's model is not in this copy (optional). The demo compares a plain threshold with\n"
              f"our door rule. To add PetFace's door, save PetFace's released cat model as {paths.PETFACE_WEIGHTS}")
    print("\nNext:  python demo/server.py   then open http://localhost:8123")


if __name__ == "__main__":
    main()
