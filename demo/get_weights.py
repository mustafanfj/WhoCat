"""Download our trained WhoCat model for the demo (no training needed).

  python demo/get_weights.py

Saves best.pt and config.json into <project>/checkpoints/arc_coat_door/, next to this repository
folder, which is where demo/server.py looks. Uses only the Python standard library.
"""
import sys
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/mustafanfj/WhoCat/releases/download/v1.0"
FILES = ["config.json", "best.pt"]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from whocat import paths  # noqa: E402

DEST = paths.CKPT / "arc_coat_door"


def fetch(name):
    out = DEST / name
    if out.exists() and out.stat().st_size > 0:
        print(f"already there: {out}")
        return
    tmp = out.with_suffix(out.suffix + ".part")
    url = f"{RELEASE}/{name}"
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 2**20:.0f} / {total / 2**20:.0f} MB", end="", flush=True)
    print()
    tmp.replace(out)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    try:
        for name in FILES:
            fetch(name)
    except Exception as ex:  # noqa: BLE001
        sys.exit(f"Download failed ({ex}).\nGet best.pt and config.json by hand from "
                 f"https://github.com/mustafanfj/WhoCat/releases and put them in {DEST}")
    print(f"\nOur model is ready in {DEST}.")
    pf = paths.PETFACE_WEIGHTS
    if pf.exists():
        print(f"PetFace's model found at {pf}: the demo will compare both doors.")
    else:
        print("Optional: to also show the PetFace door, download PetFace's released cat model\n"
              "(https://github.com/mapooon/PetFace, research use) and save it as\n"
              f"  {pf}")
    print("\nNext:  python demo/server.py   then open http://localhost:8123")


if __name__ == "__main__":
    main()
