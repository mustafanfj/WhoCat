# WhoCat demo

A front-door simulator. One visitor, two cat doors side by side, each showing its decision, both checks and every score:
- the **WhoCat door**: our model with the full door rule (a threshold tuned on look-alike cats, and a memory of the neighbours it has turned away);
- the other door: our model with a plain threshold, or PetFace's model, depending on what is installed. Change either door under **Adjust the doors**.

## Quick start (any laptop)

```
git clone https://github.com/mustafanfj/WhoCat.git
cd WhoCat
pip install -r demo/requirements.txt
python demo/get_weights.py        # downloads our model (once)
python demo/server.py             # then open http://localhost:8123
```

Open the page as `http://localhost:8123`: browsers allow the camera only on `localhost` or `https`. The models run on the CPU if there is no GPU; a photo takes about one to three seconds. No internet is needed after the download.

`get_weights.py` puts `best.pt` and `config.json` in `checkpoints/arc_coat_door/` inside the repository. If the download is blocked, get both files from https://github.com/mustafanfj/WhoCat/releases and put them there by hand.

## What it can show

| You have | What the demo does |
|---|---|
| Our model only (the quick start) | Your own cat: webcam or uploaded photos, our model with a plain threshold vs the full door rule, neighbour memory, door-camera view, the paper's scoreboard |
| + PetFace's cat model in `checkpoints/petface/cat.pt` | Also PetFace's model as the second door, on every photo |
| + the sample-home bundle in `demo/web/bundle/` | Also 12 held-out PetFace homes, the five-scene guided tour, and a scoreboard counted on those homes |

PetFace's model comes from https://github.com/mapooon/PetFace. The bundle is made from the PetFace dataset with `python demo/export_bundle.py --ours ckpt:<path>/best.pt` (needs the dataset and our embedding caches). Both are under PetFace's non-commercial research licence, so they are not in this repository.

## Other ways to run it

- **Sample homes without Python models:** with the bundle in place, `python -m http.server 8123 --directory demo/web --bind 127.0.0.1` serves the page; the tour and scoreboard work, the camera does not.
- **One file:** `python demo/build_single.py` writes `demo/WhoCat_demo.html` (sample homes, opens by double-click).
- **On a remote GPU machine:** run `python demo/server.py` there and `ssh -L 8123:localhost:8123 <machine>` on your laptop, then open http://localhost:8123 on the laptop; the camera works because the page is still `localhost`.

## How it decides

Photos go through OpenCV's cat-face detector, cropped the way PetFace's aligned faces are framed (calibrated on 600 PetFace validation images); if no face is found, the centre square is used. Each model turns the face into a 512-d embedding; the enrolment embeddings are averaged into the owner template. The page then applies the door rule in the browser with the thresholds from `demo/web/models.json`, which were tuned on PetFace validation homes. Nothing is written to disk.
