# WhoCat demo

A front-door simulator. One visitor, two cat flaps side by side: the **normal door** (PetFace's face model, one threshold) and the **WhoCat door** (our model, a threshold tuned on look-alike cats, and a memory of neighbours it has kept out).

- **Sample homes**: 12 homes of PetFace validation cats, each with its owner and look-alike neighbours. Pick a visitor and see both doors' checks, scores and decisions. Works without the models.
- **Your own cat** (live server only): take one to three photos of your cat with the webcam or upload them, then photograph visitors. Mark a visitor as a neighbour to see the memory at work.

Every decision is computed in the browser from the 512-d face embeddings; the server only turns photos into embeddings.

## Run it on your own computer (Windows, macOS or Linux)

You need Python 3.10+, this repository, the exported bundle and two weight files.

```
project/
  code/                                  this repository (any folder name works)
    demo/web/bundle/                     the exported bundle: bundle.json, img/, img_door/
  checkpoints/
    arc_coat_door/best.pt, config.json   our model
    petface/cat.pt                       PetFace's released cat model
```

```bash
cd project/code
pip install torch torchvision timm opencv-python pillow numpy
python demo/server.py --port 8123
```

Open **http://localhost:8123**. The browser allows the camera only on `localhost` or `https`, so use that address and not the computer's IP. Stop with Ctrl+C. The models run on the CPU if there is no GPU; each photo takes one to three seconds.

On Windows PowerShell, if the weights sit somewhere else, point the server at the project folder first: `$env:WHOCAT_ROOT = "C:\path\to\project"`.

## One-file version (sample homes only)

`python demo/build_single.py` packs the page, the bundle and all images into `demo/WhoCat_demo.html`, which opens by double-click. It has no camera mode, since the models are not inside it.

## Static mode (sample homes only)

```bash
python -m http.server 8123 --directory demo/web --bind 127.0.0.1
```

## Run it on the lab machine instead

```bash
ssh -L 8123:localhost:8123 <lab-machine>
python demo/server.py --port 8123      # on the lab machine
```

Then open http://localhost:8123 on your laptop; the camera works because the page is still `localhost`.

## Making the bundle

`python demo/export_bundle.py --ours ckpt:../checkpoints/arc_coat_door/best.pt` on the training machine (needs the embedding caches). The bundle holds PetFace face crops, so it is not committed; keep it for research use as the PetFace licence requires.

Photos you take go through the OpenCV cat-face detector with the crop calibrated on PetFace's aligned faces; if no face is found the centre square is used. Nothing is written to disk.
