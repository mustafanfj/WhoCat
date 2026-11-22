# WhoCat edge package

Everything needed to run the WhoCat cat-face door on a **Raspberry Pi 4** or a **Jetson Nano**, and to test it
on the campus cat videos. No PyTorch and no dataset are needed: only Python, numpy, OpenCV and onnxruntime.

```
models/                 the face models (ONNX) and the cat-face detector
  whocat_small_*.onnx     MobileNetV3, made for the edge (fp32 and int8 = 8-bit, about 4x smaller)
  whocat_big_*.onnx       DINOv2-B, our most accurate model (for comparison; slow on the Pi)
config.json             image size, normalisation, decision thresholds, detector crop settings
ACCURACY_ON_PETFACE.md  accuracy of each file on PetFace (checked on a GPU machine before export)
whocat_edge.py          shared code (detector, crop, model, door decision)
extract_faces.py        step 1 for the videos: find and save cat-face crops
eval_videos.py          step 2 for the videos: the campus-cat test (E6)
benchmark.py            speed test on the device (E7)
enroll.py, door.py      live demo: enroll a cat, then run the door on a camera or a video
```

## 1. Set up the device

**Raspberry Pi 4** (Raspberry Pi OS **64-bit**; check with `uname -m`, it must print `aarch64`):
```bash
sudo apt update && sudo apt install -y python3-venv python3-picamera2
python3 -m venv ~/whocat-env --system-site-packages
source ~/whocat-env/bin/activate
pip install numpy onnxruntime opencv-python-headless
```

**Jetson Nano** (JetPack 4.6, Python 3.6):
```bash
sudo apt update && sudo apt install -y python3-pip
pip3 install --upgrade pip
# onnxruntime with GPU support: download the onnxruntime_gpu wheel for your JetPack version and Python 3.6
# from the "ONNX Runtime" section of the Jetson Zoo page (elinux.org/Jetson_Zoo), then:
pip3 install onnxruntime_gpu-*.whl
python3 -c "import cv2, onnxruntime as o; print(cv2.__version__, o.get_available_providers())"
```
OpenCV comes with JetPack. The last line should list `CUDAExecutionProvider`.

Copy this folder to the device (USB stick or `scp -r edge_package pi@<ip>:~/`), then `cd edge_package`.

## 2. Speed test (E7), on each device
```bash
python3 benchmark.py --n 100                       # quick, synthetic frames
python3 benchmark.py --n 100 --video some_clip.mp4 # with a real cat video
```
On the Jetson also run it with `--cpu-only` (CPU vs GPU), and the TensorRT speed test:
```bash
/usr/src/tensorrt/bin/trtexec --onnx=models/whocat_small_fp32.onnx --fp16 --shapes=image:1x3x224x224
/usr/src/tensorrt/bin/trtexec --onnx=models/whocat_big_fp32.onnx --fp16 --shapes=image:1x3x224x224
```
(copy the "mean" GPU compute time from the end of each trtexec output).
Keep the device cool and plugged into its proper power supply; note if it has a fan.

## 3. Campus cat videos (E6), on any computer (a laptop is fine)
Put the videos in one folder per cat; each cat needs **at least 2 clips**, ideally filmed at different times:
```
videos/ginger1/clip1.mp4   videos/ginger1/clip2.mp4
videos/gray_tom/clip1.mp4  videos/gray_tom/clip2.mp4 ...
```
The first clip of each cat (alphabetical order) is used to enroll it; the others are visits.
Optionally write `similar.txt` here, one pair of similar-looking cats per line: `ginger1,ginger2`.

```bash
python3 extract_faces.py --videos /path/to/videos
```
**Check the crops:** open `faces/<cat>/<clip>_sheet.jpg`. Delete any crop in `faces/<cat>/<clip>/` that is not
that cat's face (another cat, a wrong box). If a clip finds very few faces, see "Troubleshooting".
```bash
python3 eval_videos.py --models small big
```
Results: `results/e6_summary.md`.

## 4. Live demo (optional)
```bash
python3 enroll.py --name ginger1 --source videos/ginger1/clip1.mp4
python3 enroll.py --name ginger2 --source videos/ginger2/clip1.mp4
python3 door.py --owner ginger1 --source videos/ginger2/clip2.mp4 --show   # should STAY CLOSED
python3 door.py --owner ginger1 --source 0 --show                          # USB camera; --source picam for the Pi camera
```

## 5. What to send back
Zip and send: `results/` (all files), `faces/summary.csv`, the `faces/*/*_sheet.jpg` images, `similar.txt`
if you made one, and a note with: device models, RAM, power supply/fan, camera used, and anything that went wrong.

## Troubleshooting
- **Few faces found:** the detector only finds fairly frontal cat faces. In `config.json` → `detector`, lower
  `min_neighbors` to 2 (more faces, more wrong boxes) or `scale_factor` to 1.03. Film the cat facing the camera.
- **`onnxruntime` install fails on the Pi:** the OS is 32-bit; reinstall Raspberry Pi OS 64-bit.
- **Jetson: TensorRT error loading the model:** use `benchmark.py --cpu-only` and report the error text.
