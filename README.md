# Computer Vision -- PAROL6 camera-in-hand project for object sorting and classification
## About

## Requirements
- Python 3.11+
- [Intel RealSense SDK 2.0](https://github.com/realsenseai/librealsense/)
- [PAROL6 API](https://github.com/PCrnjak/PAROL6-python-API)
#### For real-life testing
- Intel RealSense Depth Camera D435
- Source Robotics PAROL6 robotic arm
## Installation
The code should theoretically be OS-agnostic but for most stability it is recommended to use it in a Linux environment.

Enter an env like conda or venv before installing to avoid package conflicts.

```bash
pip install -r requirements.txt
```
> [!NOTE]
> if you have an NVIDIA GPU install the CUDA build of torch from pytorch.org first

RealSense Viewer should be used to test the hardware before starting the perception program 
## Usage
### Perception
How to use the object detection software with hardware and simulation
```bash
python perception/main.py                                  # real D435
python perception/main.py --sim ~/media/lab.mp4            # video
python perception/main.py --sim ~/media/desk.jpg           # single image (reused every frame)
python perception/main.py --sim ~/media/frames/ --loop     # folder of images, looping
python perception/main.py --sim lab.mp4 --sim-depth-m 0.8  # change the fake distance
python perception/main.py --sim lab.mp4 --sim-depth d.png  # real depth map, 16-bit PNG in mm
python perception/main.py --model models/yolo11s-seg-custom.pt  # finetuned weights
```

#### Surface Detection (Table/Plane Recognition)
The perception pipeline finds the surface the camera is directly above, i.e. the table the arm
stands on, so the arm can use it later. It works on cluttered, slightly warped and irregularly
shaped surfaces: up-facing depth pixels are grouped by height, each candidate plane is fitted
robustly and grown with a tolerance that follows the D435's depth noise, and the candidate nearest
the point straight below the camera is kept (so the floor or a box on the table is not chosen).
Objects standing on the surface are cut out of it; the usable area with them filled in is
shaded magenta and its 3D position, extent, area and flatness are printed. The plane is
smoothed over frames.

```bash
# Real camera - surface detection runs automatically
python perception/main.py

# Camera tilted 60 degrees down (90 = looking straight down)
python perception/main.py --surface-pitch 60

# Simulation with a real depth map (16-bit PNG in millimetres)
python perception/main.py --sim ~/media/desk.jpg --sim-depth depth.png
```

Surface detection options (`perception/main.py`, defaults in `perception/surface.py`):
- `--surface-pitch`: camera tilt below horizontal in degrees, used to know which way is up (default 0)
- `--surface-min-height` / `--surface-max-height`: distance of the surface below the camera (default 0.2m / 1.5m)
- `--surface-angle-thresh`: max tilt from horizontal (default 15°)
- `--surface-min-area`: minimum visible surface area (default 0.05 m²)
- `--surface-tol`: how uneven the surface may be; objects taller than this are cut out (default 0.015m)
### Finetuning
YOLO is finetuned for instance segmentation (an outline per object) on your own objects. The
labelled images are made elsewhere, as rendered views of the 3D models or as frames labelled in
[CVAT](https://www.cvat.ai/) and exported as YOLO segmentation. Put them in `dataset/source/`:
```
dataset/source/
    classes.txt          class names, one per line, in class-id order
    images/**/*.jpg
    labels/**/*.txt      same path and name as the image: "<class id> x1 y1 x2 y2 ..." (0-1)
```
Build the dataset, then train (or follow the same steps in `perception/finetune.ipynb`):
```bash
python dataset/build_dataset.py              # check labels, split 70/20/10, write data.yaml
python dataset/build_dataset.py --by-folder  # keep each folder under images/ in one split
python perception/finetune.py                # writes models/yolo11s-seg-custom.pt
```
Labels must be polygons; bounding-box-only labels are rejected. Images with no label file are kept as
background images. Check the outlines in `dataset/preview/` before training. Rebuild with `--overwrite`.
Training logs and plots go to `runs/`.
### Calibration
TODO: make the parol6 arm go to home at the start of program and get coordinates from perception.
## Copyright
Check [LICENSE](LICENSE)
