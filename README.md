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
The perception pipeline now includes automatic detection of horizontal planar surfaces
(e.g. tables, floors) using depth data. When a surface is detected, its boundary is drawn
in magenta and its 3D position/extent is printed.

```bash
# Real camera - surface detection runs automatically
python perception/main.py

# Simulation with synthetic depth (flat plane at fixed distance)
python perception/main.py --sim ~/media/desk.jpg --sim-depth-m 1.0

# Simulation with a real depth map (16-bit PNG in millimetres)
python perception/main.py --sim ~/media/desk.jpg --sim-depth depth.png
```

Surface detection parameters (in `perception/surface.py`):
- `max_height` / `min_height`: height range of surface above camera (default 1.5m / 0.2m)
- `angle_thresh_deg`: max tilt from horizontal (default 15°)
- `min_area_m2`: minimum surface area (default 0.05 m²)
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
