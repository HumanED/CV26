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
### Finetuning
YOLO can be finetuned to detect and outline your own objects. Record one video per object,
roughly centred in view and from several angles and distances, and name it after the object:
```
dataset/video/
    bottle.mp4  can.mp4  wrapper.mp4 ...
```
Build the dataset, then train (or follow the same steps in `perception/finetune.ipynb`):
```bash
python dataset/build_dataset.py              # frames -> SAM 2 outlines -> train/val/test split
python perception/finetune.py                # writes models/yolo11s-seg-custom.pt
```
`build_dataset.py` keeps every 5th frame (`--every`), outlines the object at the image centre with
SAM 2 and splits each object's frames 70/20/10 into train/val/test (`--ratios`). Check the outlines in
`dataset/preview/` before training, and delete the image and label of any frame SAM got wrong. Rebuild
with `--overwrite`. Training logs and plots go to `runs/`.
### Calibration
TODO: make the parol6 arm go to home at the start of program and get coordinates from perception.
## Copyright
Check [LICENSE](LICENSE)
