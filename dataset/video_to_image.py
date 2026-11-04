"""Save the frames of a video as JPEG images.

build_dataset.py calls extract_frames() for every video in dataset/video/. It can
also be run on its own, from any folder:
    python dataset/video_to_image.py dataset/video/bottle.mp4            # -> dataset/frames/bottle/
    python dataset/video_to_image.py dataset/video/bottle.mp4 --every 5  # keep every 5th frame
"""
import argparse
from pathlib import Path

import cv2

DATASET_DIR = Path(__file__).resolve().parent  # dataset/, so paths work from any folder
FRAMES_DIR = DATASET_DIR / "frames"  # default output: frames/<video name>/


def extract_frames(video: str | Path, out_dir: str | Path, every: int = 1) -> list[Path]:
    """Save every `every`-th frame of `video` to out_dir as <video name>_<frame>.jpg.
    Returns the saved paths in frame order."""
    if every < 1:
        raise ValueError(f"every must be 1 or more, got {every}")
    video, out_dir = Path(video), Path(out_dir)

    cam = cv2.VideoCapture(str(video))
    try:
        if not cam.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        out_dir.mkdir(parents=True, exist_ok=True)

        saved: list[Path] = []
        index = 0  # frame number in the video
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            if index % every == 0:
                path = out_dir / f"{video.stem}_{index:05d}.jpg"
                if not cv2.imwrite(str(path), frame):
                    raise OSError(f"Could not write {path}")
                saved.append(path)
            index += 1
    finally:
        cam.release()
    return saved


def parse_args() -> argparse.Namespace:
    """Parse the video path and frame options."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", help="video file to convert")
    ap.add_argument("--out", metavar="DIR",
                    help="output folder (default dataset/frames/<video name>/)")
    ap.add_argument("--every", type=int, default=1, help="keep every Nth frame (default 1)")
    return ap.parse_args()


def main() -> None:
    """Extract frames from one video and report how many were saved."""
    args = parse_args()
    out = Path(args.out) if args.out else FRAMES_DIR / Path(args.video).stem
    saved = extract_frames(args.video, out, args.every)
    print(f"Saved {len(saved)} frames to {out}")


if __name__ == "__main__":
    main()
