"""Build a YOLO segmentation dataset from one video per object.

Each video in dataset/video/ is one class named after the file (bottle.mp4 -> "bottle").
Film each object roughly centred in view, from different angles and distances.

Steps:
    1. keep every Nth frame of each video (neighbouring frames are nearly identical)
    2. outline the object in each frame with SAM 2, prompted at the image centre
    3. split each class's frames at random into train/val/test (70/20/10 by default)
    4. write the images, polygon labels and data.yaml for perception/finetune.py

Output (all generated, ignored by git):
    dataset/frames/<class>/*.jpg            extracted frames
    dataset/{train,val,test}/images/*.jpg
    dataset/{train,val,test}/labels/*.txt   "<class id> x1 y1 x2 y2 ..." (0-1 of image size)
    dataset/preview/<class>/*.jpg           outlines drawn on each frame, for checking
    dataset/data.yaml

Look through dataset/preview/ before training. If SAM outlined the wrong thing in a
frame, delete that frame's image and label from dataset/<split>/images and labels.

Run from any folder:
    python dataset/build_dataset.py                # every 5th frame, SAM 2.1 base
    python dataset/build_dataset.py --every 10 --overwrite
"""
import argparse
import json
import random
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from video_to_image import extract_frames

DATASET_DIR = Path(__file__).resolve().parent  # dataset/, so paths work from any folder
VIDEO_DIR = DATASET_DIR / "video"  # input: one video per object
FRAMES_DIR = DATASET_DIR / "frames"
PREVIEW_DIR = DATASET_DIR / "preview"
DATA_YAML = DATASET_DIR / "data.yaml"
SPLITS = ("train", "val", "test")
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}  # same list as perception/sim.py

EVERY = 5  # keep every Nth video frame
RATIOS = (0.7, 0.2, 0.1)  # train, val, test share of each class's frames
SEED = 0  # random seed for the split, so rebuilding gives the same split
SAM_MODEL = str(DATASET_DIR.parent / "models" / "sam2.1_b.pt")  # downloaded here on first use
MIN_AREA, MAX_AREA = 0.005, 0.9  # accepted mask size as a share of the frame
SIMPLIFY = 0.002  # outline simplification, as a share of its perimeter (0 = keep every point)


def find_videos(video_dir: Path) -> dict[str, Path]:
    """Map class name -> video path, sorted by class name. One video per class."""
    if not video_dir.is_dir():
        raise ValueError(f"Video folder not found: {video_dir}")
    videos: dict[str, Path] = {}
    for path in sorted(video_dir.iterdir()):
        if not (path.is_file() and path.suffix.lower() in VID_EXT):
            continue
        if path.stem in videos:
            raise ValueError(f"Two videos for class '{path.stem}': "
                             f"{videos[path.stem].name} and {path.name}")
        videos[path.stem] = path
    if not videos:
        raise ValueError(f"No videos ({', '.join(sorted(VID_EXT))}) in {video_dir}")
    return dict(sorted(videos.items()))


def mask_to_polygon(mask: np.ndarray, min_area: float = MIN_AREA,
                    max_area: float = MAX_AREA,
                    simplify: float = SIMPLIFY) -> np.ndarray | None:
    """Outline of the largest region in a binary mask as an (N, 2) array of pixel
    coordinates. Returns None if the mask covers less than min_area or more than
    max_area of the image (SAM most likely outlined the background or nothing)."""
    mask = (mask > 0).astype(np.uint8)
    share = mask.sum() / mask.size
    if not (min_area <= share <= max_area):
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if simplify > 0:
        contour = cv2.approxPolyDP(contour, simplify * cv2.arcLength(contour, True), True)
    poly = contour.reshape(-1, 2)
    return poly if len(poly) >= 3 else None


def polygon_to_yolo(class_id: int, poly: np.ndarray, width: int, height: int) -> str:
    """One YOLO segmentation label line: class id, then x y pairs scaled to 0-1."""
    norm = np.clip(poly / np.array([width, height], dtype=float), 0.0, 1.0)
    return " ".join([str(class_id)] + [f"{v:.6f}" for v in norm.reshape(-1)])


def split_items(items: Sequence[Any], ratios: Sequence[float] = RATIOS,
                seed: int = SEED) -> dict[str, list[Any]]:
    """Shuffle items and divide them into train/val/test by ratios. Test gets the
    remainder after rounding, so every item lands in exactly one split."""
    if len(ratios) != 3 or any(r < 0 for r in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"ratios must be 3 non-negative values summing to 1, got {ratios}")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    n_train = round(len(shuffled) * ratios[0])
    n_val = round(len(shuffled) * ratios[1])
    return {"train": shuffled[:n_train],
            "val": shuffled[n_train:n_train + n_val],
            "test": shuffled[n_train + n_val:]}


def write_data_yaml(path: Path, root: Path, names: Sequence[str]) -> None:
    """Write the Ultralytics dataset file. Names are JSON-quoted, which is valid YAML."""
    lines = [f"path: {json.dumps(str(root))}"]
    lines += [f"{split}: {split}/images" for split in SPLITS]
    lines.append("names:")
    lines += [f"  {i}: {json.dumps(name)}" for i, name in enumerate(names)]
    path.write_text("\n".join(lines) + "\n")


def generated_paths(root: Path) -> list[Path]:
    """Every file or folder this script writes, so it can be cleared before a rebuild."""
    paths = [root / "frames", root / "preview", root / "data.yaml"]
    paths += [root / split / sub for split in SPLITS
              for sub in ("images", "labels", "labels.cache")]  # cache written by Ultralytics
    return paths


def clear_outputs(root: Path, overwrite: bool) -> None:
    """Delete earlier output, or stop if it exists and overwrite is not set."""
    existing = [p for p in generated_paths(root) if p.exists()]
    if existing and not overwrite:
        raise SystemExit("Earlier output found (e.g. "
                         f"{existing[0].relative_to(root.parent)}). "
                         "Pass --overwrite to delete it and rebuild.")
    for path in existing:
        shutil.rmtree(path) if path.is_dir() else path.unlink()


def pick_device() -> str:
    """Fastest available torch device for SAM: CUDA GPU, Apple GPU, then CPU."""
    import torch  # lazy, so the dataset helpers and their tests do not need torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SamLabeler:
    """Outlines the object at the centre of each frame with an Ultralytics SAM model."""

    def __init__(self, weights: str, device: str) -> None:
        """Load the SAM weights (downloaded on first use)."""
        from ultralytics import SAM  # lazy, so the dataset helpers and tests skip ultralytics

        self.model = SAM(weights)
        self.device = device

    def outline(self, image: np.ndarray) -> np.ndarray | None:
        """Pixel outline of the object under the image centre, or None if rejected."""
        h, w = image.shape[:2]
        result = self.model(image, points=[[w / 2, h / 2]], labels=[1],
                            device=self.device, verbose=False)[0]
        if result.masks is None or len(result.masks.data) == 0:
            return None
        mask = result.masks.data[0].cpu().numpy()
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return mask_to_polygon(mask)


def build(every: int, ratios: Sequence[float], seed: int, sam_weights: str,
          device: str, preview: bool, overwrite: bool) -> None:
    """Run every step from videos to a ready-to-train dataset."""
    if every < 1:
        raise ValueError(f"every must be 1 or more, got {every}")
    split_items([], ratios)  # check the ratios before the slow steps
    videos = find_videos(VIDEO_DIR)
    clear_outputs(DATASET_DIR, overwrite)
    for split in SPLITS:
        for sub in ("images", "labels"):
            (DATASET_DIR / split / sub).mkdir(parents=True)

    labeler = SamLabeler(sam_weights, device)
    names = list(videos)
    counts: dict[str, dict[str, int]] = {}

    for class_id, (name, video) in enumerate(videos.items()):
        frames = extract_frames(video, FRAMES_DIR / name, every)
        labelled: list[tuple[Path, str]] = []  # (frame path, label line)
        for i, frame_path in enumerate(frames, 1):
            image = cv2.imread(str(frame_path))
            poly = labeler.outline(image) if image is not None else None
            print(f"\r[{name}] labelling {i}/{len(frames)}", end="", flush=True)
            if poly is None:
                continue
            h, w = image.shape[:2]
            labelled.append((frame_path, polygon_to_yolo(class_id, poly, w, h)))
            if preview:
                out = PREVIEW_DIR / name / frame_path.name
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.polylines(image, [poly.astype(np.int32)], True, (0, 255, 0), 2)
                cv2.imwrite(str(out), image)
        print()

        counts[name] = {"frames": len(frames), "skipped": len(frames) - len(labelled)}
        for split, items in split_items(labelled, ratios, seed).items():
            for frame_path, line in items:
                shutil.copy2(frame_path, DATASET_DIR / split / "images" / frame_path.name)
                label = DATASET_DIR / split / "labels" / f"{frame_path.stem}.txt"
                label.write_text(line + "\n")
            counts[name][split] = len(items)

    write_data_yaml(DATA_YAML, DATASET_DIR, names)

    print(f"\n{'class':<20}{'frames':>8}{'skipped':>9}{'train':>7}{'val':>6}{'test':>6}")
    for name, c in counts.items():
        print(f"{name:<20}{c['frames']:>8}{c['skipped']:>9}"
              f"{c['train']:>7}{c['val']:>6}{c['test']:>6}")
        if c["val"] == 0:
            print(f"  warning: '{name}' has no validation frames; use a longer video or lower --every")
    print(f"\nWrote {DATA_YAML}")
    if preview:
        print(f"Check the outlines in {PREVIEW_DIR} before training")


def parse_args() -> argparse.Namespace:
    """Parse frame, split and labelling options."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--every", type=int, default=EVERY,
                    help=f"keep every Nth video frame (default {EVERY})")
    ap.add_argument("--ratios", type=float, nargs=3, default=RATIOS,
                    metavar=("TRAIN", "VAL", "TEST"),
                    help="split shares, summing to 1 (default 0.7 0.2 0.1)")
    ap.add_argument("--seed", type=int, default=SEED, help=f"split seed (default {SEED})")
    ap.add_argument("--sam", default=SAM_MODEL, help="SAM weights (default models/sam2.1_b.pt)")
    ap.add_argument("--device", help="torch device, e.g. cpu, mps, 0 (default: fastest)")
    ap.add_argument("--no-preview", action="store_true", help="skip writing dataset/preview/")
    ap.add_argument("--overwrite", action="store_true",
                    help="delete earlier output before building")
    return ap.parse_args()


def main() -> None:
    """Build the dataset from the command-line options."""
    args = parse_args()
    build(args.every, args.ratios, args.seed, args.sam, args.device or pick_device(),
          not args.no_preview, args.overwrite)


if __name__ == "__main__":
    main()
