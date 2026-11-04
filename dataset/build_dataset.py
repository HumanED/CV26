"""Load a labelled YOLO instance segmentation dataset and split it for finetuning.

The images and labels are made elsewhere (rendered views of the 3D models, or
frames labelled in CVAT and exported as YOLO segmentation). Put them in dataset/source/:
    dataset/source/
        classes.txt         class names, one per line, in class-id order
                            (obj.names or a data.yaml with "names" also work)
        images/**/*.jpg     any nesting; .png, .bmp and .webp also work
        labels/**/*.txt     same relative path and name as the image, one line per object:
                            "<class id> x1 y1 x2 y2 x3 y3 ..." (polygon, 0-1 of image size)

Every label line must be a polygon (instance segmentation); bounding-box-only lines
are rejected. Images without a label file are kept as background images (no objects).

Steps:
    1. pair each image with its label and check every label line
    2. split at random into train/val/test (70/20/10 by default). With --by-folder,
       each folder under images/ stays together, so whole sequences can be held out
    3. copy into the layout Ultralytics expects and write data.yaml

Output (all generated, ignored by git):
    dataset/{train,val,test}/images/*
    dataset/{train,val,test}/labels/*.txt
    dataset/preview/*.jpg       outlines drawn on each image, for checking
    dataset/data.yaml

Run from any folder:
    python dataset/build_dataset.py
    python dataset/build_dataset.py --source ~/renders --by-folder --overwrite
"""
import argparse
import json
import random
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DATASET_DIR = Path(__file__).resolve().parent  # dataset/, so paths work from any folder
SOURCE_DIR = DATASET_DIR / "source"  # input: images/, labels/ and class names
SPLITS = ("train", "val", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}  # same list as perception/sim.py
NAME_FILES = ("classes.txt", "obj.names", "data.yaml")  # checked in this order

RATIOS = (0.7, 0.2, 0.1)  # train, val, test share
SEED = 0  # random seed for the split, so rebuilding gives the same split


@dataclass
class Sample:
    """One image, its label file (None for a background image) and its split group."""
    image: Path
    label: Path | None
    group: str  # samples with the same group always land in the same split


def read_names(source: Path) -> list[str]:
    """Class names in class-id order, from the first name file found in source."""
    for name in NAME_FILES:
        path = source / name
        if not path.is_file():
            continue
        if path.suffix == ".yaml":
            import yaml  # lazy: PyYAML comes with ultralytics, but tests run without it

            names = (yaml.safe_load(path.read_text()) or {}).get("names")
            if isinstance(names, dict):  # {0: a, 1: b}
                names = [names[i] for i in sorted(names)]
        else:
            names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if not names:
            raise SystemExit(f"No class names in {path}")
        return [str(n) for n in names]
    raise SystemExit(f"No class names in {source}: add one of {', '.join(NAME_FILES)}")


def parse_label(path: Path, n_classes: int) -> list[tuple[int, np.ndarray]]:
    """(class id, (N, 2) polygon scaled 0-1) for every line of a YOLO segmentation label."""
    objects = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        values = line.split()
        if not values:
            continue
        where = f"{path}:{line_no}"
        try:
            class_id, coords = int(values[0]), np.array(values[1:], dtype=float)
        except ValueError:
            raise ValueError(f"{where}: not a number") from None
        if not 0 <= class_id < n_classes:
            raise ValueError(f"{where}: class id {class_id} not in 0-{n_classes - 1}")
        if len(coords) == 4:
            raise ValueError(f"{where}: bounding box only; instance segmentation needs "
                             "polygon points (export as YOLO segmentation)")
        if len(coords) < 6 or len(coords) % 2:
            raise ValueError(f"{where}: needs x y pairs for at least 3 points")
        if (coords < 0).any() or (coords > 1).any():
            raise ValueError(f"{where}: coordinates must be 0-1 of the image size")
        objects.append((class_id, coords.reshape(-1, 2)))
    return objects


def find_samples(source: Path, by_folder: bool = False) -> list[Sample]:
    """Pair every image under source/images with source/labels/<same path>.txt.
    The group is the image's first folder under images/ if by_folder, else the image."""
    images_dir, labels_dir = source / "images", source / "labels"
    if not images_dir.is_dir():
        raise SystemExit(f"Image folder not found: {images_dir}")

    samples: list[Sample] = []
    seen: dict[str, Path] = {}  # output file name -> image, to catch name clashes
    for image in sorted(p for p in images_dir.rglob("*")
                        if p.is_file() and p.suffix.lower() in IMG_EXT):
        rel = image.relative_to(images_dir)
        if rel.stem in seen:
            raise SystemExit(f"Two images named '{rel.stem}': {seen[rel.stem]} and {image}")
        seen[rel.stem] = image
        label = labels_dir / rel.with_suffix(".txt")
        group = rel.parts[0] if by_folder and len(rel.parts) > 1 else str(rel)
        samples.append(Sample(image, label if label.is_file() else None, group))
    if not samples:
        raise SystemExit(f"No images ({', '.join(sorted(IMG_EXT))}) in {images_dir}")
    return samples


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


def split_samples(samples: Sequence[Sample], ratios: Sequence[float] = RATIOS,
                  seed: int = SEED) -> dict[str, list[Sample]]:
    """Split by group, so samples sharing a group never end up in different splits."""
    groups = sorted({s.group for s in samples})
    group_split = {g: split for split, gs in split_items(groups, ratios, seed).items()
                   for g in gs}
    out: dict[str, list[Sample]] = {split: [] for split in SPLITS}
    for s in samples:
        out[group_split[s.group]].append(s)
    return out


def write_data_yaml(path: Path, root: Path, names: Sequence[str]) -> None:
    """Write the Ultralytics dataset file. Names are JSON-quoted, which is valid YAML."""
    lines = [f"path: {json.dumps(root.as_posix())}"]  # forward slashes on every OS
    lines += [f"{split}: {split}/images" for split in SPLITS]
    lines.append("names:")
    lines += [f"  {i}: {json.dumps(name)}" for i, name in enumerate(names)]
    path.write_text("\n".join(lines) + "\n")


def generated_paths(root: Path) -> list[Path]:
    """Every file or folder this script writes, so it can be cleared before a rebuild."""
    paths = [root / "preview", root / "data.yaml"]
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


def draw_preview(image_path: Path, objects: list[tuple[int, np.ndarray]],
                 names: Sequence[str], out: Path) -> None:
    """Save the image with each object's outline and class name drawn on it."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read image {image_path}")
    h, w = image.shape[:2]
    for class_id, poly in objects:
        pts = (poly * [w, h]).astype(np.int32)
        cv2.polylines(image, [pts], True, (0, 255, 0), 2)
        x, y = pts.min(axis=0)
        cv2.putText(image, names[class_id], (int(x), max(int(y) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(str(out), image)


def build(source: Path = SOURCE_DIR, out_root: Path = DATASET_DIR,
          ratios: Sequence[float] = RATIOS, seed: int = SEED, by_folder: bool = False,
          preview: bool = True, overwrite: bool = False) -> dict[str, list[int]]:
    """Check, split and copy the source dataset. Returns split -> [images, objects]."""
    split_items([], ratios)  # check the ratios before doing any work
    names = read_names(source)
    samples = find_samples(source, by_folder)
    labels = {s.image: parse_label(s.label, len(names)) if s.label else [] for s in samples}

    clear_outputs(out_root, overwrite)
    for split in SPLITS:
        for sub in ("images", "labels"):
            (out_root / split / sub).mkdir(parents=True)
    if preview:
        (out_root / "preview").mkdir()

    counts = {split: [0, 0] for split in SPLITS}  # split -> [images, objects]
    per_class = [0] * len(names)
    for split, members in split_samples(samples, ratios, seed).items():
        for s in members:
            objects = labels[s.image]
            shutil.copy2(s.image, out_root / split / "images" / s.image.name)
            label_out = out_root / split / "labels" / f"{s.image.stem}.txt"
            if s.label:
                shutil.copy2(s.label, label_out)
            else:
                label_out.write_text("")
            if preview:
                draw_preview(s.image, objects, names,
                             out_root / "preview" / f"{s.image.stem}.jpg")
            counts[split][0] += 1
            counts[split][1] += len(objects)
            for class_id, _ in objects:
                per_class[class_id] += 1

    write_data_yaml(out_root / "data.yaml", out_root, names)

    print(f"{'split':<8}{'images':>8}{'objects':>9}")
    for split, (n_images, n_objects) in counts.items():
        print(f"{split:<8}{n_images:>8}{n_objects:>9}")
    print("objects per class: "
          + ", ".join(f"{name} {n}" for name, n in zip(names, per_class)))
    backgrounds = sum(1 for s in samples if s.label is None)
    if backgrounds:
        print(f"{backgrounds} images have no label file and are used as background images")
    for name, n in zip(names, per_class):
        if n == 0:
            print(f"warning: class '{name}' has no labelled objects")
    if counts["val"][0] == 0:
        print("warning: no validation images; add more images or folders, or change --ratios")
    print(f"Wrote {out_root / 'data.yaml'}")
    return counts


def parse_args() -> argparse.Namespace:
    """Parse source, split and preview options."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=SOURCE_DIR,
                    help="folder with images/, labels/ and classes.txt (default dataset/source)")
    ap.add_argument("--ratios", type=float, nargs=3, default=RATIOS,
                    metavar=("TRAIN", "VAL", "TEST"),
                    help="split shares, summing to 1 (default 0.7 0.2 0.1)")
    ap.add_argument("--seed", type=int, default=SEED, help=f"split seed (default {SEED})")
    ap.add_argument("--by-folder", action="store_true",
                    help="keep each folder under images/ together in one split")
    ap.add_argument("--no-preview", action="store_true", help="skip writing dataset/preview/")
    ap.add_argument("--overwrite", action="store_true",
                    help="delete earlier output before building")
    return ap.parse_args()


def main() -> None:
    """Build the dataset from the command-line options."""
    args = parse_args()
    try:
        build(args.source.expanduser(), DATASET_DIR, args.ratios, args.seed, args.by_folder,
              not args.no_preview, args.overwrite)
    except ValueError as err:
        raise SystemExit(f"Error: {err}") from None


if __name__ == "__main__":
    main()
