"""Finetune the YOLO segmentation model on the dataset from dataset/build_dataset.py.

Starts from the pretrained YOLO11s-seg weights, freezes the first layers so a small
dataset only retrains the later ones, and scores the result on the test split.
The best weights are copied to models/yolo11s-seg-custom.pt.

Run from the repository root:
    python perception/finetune.py                        # defaults below
    python perception/finetune.py --epochs 100 --batch 16
    python perception/main.py --model models/yolo11s-seg-custom.pt
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repository root, so paths work from any folder
DATA = ROOT / "dataset" / "data.yaml"  # written by dataset/build_dataset.py
BASE_MODEL = ROOT / "models" / "yolo11s-seg.pt"  # pretrained starting weights
OUT_MODEL = ROOT / "models" / "yolo11s-seg-custom.pt"  # where the finetuned weights go
RUNS_DIR = ROOT / "runs"  # Ultralytics training logs, plots and checkpoints

EPOCHS = 50  # passes over the training set
IMGSZ = 640  # training image size (px); matches the camera width
BATCH = 8  # images per step; lower it if the GPU runs out of memory
FREEZE = 15  # first N layers kept fixed (0-10 backbone, 11-22 neck, 23 head)
PATIENCE = 15  # stop early after this many epochs without improvement


def pick_device() -> str:
    """Fastest available torch device: CUDA GPU, Apple GPU, then CPU."""
    import torch  # lazy, so --help works without torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def has_images(data: Path, split: str) -> bool:
    """True if the dataset's <split>/images folder contains any files."""
    folder = data.parent / split / "images"
    return folder.is_dir() and any(folder.iterdir())


def finetune(data: Path = DATA, base_model: Path = BASE_MODEL, out: Path = OUT_MODEL,
             epochs: int = EPOCHS, imgsz: int = IMGSZ, batch: int = BATCH,
             freeze: int = FREEZE, patience: int = PATIENCE,
             device: str | None = None, name: str = "finetune") -> Path:
    """Train, copy the best weights to `out`, score them on the test split and
    return `out`."""
    from ultralytics import YOLO  # lazy, so --help works without ultralytics

    if not data.is_file():
        raise SystemExit(f"Dataset file not found: {data}\n"
                         "Build it first: python dataset/build_dataset.py")
    for split in ("train", "val"):
        if not has_images(data, split):
            raise SystemExit(f"No {split} images in {data.parent / split / 'images'}")
    device = device or pick_device()

    model = YOLO(str(base_model))
    model.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch,
                freeze=freeze, patience=patience, device=device, optimizer="auto",
                project=str(RUNS_DIR), name=name, seed=0)

    best = Path(model.trainer.best)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    print(f"Best weights: {best}\nCopied to:    {out}")

    if has_images(data, "test"):
        metrics = YOLO(str(out)).val(data=str(data), split="test", imgsz=imgsz,
                                     batch=batch, device=device, project=str(RUNS_DIR),
                                     name=f"{name}-test")
        print(f"Test mask mAP50-95: {metrics.seg.map:.3f} | mAP50: {metrics.seg.map50:.3f}")
    else:
        print("No test images, skipping the test score")
    return out


def parse_args() -> argparse.Namespace:
    """Parse training options."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DATA, help="dataset YAML")
    ap.add_argument("--model", type=Path, default=BASE_MODEL, help="starting weights")
    ap.add_argument("--out", type=Path, default=OUT_MODEL, help="where to copy the best weights")
    ap.add_argument("--epochs", type=int, default=EPOCHS, help=f"default {EPOCHS}")
    ap.add_argument("--imgsz", type=int, default=IMGSZ, help=f"default {IMGSZ}")
    ap.add_argument("--batch", type=int, default=BATCH, help=f"default {BATCH}")
    ap.add_argument("--freeze", type=int, default=FREEZE,
                    help=f"layers kept fixed (default {FREEZE})")
    ap.add_argument("--patience", type=int, default=PATIENCE,
                    help=f"early-stop epochs (default {PATIENCE})")
    ap.add_argument("--device", help="torch device, e.g. cpu, mps, 0 (default: fastest)")
    ap.add_argument("--name", default="finetune", help="run name under runs/")
    return ap.parse_args()


def main() -> None:
    """Finetune with the command-line options."""
    args = parse_args()
    finetune(args.data, args.model, args.out, args.epochs, args.imgsz, args.batch,
             args.freeze, args.patience, args.device, args.name)


if __name__ == "__main__":
    main()
