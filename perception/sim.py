"""Hardware-free frame source for testing the perception pipeline.

Accepts an image, a folder of images or a video. Colour frames come from the
media. Depth is synthetic (a constant distance) unless a 16-bit PNG
depth map is passed in millimetres.
"""
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float


def deproject(intr, pixel, z):
    """Pixel plus depth (m) to a camera-frame point (x right, y down, z forward).
    Works with both sim Intrinsics and rs.intrinsics (same attribute names)."""
    x = (pixel[0] - intr.ppx) / intr.fx
    y = (pixel[1] - intr.ppy) / intr.fy
    return [x * z, y * z, z]


def _letterbox(img, w, h, interp=cv2.INTER_AREA):
    """Resize keeping aspect ratio and pad, so object sizes are not distorted."""
    s = min(w / img.shape[1], h / img.shape[0])
    nw, nh = max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))
    out = np.zeros((h, w) + img.shape[2:], img.dtype)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    return out


class SimSource:
    def __init__(self, path, size=(640, 480), fps=30, depth_m=1.5,
                 depth_path=None, loop=False):
        try:
            self.w, self.h = size
        except (TypeError, ValueError) as exc:
            raise ValueError("Sim size must contain positive integer width and height") from exc
        if any(isinstance(value, (bool, np.bool_))
               or not isinstance(value, (int, np.integer)) or value <= 0
               for value in (self.w, self.h)):
            raise ValueError("Sim size must contain positive integer width and height")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Sim FPS must be finite and greater than zero")
        self.period = 1.0 / fps
        if not np.isfinite(self.period):
            raise ValueError("Sim FPS must give a finite frame interval")
        self.depth_scale = 0.001  # depth image is in millimetres
        self.loop = loop
        self._last = 0.0

        # D435 colour stream is roughly fx = fy = 615 at 640x480
        f = 615.0 * self.w / 640
        self.intr = Intrinsics(self.w, self.h, f, f, self.w / 2, self.h / 2)

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Sim media not found: {p}")

        self._cap = None
        self._images = []
        self._idx = 0
        if p.is_dir():
            self._images = sorted(f for f in p.iterdir()
                                  if f.is_file() and f.suffix.lower() in IMG_EXT)
            if not self._images:
                raise ValueError(f"No images in {p}")
        elif p.suffix.lower() in IMG_EXT:
            self._images = [p]
        elif p.suffix.lower() not in VID_EXT:
            raise ValueError(f"Unsupported media type: {p.suffix}")

        if depth_path:
            if Path(depth_path).suffix.lower() != ".png":
                raise ValueError("--sim-depth must be a single-channel 16-bit PNG in millimetres")
            d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if d is None or d.dtype != np.uint16 or d.ndim != 2:
                raise ValueError("--sim-depth must be a single-channel 16-bit PNG in millimetres")
            self._depth = d
        else:
            if not np.isfinite(depth_m) or not (0.001 <= depth_m <= 65.535):
                raise ValueError("--sim-depth-m must be finite and between 0.001 and 65.535 metres")
            self._depth_mm = round(depth_m * 1000)
            self._depth = None

        # Validate depth before opening a video, so constructor errors cannot
        # leave an acquired capture behind.
        if not self._images:
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                cap.release()
                raise ValueError(f"Could not open video {p}")
            self._cap = cap

        self.static = p.is_file() and self._cap is None

    def read(self):
        """Returns (color_bgr, depth_u16, intrinsics) or None at end of media."""
        wait = self.period - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

        if self._cap is not None:
            ok, frame = self._cap.read()
            if not ok and self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            if not ok:
                return None
        else:
            if self._idx >= len(self._images):
                if not self.loop:
                    return None
                self._idx = 0
            image_path = self._images[self._idx]
            frame = cv2.imread(str(image_path))
            if not self.static:
                self._idx += 1
            if frame is None:
                raise ValueError(f"Could not read sim image {image_path}")

        if self._depth is None:
            depth = np.full(frame.shape[:2], self._depth_mm, np.uint16)
        else:
            if self._depth.shape != frame.shape[:2]:
                raise ValueError("--sim-depth dimensions must match each colour frame before resizing")
            depth = self._depth

        return (_letterbox(frame, self.w, self.h),
                _letterbox(depth, self.w, self.h, cv2.INTER_NEAREST), self.intr)

    def stop(self):
        if self._cap is not None:
            self._cap.release()
