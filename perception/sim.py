"""Hardware-free frame source for testing the perception pipeline.

Accepts an image, a folder of images or a video. Colour frames come from the
media. Depth is synthetic (a constant distance) unless a 16-bit PNG
depth map is passed in millimetres.
"""
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}  # file types loaded as still images
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}  # file types loaded as video


@dataclass
class Intrinsics:
    """Camera lens parameters (image size, focal length, image centre) used to turn
    a pixel plus depth into a 3D point. Field names match the RealSense SDK's
    intrinsics, so deproject() accepts either."""
    width: int  # image width (px)
    height: int  # image height (px)
    fx: float  # horizontal focal length (px)
    fy: float  # vertical focal length (px)
    ppx: float  # x pixel of the image centre (where the lens points)
    ppy: float  # y pixel of the image centre


def deproject(intr: Intrinsics, pixel: Sequence[float], z: float) -> list[float]:
    """Convert a pixel (u, v) and its measured depth z (m) into a 3D point (m)
    relative to the camera:
        X = (u - ppx) / fx * z    metres right of the lens centre
        Y = (v - ppy) / fy * z    metres below the lens centre
        Z = z                     metres straight out from the lens
    Returns [X, Y, Z]. Works with both sim Intrinsics and rs.intrinsics
    (same attribute names)."""
    x = (pixel[0] - intr.ppx) / intr.fx
    y = (pixel[1] - intr.ppy) / intr.fy
    return [x * z, y * z, z]


def _letterbox(img: np.ndarray, w: int, h: int,
               interp: int = cv2.INTER_AREA) -> np.ndarray:
    """Resize keeping aspect ratio and pad, so object sizes are not distorted."""
    s = min(w / img.shape[1], h / img.shape[0])
    nw, nh = int(img.shape[1] * s), int(img.shape[0] * s)
    out = np.zeros((h, w) + img.shape[2:], img.dtype)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    return out


class SimSource:
    """Frame source that replays an image, image folder or video with synthetic depth."""

    def __init__(self, path: str | Path, size: tuple[int, int] = (640, 480),
                 fps: float = 30, depth_m: float = 1.5,
                 depth_path: str | Path | None = None, loop: bool = False) -> None:
        """Open the media at path and prepare a constant or file-based depth map."""
        self.w, self.h = size  # output frame width and height (px)
        self.period = 1.0 / fps  # seconds between frames
        self.depth_scale = 0.001  # depth image is in millimetres
        self.loop = loop  # restart the media when it ends
        self._last = 0.0  # time the previous frame was returned, for pacing

        # D435 colour stream is roughly fx = fy = 615 at 640x480
        f = 615.0 * self.w / 640
        self.intr = Intrinsics(self.w, self.h, f, f,  # fake D435-like lens parameters
                               self.w / 2, self.h / 2)

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Sim media not found: {p}")

        self._cap: cv2.VideoCapture | None = None  # open video, or None for images
        self._images: list[Path] = []  # image files to play, in order
        self._idx = 0  # index of the next image to return
        if p.is_dir():
            self._images = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
            if not self._images:
                raise ValueError(f"No images in {p}")
        elif p.suffix.lower() in IMG_EXT:
            self._images = [p]
        elif p.suffix.lower() in VID_EXT:
            self._cap = cv2.VideoCapture(str(p))
            if not self._cap.isOpened():
                raise ValueError(f"Could not open video {p}")
        else:
            raise ValueError(f"Unsupported media type: {p.suffix}")

        if depth_path:
            d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if d is None or d.dtype != np.uint16:
                raise ValueError("--sim-depth must be a 16-bit PNG in millimetres")
            self._depth = _letterbox(d, self.w, self.h, cv2.INTER_NEAREST)  # depth map (mm) from file
        else:
            self._depth = np.full((self.h, self.w), int(depth_m * 1000), np.uint16)  # constant depth (mm)

        self.static = self._cap is None and len(self._images) == 1  # single image repeated forever

    def read(self) -> tuple[np.ndarray, np.ndarray, Intrinsics] | None:
        """Returns (color_bgr, depth_u16, intrinsics) or None at end of media."""
        wait = self.period - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

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
            frame = cv2.imread(str(self._images[self._idx]))
            if not self.static:
                self._idx += 1
            if frame is None:
                return None

        return _letterbox(frame, self.w, self.h), self._depth.copy(), self.intr

    def stop(self) -> None:
        """Release the video capture, if one is open."""
        if self._cap is not None:
            self._cap.release()
