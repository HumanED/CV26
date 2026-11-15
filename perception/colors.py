"""Filament colour names, shared by the training renderer and the live detector.

YOLO is trained to recognise the shape only, whatever its colour, so one model works
for any filament. The colour (or colours, for multi-colour prints) of each detection
is read from the pixels inside its mask and matched to the nearest named colour.
Add or change entries in PALETTE to match the filaments you print with.
"""
from collections.abc import Sequence

import cv2
import numpy as np

# colour name -> sRGB hex, as the filament looks under normal indoor light
PALETTE: dict[str, str] = {
    "red": "#c8202a",
    "orange": "#e0782a",
    "yellow": "#f2c81e",
    "green": "#2a8a3a",
    "blue": "#2060b0",
    "navy": "#1a3a60",
    "purple": "#6a3a9a",
    "pink": "#e87aa8",
    "brown": "#7a4a2a",
    "white": "#f0f0f0",
    "grey": "#8a8a8a",
    "black": "#262626",
}


def parse_color(spec: str, palette: dict[str, str] = PALETTE) -> np.ndarray:
    """sRGB colour (3,) in 0-1 from a palette name or a '#rrggbb' hex string."""
    text = palette.get(spec.lower(), spec)
    if not (text.startswith("#") and len(text) == 7):
        raise ValueError(f"Unknown colour '{spec}': use one of {', '.join(palette)} or #rrggbb")
    try:
        return np.array([int(text[i:i + 2], 16) for i in (1, 3, 5)], float) / 255
    except ValueError:
        raise ValueError(f"Bad hex colour '{spec}'") from None


def _lab(rgb01: np.ndarray) -> np.ndarray:
    """CIE Lab of sRGB colours (N, 3) in 0-1."""
    img = np.asarray(rgb01, np.float32).reshape(-1, 1, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2Lab).reshape(-1, 3)


def name_colors(color_bgr: np.ndarray, poly: np.ndarray,
                palette: dict[str, str] = PALETTE, min_share: float = 0.2,
                max_colors: int = 3) -> list[str]:
    """Names of the colours covering at least min_share of the object, most common first.

    Each pixel inside the outline (shrunk a little to skip the background at its edge)
    votes for its nearest palette colour in Lab. Lightness counts less than hue,
    so shading and highlights change the vote less. A multi-colour print gets
    several names, up to max_colors.
    """
    mask = np.zeros(color_bgr.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.asarray(poly, np.int32)], 1)
    shrunk = cv2.erode(mask, np.ones((5, 5), np.uint8))
    pixels = color_bgr[(shrunk if shrunk.any() else mask) > 0]
    if len(pixels) == 0:
        return []
    if len(pixels) > 2000:
        pixels = pixels[np.linspace(0, len(pixels) - 1, 2000).astype(int)]

    names = list(palette)
    ref = _lab(np.stack([parse_color(n, palette) for n in names]))
    lab = _lab(pixels[:, ::-1] / 255.0)
    diff = lab[:, None, :] - ref[None, :, :]
    diff[..., 0] *= 0.4
    votes = np.bincount(np.argmin((diff ** 2).sum(-1), axis=1), minlength=len(names))
    order = np.argsort(-votes)
    share = votes / votes.sum()
    return [names[i] for i in order[:max_colors] if share[i] >= min_share] or [names[order[0]]]


def describe(label: str, colors: Sequence[str]) -> str:
    """'orange cube', or 'blue+white star' for a multi-colour print."""
    return f"{'+'.join(colors)} {label}" if colors else label
