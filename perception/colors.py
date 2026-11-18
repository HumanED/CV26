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


ACHROMATIC_C = 12.0  # Lab chroma below this is white, grey or black
SAME_HUE_DEG = 30.0  # coloured regions closer in hue than this are one filament, lit differently
SAME_GREY_L = 40.0  # grey regions closer in lightness than this are one filament, lit differently


def _hue_gap(h1: float, h2: float) -> float:
    """Angle (degrees) between two hues."""
    return abs((h1 - h2 + 180) % 360 - 180)


def name_colors(color_bgr: np.ndarray, poly: np.ndarray,
                palette: dict[str, str] = PALETTE, min_share: float = 0.2,
                max_colors: int = 3) -> list[str]:
    """Names of the colours covering at least min_share of the object, most common first.

    The pixels inside the outline (shrunk a little to skip the background at its edge)
    are grouped into up to 4 regions of similar colour. Regions that only differ by
    lighting are merged: coloured regions of nearly the same hue (a red face in shadow
    is a darker, greyer red, not brown) and grey regions of similar lightness (white in
    shadow is not grey). Each remaining region is named from its brightest, most
    saturated pixels, the ones that show the filament's real colour, and only compared
    with palette colours of its kind (coloured or white/grey/black). A multi-colour
    print gets several names, up to max_colors.
    """
    mask = np.zeros(color_bgr.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.asarray(poly, np.int32)], 1)
    shrunk = cv2.erode(mask, np.ones((5, 5), np.uint8))
    pixels = color_bgr[(shrunk if shrunk.any() else mask) > 0]
    if len(pixels) == 0:
        return []
    if len(pixels) > 2000:
        pixels = pixels[np.linspace(0, len(pixels) - 1, 2000).astype(int)]

    lab = _lab(pixels[:, ::-1] / 255.0)
    light, chroma = lab[:, 0], np.hypot(lab[:, 1], lab[:, 2])
    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))

    # Regions of similar colour; lightness counts half, as shading changes it most
    k = min(4, len(lab))
    cv2.setRNGSeed(0)
    feats = np.column_stack([lab[:, 1], lab[:, 2], 0.5 * light]).astype(np.float32)
    _, region, _ = cv2.kmeans(feats, k, None,
                              (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
                              3, cv2.KMEANS_PP_CENTERS)
    region = region.ravel()
    stats = []  # (region id, coloured?, median lightness, mean hue)
    for r in np.unique(region):
        sel = region == r
        w = chroma[sel]
        mean_hue = np.degrees(np.arctan2((w * np.sin(np.radians(hue[sel]))).sum(),
                                         (w * np.cos(np.radians(hue[sel]))).sum()))
        stats.append((r, np.median(chroma[sel]) >= ACHROMATIC_C, np.median(light[sel]), mean_hue))

    # Merge regions that are one filament under different light
    group = {r: r for r, *_ in stats}
    def root(r):
        while group[r] != r:
            r = group[r]
        return r
    for i, (r1, col1, l1, h1) in enumerate(stats):
        for r2, col2, l2, h2 in stats[i + 1:]:
            same = (col1 and col2 and _hue_gap(h1, h2) < SAME_HUE_DEG) or \
                   (not col1 and not col2 and abs(l1 - l2) < SAME_GREY_L)
            if same:
                group[root(r2)] = root(r1)
    region = np.array([root(r) for r in region])

    names = list(palette)
    ref = _lab(np.stack([parse_color(n, palette) for n in names]))
    ref_coloured = np.hypot(ref[:, 1], ref[:, 2]) >= ACHROMATIC_C
    shares: dict[str, float] = {}
    for g in np.unique(region):
        sel = region == g
        coloured = np.median(chroma[sel]) >= ACHROMATIC_C
        # The filament's own colour shows where it is lit best
        key = chroma[sel] if coloured else light[sel]
        best = lab[sel][key >= np.percentile(key, 70)]
        target = np.median(best, axis=0)
        allowed = ref_coloured == coloured
        if not allowed.any():
            allowed[:] = True
        diff = ref - target
        diff[:, 0] *= 0.5
        dist = np.where(allowed, (diff ** 2).sum(axis=1), np.inf)
        name = names[int(np.argmin(dist))]
        shares[name] = shares.get(name, 0.0) + sel.mean()

    ranked = sorted(shares, key=lambda n: -shares[n])
    return [n for n in ranked[:max_colors] if shares[n] >= min_share] or ranked[:1]


def describe(label: str, colors: Sequence[str]) -> str:
    """'orange cube', or 'blue+white star' for a multi-colour print."""
    return f"{'+'.join(colors)} {label}" if colors else label
