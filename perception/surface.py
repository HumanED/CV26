"""Support-surface detection from depth data.

Finds the surface the camera (and the arm it is mounted on) is directly above, e.g. the
table the PAROL6 stands on, so the arm can later place objects on it. Cluttered, slightly
warped or irregularly shaped surfaces are handled:

1. Depth -> point cloud -> per-pixel surface normals (smoothed over a small window).
2. Pixels whose normal points "up" are binned by their offset along the dominant up-facing
   normal; each peak is a candidate horizontal surface (table, floor, box top, shelf).
3. Each candidate plane is fitted robustly and grown into connected regions with a
   depth-dependent tolerance, so sensor noise at range and small bumps or warps are accepted
   while objects standing on the surface become holes in it.
4. The region closest to the point directly below the camera wins, ties going to the larger
   region, so the table beats a box top on it and the floor further away.

The result is a plane equation in camera coordinates plus the surface outline in the image.
"""
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np

DEFAULT_UP = np.array([0.0, -1.0, 0.0])  # world up in camera coords for a level camera (Y points down)
NOISE_COEF = 0.004  # D435 depth RMS error ~ 0.004 * z² (m): 50 mm baseline, ~0.08 px subpixel
NORMAL_RADIUS = 3  # half-size (px) of the window used to estimate normals
MIN_PIXELS = 300  # fewest pixels for a candidate surface
BIN_M = 0.005  # height histogram bin size (m)
MAX_CANDIDATES = 6  # height peaks examined per frame
TIE_M = 0.10  # surfaces whose distance from below the camera differ by less than this tie (m)


@dataclass
class Plane:
    """A detected plane in camera coordinates.

    Equation: normal · (X, Y, Z) + d = 0  where normal is unit length and points up
    (towards the camera side), so normal · p + d is the height of p above the surface.
    For a level camera, normal ≈ (0, -1, 0).
    """
    normal: np.ndarray          # (3,) unit normal vector in camera coords
    d: float                    # plane offset: normal · point + d = 0
    inlier_mask: np.ndarray     # (H, W) bool mask of pixels lying on the surface
    boundary: np.ndarray        # (N, 2) polygon vertices in image coords (u, v)
    center_3d: np.ndarray       # (3,) area-weighted centroid of the surface in camera coords (m)
    extent: tuple[float, float] # (width, depth) of the minimum-area rectangle around it (m)
    support_mask: np.ndarray    # (H, W) inlier_mask with holes (objects on the surface) filled
    corners_3d: np.ndarray      # (4, 3) corners of that rectangle on the plane, camera coords (m)
    area: float                 # true area of the visible surface pixels (m²)
    flatness: float             # RMS distance of the surface pixels from the plane (m)

    def height(self, points: np.ndarray) -> np.ndarray:
        """Height (m) of camera-frame points above the surface; negative below it."""
        return np.asarray(points) @ self.normal + self.d


def up_from_pitch(pitch_deg: float) -> np.ndarray:
    """World up in camera coords for a camera tilted down by pitch_deg (0 = level, 90 = straight down)."""
    p = np.radians(pitch_deg)
    return np.array([0.0, -np.cos(p), -np.sin(p)])


def depth_to_pointcloud(depth: np.ndarray, intr, depth_scale: float) -> np.ndarray:
    """Convert depth image to 3D point cloud in camera coordinates.

    Args:
        depth: (H, W) uint16 depth image in raw units
        intr: camera intrinsics (fx, fy, ppx, ppy)
        depth_scale: metres per raw depth unit

    Returns:
        (H, W, 3) float32 array of (X, Y, Z) in metres, or (0,0,0) where depth=0
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32) * depth_scale
    x = (u - intr.ppx) / intr.fx * z
    y = (v - intr.ppy) / intr.fy * z
    pts = np.stack([x, y, z], axis=-1).astype(np.float32)
    pts[depth == 0] = 0
    return pts


def estimate_normals(pts: np.ndarray, valid: np.ndarray,
                     radius: int = NORMAL_RADIUS) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel unit normals facing the camera, from a smoothed point cloud.

    The cloud is averaged over a (2r+1)² window, ignoring pixels without depth, then the
    normal is the cross product of the horizontal and vertical differences r pixels apart.
    Returns (normals (H, W, 3), ok (H, W) bool); ok is False where there was too little
    depth around the pixel to trust the normal.
    """
    k = 2 * radius + 1
    r = radius
    wgt = valid.astype(np.float32)
    cnt = cv2.boxFilter(wgt, -1, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)
    tot = cv2.boxFilter(np.ascontiguousarray(pts * wgt[..., None]), -1, (k, k),
                        normalize=False, borderType=cv2.BORDER_CONSTANT)
    sm = tot / np.maximum(cnt, 1.0)[..., None]
    dense = valid & (cnt >= 0.5 * k * k)

    du = np.zeros_like(sm)
    dv = np.zeros_like(sm)
    du[:, r:-r] = sm[:, 2 * r:] - sm[:, :-2 * r]
    dv[r:-r] = sm[2 * r:] - sm[:-2 * r]
    ok = np.zeros_like(dense)
    ok[r:-r, r:-r] = (dense[r:-r, r:-r] & dense[r:-r, 2 * r:] & dense[r:-r, :-2 * r]
                      & dense[2 * r:, r:-r] & dense[:-2 * r, r:-r])

    n = np.cross(dv, du)  # +Y x +X = -Z, i.e. towards the camera
    norm = np.linalg.norm(n, axis=-1)
    ok &= norm > 1e-9
    n /= np.maximum(norm, 1e-9)[..., None]
    return n, ok


def _lstsq_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares plane through points: (unit normal, d)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, vecs = np.linalg.eigh(centered.T @ centered)
    normal = vecs[:, 0]  # smallest eigenvalue direction = plane normal
    return normal, float(-normal @ centroid)


def _robust_plane(points: np.ndarray, normal: np.ndarray, d: float,
                  tol: float, rounds: int = 3) -> tuple[np.ndarray, float]:
    """Refit a plane on points within tol of it, shrinking the band to 2.5 sigma each round.
    Keeps the normal oriented like the one passed in."""
    if len(points) > 20000:  # an even spread of 20k points fits as well and much faster
        points = points[::len(points) // 20000]
    for _ in range(rounds):
        res = points @ normal + d
        keep = np.abs(res) < tol
        if keep.sum() < 3:
            break
        new_n, new_d = _lstsq_plane(points[keep])
        if new_n @ normal < 0:
            new_n, new_d = -new_n, -new_d
        normal, d = new_n, new_d
        tol = max(2.5 * float(np.std(res[keep])), 0.002)
    return normal, d


def fit_plane_ransac(points: np.ndarray,
                     max_iter: int = 1000,
                     dist_thresh: float = 0.01,
                     min_inliers: int = 1000,
                     confidence: float = 0.999,
                     rng: Optional[np.random.Generator] = None
                     ) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Fit a plane to 3D points using RANSAC.

    Hypotheses are scored in vectorised batches on at most 20k points, and the number of
    iterations adapts to the inlier ratio found so far (stopping once `confidence` is
    reached). The best plane is then refined by least squares on its inliers.

    Args:
        points: (N, 3) array of 3D points
        max_iter: maximum RANSAC iterations
        dist_thresh: inlier distance threshold (metres)
        min_inliers: minimum inliers to accept a plane
        confidence: probability of having drawn at least one all-inlier sample before stopping
        rng: random generator, for repeatable results

    Returns:
        (normal, d, inlier_mask) or None if no good plane found
        normal is unit length with normal[1] <= 0 (up for a level camera),
        plane eq: normal · p + d = 0
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 3 or n < min_inliers:
        return None
    rng = np.random.default_rng() if rng is None else rng

    score_pts = points if n <= 20000 else points[rng.choice(n, 20000, replace=False)]
    m = len(score_pts)
    best_count, best = 0, None
    needed, it, batch = max_iter, 0, 64

    while it < min(needed, max_iter):
        b = min(batch, max_iter - it)
        it += b
        tri = points[rng.integers(0, n, size=(b, 3))]  # (b, 3 points, xyz)
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        norms = np.linalg.norm(normals, axis=1)
        good = norms > 1e-9
        if not good.any():
            continue
        normals = normals[good] / norms[good, None]
        ds = -np.einsum("ij,ij->i", normals, tri[good, 0])
        counts = (np.abs(score_pts @ normals.T + ds) < dist_thresh).sum(axis=0)
        i = int(np.argmax(counts))
        if counts[i] > best_count:
            best_count, best = int(counts[i]), (normals[i], float(ds[i]))
            ratio = best_count / m
            if ratio >= 1.0:
                break
            needed = int(np.ceil(np.log(1 - confidence) / np.log(1 - ratio ** 3)))

    if best is None:
        return None
    normal, d = best
    for _ in range(3):
        mask = np.abs(points @ normal + d) < dist_thresh
        if mask.sum() < 3:
            return None
        normal, d = _lstsq_plane(points[mask])
    mask = np.abs(points @ normal + d) < dist_thresh
    if mask.sum() < min_inliers:
        return None

    # Ensure normal points roughly upward (negative Y in camera coords)
    if normal[1] > 0:
        normal, d = -normal, -d
    return normal, d, mask


def extract_boundary(mask: np.ndarray, min_area: int = 500) -> Optional[np.ndarray]:
    """Extract the outer boundary polygon from a binary mask.

    Args:
        mask: (H, W) bool or uint8 mask
        min_area: minimum contour area to consider

    Returns:
        (N, 2) polygon vertices in image coordinates, or None
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Take the largest contour
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None

    # Simplify polygon
    eps = 0.005 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True)
    return approx.reshape(-1, 2).astype(np.float32)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes inside a mask (e.g. where objects stand on the surface)."""
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask_u8)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool) | mask


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal in-plane axes: u follows the camera X axis, v = normal x u."""
    u = np.array([1.0, 0.0, 0.0]) - normal[0] * normal
    if np.linalg.norm(u) < 1e-6:
        u = np.array([0.0, 0.0, 1.0]) - normal[2] * normal
    u /= np.linalg.norm(u)
    return u, np.cross(normal, u)


def _min_area_rect(points: np.ndarray, normal: np.ndarray
                   ) -> tuple[tuple[float, float], np.ndarray]:
    """Minimum-area rectangle around points projected onto the plane.

    Returns ((width, depth), corners (4, 3)); width is the side closer to the camera's X axis.
    """
    u, v = _plane_basis(normal)
    origin = points.mean(axis=0)
    uv = ((points - origin) @ np.stack([u, v], axis=1)).astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(uv)).astype(np.float64)
    corners = origin + box[:, :1] * u + box[:, 1:] * v
    e1, e2 = box[1] - box[0], box[2] - box[1]
    l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
    if abs(e1[0]) * l2 >= abs(e2[0]) * l1:  # e1 is more aligned with u
        return (float(l1), float(l2)), corners
    return (float(l2), float(l1)), corners


def compute_extent(points: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    """Compute the 2D extent of points projected onto the plane.

    Returns (width, depth) in metres: the sides of the minimum-area rectangle around the
    points, width being the side closer to the camera's X axis.
    """
    return _min_area_rect(points, normal)[0]


def _pixel_area(pts: np.ndarray, normal: np.ndarray, intr) -> np.ndarray:
    """Area (m²) each pixel covers on a plane with this normal: z³ / (fx·fy·|n·p|)."""
    z = pts[:, 2]
    return z ** 3 / (intr.fx * intr.fy * np.maximum(np.abs(pts @ normal), 1e-6 * z))


def _dominant_normal(normals: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Mean of the up-facing normals, refined on the ones close to that mean."""
    n = normals.mean(axis=0)
    for cos_t in (np.cos(np.radians(8)), np.cos(np.radians(4))):
        n /= np.linalg.norm(n)
        close = normals @ n > cos_t
        if close.sum() < MIN_PIXELS:
            break
        n = normals[close].mean(axis=0)
    n /= np.linalg.norm(n)
    return n if n @ up > 0 else -n


def _height_peaks(offsets: np.ndarray, lo: float, hi: float) -> list[float]:
    """Offsets of the strongest peaks in a histogram of plane offsets, strongest first."""
    bins = np.arange(lo - BIN_M, hi + 2 * BIN_M, BIN_M)
    hist, _ = np.histogram(offsets, bins=bins)
    hist = np.convolve(hist, [1, 2, 3, 2, 1], mode="same") / 9.0
    centres = (bins[:-1] + bins[1:]) / 2
    peaks = [i for i in range(1, len(hist) - 1)
             if hist[i] >= hist[i - 1] and hist[i] > hist[i + 1] and hist[i] * 3 >= MIN_PIXELS]
    peaks.sort(key=lambda i: -hist[i])
    chosen: list[float] = []
    for i in peaks:
        if all(abs(centres[i] - c) > 0.02 for c in chosen):
            chosen.append(float(centres[i]))
        if len(chosen) == MAX_CANDIDATES:
            break
    return chosen


def detect_horizontal_surface(depth: np.ndarray,
                               intr,
                               depth_scale: float,
                               max_height: float = 1.5,
                               min_height: float = 0.2,
                               angle_thresh_deg: float = 15.0,
                               min_area_m2: float = 0.05,
                               up: Optional[np.ndarray] = None,
                               surface_tol: float = 0.015,
                               normal_tol_deg: float = 25.0,
                               max_range: float = 3.0,
                               step: int = 1) -> Optional[Plane]:
    """Detect the horizontal surface directly below the camera (e.g. the robot's table).

    Args:
        depth: (H, W) uint16 depth image
        intr: camera intrinsics
        depth_scale: metres per raw depth unit
        max_height: maximum distance of surface below camera (metres)
        min_height: minimum distance of surface below camera (metres)
        angle_thresh_deg: maximum tilt from horizontal (degrees)
        min_area_m2: minimum visible surface area in square metres
        up: world up direction in camera coords; defaults to a level camera (0, -1, 0).
            For a tilted camera use up_from_pitch(), or pass the arm's camera rotation
            applied to world up once it is known.
        surface_tol: how far (m) a surface point may stray from the plane, on top of the
            depth noise, and still count as surface; allows for warps, mats and cloth.
            Objects taller than this on the surface are cut out of it.
        normal_tol_deg: how far (degrees) a pixel's normal may differ from the plane's
        max_range: ignore depth beyond this distance (m), where the D435 gets noisy
        step: work on every step-th pixel (2 is ~4x faster and nearly as accurate);
            the returned masks and boundary are still full resolution

    Returns:
        Plane object with boundary, normal, center, etc. or None if no surface found
    """
    up = np.asarray(DEFAULT_UP if up is None else up, dtype=np.float64)
    up = up / np.linalg.norm(up)
    full_h, full_w = depth.shape
    if step > 1:
        depth = depth[::step, ::step]
        intr = SimpleNamespace(fx=intr.fx / step, fy=intr.fy / step,
                               ppx=intr.ppx / step, ppy=intr.ppy / step)

    pts = depth_to_pointcloud(depth, intr, depth_scale)
    z = pts[:, :, 2]
    valid = (depth > 0) & (z > 0) & (z < max_range)
    if valid.sum() < MIN_PIXELS:
        return None
    normals, normal_ok = estimate_normals(pts, valid)

    # Pixels that face up and sit in the allowed height band below the camera
    below = -(pts @ up.astype(np.float32))  # metres below the camera along gravity
    cos_tilt = np.cos(np.radians(angle_thresh_deg))
    facing_up = (normal_ok & (normals @ up.astype(np.float32) > cos_tilt)
                 & (below >= min_height) & (below <= max_height))
    if facing_up.sum() < MIN_PIXELS:
        return None

    # Horizontal surfaces are parallel, so one dominant normal serves every candidate and
    # each surface shows up as a peak in the offsets along it (tolerates a slightly wrong up)
    n0 = _dominant_normal(normals[facing_up].astype(np.float64), up)
    offsets = -(pts[facing_up] @ n0)  # plane offset d of each pixel
    peaks = _height_peaks(offsets, float(offsets.min()), float(offsets.max()))

    cos_normal = np.cos(np.radians(normal_tol_deg))
    tol_img = surface_tol + NOISE_COEF * z ** 2
    normals_agree = normal_ok & (normals @ n0.astype(np.float32) > cos_normal)
    kernel = np.ones((3, 3), np.uint8)

    candidates = []  # (footprint distance, area, component mask, normal, d)
    for off in peaks:
        # Fit the plane on the up-facing pixels around this peak, then grow it over all pixels
        seed = facing_up.copy()
        seed[facing_up] = np.abs(offsets - off) < 0.02
        seed_pts = pts[seed].astype(np.float64)
        if len(seed_pts) < MIN_PIXELS:
            continue
        normal, d = _robust_plane(seed_pts, n0, off, 0.02)

        dist = pts @ normal.astype(np.float32) + np.float32(d)
        grown = valid & normals_agree & (np.abs(dist) < tol_img)
        grown = cv2.morphologyEx(grown.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)

        for lab in range(1, n_lab):
            if stats[lab, cv2.CC_STAT_AREA] < MIN_PIXELS:
                continue
            comp = labels == lab
            comp_pts = pts[comp].astype(np.float64)
            c_normal, c_d = _robust_plane(comp_pts, normal, d, float(tol_img[comp].mean()))

            if np.degrees(np.arccos(np.clip(c_normal @ up, -1, 1))) > angle_thresh_deg:
                continue
            if not (min_height <= c_d <= max_height):  # c_d = height of camera above plane
                continue
            area = float(_pixel_area(comp_pts, c_normal, intr).sum())
            if area < min_area_m2:
                continue

            # Horizontal distance from the point on the plane directly below the camera
            foot = -c_d * c_normal
            rel = comp_pts - foot
            in_plane = rel - np.outer(rel @ c_normal, c_normal)
            reach = float(np.percentile(np.linalg.norm(in_plane, axis=1), 2))
            candidates.append((reach, area, comp, c_normal, c_d))

    if not candidates:
        return None

    # The surface the camera stands over: nearest to its footprint, larger area on a tie
    nearest = min(c[0] for c in candidates)
    _, area, inlier_mask, normal, d = max((c for c in candidates if c[0] <= nearest + TIE_M),
                                          key=lambda c: c[1])

    boundary = extract_boundary(inlier_mask, min_area=500 // step ** 2)
    if boundary is None:
        return None

    inlier_pts = pts[inlier_mask].astype(np.float64)
    res = inlier_pts @ normal + d
    flatness = float(np.sqrt(np.mean(res ** 2)))
    band = max(2.5 * flatness, 0.002)  # drops stray edge pixels from the centre and outline

    core = np.abs(res) < band
    core_pts = inlier_pts[core] if core.sum() >= 3 else inlier_pts
    weights = _pixel_area(core_pts, normal, intr)
    center_3d = (core_pts * weights[:, None]).sum(axis=0) / weights.sum()
    center_3d -= (center_3d @ normal + d) * normal  # onto the plane

    # The rectangle only depends on the outline, so fit it to the edge pixels
    edge = inlier_mask & ~cv2.erode(inlier_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    edge_pts = pts[edge].astype(np.float64)
    edge_pts = edge_pts[np.abs(edge_pts @ normal + d) < band]
    if len(edge_pts) < 3:
        edge_pts = core_pts
    extent, corners_3d = _min_area_rect(edge_pts - np.outer(edge_pts @ normal + d, normal), normal)

    support_mask = fill_holes(inlier_mask)
    if step > 1:
        boundary = boundary * step
        inlier_mask, support_mask = (
            cv2.resize(m.astype(np.uint8), (full_w, full_h), interpolation=cv2.INTER_NEAREST) > 0
            for m in (inlier_mask, support_mask))

    return Plane(
        normal=normal,
        d=float(d),
        inlier_mask=inlier_mask,
        boundary=boundary,
        center_3d=center_3d,
        extent=extent,
        support_mask=support_mask,
        corners_3d=corners_3d,
        area=area,
        flatness=flatness,
    )


class SurfaceTracker:
    """Smooths the detected surface over frames and ignores one-frame flickers.

    Single-frame plane estimates jitter by a few mm with D435 noise; while the camera is
    still the surface does not move, so averaging the plane over frames is more accurate.
    A detection that disagrees with the tracked plane replaces it only after it has been
    seen switch_frames times in a row. If nothing is found, the last plane is kept for
    hold_frames frames. Call reset() whenever the arm moves the camera.
    """

    def __init__(self, alpha: float = 0.3, max_angle_deg: float = 5.0,
                 max_offset_m: float = 0.03, switch_frames: int = 5,
                 hold_frames: int = 10) -> None:
        self.alpha = alpha  # weight of the newest plane (1 = no smoothing)
        self.max_cos = np.cos(np.radians(max_angle_deg))  # planes closer than this angle agree
        self.max_offset_m = max_offset_m  # planes whose offsets differ less than this agree
        self.switch_frames = switch_frames
        self.hold_frames = hold_frames
        self.reset()

    def reset(self) -> None:
        """Forget the tracked surface."""
        self.plane: Optional[Plane] = None
        self.missed = 0  # frames in a row with no detection
        self.disagree = 0  # frames in a row with a detection that disagrees

    def update(self, plane: Optional[Plane]) -> Optional[Plane]:
        """Feed this frame's detection (or None); returns the tracked surface."""
        if plane is None:
            self.missed += 1
            if self.missed > self.hold_frames:
                self.plane = None
            return self.plane
        self.missed = 0

        old = self.plane
        if old is None:
            self.plane = plane
            return plane
        agrees = (plane.normal @ old.normal > self.max_cos
                  and abs(plane.d - old.d) < self.max_offset_m)
        if not agrees:
            self.disagree += 1
            if self.disagree >= self.switch_frames:
                self.disagree = 0
                self.plane = plane
            return self.plane
        self.disagree = 0

        a = self.alpha
        normal = (1 - a) * old.normal + a * plane.normal
        normal /= np.linalg.norm(normal)
        d = (1 - a) * old.d + a * plane.d
        # Keep the newest outline and points, moved onto the smoothed plane
        center = plane.center_3d - (plane.center_3d @ normal + d) * normal
        corners = plane.corners_3d - np.outer(plane.corners_3d @ normal + d, normal)
        self.plane = replace(plane, normal=normal, d=float(d),
                             center_3d=center, corners_3d=corners)
        return self.plane


def draw_surface(color: np.ndarray, plane: Plane, intr, depth_scale: float) -> None:
    """Draw the detected surface boundary and info on the color image."""
    # Shade the usable surface (holes where objects stand are included)
    overlay = color.copy()
    overlay[plane.support_mask] = (255, 0, 255)
    cv2.addWeighted(overlay, 0.2, color, 0.8, 0, color)

    # Draw boundary polygon
    cv2.polylines(color, [plane.boundary.astype(np.int32)], True, (255, 0, 255), 3)

    # Draw center point projected to image
    cx = plane.center_3d[0] * intr.fx / plane.center_3d[2] + intr.ppx
    cy = plane.center_3d[1] * intr.fy / plane.center_3d[2] + intr.ppy
    cv2.circle(color, (int(cx), int(cy)), 8, (255, 0, 255), -1)

    # Draw info text
    w, d = plane.extent
    text = (f"Surface: {w:.2f}x{d:.2f}m {plane.area:.2f}m2 | "
            f"{plane.d:.2f}m below | flat {plane.flatness * 1000:.0f}mm")
    cv2.putText(color, text, (int(cx) + 10, int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
