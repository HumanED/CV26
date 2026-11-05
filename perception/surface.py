"""Surface/plane detection from depth data.

Detects horizontal planar surfaces (e.g. tables, floors) using RANSAC plane fitting
on the depth point cloud. Returns the surface boundary as a 2D polygon in image
coordinates and the plane equation in camera coordinates.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class Plane:
    """A detected plane in camera coordinates.

    Equation: normal · (X, Y, Z) + d = 0  where normal is unit length.
    For a horizontal surface like a table, normal ≈ (0, -1, 0) (pointing up).
    """
    normal: np.ndarray          # (3,) unit normal vector in camera coords
    d: float                    # plane offset: normal · point + d = 0
    inlier_mask: np.ndarray     # (H, W) bool mask of inlier pixels
    boundary: np.ndarray        # (N, 2) polygon vertices in image coords (u, v)
    center_3d: np.ndarray       # (3,) centroid of inliers in camera coords (m)
    extent: tuple[float, float] # (width, depth) of the surface in metres


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
    pts = np.stack([x, y, z], axis=-1)
    pts[depth == 0] = 0
    return pts


def fit_plane_ransac(points: np.ndarray,
                     max_iter: int = 1000,
                     dist_thresh: float = 0.01,
                     min_inliers: int = 1000) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Fit a plane to 3D points using RANSAC.

    Args:
        points: (N, 3) array of 3D points
        max_iter: maximum RANSAC iterations
        dist_thresh: inlier distance threshold (metres)
        min_inliers: minimum inliers to accept a plane

    Returns:
        (normal, d, inlier_mask) or None if no good plane found
        normal is unit length, plane eq: normal · p + d = 0
    """
    n = points.shape[0]
    if n < 3:
        return None

    best_inliers = 0
    best_normal = None
    best_d = None
    best_mask = None

    for _ in range(max_iter):
        # Sample 3 random points
        idx = np.random.choice(n, 3, replace=False)
        p1, p2, p3 = points[idx]

        # Compute plane normal from 3 points
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm

        # Plane equation: normal · (p - p1) = 0  =>  normal · p + d = 0
        d = -np.dot(normal, p1)

        # Count inliers
        dists = np.abs(points @ normal + d)
        mask = dists < dist_thresh
        n_inliers = mask.sum()

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_normal = normal
            best_d = d
            best_mask = mask

            if best_inliers >= min_inliers:
                break

    if best_inliers < min_inliers:
        return None

    # Refine using all inliers (PCA on inlier points)
    inlier_pts = points[best_mask]
    centroid = inlier_pts.mean(axis=0)
    centered = inlier_pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]  # smallest singular value direction = plane normal
    d = -np.dot(normal, centroid)

    # Ensure normal points roughly upward (negative Y in camera coords)
    if normal[1] > 0:
        normal = -normal
        d = -d

    return normal, d, best_mask


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


def compute_extent(points: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    """Compute the 2D extent of points projected onto the plane.

    Returns (width, depth) in metres.
    """
    # Project points onto plane coordinate system
    # Find two orthonormal vectors in the plane
    if abs(normal[0]) > abs(normal[2]):
        u = np.array([-normal[1], normal[0], 0.0])
    else:
        u = np.array([0.0, -normal[2], normal[1]])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    coords = points @ np.stack([u, v], axis=1)  # (N, 2)
    width = coords[:, 0].max() - coords[:, 0].min()
    depth = coords[:, 1].max() - coords[:, 1].min()
    return float(width), float(depth)


def detect_horizontal_surface(depth: np.ndarray,
                               intr,
                               depth_scale: float,
                               max_height: float = 1.5,
                               min_height: float = 0.2,
                               angle_thresh_deg: float = 15.0,
                               min_area_m2: float = 0.05) -> Optional[Plane]:
    """Detect the largest horizontal planar surface (e.g. a table) in the depth image.

    Args:
        depth: (H, W) uint16 depth image
        intr: camera intrinsics
        depth_scale: metres per raw depth unit
        max_height: maximum distance of surface below camera (metres)
        min_height: minimum distance of surface below camera (metres)
        angle_thresh_deg: maximum tilt from horizontal (degrees)
        min_area_m2: minimum surface area in square metres

    Returns:
        Plane object with boundary, normal, center, etc. or None if no surface found
    """
    # Build point cloud
    pts = depth_to_pointcloud(depth, intr, depth_scale)
    h, w = depth.shape

    # Filter valid points within height range
    valid = (depth > 0) & (pts[:, :, 2] > 0)
    y_cam = pts[:, :, 1]  # Y is up/down in camera coords (positive = below camera)

    # For a horizontal surface like a table below the camera, points should have
    # similar positive Y (distance below camera) and plane normal close to (0, -1, 0)
    height_mask = valid & (min_height <= y_cam) & (y_cam <= max_height)
    points = pts[height_mask].reshape(-1, 3)

    if len(points) < 1000:
        return None

    # RANSAC plane fitting
    result = fit_plane_ransac(points, max_iter=500, dist_thresh=0.01, min_inliers=500)
    if result is None:
        return None

    normal, d, inlier_mask_flat = result

    # Check if plane is horizontal (normal close to up vector)
    up = np.array([0.0, -1.0, 0.0])
    angle = np.degrees(np.arccos(np.clip(np.abs(normal @ up), 0, 1)))
    if angle > angle_thresh_deg:
        return None

    # Reconstruct full-size inlier mask
    inlier_mask = np.zeros((h, w), dtype=bool)
    inlier_mask[height_mask] = inlier_mask_flat

    # Extract boundary in image coordinates
    boundary = extract_boundary(inlier_mask)
    if boundary is None:
        return None

    # Compute 3D centroid and extent of inliers
    inlier_pts = points[inlier_mask_flat]
    center_3d = inlier_pts.mean(axis=0)
    extent = compute_extent(inlier_pts, normal)
    area = extent[0] * extent[1]

    if area < min_area_m2:
        return None

    return Plane(
        normal=normal,
        d=d,
        inlier_mask=inlier_mask,
        boundary=boundary,
        center_3d=center_3d,
        extent=extent
    )


def draw_surface(color: np.ndarray, plane: Plane, intr, depth_scale: float) -> None:
    """Draw the detected surface boundary and info on the color image."""
    # Draw boundary polygon
    cv2.polylines(color, [plane.boundary.astype(np.int32)], True, (255, 0, 255), 3)

    # Draw center point projected to image
    cx = plane.center_3d[0] * intr.fx / plane.center_3d[2] + intr.ppx
    cy = plane.center_3d[1] * intr.fy / plane.center_3d[2] + intr.ppy
    cv2.circle(color, (int(cx), int(cy)), 8, (255, 0, 255), -1)

    # Draw info text
    w, d = plane.extent
    text = f"Surface: {w:.2f}x{d:.2f}m | Z={plane.center_3d[2]:.2f}m"
    cv2.putText(color, text, (int(cx) + 10, int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Optionally draw the inlier mask as overlay
    overlay = color.copy()
    overlay[plane.inlier_mask] = (255, 0, 255)
    cv2.addWeighted(overlay, 0.2, color, 0.8, 0, color)