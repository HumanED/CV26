"""Tests for surface detection module."""

import unittest
import numpy as np

from perception.sim import Intrinsics
from perception.surface import (
    depth_to_pointcloud,
    fit_plane_ransac,
    extract_boundary,
    detect_horizontal_surface,
)


class DepthToPointCloudTests(unittest.TestCase):
    """Tests for depth_to_pointcloud conversion."""

    def setUp(self):
        self.intr = Intrinsics(640, 480, 615.0, 615.0, 320.0, 240.0)
        self.depth_scale = 0.001

    def test_center_pixel_at_expected_position(self):
        """Center pixel at 1m depth should map to (0, 0, 1) in camera coords."""
        depth = np.full((480, 640), 1000, dtype=np.uint16)
        pts = depth_to_pointcloud(depth, self.intr, self.depth_scale)

        self.assertEqual(pts.shape, (480, 640, 3))
        cx, cy = 320, 240
        np.testing.assert_allclose(pts[cy, cx], [0.0, 0.0, 1.0], atol=1e-3)

    def test_corner_pixels_have_correct_depth(self):
        """All pixels should have correct Z value regardless of position."""
        depth = np.full((480, 640), 1000, dtype=np.uint16)
        pts = depth_to_pointcloud(depth, self.intr, self.depth_scale)

        self.assertEqual(pts[0, 0, 2], 1.0)
        self.assertEqual(pts[479, 639, 2], 1.0)

    def test_zero_depth_produces_zero_point(self):
        """Zero depth should produce (0, 0, 0) point."""
        depth = np.zeros((480, 640), dtype=np.uint16)
        pts = depth_to_pointcloud(depth, self.intr, self.depth_scale)

        np.testing.assert_array_equal(pts, 0)


class FitPlaneRansacTests(unittest.TestCase):
    """Tests for RANSAC plane fitting."""

    def test_fits_horizontal_plane(self):
        """RANSAC should fit a plane to horizontal planar data."""
        n = 1000
        xs = np.random.uniform(-0.5, 0.5, n)
        zs = np.random.uniform(0.5, 1.5, n)
        ys = np.full(n, -0.5)  # Y = -0.5 (0.5m above camera)
        points = np.stack([xs, ys, zs], axis=1)

        # Add small noise
        points += np.random.normal(0, 0.001, points.shape)

        result = fit_plane_ransac(points, max_iter=200, dist_thresh=0.01, min_inliers=100)
        self.assertIsNotNone(result)

        normal, d, mask = result
        self.assertGreater(mask.sum(), 800)

        # Normal should point up (negative Y in camera coords)
        self.assertLess(normal[1], -0.9)

    def test_returns_none_for_insufficient_points(self):
        """RANSAC should return None when too few points."""
        points = np.array([[0, 0, 1], [1, 0, 1]])
        result = fit_plane_ransac(points, max_iter=100, dist_thresh=0.01, min_inliers=100)
        self.assertIsNone(result)

    def test_returns_none_for_noisy_non_planar_data(self):
        """RANSAC should return None for scattered non-planar points."""
        points = np.random.uniform(-1, 1, (200, 3))
        result = fit_plane_ransac(points, max_iter=100, dist_thresh=0.01, min_inliers=100)
        self.assertIsNone(result)


class ExtractBoundaryTests(unittest.TestCase):
    """Tests for boundary extraction from mask."""

    def test_extracts_rectangle_boundary(self):
        """Should extract boundary from a rectangular mask."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True

        boundary = extract_boundary(mask, min_area=100)
        self.assertIsNotNone(boundary)
        self.assertGreaterEqual(len(boundary), 4)

    def test_returns_none_for_small_mask(self):
        """Should return None for mask below min_area."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:25, 20:25] = True  # 5x5 = 25 pixels

        boundary = extract_boundary(mask, min_area=100)
        self.assertIsNone(boundary)

    def test_returns_none_for_empty_mask(self):
        """Should return None for empty mask."""
        mask = np.zeros((100, 100), dtype=bool)
        boundary = extract_boundary(mask, min_area=100)
        self.assertIsNone(boundary)


class DetectHorizontalSurfaceTests(unittest.TestCase):
    """Tests for full surface detection pipeline."""

    def setUp(self):
        self.intr = Intrinsics(640, 480, 615.0, 615.0, 320.0, 240.0)
        self.depth_scale = 0.001

    def create_table_depth(self, table_y=0.3, z_range=(0.5, 2.0), x_range=(-0.5, 0.5)):
        """Create a synthetic depth image with a horizontal table at constant Y.

        A horizontal table in camera coordinates has constant Y (height below camera),
        with X and Z varying across its surface.
        """
        depth = np.zeros((480, 640), dtype=np.uint16)

        for v in range(480):
            for u in range(640):
                # For a point on the table at Y=table_y, solve for Z given X
                # Camera projection: x = (u - ppx) / fx * z, y = (v - ppy) / fy * z
                # For table at constant Y: y = table_y
                # => table_y = (v - ppy) / fy * z
                # => z = table_y * fy / (v - ppy)
                cam_y = table_y
                v_float = v - 240  # ppy = 240
                if v_float <= 0:
                    continue  # Table is below camera center (v > ppy)
                cam_z = table_y * 615.0 / v_float  # fy = 615
                if not (z_range[0] <= cam_z <= z_range[1]):
                    continue
                cam_x = (u - 320) / 615.0 * cam_z
                if x_range[0] <= cam_x <= x_range[1]:
                    depth[v, u] = int(cam_z / self.depth_scale)
        return depth

    def test_detects_table_at_expected_position(self):
        """Should detect a horizontal table and return correct position."""
        depth = self.create_table_depth(table_y=0.3, z_range=(0.5, 2.0), x_range=(-0.5, 0.5))

        surface = detect_horizontal_surface(
            depth, self.intr, self.depth_scale,
            max_height=1.0, min_height=0.1,
            angle_thresh_deg=15.0, min_area_m2=0.05
        )

        self.assertIsNotNone(surface)
        # Center should be around Y=0.3 (below camera), Z ~ 1.0-1.5, X=0
        self.assertAlmostEqual(surface.center_3d[1], 0.3, delta=0.1)
        self.assertAlmostEqual(surface.center_3d[0], 0.0, delta=0.1)

    def test_surface_extent_matches_table_size(self):
        """Surface extent should match the table dimensions."""
        depth = self.create_table_depth(table_y=0.3, z_range=(0.5, 2.0), x_range=(-0.5, 0.5))

        surface = detect_horizontal_surface(
            depth, self.intr, self.depth_scale,
            max_height=1.0, min_height=0.1,
            angle_thresh_deg=15.0, min_area_m2=0.05
        )

        self.assertIsNotNone(surface)
        w, d = surface.extent
        # Table is 1.0m wide (X) and spans ~1.5m in Z
        # Perspective effects can slightly increase the measured extent
        self.assertAlmostEqual(w, 1.0, delta=0.25)

    def test_returns_none_for_no_depth(self):
        """Should return None when depth image is all zeros."""
        depth = np.zeros((480, 640), dtype=np.uint16)

        surface = detect_horizontal_surface(
            depth, self.intr, self.depth_scale,
            max_height=1.0, min_height=0.1,
            angle_thresh_deg=15.0, min_area_m2=0.05
        )
        self.assertIsNone(surface)

    def test_returns_none_for_tilted_surface(self):
        """Should return None for surface tilted beyond threshold."""
        # Create a tilted plane (sloped floor)
        depth = np.zeros((480, 640), dtype=np.uint16)
        for v in range(480):
            for u in range(640):
                # Z increases with V (tilted away)
                cam_z = 1.0 + (v - 240) * 0.005  # Smaller slope to avoid overflow
                cam_x = (u - 320) / 615 * cam_z
                cam_y = (v - 240) / 615 * cam_z
                if -0.5 <= cam_x <= 0.5 and 0.1 <= cam_y <= 0.5 and cam_z > 0:
                    depth[v, u] = int(cam_z / self.depth_scale)

        surface = detect_horizontal_surface(
            depth, self.intr, self.depth_scale,
            max_height=1.0, min_height=0.1,
            angle_thresh_deg=5.0,  # Strict threshold
            min_area_m2=0.05
        )
        self.assertIsNone(surface)

    def test_returns_none_for_too_small_surface(self):
        """Should return None for surface below min_area."""
        # Very small table
        depth = self.create_table_depth(table_y=0.3, z_range=(0.5, 0.6), x_range=(-0.05, 0.05))

        surface = detect_horizontal_surface(
            depth, self.intr, self.depth_scale,
            max_height=1.0, min_height=0.1,
            angle_thresh_deg=15.0, min_area_m2=0.05
        )
        self.assertIsNone(surface)

    def test_respects_height_bounds(self):
        """Should not detect surfaces outside height bounds."""
        # Table above camera (negative Y) - should not be detected
        # Create table at negative Y using old method
        depth = np.zeros((480, 640), dtype=np.uint16)
        for v in range(480):
            for u in range(640):
                cam_z = 1.5
                cam_x = (u - 320) / 615 * cam_z
                cam_y = (v - 240) / 615 * cam_z
                if -0.5 <= cam_x <= 0.5 and -0.5 <= cam_y <= -0.3:
                    depth[v, u] = int(cam_z / self.depth_scale)

        surface = detect_horizontal_surface(
            depth, self.intr, self.depth_scale,
            max_height=1.0, min_height=0.1,  # Only surfaces below camera
            angle_thresh_deg=15.0, min_area_m2=0.05
        )
        self.assertIsNone(surface)


if __name__ == "__main__":
    unittest.main()