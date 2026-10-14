r"""Regression tests: repeatable checks that catch unintended changes early.

Automated regression tests in CI are standard engineering practice. These checks
give future PRs changing the simulator a guardrail, including AI-assisted or
"vibe coding" edits: an unexpected failure can be noticed before merging. They
help reduce repetitive manual checking during PR review; they do not replace
review or guarantee that all core code is correct. If behaviour changes on
purpose, explain the change and update the relevant expectations, rather than
removing a failing check just to get a green result.

There are 26 SimSource tests and one deproject test. Image checks write real,
tiny PNG files and compare output pixels with known arrays. Video checks replace
VideoCapture with scripted frames or failures and record seek/release calls.
The timing check uses a simulated clock; other checks skip frame delays.
Temporary files and sources are cleaned up after each test.

Scope: these tests cover simulator inputs, colour/depth alignment, playback,
cleanup, pacing arithmetic and one pixel-to-camera formula example. They do not
run YOLO, main.py's detection filters/tracking/UI, real video decoding, camera or
robot hardware, or real calibration. Passing them does not establish detection
accuracy, robot readiness or sustained real-time performance.

Run from the repository root in a separate test environment:
    python -m venv .venv-tests

Windows PowerShell:
    .\.venv-tests\Scripts\python.exe -m pip install -r requirements-test.txt
    .\.venv-tests\Scripts\python.exe -m unittest discover -s tests -v

Linux/macOS:
    .venv-tests/bin/python -m pip install -r requirements-test.txt
    .venv-tests/bin/python -m unittest discover -s tests -v

Standard testing/CI practice:
    https://docs.python.org/3/library/unittest.html
    https://docs.github.com/en/actions/get-started/continuous-integration
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from perception import sim


class SimSourceTests(unittest.TestCase):
    """26 checks for failures that could break simulated perception inputs.

    Seven check alignment and independent depth arrays; eight check numeric/depth
    validation; six check image/folder playback; four check video lifecycle;
    one checks pacing. Each method states the failure it is intended to catch.
    """

    def setUp(self):
        """Give each test a temporary folder and a known 4-by-8 colour image."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = self.write_image("frame.png", np.full((4, 8, 3), 90, np.uint8))

    def write_image(self, name, array):
        """Encode a real image fixture and check that OpenCV saved it."""
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        self.assertTrue(cv2.imwrite(str(path), array), f"Could not write fixture {path}")
        return path

    def source(self, path=None, **options):
        """Create an 8-by-8 source without pacing delays; release it after the test."""
        options.setdefault("size", (8, 8))
        source = sim.SimSource(self.image if path is None else path, **options)
        source.period = 0  # These checks exercise media behavior without frame delays.
        self.addCleanup(source.stop)
        return source

    def video_path(self):
        """Create a placeholder path; video decoding is supplied by a mock."""
        path = self.root / "video.mp4"
        path.touch()
        return path

    def test_synthetic_depth_marks_only_the_letterboxed_content(self):
        """Catch false depth in added borders or wrong frame dimensions/depth values."""
        color, depth, intrinsics = self.source(depth_m=1.234).read()
        expected_color = np.zeros((8, 8, 3), np.uint8)
        expected_color[2:6] = 90
        expected_depth = np.zeros((8, 8), np.uint16)
        expected_depth[2:6] = 1234
        np.testing.assert_array_equal(color, expected_color)
        np.testing.assert_array_equal(depth, expected_depth)
        self.assertEqual(depth.dtype, np.uint16)
        self.assertEqual((intrinsics.width, intrinsics.height), (8, 8))

    def test_registered_depth_uses_nearest_neighbors_and_shared_padding(self):
        """Catch colour/depth misalignment or invented depths from blending measurements."""
        native_color = np.array(
            [[[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]],
             [[130, 140, 150], [160, 170, 180], [190, 200, 210], [220, 230, 240]]],
            dtype=np.uint8,
        )
        native_depth = np.array([[0, 1000, 2000, 3000], [4000, 5000, 6000, 7000]],
                                dtype=np.uint16)
        image = self.write_image("registered.png", native_color)
        depth_path = self.write_image("depth.PNG", native_depth)
        color, depth, _ = self.source(image, depth_path=depth_path).read()
        expected_color = np.zeros((8, 8, 3), np.uint8)
        expected_depth = np.zeros((8, 8), np.uint16)
        expected_color[2:6] = native_color.repeat(2, axis=0).repeat(2, axis=1)
        expected_depth[2:6] = native_depth.repeat(2, axis=0).repeat(2, axis=1)
        np.testing.assert_array_equal(color, expected_color)
        np.testing.assert_array_equal(depth, expected_depth)

    def test_extreme_aspect_ratio_keeps_at_least_one_content_pixel(self):
        """Catch resize failures that shrink a very thin image to zero pixels."""
        image = self.write_image("thin.png", np.full((1, 1000, 3), 80, np.uint8))
        color, depth, _ = self.source(image).read()
        content = np.any(color != 0, axis=2)
        self.assertEqual(color.shape, (8, 8, 3))
        self.assertEqual(np.count_nonzero(content), 8)
        np.testing.assert_array_equal(depth[content], np.full(8, 1500, np.uint16))
        self.assertFalse(np.any(depth[~content]))

    def test_depth_native_dimensions_must_match_the_color_frame(self):
        """Catch pairing a depth map with a differently sized colour image."""
        depth_path = self.write_image("depth.png", np.full((8, 4), 1000, np.uint16))
        source = self.source(depth_path=depth_path)
        with self.assertRaises(ValueError):
            source.read()

    def test_depth_dimensions_are_checked_on_later_folder_frames(self):
        """Catch dimension checks that miss mismatched later folder frames."""
        folder = self.root / "frames"
        self.write_image("frames/01.png", np.full((2, 4, 3), 30, np.uint8))
        self.write_image("frames/02.png", np.full((4, 4, 3), 60, np.uint8))
        depth_path = self.write_image("depth.png", np.full((2, 4), 1000, np.uint16))
        source = self.source(folder, depth_path=depth_path)
        self.assertIsNotNone(source.read())
        with self.assertRaises(ValueError):
            source.read()

    def test_depth_dimensions_are_checked_on_later_video_frames(self):
        """Catch dimension checks that miss mismatched later video frames."""
        depth_path = self.write_image("depth.png", np.full((2, 4), 1000, np.uint16))
        capture = mock.Mock()
        capture.isOpened.return_value = True
        capture.read.side_effect = [
            (True, np.full((2, 4, 3), 30, np.uint8)),
            (True, np.full((4, 4, 3), 60, np.uint8)),
        ]
        with mock.patch.object(sim.cv2, "VideoCapture", return_value=capture):
            source = self.source(self.video_path(), depth_path=depth_path)
            self.assertIsNotNone(source.read())
            with self.assertRaises(ValueError):
                source.read()

    def test_invalid_sizes_are_rejected(self):
        """Catch invalid frame dimensions reaching image allocation or resizing."""
        for size in ((0, 8), (8, 0), (-1, 8), (8, -1), (8.0, 8), (8, 8.0),
                     (True, 8), (8, False), (8,), (8, 8, 8), None):
            with self.subTest(size=size), self.assertRaises(ValueError):
                sim.SimSource(self.image, size=size)

    def test_invalid_frame_rates_are_rejected(self):
        """Catch frame rates that would divide by zero or give invalid timing intervals."""
        for fps in (0, -1, 5e-324, float("nan"), float("inf"), -float("inf")):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                sim.SimSource(self.image, fps=fps)

    def test_invalid_synthetic_depth_is_rejected(self):
        """Catch synthetic distances outside 1-65535 mm or with nonfinite values."""
        for depth_m in (0, -1, 0.0009, 65.535001, float("nan"), float("inf"),
                        -float("inf")):
            with self.subTest(depth_m=depth_m), self.assertRaises(ValueError):
                sim.SimSource(self.image, depth_m=depth_m)

    def test_synthetic_depth_accepts_uint16_millimeter_limits(self):
        """Catch incorrect rejection or conversion at the supported depth limits."""
        for depth_m, millimeters in ((0.001, 1), (65.535, 65535)):
            with self.subTest(depth_m=depth_m):
                _, depth, _ = self.source(depth_m=depth_m).read()
                self.assertEqual(int(depth[3, 3]), millimeters)

    def test_synthetic_depth_preserves_whole_millimeter_values(self):
        """Catch whole millimetres being lost when floating-point values are truncated."""
        for depth_m, millimeters in ((1.001, 1001), (1.003, 1003)):
            with self.subTest(depth_m=depth_m):
                _, depth, _ = self.source(depth_m=depth_m).read()
                self.assertEqual(int(depth[3, 3]), millimeters)

    def test_depth_map_accepts_missing_measurements_and_ignores_synthetic_depth(self):
        """Catch zero depths being filled in or an unused synthetic option rejecting a map."""
        depth_path = self.write_image("depth.png", np.zeros((4, 8), np.uint16))
        _, depth, _ = self.source(depth_path=depth_path, depth_m=float("nan")).read()
        self.assertEqual(depth.shape, (8, 8))
        self.assertFalse(np.any(depth))

    def test_invalid_depth_maps_are_rejected_before_opening_video(self):
        """Catch invalid depth files being accepted or capture opening before rejection."""
        eight_bit = self.write_image("eight-bit.png", np.ones((4, 8), np.uint8))
        channels = self.write_image("channels.png", np.ones((4, 8, 3), np.uint16))
        valid_png = self.write_image("depth.png", np.ones((4, 8), np.uint16))
        wrong_extension = self.root / "depth.tiff"
        wrong_extension.write_bytes(valid_png.read_bytes())
        unreadable = self.root / "unreadable.png"
        unreadable.write_text("This is not an image.", encoding="utf-8")
        video = self.video_path()
        for depth_path in (eight_bit, channels, wrong_extension, unreadable,
                           self.root / "missing.png"):
            with (self.subTest(depth_path=depth_path.name),
                  mock.patch.object(sim.cv2, "VideoCapture") as open_video):
                with self.assertRaises(ValueError):
                    sim.SimSource(video, depth_path=depth_path)
                open_video.assert_not_called()

    def test_invalid_numeric_options_are_rejected_before_opening_video(self):
        """Catch video capture opening before invalid size, FPS or depth options fail."""
        video = self.video_path()
        for options in ({"size": (0, 8)}, {"fps": 0}, {"depth_m": float("nan")}):
            with (self.subTest(options=options),
                  mock.patch.object(sim.cv2, "VideoCapture") as open_video):
                with self.assertRaises(ValueError):
                    sim.SimSource(video, **options)
                open_video.assert_not_called()

    def test_single_image_repeats_without_loop(self):
        """Catch a directly supplied image stopping after its first read."""
        source = self.source()
        for _ in range(3):
            color, _, _ = source.read()
            self.assertEqual(int(color[3, 3, 0]), 90)

    def test_folders_ignore_image_named_directories_and_preserve_order_and_eof(self):
        """Catch unwanted folder entries, wrong frame order or missing end signals."""
        folder = self.root / "frames"
        (folder / "00.png").mkdir(parents=True)
        self.write_image("frames/02.png", np.full((4, 8, 3), 60, np.uint8))
        self.write_image("frames/01.png", np.full((4, 8, 3), 30, np.uint8))
        (folder / "notes.txt").write_text("Ignored.", encoding="utf-8")
        source = self.source(folder)
        self.assertEqual(int(source.read()[0][3, 3, 0]), 30)
        self.assertEqual(int(source.read()[0][3, 3, 0]), 60)
        self.assertIsNone(source.read())
        self.assertIsNone(source.read())

    def test_folder_loop_restarts_in_filename_order(self):
        """Catch loops that fail to restart at the first image in filename order."""
        folder = self.root / "frames"
        self.write_image("frames/02.png", np.full((4, 8, 3), 60, np.uint8))
        self.write_image("frames/01.png", np.full((4, 8, 3), 30, np.uint8))
        source = self.source(folder, loop=True)
        self.assertEqual([int(source.read()[0][3, 3, 0]) for _ in range(5)],
                         [30, 60, 30, 60, 30])

    def test_single_image_folder_ends_or_loops_as_requested(self):
        """Catch a one-image folder repeating without --loop or not repeating with it."""
        folder = self.root / "frames"
        self.write_image("frames/01.png", np.full((4, 8, 3), 30, np.uint8))
        source = self.source(folder)
        self.assertEqual(int(source.read()[0][3, 3, 0]), 30)
        self.assertIsNone(source.read())
        looping = self.source(folder, loop=True)
        self.assertEqual([int(looping.read()[0][3, 3, 0]) for _ in range(3)],
                         [30, 30, 30])

    def test_folder_with_no_image_files_is_rejected(self):
        """Catch image-named directories being mistaken for usable image files."""
        folder = self.root / "frames"
        (folder / "only-a-directory.png").mkdir(parents=True)
        with self.assertRaises(ValueError):
            sim.SimSource(folder)

    def test_corrupt_image_reports_its_filename_instead_of_end_of_media(self):
        """Catch corrupt images silently ending playback without naming the broken file."""
        self.write_image("frames/01.png", np.full((4, 8, 3), 30, np.uint8))
        (self.root / "frames" / "02.png").write_text("Broken image.", encoding="utf-8")
        self.write_image("frames/03.png", np.full((4, 8, 3), 60, np.uint8))
        source = self.source(self.root / "frames")
        self.assertIsNotNone(source.read())
        with self.assertRaisesRegex(ValueError, "02[.]png"):
            source.read()

    def test_returned_depth_does_not_mutate_subsequent_frames(self):
        """Catch later frames being corrupted when a caller edits an earlier depth array."""
        depth_path = self.write_image("depth.png", np.full((4, 8), 1200, np.uint16))
        for options in ({"depth_m": 1.2}, {"depth_path": depth_path}):
            with self.subTest(options=options):
                source = self.source(**options)
                _, first, _ = source.read()
                first[:] = 999
                _, second, _ = source.read()
                self.assertEqual(int(second[3, 3]), 1200)
                self.assertEqual(int(second[0, 0]), 0)
                self.assertFalse(np.shares_memory(first, second))

    def test_failed_video_open_releases_the_capture(self):
        """Catch a failed video open leaving its capture unreleased."""
        capture = mock.Mock()
        capture.isOpened.return_value = False
        with (mock.patch.object(sim.cv2, "VideoCapture", return_value=capture),
              self.assertRaises(ValueError)):
            sim.SimSource(self.video_path())
        capture.release.assert_called_once_with()

    def test_video_end_of_media_and_stop_release(self):
        """Catch unrequested video looping or a capture left open after stop()."""
        capture = mock.Mock()
        capture.isOpened.return_value = True
        capture.read.side_effect = [(True, np.full((4, 8, 3), 70, np.uint8)),
                                    (False, None)]
        with mock.patch.object(sim.cv2, "VideoCapture", return_value=capture):
            source = sim.SimSource(self.video_path(), size=(8, 8))
            source.period = 0
            try:
                self.assertEqual(int(source.read()[0][3, 3, 0]), 70)
                self.assertIsNone(source.read())
                capture.set.assert_not_called()
            finally:
                source.stop()
        capture.release.assert_called_once_with()

    def test_video_loop_seeks_to_start_and_retries(self):
        """Catch video looping that fails to seek back and retry after a read failure."""
        frame = np.full((4, 8, 3), 70, np.uint8)
        capture = mock.Mock()
        capture.isOpened.return_value = True
        capture.read.side_effect = [(True, frame), (False, None), (True, frame)]
        with mock.patch.object(sim.cv2, "VideoCapture", return_value=capture):
            source = self.source(self.video_path(), loop=True)
            self.assertIsNotNone(source.read())
            self.assertIsNotNone(source.read())
        capture.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 0)

    def test_video_with_no_decodable_frames_stops_even_when_looping(self):
        """Catch endless retries when a looping video still has no readable frame."""
        capture = mock.Mock()
        capture.isOpened.return_value = True
        capture.read.side_effect = [(False, None), (False, None)]
        with mock.patch.object(sim.cv2, "VideoCapture", return_value=capture):
            source = self.source(self.video_path(), loop=True)
            self.assertIsNone(source.read())
        capture.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 0)

    def test_pacing_uses_monotonic_time_and_waits_only_for_remaining_period(self):
        """Catch wall-clock pacing, excessive waiting or sleeping when a frame is late."""
        source = sim.SimSource(self.image, size=(8, 8), fps=10)
        self.addCleanup(source.stop)
        clock = [10.0]

        def advance_clock(delay):
            clock[0] += delay

        with (mock.patch.object(sim.time, "monotonic", side_effect=lambda: clock[0]),
              mock.patch.object(sim.time, "sleep", side_effect=advance_clock) as sleep,
              mock.patch.object(sim.time, "time", side_effect=AssertionError(
                  "Frame pacing must not depend on wall-clock corrections"))):
            self.assertIsNotNone(source.read())
            sleep.assert_not_called()
            clock[0] += 0.04
            self.assertIsNotNone(source.read())
            sleep.assert_called_once()
            self.assertAlmostEqual(sleep.call_args.args[0], 0.06)
            clock[0] += 0.2
            self.assertIsNotNone(source.read())
            self.assertEqual(sleep.call_count, 1)


class DeprojectTests(unittest.TestCase):
    """One formula check with chosen camera parameters; no real calibration."""

    def test_camera_axes_and_known_off_center_point(self):
        """Known pixels and 2 m depth give expected right/down/forward camera coordinates."""
        intrinsics = sim.Intrinsics(640, 480, 400.0, 200.0, 320.0, 240.0)
        self.assertEqual(sim.deproject(intrinsics, (320, 240), 2.0), [0.0, 0.0, 2.0])
        self.assertEqual(sim.deproject(intrinsics, (420, 340), 2.0), [0.5, 1.0, 2.0])
        self.assertEqual(sim.deproject(intrinsics, (220, 140), 2.0), [-0.5, -1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
