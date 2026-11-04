"""Regression tests for the dataset tools in dataset/.

These checks run without torch, ultralytics, SAM or a network connection. They
cover frame extraction (with VideoCapture replaced by scripted frames), finding
one video per class, turning a mask into a YOLO polygon label, the train/val/test
split, data.yaml and the guard against overwriting an earlier build.

Scope: they do not run SAM or YOLO training, so passing them does not establish
label quality or model accuracy.

Run from the repository root:
    python -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

# build_dataset.py runs with dataset/ on the path, so it imports video_to_image directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dataset"))
import build_dataset  # noqa: E402
import video_to_image  # noqa: E402


class TempDirTest(unittest.TestCase):
    """Gives each test a temporary folder that is deleted afterwards."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)


class ExtractFramesTests(TempDirTest):
    """Frame extraction from a scripted video."""

    def capture(self, n_frames):
        """A VideoCapture stand-in that yields n_frames frames with values 0, 1, 2 ..."""
        capture = mock.Mock()
        capture.isOpened.return_value = True
        capture.read.side_effect = ([(True, np.full((4, 6, 3), i, np.uint8))
                                     for i in range(n_frames)] + [(False, None)])
        return capture

    def test_every_nth_frame_is_saved_in_order_and_capture_released(self):
        """Catch skipped, reordered or misnamed frames, or a capture left open."""
        capture = self.capture(7)
        with mock.patch.object(video_to_image.cv2, "VideoCapture", return_value=capture):
            saved = video_to_image.extract_frames("clip/bottle.mp4", self.root / "out", every=3)
        self.assertEqual([p.name for p in saved],
                         ["bottle_00000.jpg", "bottle_00003.jpg", "bottle_00006.jpg"])
        self.assertTrue(all(p.is_file() for p in saved))
        capture.release.assert_called_once_with()

    def test_unopened_video_is_rejected_and_released(self):
        """Catch a missing video silently producing no frames."""
        capture = mock.Mock()
        capture.isOpened.return_value = False
        with (mock.patch.object(video_to_image.cv2, "VideoCapture", return_value=capture),
              self.assertRaises(ValueError)):
            video_to_image.extract_frames("missing.mp4", self.root)
        capture.release.assert_called_once_with()

    def test_invalid_step_is_rejected(self):
        """Catch every=0 looping forever or dividing by zero."""
        with self.assertRaises(ValueError):
            video_to_image.extract_frames("clip.mp4", self.root, every=0)


class FindVideosTests(TempDirTest):
    """One class per video file."""

    def test_classes_are_named_after_videos_and_sorted(self):
        """Catch non-video files becoming classes or class ids changing order."""
        for name in ("can.mp4", "bottle.MOV", "notes.txt"):
            (self.root / name).touch()
        (self.root / "folder.mp4").mkdir()
        self.assertEqual(list(build_dataset.find_videos(self.root)), ["bottle", "can"])

    def test_two_videos_with_the_same_name_are_rejected(self):
        """Catch one class silently replacing another."""
        (self.root / "can.mp4").touch()
        (self.root / "can.avi").touch()
        with self.assertRaises(ValueError):
            build_dataset.find_videos(self.root)

    def test_empty_or_missing_folder_is_rejected(self):
        """Catch an empty build that would train on nothing."""
        with self.assertRaises(ValueError):
            build_dataset.find_videos(self.root)
        with self.assertRaises(ValueError):
            build_dataset.find_videos(self.root / "missing")


class LabelTests(unittest.TestCase):
    """Mask to polygon to YOLO label line."""

    def test_rectangle_mask_gives_its_corners(self):
        """Catch outlines that do not follow the mask."""
        mask = np.zeros((100, 200), np.uint8)
        mask[20:60, 50:150] = 1
        poly = build_dataset.mask_to_polygon(mask)
        self.assertEqual(sorted(map(tuple, poly)), [(50, 20), (50, 59), (149, 20), (149, 59)])

    def test_largest_region_is_kept(self):
        """Catch a speck of noise being labelled instead of the object."""
        mask = np.zeros((100, 100), np.uint8)
        mask[10:60, 10:60] = 1
        mask[80:83, 80:83] = 1
        poly = build_dataset.mask_to_polygon(mask)
        self.assertTrue((poly <= 59).all())

    def test_too_small_or_too_large_masks_are_rejected(self):
        """Catch empty masks or background-sized masks becoming labels."""
        mask = np.zeros((100, 100), np.uint8)
        self.assertIsNone(build_dataset.mask_to_polygon(mask))
        mask[0:2, 0:2] = 1
        self.assertIsNone(build_dataset.mask_to_polygon(mask))
        self.assertIsNone(build_dataset.mask_to_polygon(np.ones((100, 100), np.uint8)))

    def test_yolo_line_is_class_then_normalised_points(self):
        """Catch pixel coordinates, swapped x/y or values outside 0-1 in labels."""
        poly = np.array([[0, 0], [100, 50], [210, 100]])
        line = build_dataset.polygon_to_yolo(2, poly, width=200, height=100)
        self.assertEqual(line, "2 0.000000 0.000000 0.500000 0.500000 1.000000 1.000000")


class SplitTests(unittest.TestCase):
    """Train/val/test split."""

    def test_every_item_lands_in_exactly_one_split(self):
        """Catch frames lost or duplicated between splits."""
        splits = build_dataset.split_items(range(23))
        self.assertEqual([len(splits[s]) for s in build_dataset.SPLITS], [16, 5, 2])
        self.assertEqual(sorted(sum(splits.values(), [])), list(range(23)))

    def test_same_seed_gives_same_split(self):
        """Catch rebuilds moving frames between train and test."""
        self.assertEqual(build_dataset.split_items(range(50), seed=3),
                         build_dataset.split_items(range(50), seed=3))
        self.assertNotEqual(build_dataset.split_items(range(50), seed=3),
                            build_dataset.split_items(range(50), seed=4))

    def test_invalid_ratios_are_rejected(self):
        """Catch ratios that do not cover every frame."""
        for ratios in [(0.5, 0.2, 0.1), (0.7, 0.3), (1.2, -0.1, -0.1)]:
            with self.assertRaises(ValueError):
                build_dataset.split_items(range(10), ratios)


class OutputTests(TempDirTest):
    """data.yaml and rebuild safety."""

    def test_data_yaml_lists_splits_and_class_names(self):
        """Catch a dataset file Ultralytics cannot find images or names in."""
        path = self.root / "data.yaml"
        build_dataset.write_data_yaml(path, self.root, ["bottle", "can"])
        text = path.read_text()
        self.assertIn(f'path: "{self.root}"', text)
        for line in ("train: train/images", "val: val/images", "test: test/images",
                     '  0: "bottle"', '  1: "can"'):
            self.assertIn(line + "\n", text)

    def test_earlier_build_is_kept_without_overwrite(self):
        """Catch a rerun deleting labelled frames without being asked."""
        label = self.root / "train" / "labels" / "a.txt"
        label.parent.mkdir(parents=True)
        label.write_text("0 0 0 1 1 0 1\n")
        with self.assertRaises(SystemExit):
            build_dataset.clear_outputs(self.root, overwrite=False)
        self.assertTrue(label.exists())

    def test_overwrite_deletes_only_generated_output(self):
        """Catch a rebuild deleting the videos or the .gitkeep placeholders."""
        keep = [self.root / "video" / "bottle.mp4", self.root / "train" / ".gitkeep"]
        gone = [self.root / "train" / "images" / "a.jpg", self.root / "frames" / "a.jpg",
                self.root / "data.yaml"]
        for path in keep + gone:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        build_dataset.clear_outputs(self.root, overwrite=True)
        self.assertTrue(all(p.exists() for p in keep))
        self.assertFalse(any(p.exists() for p in gone))


if __name__ == "__main__":
    unittest.main()
