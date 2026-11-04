"""Regression tests for the dataset tools in dataset/.

These checks run without torch, ultralytics or a network connection. They cover
frame extraction (with VideoCapture replaced by scripted frames), reading class
names, checking YOLO segmentation labels, pairing images with labels, the
train/val/test split, data.yaml, the guard against overwriting an earlier build
and one full build on tiny images.

Scope: they do not run YOLO training, so passing them does not establish label
quality or model accuracy.

Run from the repository root:
    python -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

# build_dataset.py runs with dataset/ on the path, so the tests import it the same way
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dataset"))
import build_dataset  # noqa: E402
import video_to_image  # noqa: E402

TRIANGLE = "0.1 0.1 0.5 0.1 0.3 0.5"  # a valid 3-point polygon


class TempDirTest(unittest.TestCase):
    """Gives each test a temporary folder that is deleted afterwards."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def write(self, rel, text=""):
        """Write a text file under the temporary folder and return its path."""
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def image(self, rel):
        """Write a real 8-by-8 image under the temporary folder and return its path."""
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        self.assertTrue(cv2.imwrite(str(path), np.full((8, 8, 3), 90, np.uint8)))
        return path


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


class NamesTests(TempDirTest):
    """Class names from the source folder."""

    def test_classes_txt_skips_blank_lines(self):
        """Catch blank lines shifting class ids."""
        self.write("classes.txt", "bottle\n\ncan\n")
        self.assertEqual(build_dataset.read_names(self.root), ["bottle", "can"])

    def test_classes_txt_wins_over_obj_names(self):
        """Catch the documented file order being ignored."""
        self.write("obj.names", "other\n")
        self.write("classes.txt", "bottle\n")
        self.assertEqual(build_dataset.read_names(self.root), ["bottle"])

    def test_missing_names_are_rejected(self):
        """Catch a build with no way to name its classes."""
        with self.assertRaises(SystemExit):
            build_dataset.read_names(self.root)


class LabelTests(TempDirTest):
    """Checking YOLO segmentation label lines."""

    def test_polygons_are_read_as_point_arrays(self):
        """Catch lost objects or swapped x/y."""
        path = self.write("a.txt", f"1 {TRIANGLE}\n\n0 0 0 1 0 1 1 0 1\n")
        objects = build_dataset.parse_label(path, n_classes=2)
        self.assertEqual([c for c, _ in objects], [1, 0])
        np.testing.assert_allclose(objects[0][1], [[0.1, 0.1], [0.5, 0.1], [0.3, 0.5]])
        self.assertEqual(objects[1][1].shape, (4, 2))

    def test_bad_lines_are_rejected_with_their_location(self):
        """Catch box labels, bad class ids or pixel coordinates reaching training."""
        bad = {"box": "0 0.5 0.5 0.2 0.2",
               "class": f"2 {TRIANGLE}",
               "two points": "0 0.1 0.1 0.2 0.2",
               "odd": "0 0.1 0.1 0.2 0.2 0.3",
               "pixels": "0 10 10 50 10 30 50",
               "text": "bottle 0.1 0.1 0.5 0.1 0.3 0.5"}
        for case, line in bad.items():
            with self.subTest(case), self.assertRaisesRegex(ValueError, r"a\.txt:1"):
                build_dataset.parse_label(self.write("a.txt", line), n_classes=2)

    def test_box_error_mentions_segmentation(self):
        """Catch a detection-format export failing with an unclear message."""
        with self.assertRaisesRegex(ValueError, "segmentation"):
            build_dataset.parse_label(self.write("a.txt", "0 0.5 0.5 0.2 0.2"), 1)


class SampleTests(TempDirTest):
    """Pairing images with labels and splitting them."""

    def test_images_pair_with_labels_by_relative_path(self):
        """Catch labels attached to the wrong image or missing labels crashing."""
        self.image("images/seq1/a.jpg")
        self.image("images/seq1/b.png")
        self.write("labels/seq1/a.txt", f"0 {TRIANGLE}")
        self.write("images/notes.txt")
        samples = build_dataset.find_samples(self.root)
        self.assertEqual([(s.image.name, s.label and s.label.name) for s in samples],
                         [("a.jpg", "a.txt"), ("b.png", None)])

    def test_by_folder_groups_by_first_folder(self):
        """Catch frames from one held-out sequence leaking into training."""
        self.image("images/seq1/x/a.jpg")
        self.image("images/seq2/b.jpg")
        self.image("images/c.jpg")
        groups = [s.group for s in build_dataset.find_samples(self.root, by_folder=True)]
        self.assertEqual(groups, ["c.jpg", "seq1", "seq2"])

    def test_duplicate_image_names_are_rejected(self):
        """Catch two images overwriting each other in the flat output folders."""
        self.image("images/seq1/a.jpg")
        self.image("images/seq2/a.png")
        with self.assertRaises(SystemExit):
            build_dataset.find_samples(self.root)

    def test_every_item_lands_in_exactly_one_split(self):
        """Catch images lost or duplicated between splits."""
        splits = build_dataset.split_items(range(23))
        self.assertEqual([len(splits[s]) for s in build_dataset.SPLITS], [16, 5, 2])
        self.assertEqual(sorted(sum(splits.values(), [])), list(range(23)))

    def test_same_seed_gives_same_split(self):
        """Catch rebuilds moving images between train and test."""
        self.assertEqual(build_dataset.split_items(range(50), seed=3),
                         build_dataset.split_items(range(50), seed=3))
        self.assertNotEqual(build_dataset.split_items(range(50), seed=3),
                            build_dataset.split_items(range(50), seed=4))

    def test_groups_stay_together(self):
        """Catch one sequence being spread over several splits."""
        samples = [build_dataset.Sample(Path(f"{g}{i}.jpg"), None, g)
                   for g in "abcdefghij" for i in range(3)]
        for members in build_dataset.split_samples(samples).values():
            for s in members:
                self.assertTrue(all(t in members for t in samples if t.group == s.group))

    def test_invalid_ratios_are_rejected(self):
        """Catch ratios that do not cover every image."""
        for ratios in [(0.5, 0.2, 0.1), (0.7, 0.3), (1.2, -0.1, -0.1)]:
            with self.assertRaises(ValueError):
                build_dataset.split_items(range(10), ratios)


class OutputTests(TempDirTest):
    """data.yaml, rebuild safety and a full build."""

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
        """Catch a rerun deleting a built dataset without being asked."""
        label = self.write("train/labels/a.txt", f"0 {TRIANGLE}\n")
        with self.assertRaises(SystemExit):
            build_dataset.clear_outputs(self.root, overwrite=False)
        self.assertTrue(label.exists())

    def test_overwrite_deletes_only_generated_output(self):
        """Catch a rebuild deleting the source data or the .gitkeep placeholders."""
        keep = [self.write("source/images/a.jpg"), self.write("train/.gitkeep")]
        gone = [self.write("train/images/a.jpg"), self.write("val/labels.cache"),
                self.write("data.yaml")]
        build_dataset.clear_outputs(self.root, overwrite=True)
        self.assertTrue(all(p.exists() for p in keep))
        self.assertFalse(any(p.exists() for p in gone))

    def test_full_build_copies_every_image_and_label(self):
        """Catch images or labels missing from the output, or bad labels getting through."""
        source = self.root / "source"
        self.write("source/classes.txt", "bottle\ncan\n")
        for i in range(10):
            self.image(f"source/images/{i:02d}.jpg")
            self.write(f"source/labels/{i:02d}.txt", f"{i % 2} {TRIANGLE}\n")
        self.image("source/images/empty.jpg")
        out = self.root / "out"

        with mock.patch("builtins.print"):
            counts = build_dataset.build(source, out)
        self.assertEqual(sum(n for n, _ in counts.values()), 11)
        self.assertEqual(sum(n for _, n in counts.values()), 10)
        for split in build_dataset.SPLITS:
            images = sorted(p.stem for p in (out / split / "images").iterdir())
            labels = sorted(p.stem for p in (out / split / "labels").iterdir())
            self.assertEqual(images, labels)
        self.assertEqual(len(list((out / "preview").iterdir())), 11)
        self.assertIn('  1: "can"', (out / "data.yaml").read_text())

        self.write("source/labels/03.txt", "5 0.1 0.1 0.5 0.1 0.3 0.5\n")
        with self.assertRaises(ValueError), mock.patch("builtins.print"):
            build_dataset.build(source, out, overwrite=True)


if __name__ == "__main__":
    unittest.main()
