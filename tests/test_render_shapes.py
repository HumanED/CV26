"""Tests for the synthetic training-image renderer and colour naming.

They run without torch or ultralytics, on small meshes written in the test, so the
3D models in 3dprint.ignore/ (not in git) are not needed.

Run from the repository root:
    python -m unittest discover -s tests -v
"""

import struct
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset"))
sys.path.insert(0, str(ROOT / "perception"))
import render_shapes  # noqa: E402
from colors import describe, name_colors, parse_color  # noqa: E402

# Unit cube (mm) as 12 outward-facing triangles
CUBE_V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                   [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float) * 20
CUBE_F = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
          (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]


def cone_tris(r=10.0, h=20.0, n=32):
    """Closed cone (mm) standing on its base."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rim = np.stack([r * np.cos(ang), r * np.sin(ang), np.zeros(n)], 1)
    tip, centre = np.array([0, 0, h]), np.zeros(3)
    tris = []
    for i in range(n):
        a, b = rim[i], rim[(i + 1) % n]
        tris += [[a, b, tip], [centre, b, a]]
    return np.array(tris)


def write_binary_stl(path, tris):
    data = bytearray(80) + struct.pack("<I", len(tris))
    for t in tris:
        data += struct.pack("<12fH", 0, 0, 0, *np.asarray(t, np.float32).reshape(-1), 0)
    path.write_bytes(bytes(data))


class TempDirTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)


class LoadMeshTests(TempDirTest):
    """Reading STL/OBJ files into metres, set on the table."""

    def test_binary_stl_is_centred_on_table_in_metres(self):
        """Catch wrong units or a model floating above or sunk into the table."""
        path = self.root / "Cube Thing.stl"
        write_binary_stl(path, CUBE_V[np.array(CUBE_F)] + [100, -50, 7])
        mesh = render_shapes.load_mesh(path)
        flat = mesh.tris.reshape(-1, 3)
        np.testing.assert_allclose(flat.min(0), [-0.01, -0.01, 0], atol=1e-6)
        np.testing.assert_allclose(flat.max(0), [0.01, 0.01, 0.02], atol=1e-6)
        np.testing.assert_allclose(mesh.com, [0, 0, 0.01], atol=1e-6)
        self.assertEqual(mesh.name, "cube_thing")
        self.assertEqual(len(mesh.verts), 8)

    def test_ascii_stl_and_obj_give_the_same_cube(self):
        """Catch a parser that drops or reorders triangles."""
        ascii_path = self.root / "a.stl"
        lines = ["solid c"]
        for f in CUBE_F:
            lines += ["facet normal 0 0 0", "outer loop"]
            lines += [f"vertex {x} {y} {z}" for x, y, z in CUBE_V[list(f)]]
            lines += ["endloop", "endfacet"]
        ascii_path.write_text("\n".join(lines + ["endsolid c"]))
        obj_path = self.root / "b.obj"
        obj_path.write_text("\n".join([f"v {x} {y} {z}" for x, y, z in CUBE_V]
                                      + [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in CUBE_F]))
        a, b = render_shapes.load_mesh(ascii_path), render_shapes.load_mesh(obj_path)
        np.testing.assert_allclose(a.tris, b.tris)

    def test_obj_polygons_are_split_into_triangles(self):
        """A quad face becomes two triangles."""
        path = self.root / "q.obj"
        path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1/1 2/2 3/3 4/4\n")
        self.assertEqual(len(render_shapes.load_mesh(path).tris), 2)

    def test_inside_out_model_is_flipped(self):
        """Faces wound the wrong way would be culled as back faces and vanish."""
        path = self.root / "inv.stl"
        write_binary_stl(path, CUBE_V[np.array(CUBE_F)][:, ::-1])
        mesh = render_shapes.load_mesh(path)
        n = np.cross(mesh.tris[:, 1] - mesh.tris[:, 0], mesh.tris[:, 2] - mesh.tris[:, 0])
        outward = np.einsum("ij,ij->i", n, mesh.tris.mean(1) - mesh.com) > 0
        self.assertTrue(outward.all())


class SettleTests(TempDirTest):
    """Dropped models must come to rest in a physically stable pose."""

    def resting(self, mesh, rot):
        v = mesh.verts @ rot.T
        c = rot @ mesh.com
        low = v[v[:, 2] < v[:, 2].min() + 1e-6][:, :2]
        hull = cv2.convexHull(low.astype(np.float32))
        return len(low), cv2.pointPolygonTest(hull, (float(c[0]), float(c[1])), True)

    def test_cube_lands_on_a_face(self):
        path = self.root / "cube.stl"
        write_binary_stl(path, CUBE_V[np.array(CUBE_F)])
        mesh = render_shapes.load_mesh(path)
        rng = np.random.default_rng(0)
        for _ in range(20):
            rot = render_shapes.settle(mesh.verts, mesh.com, render_shapes.random_rotation(rng))
            n_low, inside = self.resting(mesh, rot)
            self.assertEqual(n_low, 4)
            self.assertGreater(inside, 0)

    def test_cone_lands_on_base_or_side(self):
        """A cone rests on its base or lies on its side, never on its tip."""
        path = self.root / "cone.stl"
        write_binary_stl(path, cone_tris())
        mesh = render_shapes.load_mesh(path)
        rng = np.random.default_rng(1)
        for _ in range(20):
            rot = render_shapes.settle(mesh.verts, mesh.com, render_shapes.random_rotation(rng))
            axis = rot @ np.array([0, 0, 1.0])
            tilt = np.degrees(np.arccos(np.clip(axis[2], -1, 1)))
            # base down: 0 deg; on its side: 90 + half-angle (26.6 deg) = 116.6 deg
            self.assertTrue(tilt < 1 or abs(tilt - 116.57) < 3, tilt)


class RenderTests(TempDirTest):
    """Whole scenes and the files written for build_dataset.py."""

    def setUp(self):
        super().setUp()
        write_binary_stl(self.root / "cube.stl", CUBE_V[np.array(CUBE_F)])
        write_binary_stl(self.root / "cone.stl", cone_tris())
        self.meshes = [render_shapes.load_mesh(p)
                       for p in render_shapes.find_models([self.root])]

    def test_labels_are_valid_polygons_on_the_object(self):
        """Every label is a polygon inside the image, around pixels that differ from the table."""
        rng = np.random.default_rng(5)
        n_objects = 0
        for _ in range(6):
            img, objects = render_shapes.render(self.meshes, rng, ["red", "#00ff00"], 0.5, [])
            self.assertEqual(img.shape, (render_shapes.H, render_shapes.W, 3))
            for class_id, poly in objects:
                self.assertIn(class_id, (0, 1))
                self.assertGreaterEqual(len(poly), 3)
                self.assertTrue(((poly >= 0) & (poly <= 1)).all())
                n_objects += 1
        self.assertGreater(n_objects, 5)

    def test_multicolor_uses_several_colours(self):
        rng = np.random.default_rng(2)
        cols = render_shapes.triangle_colors(self.meshes[0], rng, ["red", "blue", "white"], 1.0)
        self.assertGreaterEqual(len(np.unique(np.round(cols, 1), axis=0)), 2)
        single = render_shapes.triangle_colors(self.meshes[0], rng, ["red"], 0.0)
        self.assertEqual(len(np.unique(single, axis=0)), 1)

    def test_generate_writes_dataset_source_layout(self):
        """Output must be what build_dataset.py reads, and reruns must not clobber silently."""
        out = self.root / "source"
        counts = render_shapes.generate([self.root], out, n_images=4, workers=1)
        self.assertEqual((out / "classes.txt").read_text().split(), ["cone", "cube"])
        self.assertEqual(len(list((out / "images" / "render").glob("*.jpg"))), 4)
        self.assertEqual(len(list((out / "labels" / "render").glob("*.txt"))), 4)
        self.assertGreater(sum(counts), 0)
        with self.assertRaises(SystemExit):
            render_shapes.generate([self.root], out, n_images=1, workers=1)

    def test_bad_colour_is_rejected_before_rendering(self):
        with self.assertRaises(ValueError):
            render_shapes.generate([self.root], self.root / "s", n_images=1,
                                   colors=["notacolour"], workers=1)


class ColorNameTests(unittest.TestCase):
    """Naming the colour of a detection from its pixels."""

    def square(self, bgr_top, bgr_bottom=None):
        img = np.full((100, 100, 3), 128, np.uint8)
        img[20:50, 20:80] = bgr_top
        img[50:80, 20:80] = bgr_bottom if bgr_bottom is not None else bgr_top
        return img, np.array([[20, 20], [79, 20], [79, 79], [20, 79]])

    def test_single_colour_with_shading(self):
        orange = (parse_color("orange")[::-1] * 255).astype(int)
        img, poly = self.square(orange, (orange * 0.75).astype(int))  # shadowed lower half
        self.assertEqual(name_colors(img, poly), ["orange"])

    def test_red_in_shadow_is_not_brown(self):
        """Lit and shaded faces of a red print as the D435 sees them (BGR, from a real frame)."""
        img, poly = self.square((18, 27, 173), (59, 62, 129))
        self.assertEqual(name_colors(img, poly), ["red"])

    def test_white_in_shadow_is_not_grey(self):
        img, poly = self.square((235, 238, 240), (140, 142, 145))
        self.assertEqual(name_colors(img, poly), ["white"])

    def test_two_colour_print(self):
        img, poly = self.square((200, 90, 30), (240, 240, 240))  # blue top, white bottom (BGR)
        self.assertEqual(sorted(name_colors(img, poly)), ["blue", "white"])
        self.assertEqual(describe("star", ["blue", "white"]), "blue+white star")

    def test_red_and_white_print_keeps_both(self):
        img, poly = self.square((18, 27, 173), (235, 238, 240))
        self.assertEqual(sorted(name_colors(img, poly)), ["red", "white"])

    def test_black_and_white_print_keeps_both(self):
        """Greys merge only when close in lightness; black and white are two filaments."""
        img, poly = self.square((30, 30, 30), (235, 238, 240))
        self.assertEqual(sorted(name_colors(img, poly)), ["black", "white"])

    def test_hex_and_unknown_colours(self):
        np.testing.assert_allclose(parse_color("#ff8000"), [1, 128 / 255, 0])
        with self.assertRaises(ValueError):
            parse_color("chartreuse-ish")


if __name__ == "__main__":
    unittest.main()
