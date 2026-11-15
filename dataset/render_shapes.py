"""Render labelled training images of the 3D-printed shapes from their 3D models.

Each image is a random scene of printed shapes lying on a table, seen by a D435-like
camera from the arm: shapes, colours, resting poses, camera height and tilt, lighting,
table texture, shadows, blur and sensor noise all vary, so a model trained on the
renders works on the real prints. YOLO only learns the shape; the colour is varied on
purpose (any listed filament colour, random colours, and multi-colour prints) and
named at run time by perception/colors.py.

Input: STL or OBJ files (millimetres, as exported for printing); every file directly in
3dprint.ignore/ by default, class name = file name. Output, ready for build_dataset.py:
    dataset/source/
        classes.txt
        images/render/render_000000.jpg ...
        labels/render/render_000000.txt  "<class id> x1 y1 x2 y2 ..." (polygon, 0-1)

Run from any folder:
    python dataset/render_shapes.py                                   # 3000 images
    python dataset/render_shapes.py --colors orange blue black        # your filaments only
    python dataset/render_shapes.py --colors random --multicolor 0.3  # any colours
    python dataset/render_shapes.py --backgrounds ~/photos/table      # photos of your table
    python dataset/build_dataset.py --overwrite
    python perception/finetune.py
"""
import argparse
import multiprocessing
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # repository root, so paths work from any folder
sys.path.insert(0, str(ROOT / "perception"))
from colors import PALETTE, parse_color  # noqa: E402

MODELS_DIR = ROOT / "3dprint.ignore"  # default 3D models
SOURCE_DIR = ROOT / "dataset" / "source"  # build_dataset.py reads from here
SUBDIR = "render"  # renders go in images/render and labels/render
MESH_EXT = {".stl", ".obj"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}  # background photos

N_IMAGES = 3000  # images to render
W, H = 640, 480  # image size, as the D435 colour stream in perception/main.py
FOCAL = 615.0  # D435 colour focal length (px) at 640x480
SUPERSAMPLE = 2  # render at 2x and shrink, for smooth edges
OBJECTS = (1, 8)  # objects per image (min, max)
EMPTY_P = 0.05  # share of images with no objects, so the model learns not to see shapes in clutter
UPRIGHT_P = 0.6  # share of objects standing as printed; the rest lie in a random stable pose
SCALE_JITTER = 0.05  # +-5% size, for printer shrink and slicer scaling
CAM_HEIGHT = (0.15, 0.6)  # camera height above the table (m)
CAM_PITCH = (30.0, 90.0)  # camera tilt below horizontal (degrees)
MAX_DIST = 1.0  # objects are placed at most this far from the camera (m)
MIN_VISIBLE = 0.3  # objects with less of their area visible are not labelled
MIN_PIXELS = 40  # nor are objects smaller than this (px)
SEED = 0


@dataclass
class Mesh:
    """A 3D model resting on z = 0, centred on x = y = 0, in metres."""
    name: str
    tris: np.ndarray  # (T, 3, 3) triangle corners, counter-clockwise seen from outside
    verts: np.ndarray  # (V, 3) unique corners
    com: np.ndarray  # (3,) centre of mass
    radius: float  # largest distance of a corner from the z axis


def _read_stl(path: Path) -> np.ndarray:
    """Triangles (T, 3, 3) from a binary or ASCII STL file."""
    data = path.read_bytes()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data, np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]),
                                count=n, offset=84)
            return rec["v"].reshape(-1, 3, 3).astype(np.float64)
    words = data.decode("ascii", errors="replace").split()
    verts = [words[i + 1:i + 4] for i, w in enumerate(words) if w == "vertex"]
    if not verts or len(verts) % 3:
        raise ValueError(f"{path}: not a valid STL file")
    return np.array(verts, float).reshape(-1, 3, 3)


def _read_obj(path: Path) -> np.ndarray:
    """Triangles (T, 3, 3) from an OBJ file; polygons are split into fans."""
    verts, tris = [], []
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            verts.append([float(x) for x in parts[1:4]])
        elif parts[0] == "f":
            idx = [int(p.split("/")[0]) for p in parts[1:]]
            idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
            tris += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
    if not tris:
        raise ValueError(f"{path}: no faces")
    return np.array(verts, float)[np.array(tris)]


def load_mesh(path: Path, units: float = 0.001) -> Mesh:
    """Read an STL/OBJ model, convert to metres (units = metres per file unit) and
    set it on the table, centred. Inside-out models are flipped."""
    path = Path(path)
    tris = (_read_stl(path) if path.suffix.lower() == ".stl" else _read_obj(path)) * units
    flat = tris.reshape(-1, 3)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    tris = tris - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])

    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6  # signed tetrahedra to the origin
    if vol.sum() < 0:
        tris = tris[:, ::-1]
        vol = -vol
    if abs(vol.sum()) > 1e-12:
        com = (vol[:, None] * (a + b + c) / 4).sum(axis=0) / vol.sum()
    else:  # open surface: use the corner mean
        com = tris.reshape(-1, 3).mean(axis=0)

    verts = np.unique(np.round(tris.reshape(-1, 3), 9), axis=0)
    radius = float(np.linalg.norm(verts[:, :2], axis=1).max())
    name = path.stem.strip().lower().replace(" ", "_")
    return Mesh(name, tris, verts, com, radius)


def find_models(paths: list[Path]) -> list[Path]:
    """Model files from files and folders (folders are not searched recursively, so
    whole print plates kept in subfolders are left out)."""
    found = []
    for p in paths:
        p = p.expanduser()
        if p.is_dir():
            found += sorted(f for f in p.iterdir() if f.suffix.lower() in MESH_EXT)
        elif p.suffix.lower() in MESH_EXT:
            found.append(p)
        else:
            raise SystemExit(f"Not a 3D model or folder: {p}")
    if not found:
        raise SystemExit(f"No {', '.join(sorted(MESH_EXT))} files in {', '.join(map(str, paths))}")
    return found


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix about a unit axis (Rodrigues)."""
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _closest_on_outline(pts: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Closest point to p on the closed outline through pts (one point, a segment or a polygon)."""
    if len(pts) == 1:
        return pts[0]
    best, best_d = pts[0], np.inf
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        ab = b - a
        t = np.clip((p - a) @ ab / max(ab @ ab, 1e-18), 0, 1)
        q = a + t * ab
        d = np.linalg.norm(p - q)
        if d < best_d:
            best, best_d = q, d
    return best


def settle(verts: np.ndarray, com: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """Rotation that lets a model dropped in orientation rot roll onto a stable pose.

    While the centre of mass is not above the contact area, the model is tipped about
    the contact edge nearest the centre of mass until the next corner touches the table,
    as a slow-moving real object would.
    """
    size = float(np.ptp(verts, axis=0).max())
    eps, tol = 1e-4 * size, 2e-3 * size
    for _ in range(200):
        v = verts @ rot.T
        c = rot @ com
        zmin = v[:, 2].min()
        support = v[v[:, 2] < zmin + eps][:, :2]
        hull = cv2.convexHull(support.astype(np.float32)).reshape(-1, 2).astype(np.float64)
        if len(hull) >= 3 and cv2.pointPolygonTest(hull.astype(np.float32),
                                                   (float(c[0]), float(c[1])), True) >= -tol:
            break
        pivot = _closest_on_outline(hull, c[:2])
        d = c[:2] - pivot
        if np.linalg.norm(d) < tol:
            if len(hull) != 2:
                break  # balanced on a point: only a sphere-like model gets here
            # balanced on an edge is unstable: tip off it to one side
            edge = hull[1] - hull[0]
            d = np.array([-edge[1], edge[0]])
            pivot = c[:2] - d / np.linalg.norm(d) * tol
        d /= np.linalg.norm(d)
        axis = np.array([-d[1], d[0], 0.0])  # z x d: tipping about it lowers the d side
        rel = v - np.array([pivot[0], pivot[1], zmin])
        x, h = rel[:, :2] @ d, rel[:, 2]
        ahead = x > eps
        if not ahead.any():
            break
        angle = float(np.arctan2(h[ahead], x[ahead]).min())  # first corner to touch down
        if angle < 1e-9:
            angle = float(np.radians(0.5))
        rot = _rot(axis, angle) @ rot
    return rot


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly random rotation matrix."""
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def random_pose(mesh: Mesh, rng: np.random.Generator, upright_p: float = UPRIGHT_P) -> np.ndarray:
    """Rotation of a model lying on the table: as printed, or in a random stable pose."""
    yaw = _rot(np.array([0.0, 0.0, 1.0]), rng.uniform(0, 2 * np.pi))
    if rng.random() < upright_p:
        return yaw
    return yaw @ settle(mesh.verts, mesh.com, random_rotation(rng))


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def random_color(rng: np.random.Generator, colors: list[str]) -> np.ndarray:
    """sRGB 0-1: one of the listed colours with a little variation, or any colour for 'random'."""
    name = colors[rng.integers(len(colors))]
    if name == "random":
        hsv = np.uint8([[[rng.integers(0, 180), rng.integers(0, 256), rng.integers(20, 256)]]])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0] / 255.0
    return np.clip(parse_color(name) + rng.normal(0, 0.03, 3), 0, 1)


def triangle_colors(mesh: Mesh, rng: np.random.Generator, colors: list[str],
                    multicolor_p: float) -> np.ndarray:
    """sRGB colour (T, 3) per triangle. Multi-colour prints get 2-3 colours, either in
    bands along the print height (filament swaps) or split by a random plane (AMS painting)."""
    n = len(mesh.tris)
    base = random_color(rng, colors)
    if rng.random() >= multicolor_p:
        return np.tile(base, (n, 1))
    k = int(rng.integers(2, 4))
    palette = np.stack([base] + [random_color(rng, colors) for _ in range(k - 1)])
    centres = mesh.tris.mean(axis=1)
    if rng.random() < 0.5:
        t = centres[:, 2] / max(centres[:, 2].max(), 1e-9)
    else:
        t = centres @ random_rotation(rng)[0]
        t = (t - t.min()) / max(np.ptp(t), 1e-9)
    cuts = np.sort(rng.uniform(0.15, 0.85, k - 1))
    return palette[np.searchsorted(cuts, t)]


def table_texture(rng: np.random.Generator, size: int) -> np.ndarray:
    """Random table top (BGR uint8): plain, wood, cutting mat or fabric, with clutter."""
    def smooth_noise(cells: int) -> np.ndarray:
        small = rng.random((cells, cells)).astype(np.float32)
        return cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)

    # mostly muted colours, like real tables, sometimes a strong one
    sat = rng.uniform(0, 0.4) if rng.random() < 0.8 else rng.uniform(0.4, 1)
    hsv = np.uint8([[[rng.integers(0, 180), sat * 255, rng.uniform(0.15, 0.95) * 255]]])
    base = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0] / 255.0
    kind = rng.integers(4)
    if kind == 0:  # plain, slightly uneven
        tex = base.astype(np.float32) * (0.85 + 0.3 * smooth_noise(4))[..., None]
    elif kind == 1:  # wood grain
        light = np.array([0.55, 0.65, 0.8]) * rng.uniform(0.5, 1.1)  # BGR
        dark = light * rng.uniform(0.5, 0.8)
        y = np.arange(size, dtype=np.float32)[:, None] / size
        grain = np.sin((y * rng.uniform(20, 60) + 3 * smooth_noise(6)) * 2 * np.pi) * 0.5 + 0.5
        tex = light * grain[..., None] + dark * (1 - grain[..., None])
    elif kind == 2:  # cutting mat with grid
        tex = np.ones((size, size, 3), np.float32) * (base * 0.6).astype(np.float32)
        step = int(rng.integers(20, 60))
        line = np.clip(base * 0.6 + 0.4, 0, 1)
        tex[::step] = line
        tex[:, ::step] = line
    else:  # fabric / paper grain
        tex = base.astype(np.float32) * (0.8 + 0.2 * rng.random((size, size), np.float32))[..., None]
    tex = tex * (0.9 + 0.2 * smooth_noise(3))[..., None]
    tex = np.clip(tex * 255, 0, 255).astype(np.uint8)

    for _ in range(rng.integers(0, 6)):  # paper, tape and pen marks, never labelled
        color = tuple(int(x) for x in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            c = rng.integers(0, size, 2)
            box = ((float(c[0]), float(c[1])), tuple(float(x) for x in rng.integers(30, size // 3, 2)),
                   float(rng.uniform(0, 180)))
            cv2.fillConvexPoly(tex, cv2.boxPoints(box).astype(np.int32), color)
        else:
            p, q = rng.integers(0, size, 2), rng.integers(0, size, 2)
            cv2.line(tex, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color,
                     int(rng.integers(1, 12)))
    return tex


def camera(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random camera: (K 3x3, R world->camera 3x3, position C (3,)). World z is up, the
    table is z = 0; camera axes are x right, y down, z forward."""
    f = FOCAL * rng.uniform(0.95, 1.05) * SUPERSAMPLE
    k = np.array([[f, 0, W * SUPERSAMPLE / 2 + rng.normal(0, 3)],
                  [0, f, H * SUPERSAMPLE / 2 + rng.normal(0, 3)], [0, 0, 1]])
    pitch = np.radians(rng.uniform(*CAM_PITCH))
    yaw = rng.uniform(0, 2 * np.pi)
    roll = np.radians(rng.normal(0, 5))
    fwd = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    down = np.cross(fwd, right)
    r = _rot(fwd, roll) @ np.stack([right, down, fwd], axis=1)
    return k, r.T, np.array([0.0, 0.0, rng.uniform(*CAM_HEIGHT)])


def table_point(k: np.ndarray, r: np.ndarray, c: np.ndarray, uv: np.ndarray) -> np.ndarray | None:
    """Where the ray through pixel uv meets the table, or None if it misses or is too far."""
    ray = r.T @ np.linalg.inv(k) @ np.array([uv[0], uv[1], 1.0])
    if ray[2] > -1e-6:
        return None
    p = c - c[2] / ray[2] * ray
    return p if np.linalg.norm(p[:2] - c[:2]) < MAX_DIST else None


def render(meshes: list[Mesh], rng: np.random.Generator, colors: list[str],
           multicolor_p: float, backgrounds: list[Path]) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    """One random scene: (BGR image, [(class id, (N, 2) polygon 0-1)])."""
    k, r, c = camera(rng)
    w, h = W * SUPERSAMPLE, H * SUPERSAMPLE

    # Background: a table texture laid on z = 0 (or a photo), wall colour above the horizon
    if backgrounds and rng.random() < 0.5:
        photo = cv2.imread(str(backgrounds[rng.integers(len(backgrounds))]))
        if photo is None:
            raise SystemExit("Could not read a background image")
        img = cv2.resize(photo, (w, h), interpolation=cv2.INTER_AREA)
        if rng.random() < 0.5:
            img = img[:, ::-1].copy()
    else:
        size, side = 1024, 2.0 * MAX_DIST  # texture px, metres it covers
        ex, ey = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        origin = c - np.array([side / 2, side / 2, c[2]])
        hom = k @ r @ np.stack([ex * side / size, ey * side / size, origin - c], axis=1)
        wall = rng.integers(0, 256, 3).astype(np.uint8)
        img = np.empty((h, w, 3), np.uint8)
        img[:] = wall
        cv2.warpPerspective(table_texture(rng, size), hom, (w, h), img,
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    # Lighting: one directional light, ambient fill, slightly glossy plastic
    elev, azim = np.radians(rng.uniform(25, 85)), rng.uniform(0, 2 * np.pi)
    light = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    ambient, diffuse = rng.uniform(0.2, 0.5), rng.uniform(0.5, 0.9)
    gloss, shine = rng.uniform(0.0, 0.35), rng.uniform(8, 60)
    tint = 1 + rng.normal(0, 0.05, 3)

    # Place objects at random visible table points, not overlapping
    n_obj = 0 if rng.random() < EMPTY_P else int(rng.integers(OBJECTS[0], OBJECTS[1] + 1))
    placed: list[tuple[int, np.ndarray, float]] = []  # (mesh index, xy, radius)
    for _ in range(n_obj * 20):
        if len(placed) == n_obj:
            break
        p = table_point(k, r, c, rng.uniform([0.05 * w, 0.05 * h], [0.95 * w, 0.95 * h]))
        if p is None:
            continue
        i = int(rng.integers(len(meshes)))
        rad = meshes[i].radius * (1 + SCALE_JITTER)
        if all(np.linalg.norm(p[:2] - q) > (rad + qr) * 0.9 for _, q, qr in placed):
            placed.append((i, p[:2], rad))

    shadow = np.zeros((h, w), np.float32)
    tris_all, ids_all, cols_all = [], [], []
    labels = []  # (object id in the id image, class id, projected area incl. outside the image)
    for obj, (i, xy, _) in enumerate(placed):
        mesh = meshes[i]
        rot = random_pose(mesh, rng) * (1 + rng.uniform(-SCALE_JITTER, SCALE_JITTER))
        world = mesh.tris @ rot.T
        world[..., 2] -= world[..., 2].min()
        world[..., :2] += xy
        tri_colors = triangle_colors(mesh, rng, colors, multicolor_p)

        # Shade each face
        normals = np.cross(world[:, 1] - world[:, 0], world[:, 2] - world[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        centres = world.mean(axis=1)
        view = c - centres
        view /= np.linalg.norm(view, axis=1, keepdims=True)
        facing = np.einsum("ij,ij->i", normals, view) > 0
        half = light + view
        half /= np.linalg.norm(half, axis=1, keepdims=True)
        lit = ambient + diffuse * np.clip(normals @ light, 0, None)
        spec = gloss * np.clip(np.einsum("ij,ij->i", normals, half), 0, None) ** shine
        shaded = srgb_to_linear(tri_colors) * lit[:, None] * tint + spec[:, None]
        bgr = (linear_to_srgb(shaded)[:, ::-1] * 255).astype(np.uint8)

        # Shadow: the model squashed onto the table along the light
        flat = world.reshape(-1, 3)
        flat = flat - flat[:, 2:3] / light[2] * light
        shadow_px = _project(k, r, c, flat)
        if shadow_px is not None:
            shadow_px = np.clip(shadow_px, -4 * w, 4 * w).astype(np.int32)
            cv2.fillConvexPoly(shadow, cv2.convexHull(shadow_px), 1.0)

        cam = (world.reshape(-1, 3) - c) @ r.T
        if (cam[:, 2] < 0.05).any():
            continue  # too close to or behind the camera
        tris_all.append(world[facing])
        ids_all.append(np.full(facing.sum(), obj + 1))
        cols_all.append(bgr[facing])
        px = _project(k, r, c, world[facing].reshape(-1, 3)).reshape(-1, 3, 2)
        e1, e2 = px[:, 1] - px[:, 0], px[:, 2] - px[:, 0]
        labels.append((obj + 1, i, 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()))

    blur = int(rng.integers(3, 10)) * SUPERSAMPLE * 2 + 1
    shadow = cv2.GaussianBlur(shadow, (blur, blur), 0)
    img = (img * (1 - rng.uniform(0.25, 0.6) * shadow[..., None])).astype(np.uint8)

    # Painter's algorithm: draw every face far to near
    ids = np.zeros((h, w), np.uint8)
    if tris_all:
        tris = np.concatenate(tris_all)
        tri_ids = np.concatenate(ids_all)
        tri_cols = np.concatenate(cols_all)
        depth = ((tris.mean(axis=1) - c) @ r.T)[:, 2]
        px = _project(k, r, c, tris.reshape(-1, 3)).reshape(-1, 3, 2)
        px = np.round(px * 16).astype(np.int32)  # 4 fractional bits
        for j in np.argsort(-depth):
            cv2.fillConvexPoly(img, px[j], tri_cols[j].tolist(), cv2.LINE_8, 4)
            cv2.fillConvexPoly(ids, px[j], int(tri_ids[j]), cv2.LINE_8, 4)

    # Camera effects
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    ids = cv2.resize(ids, (W, H), interpolation=cv2.INTER_NEAREST)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
    if rng.random() < 0.15:
        n = int(rng.integers(3, 9))
        kernel = np.zeros((n, n), np.float32)
        kernel[n // 2] = 1.0 / n
        rot_m = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), rng.uniform(0, 180), 1)
        img = cv2.filter2D(img, -1, cv2.warpAffine(kernel, rot_m, (n, n)))
    gain = rng.uniform(0.85, 1.15, 3) * rng.uniform(0.7, 1.3)
    img = img.astype(np.float32) * gain + rng.normal(0, rng.uniform(1, 6), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Labels: the visible part of each object
    objects = []
    for obj_id, class_id, area in labels:
        mask = (ids == obj_id).astype(np.uint8)
        n_px = int(mask.sum())
        if n_px < MIN_PIXELS or n_px < MIN_VISIBLE * area / SUPERSAMPLE ** 2:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 0.7, True).reshape(-1, 2)
        if len(contour) < 3:
            continue
        objects.append((class_id, contour / np.array([W, H], float)))
    return img, objects


def _project(k: np.ndarray, r: np.ndarray, c: np.ndarray, pts: np.ndarray) -> np.ndarray | None:
    """Pixel coordinates (N, 2) of world points, or None if any is behind the camera."""
    cam = (pts - c) @ r.T
    if (cam[:, 2] <= 1e-6).any():
        return None
    uvw = cam @ k.T
    return uvw[:, :2] / uvw[:, 2:]


_worker: dict = {}  # per-process state set by _init_worker


def _init_worker(meshes, colors, multicolor_p, backgrounds, out, seed) -> None:
    cv2.setNumThreads(1)  # one image per process; threads inside each would fight for cores
    _worker.update(meshes=meshes, colors=colors, multicolor_p=multicolor_p,
                   backgrounds=backgrounds, out=out, seed=seed)


def _render_one(index: int) -> list[int]:
    """Render and save image index; returns its class ids."""
    s = _worker
    rng = np.random.default_rng([s["seed"], index])
    img, objects = render(s["meshes"], rng, s["colors"], s["multicolor_p"], s["backgrounds"])
    stem = f"{SUBDIR}_{index:06d}"
    cv2.imwrite(str(s["out"] / "images" / SUBDIR / f"{stem}.jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 96))])
    lines = [f"{cid} " + " ".join(f"{v:.5f}" for v in poly.reshape(-1)) for cid, poly in objects]
    (s["out"] / "labels" / SUBDIR / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return [cid for cid, _ in objects]


def generate(model_paths: list[Path], out: Path = SOURCE_DIR, n_images: int = N_IMAGES,
             colors: list[str] | None = None, multicolor_p: float = 0.15,
             backgrounds: Path | None = None, units: float = 0.001, seed: int = SEED,
             workers: int | None = None, overwrite: bool = False) -> list[int]:
    """Render n_images into out/images/render and out/labels/render and write
    out/classes.txt. Returns the number of labelled objects per class."""
    meshes = [load_mesh(p, units) for p in find_models(model_paths)]
    names = [m.name for m in meshes]
    if len(set(names)) != len(names):
        raise SystemExit(f"Two models have the same name: {names}")
    colors = colors or list(PALETTE) + ["random"]
    for col in colors:
        if col != "random":
            parse_color(col)  # fail early on a typo
    bg_files = []
    if backgrounds:
        bg_files = sorted(p for p in backgrounds.expanduser().rglob("*") if p.suffix.lower() in IMG_EXT)
        if not bg_files:
            raise SystemExit(f"No images in {backgrounds}")

    classes = out / "classes.txt"
    if classes.is_file() and classes.read_text().split() != names and not overwrite:
        raise SystemExit(f"{classes} lists other classes; pass --overwrite to replace it")
    for sub in ("images", "labels"):
        folder = out / sub / SUBDIR
        if folder.exists():
            if not overwrite:
                raise SystemExit(f"{folder} already exists; pass --overwrite to replace it")
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
    classes.write_text("\n".join(names) + "\n")

    counts = [0] * len(names)
    args = (meshes, colors, multicolor_p, bg_files, out, seed)
    if workers == 1:  # in this process: no start-up cost, easier to debug
        _init_worker(*args)
        results = map(_render_one, range(n_images))
        pool = None
    else:
        pool = multiprocessing.Pool(workers, _init_worker, args)
        results = pool.imap_unordered(_render_one, range(n_images), 8)
    try:
        for done, ids in enumerate(results, 1):
            for cid in ids:
                counts[cid] += 1
            if done % 200 == 0 or done == n_images:
                print(f"\r{done}/{n_images} images", end="", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    print(f"\nClasses: {', '.join(f'{n} {c}' for n, c in zip(names, counts))}")
    print(f"Wrote {out / 'images' / SUBDIR}, {out / 'labels' / SUBDIR} and {classes}")
    return counts


def parse_args() -> argparse.Namespace:
    """Parse model, colour and output options."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("models", nargs="*", type=Path, default=[MODELS_DIR],
                    help="STL/OBJ files or folders of them (default 3dprint.ignore/)")
    ap.add_argument("-n", "--images", type=int, default=N_IMAGES, help=f"default {N_IMAGES}")
    ap.add_argument("--colors", nargs="+", metavar="COLOR",
                    help=f"filament colours: names ({', '.join(PALETTE)}), #rrggbb, or 'random' "
                         "for any colour (default: all names and random)")
    ap.add_argument("--multicolor", type=float, default=0.15,
                    help="share of objects printed in 2-3 colours (default 0.15)")
    ap.add_argument("--backgrounds", type=Path,
                    help="folder of photos of your table/workspace, used for half the backgrounds")
    ap.add_argument("--units", type=float, default=0.001,
                    help="metres per model unit (default 0.001, i.e. millimetres)")
    ap.add_argument("--out", type=Path, default=SOURCE_DIR, help="default dataset/source")
    ap.add_argument("--seed", type=int, default=SEED, help=f"default {SEED}")
    ap.add_argument("--workers", type=int, help="processes (default: all CPU cores)")
    ap.add_argument("--overwrite", action="store_true", help="replace earlier renders")
    return ap.parse_args()


def main() -> None:
    """Render the dataset from the command-line options."""
    args = parse_args()
    try:
        generate(args.models, args.out.expanduser(), args.images, args.colors, args.multicolor,
                 args.backgrounds, args.units, args.seed, args.workers, args.overwrite)
    except ValueError as err:
        raise SystemExit(f"Error: {err}") from None


if __name__ == "__main__":
    main()
