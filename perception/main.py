import argparse
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from sim import Intrinsics, SimSource, deproject

if TYPE_CHECKING:
    import pyrealsense2 as rs

W, H, FPS = 640, 480, 30  # camera frame width and height (px), frames per second
MODEL_PATH = "models/yolo11s-seg.pt"  # YOLO weights file, relative to the folder you run from
CONF = 0.5  # minimum YOLO confidence (0-1) to keep a detection
MIN_HITS = 3  # frames an object must be seen before it is reported
EMA_ALPHA = 0.4  # smoothing weight for the newest position (1 = no smoothing)
Z_MIN, Z_MAX = 0.3, 3.0  # accepted object distance from the camera (m)

# YOLO label -> (min, max) real object height in metres; unlisted labels allow 0-10 m
SIZE_RANGE: dict[str, tuple[float, float]] = {
}


class RealSenseSource:
    """Frame source for a live Intel RealSense D435 colour and depth stream."""

    def __init__(self) -> None:
        """Start the D435 colour and depth streams and set up alignment and filters."""
        import pyrealsense2 as rs  # lazy, so --sim works without the SDK
        self.rs = rs  # RealSense SDK module
        self.pipeline = rs.pipeline()  # handle that starts the camera and delivers frames
        config = rs.config()
        config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
        config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
        profile = self.pipeline.start(config)

        sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = sensor.get_depth_scale()  # metres per raw depth unit (usually 0.001)
        sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
        sensor.set_option(rs.option.emitter_enabled, 1)

        self.align = rs.align(rs.stream.color)  # remaps depth pixels to line up with colour pixels
        self.spatial = rs.spatial_filter()  # smooths depth noise within a frame
        self.temporal = rs.temporal_filter()  # smooths depth noise across frames

    def read(self) -> tuple[np.ndarray, np.ndarray, "rs.intrinsics"]:
        """Returns (color_bgr, depth_u16, intrinsics) with depth aligned to colour."""
        frames = self.align.process(self.pipeline.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return self.read()
        depth_frame = self.temporal.process(self.spatial.process(depth_frame))
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        return (np.asanyarray(color_frame.get_data()),
                np.asanyarray(depth_frame.get_data()), intr)

    def stop(self) -> None:
        """Stop the camera pipeline."""
        self.pipeline.stop()


def mask_depth_m(depth_img: np.ndarray, poly: np.ndarray,
                 depth_scale: float) -> float | None:
    """Distance (m) to the object: the median depth of the pixels inside its YOLO
    outline, after shrinking the outline by a few pixels so background pixels at
    the edges are not counted. Returns None if fewer than 50 pixels have depth."""
    mask = np.zeros(depth_img.shape, np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    vals = depth_img[(mask > 0) & (depth_img > 0)]
    if vals.size < 50:
        return None
    return float(np.median(vals)) * depth_scale


def parse_args() -> argparse.Namespace:
    """Parse command-line options for sim mode."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", metavar="PATH",
                    help="image, folder of images or video to use instead of the D435")
    ap.add_argument("--sim-depth", metavar="PNG",
                    help="optional 16-bit PNG depth map (mm) used for every frame")
    ap.add_argument("--sim-depth-m", type=float, default=1.5,
                    help="constant synthetic depth in metres (default 1.5)")
    ap.add_argument("--loop", action="store_true", help="loop the sim media")
    return ap.parse_args()


def find_path() -> None:
    """Plan servo coordinates from the object to its category's drop-off location. Not yet implemented."""
    pass
    #TODO: Draw a path to its designated location based on its category, could also be done in integration

def catogorize(name: str, x: float, y: float, z: float) -> None:
    """Assign a detected object to a sorting category. Not yet implemented."""
    pass
    #TODO: Implement categorizaiton



def main() -> None:
    """Run detection, tracking and 3D localisation on each frame until quit or end of media."""
    args: argparse.Namespace = parse_args()
    model: YOLO = YOLO(MODEL_PATH)

    source: SimSource | RealSenseSource
    if args.sim:
        source = SimSource(args.sim, (W, H), FPS, args.sim_depth_m,
                           args.sim_depth, args.loop)
        print(f"[sim] {args.sim} | synthetic depth, so Z and size checks are not real")
    else:
        source = RealSenseSource()

    tracks: dict[int, dict[str, Any]] = {}  # track ID -> {"xyz": smoothed position (m), "hits": frames seen}

    try:
        while True:
            data: tuple[np.ndarray, np.ndarray, Intrinsics | rs.intrinsics] | None = source.read()
            if data is None:
                print("End of media")
                break
            color, depth, intr = data

            result: Results = model.track(color, conf=CONF, persist=True, verbose=False)[0]

            if result.boxes.id is not None and result.masks is not None:
                for box, poly, raw_id in zip(result.boxes, result.masks.xy, result.boxes.id):
                    tid: int = int(raw_id)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label: str = model.names[int(box.cls[0])]
                    conf: float = float(box.conf[0])

                    z: float | None = mask_depth_m(depth, poly, source.depth_scale)
                    if z is None or not (Z_MIN <= z <= Z_MAX):
                        continue

                    height_m: float = (y2 - y1) * z / intr.fy
                    lo, hi = SIZE_RANGE.get(label, (0.0, 10.0))
                    if not (lo <= height_m <= hi):
                        continue

                    cx, cy = poly.mean(axis=0)
                    xyz: np.ndarray = np.array(deproject(intr, [cx, cy], z))

                    t: dict[str, Any] = tracks.setdefault(tid, {"xyz": xyz, "hits": 0})
                    t["xyz"] = EMA_ALPHA * xyz + (1 - EMA_ALPHA) * t["xyz"]
                    t["hits"] += 1
                    if t["hits"] < MIN_HITS:
                        continue

                    X, Y, Z = t["xyz"]
                    print(f"#{tid} {label}: ({X:.2f}, {Y:.2f}, {Z:.2f}) m")
                    cv2.polylines(color, [poly.astype(np.int32)], True, (0, 255, 0), 2)
                    cv2.putText(color, f"#{tid} {label} {conf:.2f} | {Z:.2f}m",
                                (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1)

            cv2.imshow("D435 detections", color)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        source.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting perception program...")
    main()
