import argparse

import cv2
import numpy as np
from ultralytics import YOLO

from sim import SimSource, deproject

W, H, FPS = 640, 480, 30
MODEL_PATH = "models/yolo11s-seg.pt"
CONF = 0.5
MIN_HITS = 3
EMA_ALPHA = 0.4
Z_MIN, Z_MAX = 0.3, 3.0

SIZE_RANGE = {
}


class RealSenseSource:
    def __init__(self):
        import pyrealsense2 as rs  # lazy, so --sim works without the SDK
        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
        config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
        profile = self.pipeline.start(config)

        sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = sensor.get_depth_scale()
        sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
        sensor.set_option(rs.option.emitter_enabled, 1)

        self.align = rs.align(rs.stream.color)
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()

    def read(self):
        frames = self.align.process(self.pipeline.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return self.read()
        depth_frame = self.temporal.process(self.spatial.process(depth_frame))
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        return (np.asanyarray(color_frame.get_data()),
                np.asanyarray(depth_frame.get_data()), intr)

    def stop(self):
        self.pipeline.stop()


def mask_depth_m(depth_img, poly, depth_scale):
    mask = np.zeros(depth_img.shape, np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    vals = depth_img[(mask > 0) & (depth_img > 0)]
    if vals.size < 50:
        return None
    return float(np.median(vals)) * depth_scale


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", metavar="PATH",
                    help="image, folder of images or video to use instead of the D435")
    ap.add_argument("--sim-depth", metavar="PNG",
                    help="optional 16-bit PNG depth map (mm) used for every frame")
    ap.add_argument("--sim-depth-m", type=float, default=1.5,
                    help="constant synthetic depth in metres (default 1.5)")
    ap.add_argument("--loop", action="store_true", help="loop the sim media")
    return ap.parse_args()
def find_path():
    print()
    #TODO: Draw a path to its designated location based on its category, could also be done in integration
def catogorize(name,x,y,z):
    print()
    #TODO

def main():
    args = parse_args()
    model = YOLO(MODEL_PATH)

    if args.sim:
        source = SimSource(args.sim, (W, H), FPS, args.sim_depth_m,
                           args.sim_depth, args.loop)
        if args.sim_depth:
            print(f"[sim] {args.sim} | supplied depth map; camera intrinsics are approximate")
        else:
            print(f"[sim] {args.sim} | synthetic depth, so Z and size checks are not real")
    else:
        source = RealSenseSource()

    tracks = {}

    try:
        while True:
            data = source.read()
            if data is None:
                print("End of media")
                break
            color, depth, intr = data

            result = model.track(color, conf=CONF, persist=True, verbose=False)[0]

            if result.boxes.id is not None and result.masks is not None:
                for box, poly, tid in zip(result.boxes, result.masks.xy, result.boxes.id):
                    tid = int(tid)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = model.names[int(box.cls[0])]
                    conf = float(box.conf[0])

                    z = mask_depth_m(depth, poly, source.depth_scale)
                    if z is None or not (Z_MIN <= z <= Z_MAX):
                        continue

                    height_m = (y2 - y1) * z / intr.fy
                    lo, hi = SIZE_RANGE.get(label, (0.0, 10.0))
                    if not (lo <= height_m <= hi):
                        continue

                    cx, cy = poly.mean(axis=0)
                    xyz = np.array(deproject(intr, [cx, cy], z))

                    t = tracks.setdefault(tid, {"xyz": xyz, "hits": 0})
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
