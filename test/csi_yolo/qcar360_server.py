#!/usr/bin/env python3
"""
QCar 2 — 360° Perception Server
=================================
Architecture
------------
  4 CSI cameras → 4 capture threads (parallel)
                → 4 YOLO inference threads (parallel, one model clone each)
                → AngleMapper (pixel bbox → absolute world angle)
                → Flask-SocketIO (pushes frames + detections to browser)

Camera layout (bird's-eye, QCar front = 0°)
---------------------------------------------
  CAM 0  FRONT   centre =   0°   covers  -80° …  +80°
  CAM 1  RIGHT   centre =  90°   covers  +10° … +170°
  CAM 2  BACK    centre = 180°   covers +100° … +260°  (= -260° … -100°)
  CAM 3  LEFT    centre = 270°   covers +190° … +350°  (= -170° …  -10°)

  H-FOV per camera = 160°   (configurable, see CONFIG block)
  V-FOV per camera = 120°   (for reference only, not used in angle output)

YOLO strategy
-------------
  We run YOLO once per camera frame, NOT on a stitched panorama.
  Reasons:
    • No geometric distortion from stitching narrow-FOV fisheye strips
    • Each camera pixel coordinate maps cleanly to an angle via tan()
    • Four parallel inference threads → same wall-clock latency as one
    • No bounding-box split artefacts at stitch seams

Angle formula
-------------
  For a detection bounding box [bx, by, bw, bh] in pixel space
  (top-left corner + width/height, normalised 0–1 or absolute pixels):

      x_left   = bx          (left edge of box)
      x_right  = bx + bw     (right edge of box)
      x_centre = bx + bw/2   (centre of box)

  Horizontal offset from image centre (normalised -0.5 … +0.5):
      dx_left   = x_left  / frame_width  - 0.5
      dx_right  = x_right / frame_width  - 0.5

  Angle offset from camera optical axis (pinhole model):
      θ_left  = arctan(dx_left  * 2 * tan(HFOV/2))
      θ_right = arctan(dx_right * 2 * tan(HFOV/2))

  World angle (with QCar front = 0°, clockwise positive):
      world_left  = camera_centre_angle + θ_left
      world_right = camera_centre_angle + θ_right
      (normalised to 0–360°)

Streaming
---------
  • /video/<cam_id>   — MJPEG stream for each individual camera
  • /video/panorama   — side-by-side composite (0=front,1=right,2=back,3=left)
  • Socket.IO event 'detections' — JSON list of all current detections with angles
  • Socket.IO event 'frame_ready' — fired each cycle; browser requests /video/*

Usage
-----
  pip install flask flask-socketio numpy opencv-python
  # On Jetson also: pip install ultralytics   (YOLOv8)

  python qcar360_server.py --model yolov8n.pt --port 5000
  python qcar360_server.py --model yolov8n.engine --engine  # TensorRT
  python qcar360_server.py --demo   # synthetic frames, no hardware needed
"""

import argparse
import math
import threading
import time
import queue
import json
import logging
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, render_template_string
from flask_socketio import SocketIO

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  — edit these to match your hardware
# ══════════════════════════════════════════════════════════════════════════════
FRAME_WIDTH   = 820
FRAME_HEIGHT  = 410
FRAME_RATE    = 30.0          # capture fps (120 if hardware supports)
YOLO_WIDTH    = 640           # YOLO input resolution (square)
JPEG_QUALITY  = 75            # MJPEG stream quality
CONF_THRESH   = 0.40          # minimum YOLO confidence

# Camera indices and their centre angles (clockwise from front=0°)
CAMERA_CONFIG = [
    {"id": 0, "name": "FRONT",  "centre_deg": 0.0},
    {"id": 1, "name": "RIGHT",  "centre_deg": 90.0},
    {"id": 2, "name": "BACK",   "centre_deg": 180.0},
    {"id": 3, "name": "LEFT",   "centre_deg": 270.0},
]

# FOV
H_FOV_DEG = 160.0   # horizontal FOV per camera (4 × 160° → 360° with 80° overlap)
V_FOV_DEG = 120.0   # vertical FOV per camera (informational)

# COCO class names (80 classes, indices 0–79)
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","yield sign","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
    "cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave",
    "oven","toaster","sink","refrigerator","book","clock","vase",
    "scissors","teddy bear","hair drier","toothbrush",
]

# Colour palette for class drawing (HSV hue spread)
def _class_colour(cls_id: int):
    hue = int((cls_id * 137.508) % 180)
    hsv = np.uint8([[[hue, 220, 220]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

CLASS_COLOURS = [_class_colour(i) for i in range(len(COCO_CLASSES))]

# ══════════════════════════════════════════════════════════════════════════════
# Angle math
# ══════════════════════════════════════════════════════════════════════════════

_half_hfov_rad = math.radians(H_FOV_DEG / 2.0)
_tan_half_hfov  = math.tan(_half_hfov_rad)

def pixel_x_to_angle_offset(px: float, frame_w: int) -> float:
    """
    Map a pixel x-coordinate to an angular offset from camera optical axis.
    Returns degrees. Positive = right of centre.
    Uses pinhole model: θ = arctan(dx_norm * 2 * tan(HFOV/2))
    where dx_norm ∈ [-0.5, +0.5].
    """
    dx_norm = px / frame_w - 0.5
    angle_rad = math.atan(dx_norm * 2.0 * _tan_half_hfov)
    return math.degrees(angle_rad)


def world_angle(cam_centre_deg: float, offset_deg: float) -> float:
    """Combine camera centre + local offset → world angle [0, 360)."""
    return (cam_centre_deg + offset_deg) % 360.0


def angles_for_box(bx: float, by: float, bw: float, bh: float,
                   frame_w: int, cam_centre_deg: float):
    """
    Given a bounding box (x, y, w, h in pixels, x/y = top-left),
    return (world_left_deg, world_right_deg, world_centre_deg).
    """
    x_left   = bx
    x_right  = bx + bw
    x_centre = bx + bw / 2.0

    off_left   = pixel_x_to_angle_offset(x_left,   frame_w)
    off_right  = pixel_x_to_angle_offset(x_right,  frame_w)
    off_centre = pixel_x_to_angle_offset(x_centre, frame_w)

    return (
        world_angle(cam_centre_deg, off_left),
        world_angle(cam_centre_deg, off_right),
        world_angle(cam_centre_deg, off_centre),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Camera capture (Quanser + OpenCV fallback)
# ══════════════════════════════════════════════════════════════════════════════

class CameraCapture:
    """Thread-safe wrapper around one CSI camera."""

    def __init__(self, cam_id: int, use_demo: bool = False):
        self.cam_id    = cam_id
        self.use_demo  = use_demo
        self._frame    = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        self._lock     = threading.Lock()
        self._running  = False
        self._thread: Optional[threading.Thread] = None
        self._cap      = None
        self._use_quanser = False

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self._release()

    def get_frame(self) -> np.ndarray:
        with self._lock:
            return self._frame.copy()

    # ── internal ──────────────────────────────────────────────────────────────

    def _open(self):
        if self.use_demo:
            return  # no hardware

        # Try Quanser first
        try:
            from quanser.multimedia import (
                VideoCapture as QVC, ImageFormat, ImageDataType)
            self._cap = QVC(
                f"video://localhost:{self.cam_id}",
                FRAME_RATE, FRAME_WIDTH, FRAME_HEIGHT,
                ImageFormat.ROW_MAJOR_INTERLEAVED_BGR,
                ImageDataType.UINT8, None, 0)
            self._cap.start()
            self._use_quanser = True
            logging.info(f"[CAM {self.cam_id}] Opened via quanser.multimedia")
            return
        except Exception as e:
            logging.warning(f"[CAM {self.cam_id}] Quanser open failed ({e}), trying OpenCV")

        # OpenCV fallback
        cap = cv2.VideoCapture(self.cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          FRAME_RATE)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.cam_id}")
        self._cap = cap
        self._use_quanser = False
        logging.info(f"[CAM {self.cam_id}] Opened via cv2.VideoCapture")

    def _release(self):
        if self._cap is None:
            return
        try:
            if self._use_quanser:
                self._cap.stop()
                self._cap.close()
            else:
                self._cap.release()
        except Exception:
            pass
        self._cap = None

    def _loop(self):
        if not self.use_demo:
            self._open()

        buf = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        t   = 0.0

        while self._running:
            try:
                if self.use_demo:
                    frame = self._make_demo_frame(t)
                    t += 1.0 / FRAME_RATE
                    time.sleep(1.0 / FRAME_RATE)
                elif self._use_quanser:
                    got = self._cap.read(buf)
                    frame = buf.copy() if got else None
                else:
                    ret, frame = self._cap.read()
                    if not ret:
                        frame = None

                if frame is not None:
                    with self._lock:
                        self._frame = frame
            except Exception as ex:
                logging.error(f"[CAM {self.cam_id}] capture error: {ex}")
                time.sleep(0.05)

    def _make_demo_frame(self, t: float) -> np.ndarray:
        """Generate a synthetic frame for testing without hardware."""
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        # Gradient background per camera
        colours = [(0,60,120), (0,120,60), (120,60,0), (80,0,120)]
        c = colours[self.cam_id % 4]
        frame[:] = c
        # Moving circle to simulate a detection target
        cx = int((math.sin(t * 0.7 + self.cam_id) * 0.35 + 0.5) * FRAME_WIDTH)
        cy = FRAME_HEIGHT // 2
        cv2.circle(frame, (cx, cy), 40, (0, 255, 180), -1)
        cv2.putText(frame, f"CAM {self.cam_id} DEMO",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        return frame


# ══════════════════════════════════════════════════════════════════════════════
# YOLO inference worker (one per camera thread)
# ══════════════════════════════════════════════════════════════════════════════

class YOLOWorker:
    """Runs YOLO on frames from one camera, produces angle-annotated detections."""

    def __init__(self, cam_cfg: dict, model_path: str,
                 use_engine: bool = False, use_demo: bool = False):
        self.cam_id         = cam_cfg["id"]
        self.cam_name       = cam_cfg["name"]
        self.centre_deg     = cam_cfg["centre_deg"]
        self.model_path     = model_path
        self.use_engine     = use_engine
        self.use_demo       = use_demo

        self._model         = None
        self._latest_det    = []          # list of detection dicts
        self._latest_frame  = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        self._lock          = threading.Lock()
        self._running       = False
        self._thread: Optional[threading.Thread] = None

    def start(self, capture: CameraCapture):
        self._capture  = capture
        self._running  = True
        self._thread   = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def get_latest(self):
        """Returns (annotated_frame, detections_list) — thread-safe."""
        with self._lock:
            return self._latest_frame.copy(), list(self._latest_det)

    # ── internal ──────────────────────────────────────────────────────────────

    def _load_model(self):
        if self.use_demo:
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            logging.info(f"[YOLO CAM {self.cam_id}] Model loaded: {self.model_path}")
        except ImportError:
            logging.warning("[YOLO] ultralytics not installed — running in demo mode")
            self.use_demo = True
        except Exception as ex:
            logging.error(f"[YOLO CAM {self.cam_id}] Model load error: {ex}")
            self.use_demo = True

    def _infer(self, frame: np.ndarray):
        """Run YOLO on frame, return list of raw boxes."""
        resized = cv2.resize(frame, (YOLO_WIDTH, YOLO_WIDTH))
        results = self._model(resized, conf=CONF_THRESH, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                # xyxy in resized space → scale back to original
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                scale_x = FRAME_WIDTH  / YOLO_WIDTH
                scale_y = FRAME_HEIGHT / YOLO_WIDTH
                bx = x1 * scale_x
                by = y1 * scale_y
                bw = (x2 - x1) * scale_x
                bh = (y2 - y1) * scale_y
                boxes.append((cls_id, conf, bx, by, bw, bh))
        return boxes

    def _demo_detections(self, frame: np.ndarray, t: float):
        """Fake detections that move across the frame — for testing."""
        cx = int((math.sin(t * 0.7 + self.cam_id) * 0.35 + 0.5) * FRAME_WIDTH)
        cy = FRAME_HEIGHT // 2
        bx = cx - 45
        by = cy - 45
        return [(0, 0.92, float(bx), float(by), 90.0, 90.0)]  # person

    def _loop(self):
        self._load_model()
        t = 0.0

        while self._running:
            try:
                frame = self._capture.get_frame()

                if self.use_demo:
                    raw_boxes = self._demo_detections(frame, t)
                    t += 1.0 / FRAME_RATE
                    time.sleep(1.0 / FRAME_RATE)
                else:
                    raw_boxes = self._infer(frame)

                detections = []
                annotated  = frame.copy()

                for (cls_id, conf, bx, by, bw, bh) in raw_boxes:
                    # Skip out-of-bounds boxes
                    bx = max(0.0, bx)
                    by = max(0.0, by)
                    bw = min(bw, FRAME_WIDTH  - bx)
                    bh = min(bh, FRAME_HEIGHT - by)
                    if bw <= 0 or bh <= 0:
                        continue

                    ang_left, ang_right, ang_centre = angles_for_box(
                        bx, by, bw, bh, FRAME_WIDTH, self.centre_deg)

                    cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)
                    colour   = CLASS_COLOURS[cls_id % len(CLASS_COLOURS)]

                    det = {
                        "cam_id":      self.cam_id,
                        "cam_name":    self.cam_name,
                        "class_id":    cls_id,
                        "class_name":  cls_name,
                        "confidence":  round(conf, 3),
                        "bbox_px":     [round(bx), round(by), round(bw), round(bh)],
                        "angle_left":  round(ang_left,   1),
                        "angle_right": round(ang_right,  1),
                        "angle_centre":round(ang_centre, 1),
                    }
                    detections.append(det)

                    # Draw on frame
                    x1, y1 = int(bx), int(by)
                    x2, y2 = int(bx + bw), int(by + bh)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

                    label = (f"{cls_name} {conf:.2f} | "
                             f"{ang_left:.0f}°–{ang_right:.0f}°")
                    lw, lh = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(annotated,
                                  (x1, max(0, y1 - lh - 6)),
                                  (x1 + lw + 4, y1), colour, -1)
                    cv2.putText(annotated, label,
                                (x1 + 2, max(lh, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1,
                                cv2.LINE_AA)

                # HUD overlay: camera name + FOV arc indicator
                cv2.rectangle(annotated, (0, 0), (FRAME_WIDTH, 32), (0,0,0), -1)
                cv2.putText(annotated,
                            f"CAM {self.cam_id} | {self.cam_name} | "
                            f"centre={self.centre_deg:.0f}° | "
                            f"FOV {self.centre_deg - H_FOV_DEG/2:.0f}°→"
                            f"{self.centre_deg + H_FOV_DEG/2:.0f}°",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 128), 1, cv2.LINE_AA)

                with self._lock:
                    self._latest_frame = annotated
                    self._latest_det   = detections

            except Exception as ex:
                logging.error(f"[YOLO CAM {self.cam_id}] error: {ex}")
                time.sleep(0.05)


# ══════════════════════════════════════════════════════════════════════════════
# Detection aggregator — merges outputs from all 4 workers
# ══════════════════════════════════════════════════════════════════════════════

class DetectionAggregator:
    """
    Collects detections from all cameras every cycle and:
      1. Deduplicates objects that appear in overlapping FOV zones
         (same class, world-angle overlap > 30°)
      2. Emits a clean sorted list: angle_left asc
      3. Produces terminal-friendly text like:
             person   10.0° – 30.0°   (CAM 0 FRONT, conf 0.91)
    """

    OVERLAP_THRESHOLD_DEG = 30.0

    def merge(self, all_detections: list) -> list:
        """all_detections: flat list of det dicts from all cameras."""
        merged   = []
        used     = [False] * len(all_detections)

        for i, d in enumerate(all_detections):
            if used[i]:
                continue
            group = [d]
            used[i] = True
            for j in range(i + 1, len(all_detections)):
                if used[j]:
                    continue
                e = all_detections[j]
                if e["class_id"] != d["class_id"]:
                    continue
                # Check angular overlap
                overlap = self._arc_overlap(
                    d["angle_left"], d["angle_right"],
                    e["angle_left"], e["angle_right"])
                if overlap >= self.OVERLAP_THRESHOLD_DEG:
                    group.append(e)
                    used[j] = True

            # Best representative = highest confidence
            best = max(group, key=lambda x: x["confidence"])
            merged.append(best)

        merged.sort(key=lambda x: x["angle_left"])
        return merged

    @staticmethod
    def _arc_overlap(l1, r1, l2, r2):
        """Angular overlap between two arcs on a circle [0,360)."""
        # normalise so l <= r (handle wrap-around naively for now)
        def norm(a):
            return a % 360.0
        l1, r1 = norm(l1), norm(r1)
        l2, r2 = norm(l2), norm(r2)
        if r1 < l1:
            r1 += 360.0
        if r2 < l2:
            r2 += 360.0
        overlap = min(r1, r2) - max(l1, l2)
        return max(0.0, overlap)


# ══════════════════════════════════════════════════════════════════════════════
# Panorama composer — for /video/panorama endpoint
# ══════════════════════════════════════════════════════════════════════════════

def compose_panorama(frames: list) -> np.ndarray:
    """
    Arrange 4 frames in one horizontal strip:
      [FRONT | RIGHT | BACK | LEFT]
    Each frame is resized to a common height.
    """
    target_h = 200
    strips = []
    labels = ["FRONT (0°)", "RIGHT (90°)", "BACK (180°)", "LEFT (270°)"]
    for i, f in enumerate(frames):
        h, w = f.shape[:2]
        new_w = int(w * target_h / h)
        strip = cv2.resize(f, (new_w, target_h))
        cv2.putText(strip, labels[i], (6, target_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)
        # Divider line
        cv2.line(strip, (new_w-1, 0), (new_w-1, target_h), (80,80,80), 2)
        strips.append(strip)
    return np.hstack(strips)


# ══════════════════════════════════════════════════════════════════════════════
# Flask + SocketIO app
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["SECRET_KEY"] = "qcar2-360"
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# These are populated in main()
_captures: list  = []
_workers:  list  = []
_aggregator       = DetectionAggregator()


def _jpeg_encode(frame: np.ndarray) -> bytes:
    _, buf = cv2.imencode('.jpg', frame,
                          [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes()


def _mjpeg_stream(frame_fn):
    """Generator that yields MJPEG frames from a callable frame_fn()."""
    while True:
        frame = frame_fn()
        data  = _jpeg_encode(frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               data + b"\r\n")


@app.route("/video/<int:cam_id>")
def video_single(cam_id):
    if cam_id < 0 or cam_id >= len(_workers):
        return f"Camera {cam_id} not found", 404
    worker = _workers[cam_id]
    return Response(
        _mjpeg_stream(lambda w=worker: w.get_latest()[0]),
        mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video/panorama")
def video_panorama():
    def get_pano():
        frames = [w.get_latest()[0] for w in _workers]
        return compose_panorama(frames)
    return Response(
        _mjpeg_stream(get_pano),
        mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/detections")
def detections_json():
    """REST endpoint — latest merged detections as JSON."""
    all_dets = []
    for w in _workers:
        _, dets = w.get_latest()
        all_dets.extend(dets)
    merged = _aggregator.merge(all_dets)
    return {"detections": merged, "timestamp": time.time()}


@app.route("/")
def index():
    """Serve the single-page dashboard (HTML is in the companion file)."""
    try:
        with open("qcar360_dashboard.html") as f:
            return f.read()
    except FileNotFoundError:
        return ("<h2 style='font-family:monospace;color:#0f0;background:#111;"
                "padding:2rem'>qcar360_dashboard.html not found — "
                "place it alongside this server script.</h2>")


# ── Background pusher thread ──────────────────────────────────────────────────

def _push_detections_loop():
    """Continuously push merged detections to all connected browser clients."""
    while True:
        try:
            all_dets = []
            for w in _workers:
                _, dets = w.get_latest()
                all_dets.extend(dets)
            merged = _aggregator.merge(all_dets)

            # Terminal output
            if merged:
                print(f"\r\033[K[{time.strftime('%H:%M:%S')}] Detections:", end=" ")
                parts = [f"{d['class_name']} {d['angle_left']:.0f}°–{d['angle_right']:.0f}°"
                         for d in merged]
                print(" | ".join(parts), end="", flush=True)

            socketio.emit("detections", {
                "detections": merged,
                "timestamp":  time.time()
            })
        except Exception as ex:
            logging.error(f"[pusher] {ex}")
        time.sleep(0.1)   # 10 Hz detection updates


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global CONF_THRESH
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="QCar 2 360° Perception Server")
    parser.add_argument("--model",  default="yolov8n.pt",
                        help="Path to YOLO model (.pt or .engine)")
    parser.add_argument("--engine", action="store_true",
                        help="Model is a TensorRT .engine file")
    parser.add_argument("--demo",   action="store_true",
                        help="Run with synthetic frames (no hardware)")
    parser.add_argument("--port",   type=int, default=5000)
    parser.add_argument("--conf",   type=float, default=CONF_THRESH,
                        help=f"YOLO confidence threshold (default {CONF_THRESH})")
    args = parser.parse_args()

    CONF_THRESH = args.conf

    print("=" * 60)
    print("  QCar 2 — 360° Perception Server")
    print("=" * 60)
    print(f"  Model  : {args.model}")
    print(f"  Mode   : {'DEMO (synthetic)' if args.demo else 'LIVE (hardware)'}")
    print(f"  H-FOV  : {H_FOV_DEG}° per camera")
    print(f"  Conf   : {CONF_THRESH}")
    print(f"  Port   : {args.port}")
    print("-" * 60)

    # Start cameras
    for cfg in CAMERA_CONFIG:
        cap = CameraCapture(cfg["id"], use_demo=args.demo)
        cap.start()
        _captures.append(cap)
        print(f"  Camera {cfg['id']} ({cfg['name']}) started")

    time.sleep(0.3)  # let cameras warm up

    # Start YOLO workers
    for cfg, cap in zip(CAMERA_CONFIG, _captures):
        w = YOLOWorker(cfg, args.model,
                       use_engine=args.engine,
                       use_demo=args.demo)
        w.start(cap)
        _workers.append(w)
        print(f"  YOLO worker {cfg['id']} ({cfg['name']}) started")

    # Start detection push thread
    push_thread = threading.Thread(target=_push_detections_loop, daemon=True)
    push_thread.start()

    print(f"\n  Dashboard → http://0.0.0.0:{args.port}")
    print(f"  Panorama  → http://0.0.0.0:{args.port}/video/panorama")
    print(f"  Single    → http://0.0.0.0:{args.port}/video/0  (cam 0–3)")
    print(f"  JSON API  → http://0.0.0.0:{args.port}/detections")
    print("=" * 60)
    print("  Press Ctrl+C to stop\n")

    try:
        socketio.run(app, host="0.0.0.0", port=args.port,
                     use_reloader=False, log_output=False)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down …")
    finally:
        for w in _workers:
            w.stop()
        for c in _captures:
            c.stop()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
