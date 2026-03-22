#!/usr/bin/env python3
"""
QCar 2 — 360° Perception Server  (fixed build)
Fixes applied vs previous version:
  • MJPEG stream now sleeps between frames → browser actually renders them
  • Camera _loop now re-opens on read failure instead of silently looping
  • signal handler added so Ctrl+C actually kills the process
  • FRAME_HEIGHT corrected to 616 (actual IMX219 output)
  • Model path shell variable expansion fixed ($MODELSPATH -> os.path.expandvars)
  • SocketIO pusher: guard against empty workers list at startup
  • Panorama: graceful empty-frame fallback while cameras warm up
  • HTML served using script's own directory, not cwd
"""

import argparse
import math
import os
import signal
import sys
import threading
import time
import logging
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response
from flask_socketio import SocketIO

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
FRAME_WIDTH   = 820
FRAME_HEIGHT  = 616          # actual IMX219 height
FRAME_RATE    = 30.0
YOLO_WIDTH    = 640
JPEG_QUALITY  = 75
CONF_THRESH   = 0.40
STREAM_FPS    = 15.0         # MJPEG push rate

CAMERA_CONFIG = [
    {"id": 0, "name": "FRONT", "centre_deg":   0.0},
    {"id": 1, "name": "RIGHT", "centre_deg":  90.0},
    {"id": 2, "name": "BACK",  "centre_deg": 180.0},
    {"id": 3, "name": "LEFT",  "centre_deg": 270.0},
]

H_FOV_DEG = 160.0
V_FOV_DEG = 120.0

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

def _class_colour(cls_id: int):
    hue = int((cls_id * 137.508) % 180)
    hsv = np.uint8([[[hue, 220, 220]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

CLASS_COLOURS = [_class_colour(i) for i in range(len(COCO_CLASSES))]

# ══════════════════════════════════════════════════════════════════════════════
# Angle math
# ══════════════════════════════════════════════════════════════════════════════
_tan_half_hfov = math.tan(math.radians(H_FOV_DEG / 2.0))

def pixel_x_to_angle_offset(px: float, frame_w: int) -> float:
    dx_norm = px / frame_w - 0.5
    return math.degrees(math.atan(dx_norm * 2.0 * _tan_half_hfov))

def world_angle(cam_centre_deg: float, offset_deg: float) -> float:
    return (cam_centre_deg + offset_deg) % 360.0

def angles_for_box(bx, by, bw, bh, frame_w, cam_centre_deg):
    off_l = pixel_x_to_angle_offset(bx,          frame_w)
    off_r = pixel_x_to_angle_offset(bx + bw,     frame_w)
    off_c = pixel_x_to_angle_offset(bx + bw/2.0, frame_w)
    return (world_angle(cam_centre_deg, off_l),
            world_angle(cam_centre_deg, off_r),
            world_angle(cam_centre_deg, off_c))

# ══════════════════════════════════════════════════════════════════════════════
# Camera capture
# ══════════════════════════════════════════════════════════════════════════════
_BLANK = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

class CameraCapture:
    def __init__(self, cam_id: int, use_demo: bool = False):
        self.cam_id   = cam_id
        self.use_demo = use_demo
        self._frame   = _BLANK.copy()
        self._lock    = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True,
                                         name=f"cam{self.cam_id}")
        self._thread.start()

    def stop(self):
        self._running = False

    def get_frame(self) -> np.ndarray:
        with self._lock:
            return self._frame.copy()

    def _open_cv(self):
        cap = cv2.VideoCapture(self.cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          FRAME_RATE)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        if not cap.isOpened():
            cap.release()
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"[CAM {self.cam_id}] opened {w}x{h} @ {f:.0f}fps")
        return cap

    def _loop(self):
        t   = 0.0
        cap = None

        if not self.use_demo:
            cap = self._open_cv()
            if cap is None:
                logging.error(f"[CAM {self.cam_id}] failed to open")

        while self._running:
            try:
                if self.use_demo:
                    frame = self._demo_frame(t)
                    t += 1.0 / FRAME_RATE
                    time.sleep(1.0 / FRAME_RATE)
                elif cap is None:
                    time.sleep(2.0)
                    cap = self._open_cv()
                    continue
                else:
                    ret, frame = cap.read()
                    if not ret:
                        logging.warning(f"[CAM {self.cam_id}] read failed, reopening")
                        cap.release()
                        cap = None
                        time.sleep(0.5)
                        continue

                with self._lock:
                    self._frame = frame

            except Exception as ex:
                logging.error(f"[CAM {self.cam_id}] {ex}")
                time.sleep(0.1)

        if cap is not None:
            cap.release()

    def _demo_frame(self, t: float) -> np.ndarray:
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        colours = [(30,60,120),(30,120,60),(120,60,30),(80,30,120)]
        frame[:] = colours[self.cam_id % 4]
        cx = int((math.sin(t * 0.7 + self.cam_id) * 0.35 + 0.5) * FRAME_WIDTH)
        cy = FRAME_HEIGHT // 2
        cv2.circle(frame, (cx, cy), 50, (0, 255, 180), -1)
        cv2.putText(frame, f"CAM {self.cam_id}  DEMO",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 2)
        return frame

# ══════════════════════════════════════════════════════════════════════════════
# YOLO worker
# ══════════════════════════════════════════════════════════════════════════════
class YOLOWorker:
    def __init__(self, cam_cfg: dict, model_path: str,
                 use_engine: bool = False, use_demo: bool = False):
        self.cam_id     = cam_cfg["id"]
        self.cam_name   = cam_cfg["name"]
        self.centre_deg = cam_cfg["centre_deg"]
        self.model_path = model_path
        self.use_engine = use_engine
        self.use_demo   = use_demo
        self._model     = None
        self._latest_det   = []
        self._latest_frame = _BLANK.copy()
        self._lock         = threading.Lock()
        self._running      = False
        self._thread: Optional[threading.Thread] = None

    def start(self, capture: CameraCapture):
        self._capture = capture
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True,
                                         name=f"yolo{self.cam_id}")
        self._thread.start()

    def stop(self):
        self._running = False

    def get_latest(self):
        with self._lock:
            return self._latest_frame.copy(), list(self._latest_det)

    def _load_model(self):
        if self.use_demo:
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            logging.info(f"[YOLO {self.cam_id}] loaded  {self.model_path}")
        except Exception as ex:
            logging.warning(f"[YOLO {self.cam_id}] load failed ({ex}) -> demo mode")
            self.use_demo = True

    def _infer(self, frame: np.ndarray):
        sq = cv2.resize(frame, (YOLO_WIDTH, YOLO_WIDTH))
        results = self._model(sq, conf=CONF_THRESH, verbose=False)
        sx = FRAME_WIDTH  / YOLO_WIDTH
        sy = FRAME_HEIGHT / YOLO_WIDTH
        boxes = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append((cls_id, conf,
                               x1*sx, y1*sy,
                               (x2-x1)*sx, (y2-y1)*sy))
        return boxes

    def _demo_boxes(self, t: float):
        cx = int((math.sin(t * 0.7 + self.cam_id) * 0.35 + 0.5) * FRAME_WIDTH)
        cy = FRAME_HEIGHT // 2
        return [(0, 0.92, float(cx-50), float(cy-50), 100.0, 100.0)]

    def _loop(self):
        self._load_model()
        t = 0.0

        while self._running:
            try:
                frame = self._capture.get_frame()

                if self.use_demo:
                    raw = self._demo_boxes(t)
                    t  += 1.0 / FRAME_RATE
                    time.sleep(1.0 / FRAME_RATE)
                else:
                    raw = self._infer(frame)

                dets      = []
                annotated = frame.copy()

                for (cls_id, conf, bx, by, bw, bh) in raw:
                    bx = max(0.0, bx)
                    by = max(0.0, by)
                    bw = min(bw, FRAME_WIDTH  - bx)
                    bh = min(bh, FRAME_HEIGHT - by)
                    if bw <= 2 or bh <= 2:
                        continue

                    al, ar, ac = angles_for_box(bx, by, bw, bh,
                                                FRAME_WIDTH, self.centre_deg)
                    cls_name = (COCO_CLASSES[cls_id]
                                if cls_id < len(COCO_CLASSES) else str(cls_id))
                    col = CLASS_COLOURS[cls_id % len(CLASS_COLOURS)]

                    dets.append({
                        "cam_id":       self.cam_id,
                        "cam_name":     self.cam_name,
                        "class_id":     cls_id,
                        "class_name":   cls_name,
                        "confidence":   round(conf, 3),
                        "bbox_px":      [round(bx), round(by),
                                         round(bw), round(bh)],
                        "angle_left":   round(al, 1),
                        "angle_right":  round(ar, 1),
                        "angle_centre": round(ac, 1),
                    })

                    x1, y1 = int(bx), int(by)
                    x2, y2 = int(bx+bw), int(by+bh)
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), col, 2)
                    label = f"{cls_name} {conf:.2f} | {al:.0f}deg-{ar:.0f}deg"
                    lw, lh = cv2.getTextSize(label,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    ty = max(lh+4, y1)
                    cv2.rectangle(annotated, (x1, ty-lh-4), (x1+lw+4, ty),
                                  col, -1)
                    cv2.putText(annotated, label, (x1+2, ty-2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1,
                                cv2.LINE_AA)

                # HUD
                cv2.rectangle(annotated, (0,0), (FRAME_WIDTH, 30), (0,0,0), -1)
                cv2.putText(annotated,
                    f"CAM{self.cam_id} {self.cam_name} | "
                    f"{self.centre_deg-H_FOV_DEG/2:.0f}->  "
                    f"{self.centre_deg+H_FOV_DEG/2:.0f} deg  "
                    f"[{len(dets)} obj]",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0,255,128), 1, cv2.LINE_AA)

                with self._lock:
                    self._latest_frame = annotated
                    self._latest_det   = dets

            except Exception as ex:
                logging.error(f"[YOLO {self.cam_id}] {ex}")
                time.sleep(0.05)

# ══════════════════════════════════════════════════════════════════════════════
# Detection aggregator
# ══════════════════════════════════════════════════════════════════════════════
class DetectionAggregator:
    OVERLAP_DEG = 30.0

    def merge(self, all_dets: list) -> list:
        used   = [False] * len(all_dets)
        merged = []
        for i, d in enumerate(all_dets):
            if used[i]:
                continue
            group = [d]
            used[i] = True
            for j in range(i+1, len(all_dets)):
                if used[j]:
                    continue
                e = all_dets[j]
                if e["class_id"] != d["class_id"]:
                    continue
                if self._overlap(d["angle_left"], d["angle_right"],
                                 e["angle_left"], e["angle_right"]) >= self.OVERLAP_DEG:
                    group.append(e)
                    used[j] = True
            merged.append(max(group, key=lambda x: x["confidence"]))
        merged.sort(key=lambda x: x["angle_left"])
        return merged

    @staticmethod
    def _overlap(l1, r1, l2, r2):
        l1 %= 360; r1 %= 360; l2 %= 360; r2 %= 360
        if r1 < l1: r1 += 360
        if r2 < l2: r2 += 360
        return max(0.0, min(r1, r2) - max(l1, l2))

# ══════════════════════════════════════════════════════════════════════════════
# Panorama
# ══════════════════════════════════════════════════════════════════════════════
def compose_panorama(frames: list) -> np.ndarray:
    TARGET_H = 180
    labels = ["FRONT 0deg", "RIGHT 90deg", "BACK 180deg", "LEFT 270deg"]
    strips = []
    for i, f in enumerate(frames):
        h, w = f.shape[:2]
        nw   = max(1, int(w * TARGET_H / h))
        s    = cv2.resize(f, (nw, TARGET_H))
        cv2.putText(s, labels[i], (6, TARGET_H-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)
        cv2.line(s, (nw-1,0), (nw-1, TARGET_H), (60,60,60), 2)
        strips.append(s)
    return np.hstack(strips)

# ══════════════════════════════════════════════════════════════════════════════
# Flask + SocketIO
# ══════════════════════════════════════════════════════════════════════════════
app      = Flask(__name__)
app.config["SECRET_KEY"] = "qcar2-360"
socketio = SocketIO(app, async_mode="threading",
                    cors_allowed_origins="*",
                    ping_timeout=10, ping_interval=5)

_captures:   list = []
_workers:    list = []
_aggregator        = DetectionAggregator()
_STREAM_INTERVAL   = 1.0 / STREAM_FPS


def _jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode('.jpg', frame,
                           [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else b""


def _mjpeg_gen(frame_fn):
    """
    THE KEY FIX: explicit sleep so the generator does not spin at 100% CPU.
    Without this sleep the browser TCP buffer fills instantly, it can't
    parse JPEG boundaries fast enough, and renders nothing.
    """
    while True:
        t0    = time.time()
        frame = frame_fn()
        data  = _jpeg(frame)
        if data:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n"
                   + data + b"\r\n")
        elapsed = time.time() - t0
        rem     = _STREAM_INTERVAL - elapsed
        if rem > 0:
            time.sleep(rem)


@app.route("/video/<int:cam_id>")
def video_single(cam_id):
    if cam_id < 0 or cam_id >= len(_workers):
        return f"Camera {cam_id} not found", 404
    w = _workers[cam_id]
    return Response(_mjpeg_gen(lambda ww=w: ww.get_latest()[0]),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video/panorama")
def video_panorama():
    def pano():
        if not _workers:
            return _BLANK
        frames = [w.get_latest()[0] for w in _workers]
        return compose_panorama(frames)
    return Response(_mjpeg_gen(pano),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/detections")
def detections_json():
    all_dets = []
    for w in _workers:
        _, dets = w.get_latest()
        all_dets.extend(dets)
    return {"detections": _aggregator.merge(all_dets), "ts": time.time()}


@app.route("/")
def index():
    # Always look next to the script file, not cwd
    here = os.path.dirname(os.path.abspath(__file__))
    html = os.path.join(here, "qcar360_dashboard.html")
    try:
        with open(html) as f:
            return f.read()
    except FileNotFoundError:
        return ("<h2 style='font-family:monospace;color:#0f0;background:#111;"
                "padding:2rem'>qcar360_dashboard.html not found next to "
                "qcar360_server.py</h2>")


def _push_loop():
    while True:
        try:
            if _workers:
                all_dets = []
                for w in _workers:
                    _, dets = w.get_latest()
                    all_dets.extend(dets)
                merged = _aggregator.merge(all_dets)

                if merged:
                    parts = [
                        f"{d['class_name']} "
                        f"{d['angle_left']:.0f}deg-{d['angle_right']:.0f}deg"
                        for d in merged
                    ]
                    print(f"\r\033[K[{time.strftime('%H:%M:%S')}] "
                          + " | ".join(parts), end="", flush=True)

                socketio.emit("detections",
                              {"detections": merged, "ts": time.time()})
        except Exception as ex:
            logging.error(f"[push] {ex}")
        time.sleep(0.1)

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global CONF_THRESH

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="QCar 2 360 Perception Server")
    parser.add_argument("--model",  default="yolov8n.pt")
    parser.add_argument("--engine", action="store_true")
    parser.add_argument("--demo",   action="store_true")
    parser.add_argument("--port",   type=int, default=5000)
    parser.add_argument("--conf",   type=float, default=CONF_THRESH)
    args = parser.parse_args()

    CONF_THRESH = args.conf

    # Expand shell variables like $MODELSPATH
    model_path = os.path.expandvars(args.model)

    print("=" * 60)
    print("  QCar 2 -- 360 Perception Server")
    print("=" * 60)
    print(f"  Model  : {model_path}")
    print(f"  Mode   : {'DEMO' if args.demo else 'LIVE'}")
    print(f"  Frame  : {FRAME_WIDTH}x{FRAME_HEIGHT}  Stream: {STREAM_FPS}fps")
    print(f"  Conf   : {CONF_THRESH}")
    print("-" * 60)

    # Ctrl+C handler — force-kill everything cleanly
    def _shutdown(sig, frame):
        print("\n[INFO] Shutting down ...")
        for w in _workers:
            w.stop()
        for c in _captures:
            c.stop()
        time.sleep(0.4)
        os._exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Start cameras
    for cfg in CAMERA_CONFIG:
        cap = CameraCapture(cfg["id"], use_demo=args.demo)
        cap.start()
        _captures.append(cap)
        print(f"  Camera {cfg['id']} ({cfg['name']}) started")

    time.sleep(0.4)

    # Start YOLO workers
    for cfg, cap in zip(CAMERA_CONFIG, _captures):
        w = YOLOWorker(cfg, model_path,
                       use_engine=args.engine,
                       use_demo=args.demo)
        w.start(cap)
        _workers.append(w)
        print(f"  YOLO   {cfg['id']} ({cfg['name']}) started")

    threading.Thread(target=_push_loop, daemon=True, name="push").start()

    print(f"\n  Dashboard -> http://10.1.77.73:{args.port}")
    print(f"  CAM 0-3   -> http://10.1.77.73:{args.port}/video/0")
    print(f"  Panorama  -> http://10.1.77.73:{args.port}/video/panorama")
    print(f"  JSON API  -> http://10.1.77.73:{args.port}/detections")
    print("=" * 60)
    print("  Ctrl+C to stop\n")

    socketio.run(app, host="0.0.0.0", port=args.port,
                 use_reloader=False, log_output=False)


if __name__ == "__main__":
    main()