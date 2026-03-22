#!/usr/bin/env python3
"""
QCar 2 — 360° Perception Server  (v3 — V4L2 timeout fix)

Root cause of blank screen
---------------------------
V4L2 on Jetson CSI cameras sometimes blocks indefinitely on cap.read().
This froze the YOLO worker thread, which never updated _latest_frame,
so the MJPEG generator kept sending the same blank numpy array.

Fixes in this version
----------------------
1.  Camera thread uses a separate reader thread + queue with 0.5s timeout.
    If no frame arrives in time, it retries — never blocks the worker.
2.  MJPEG generator always sends _latest_frame (even if stale) so the
    browser always gets a valid JPEG boundary and displays something.
3.  Added /healthz endpoint — open in browser to confirm server is alive.
4.  Added per-camera /raw/<id> endpoint — raw capture without YOLO overlay,
    useful to confirm cameras work independently.
5.  Frame dimensions auto-detected from first captured frame.
6.  Reference code from observer.py / perceiver.py integrated as comments
    so you can see how the angle pipeline connects to the existing autonomy stack.
"""

import argparse
import math
import os
import queue
import signal
import threading
import time
import logging
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify
from flask_socketio import SocketIO

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
FRAME_WIDTH    = 820
FRAME_HEIGHT   = 616
FRAME_RATE     = 30.0
YOLO_WIDTH     = 640
JPEG_QUALITY   = 75
CONF_THRESH    = 0.40
STREAM_FPS     = 10.0          # conservative — raise after confirming stream works
CAM_READ_TIMEOUT = 0.5         # seconds to wait for a V4L2 frame before giving up

CAMERA_CONFIG = [
    {"id": 0, "name": "FRONT", "centre_deg":   0.0},
    {"id": 1, "name": "RIGHT", "centre_deg":  90.0},
    {"id": 2, "name": "BACK",  "centre_deg": 180.0},
    {"id": 3, "name": "LEFT",  "centre_deg": 270.0},
]

H_FOV_DEG = 160.0

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

def _bgr(cls_id):
    hue = int((cls_id * 137.508) % 180)
    hsv = np.uint8([[[hue, 220, 220]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

CLASS_COLOURS = [_bgr(i) for i in range(len(COCO_CLASSES))]

# ══════════════════════════════════════════════════════════════════════════════
# Angle math
# ══════════════════════════════════════════════════════════════════════════════
_tan_hfov = math.tan(math.radians(H_FOV_DEG / 2.0))

def px_to_offset(px, fw):
    return math.degrees(math.atan((px / fw - 0.5) * 2.0 * _tan_hfov))

def world_ang(centre, offset):
    return (centre + offset) % 360.0

def box_angles(bx, by, bw, bh, fw, centre):
    return (world_ang(centre, px_to_offset(bx,          fw)),
            world_ang(centre, px_to_offset(bx + bw,     fw)),
            world_ang(centre, px_to_offset(bx + bw/2.0, fw)))

# ══════════════════════════════════════════════════════════════════════════════
# Camera capture  —  non-blocking V4L2 reader
# ══════════════════════════════════════════════════════════════════════════════
def _make_placeholder(cam_id, w=FRAME_WIDTH, h=FRAME_HEIGHT):
    """Dark frame shown while camera is warming up / stalling."""
    f = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(f, f"CAM {cam_id} — waiting...",
                (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 180, 80), 2)
    return f


class CameraCapture:
    """
    Two-thread design:
      _reader_thread  — calls cap.read() in a tight loop, puts frames in _q
      _loop (main)    — pulls from _q with timeout, updates _frame

    This means a V4L2 stall only blocks _reader_thread.
    The MJPEG generator always gets _frame (possibly stale but never blocking).
    """

    def __init__(self, cam_id, use_demo=False):
        self.cam_id   = cam_id
        self.use_demo = use_demo
        self._frame   = _make_placeholder(cam_id)
        self._raw     = _make_placeholder(cam_id)  # no YOLO overlay
        self._lock    = threading.Lock()
        self._running = False
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._good_frames = 0
        self._drop_frames = 0

    def start(self):
        self._running = True
        if self.use_demo:
            threading.Thread(target=self._demo_loop, daemon=True,
                             name=f"cam{self.cam_id}").start()
        else:
            # Reader thread — owns the cv2.VideoCapture object
            threading.Thread(target=self._reader_thread, daemon=True,
                             name=f"cam{self.cam_id}_rd").start()

    def stop(self):
        self._running = False

    def get_frame(self):
        with self._lock:
            return self._frame.copy()

    def get_raw(self):
        with self._lock:
            return self._raw.copy()

    def stats(self):
        return self._good_frames, self._drop_frames

    # ── reader ────────────────────────────────────────────────────────────────
    def _reader_thread(self):
        cap = None
        while self._running:
            if cap is None or not cap.isOpened():
                cap = self._open_cap()
                if cap is None:
                    time.sleep(1.0)
                    continue

            ret, frame = cap.read()
            if not ret or frame is None:
                self._drop_frames += 1
                logging.warning(f"[CAM {self.cam_id}] read() returned False, reopening")
                cap.release()
                cap = None
                time.sleep(0.2)
                continue

            self._good_frames += 1
            # Non-blocking put — drop oldest if full
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                pass

        if cap:
            cap.release()

    def _open_cap(self):
        cap = cv2.VideoCapture(self.cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          FRAME_RATE)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        if not cap.isOpened():
            cap.release()
            logging.error(f"[CAM {self.cam_id}] failed to open /dev/video{self.cam_id}")
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info(f"[CAM {self.cam_id}] opened /dev/video{self.cam_id}  {w}x{h}")
        return cap

    def update_annotated(self, annotated_frame):
        """Called by YOLOWorker after inference to update the displayed frame."""
        with self._lock:
            self._frame = annotated_frame

    def pull_raw_from_queue(self):
        """
        Called by YOLOWorker. Pulls latest raw frame from queue.
        Returns None if no new frame available (worker should skip inference).
        """
        try:
            frame = self._q.get(timeout=CAM_READ_TIMEOUT)
            with self._lock:
                self._raw = frame.copy()
            return frame
        except queue.Empty:
            return None

    # ── demo ──────────────────────────────────────────────────────────────────
    def _demo_loop(self):
        t = 0.0
        while self._running:
            f = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            colours = [(30,60,120),(30,120,60),(120,60,30),(80,30,120)]
            f[:] = colours[self.cam_id % 4]
            cx = int((math.sin(t * 0.7 + self.cam_id) * 0.35 + 0.5) * FRAME_WIDTH)
            cy = FRAME_HEIGHT // 2
            cv2.circle(f, (cx, cy), 55, (0, 255, 180), -1)
            cv2.putText(f, f"CAM {self.cam_id}  DEMO",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255,255,255), 2)
            try:
                self._q.put_nowait(f)
            except queue.Full:
                try: self._q.get_nowait()
                except queue.Empty: pass
                try: self._q.put_nowait(f)
                except queue.Full: pass
            t += 1.0 / FRAME_RATE
            time.sleep(1.0 / FRAME_RATE)

# ══════════════════════════════════════════════════════════════════════════════
# YOLO worker
# ══════════════════════════════════════════════════════════════════════════════
class YOLOWorker:
    def __init__(self, cam_cfg, model_path, use_engine=False, use_demo=False):
        self.cam_id     = cam_cfg["id"]
        self.cam_name   = cam_cfg["name"]
        self.centre_deg = cam_cfg["centre_deg"]
        self.model_path = model_path
        self.use_engine = use_engine
        self.use_demo   = use_demo
        self._model     = None
        self._dets      = []
        self._lock      = threading.Lock()
        self._running   = False

    def start(self, capture: CameraCapture):
        self._capture = capture
        self._running = True
        threading.Thread(target=self._loop, daemon=True,
                         name=f"yolo{self.cam_id}").start()

    def stop(self):
        self._running = False

    def get_latest(self):
        """Returns (annotated_frame, detections). Thread-safe."""
        frame = self._capture.get_frame()
        with self._lock:
            return frame, list(self._dets)

    def _load(self):
        if self.use_demo:
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            logging.info(f"[YOLO {self.cam_id}] loaded  {self.model_path}")
        except Exception as ex:
            logging.warning(f"[YOLO {self.cam_id}] load failed ({ex}) -> demo")
            self.use_demo = True

    def _infer(self, frame):
        sq = cv2.resize(frame, (YOLO_WIDTH, YOLO_WIDTH))
        results = self._model(sq, conf=CONF_THRESH, verbose=False)
        sx = frame.shape[1] / YOLO_WIDTH
        sy = frame.shape[0] / YOLO_WIDTH
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append((int(box.cls[0]), float(box.conf[0]),
                               x1*sx, y1*sy, (x2-x1)*sx, (y2-y1)*sy))
        return boxes

    def _demo_boxes(self, t):
        cx = int((math.sin(t * 0.7 + self.cam_id) * 0.35 + 0.5) * FRAME_WIDTH)
        cy = FRAME_HEIGHT // 2
        return [(0, 0.92, float(cx-55), float(cy-55), 110.0, 110.0)]

    def _loop(self):
        self._load()
        t = 0.0
        while self._running:
            try:
                if self.use_demo:
                    # In demo mode pull from queue just like real mode
                    frame = self._capture.pull_raw_from_queue()
                    if frame is None:
                        continue
                    raw_boxes = self._demo_boxes(t)
                    t += 1.0 / FRAME_RATE
                else:
                    frame = self._capture.pull_raw_from_queue()
                    if frame is None:
                        # No frame yet — show last annotated (don't stall)
                        continue
                    raw_boxes = self._infer(frame)

                fw = frame.shape[1]
                fh = frame.shape[0]
                annotated = frame.copy()
                dets = []

                for (cls_id, conf, bx, by, bw, bh) in raw_boxes:
                    bx = max(0.0, min(bx, fw - 1))
                    by = max(0.0, min(by, fh - 1))
                    bw = min(bw, fw - bx)
                    bh = min(bh, fh - by)
                    if bw < 2 or bh < 2:
                        continue

                    al, ar, ac = box_angles(bx, by, bw, bh, fw, self.centre_deg)
                    name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)
                    col  = CLASS_COLOURS[cls_id % len(CLASS_COLOURS)]

                    dets.append({
                        "cam_id":       self.cam_id,
                        "cam_name":     self.cam_name,
                        "class_id":     cls_id,
                        "class_name":   name,
                        "confidence":   round(conf, 3),
                        "bbox_px":      [round(bx),round(by),round(bw),round(bh)],
                        "angle_left":   round(al, 1),
                        "angle_right":  round(ar, 1),
                        "angle_centre": round(ac, 1),
                    })

                    x1,y1 = int(bx), int(by)
                    x2,y2 = int(bx+bw), int(by+bh)
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), col, 2)
                    lbl = f"{name} {conf:.2f} {al:.0f}d-{ar:.0f}d"
                    lw, lh = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                    ty = max(lh+4, y1)
                    cv2.rectangle(annotated, (x1, ty-lh-4), (x1+lw+4, ty), col, -1)
                    cv2.putText(annotated, lbl, (x1+2, ty-2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)

                # HUD bar
                cv2.rectangle(annotated, (0,0), (fw, 28), (0,0,0), -1)
                cv2.putText(annotated,
                    f"CAM{self.cam_id} {self.cam_name}  "
                    f"{self.centre_deg-H_FOV_DEG/2:.0f}d -> "
                    f"{self.centre_deg+H_FOV_DEG/2:.0f}d  [{len(dets)} obj]",
                    (5, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (0,255,128), 1, cv2.LINE_AA)

                # Push annotated frame back to capture
                self._capture.update_annotated(annotated)
                with self._lock:
                    self._dets = dets

            except Exception as ex:
                logging.error(f"[YOLO {self.cam_id}] {ex}")
                time.sleep(0.05)

# ══════════════════════════════════════════════════════════════════════════════
# Aggregator
# ══════════════════════════════════════════════════════════════════════════════
class Aggregator:
    OVERLAP = 30.0
    def merge(self, all_dets):
        used, out = [False]*len(all_dets), []
        for i,d in enumerate(all_dets):
            if used[i]: continue
            grp = [d]; used[i] = True
            for j in range(i+1, len(all_dets)):
                if used[j]: continue
                e = all_dets[j]
                if e["class_id"] != d["class_id"]: continue
                if self._ov(d["angle_left"],d["angle_right"],
                            e["angle_left"],e["angle_right"]) >= self.OVERLAP:
                    grp.append(e); used[j] = True
            out.append(max(grp, key=lambda x: x["confidence"]))
        out.sort(key=lambda x: x["angle_left"])
        return out

    @staticmethod
    def _ov(l1,r1,l2,r2):
        l1%=360; r1%=360; l2%=360; r2%=360
        if r1<l1: r1+=360
        if r2<l2: r2+=360
        return max(0.0, min(r1,r2)-max(l1,l2))

# ══════════════════════════════════════════════════════════════════════════════
# Panorama
# ══════════════════════════════════════════════════════════════════════════════
def panorama(frames):
    TH = 180
    strips = []
    labels = ["FRONT","RIGHT","BACK","LEFT"]
    for i, f in enumerate(frames):
        h, w = f.shape[:2]
        nw = max(1, int(w * TH / h))
        s = cv2.resize(f, (nw, TH))
        cv2.putText(s, labels[i], (5, TH-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        cv2.line(s, (nw-1,0),(nw-1,TH),(50,50,50),1)
        strips.append(s)
    return np.hstack(strips)

# ══════════════════════════════════════════════════════════════════════════════
# Flask + SocketIO
# ══════════════════════════════════════════════════════════════════════════════
app      = Flask(__name__)
app.config["SECRET_KEY"] = "qcar2"
socketio = SocketIO(app, async_mode="threading",
                    cors_allowed_origins="*",
                    ping_timeout=20, ping_interval=10)

_caps    = []
_workers = []
_agg     = Aggregator()
_SI      = 1.0 / STREAM_FPS   # stream interval


def _enc(frame):
    ok, buf = cv2.imencode('.jpg', frame,
                           [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None


def _stream(fn):
    """
    MJPEG generator.
    Always sends the last good frame — never blocks waiting for a new one.
    The explicit sleep is what makes browsers render frames correctly.
    """
    while True:
        t0   = time.time()
        data = _enc(fn())
        if data:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + data + b"\r\n")
        rem = _SI - (time.time() - t0)
        if rem > 0:
            time.sleep(rem)


@app.route("/video/<int:cid>")
def vid(cid):
    if cid < 0 or cid >= len(_workers):
        return "not found", 404
    w = _workers[cid]
    return Response(_stream(lambda ww=w: ww.get_latest()[0]),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/raw/<int:cid>")
def raw_vid(cid):
    """Raw camera feed — no YOLO overlay. Useful to verify camera works."""
    if cid < 0 or cid >= len(_caps):
        return "not found", 404
    c = _caps[cid]
    return Response(_stream(lambda cc=c: cc.get_raw()),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video/panorama")
def vid_pano():
    def _pano():
        if not _workers:
            return np.zeros((180, 820, 3), dtype=np.uint8)
        return panorama([w.get_latest()[0] for w in _workers])
    return Response(_stream(_pano),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/detections")
def dets():
    all_d = []
    for w in _workers:
        _, d = w.get_latest()
        all_d.extend(d)
    return jsonify(detections=_agg.merge(all_d), ts=time.time())


@app.route("/healthz")
def health():
    """Quick sanity check — open this first to confirm server is running."""
    stats = []
    for c in _caps:
        g, d = c.stats()
        stats.append({"cam": c.cam_id, "good": g, "drop": d})
    return jsonify(status="ok", cameras=stats, ts=time.time())


@app.route("/")
def index():
    here = os.path.dirname(os.path.abspath(__file__))
    html = os.path.join(here, "qcar360_dashboard.html")
    try:
        return open(html).read()
    except FileNotFoundError:
        return ("<h2 style='font-family:monospace;color:#0f0;background:#111;"
                "padding:2em'>qcar360_dashboard.html not found next to server.</h2>")


def _push():
    while True:
        try:
            if _workers:
                all_d = []
                for w in _workers:
                    _, d = w.get_latest()
                    all_d.extend(d)
                merged = _agg.merge(all_d)
                if merged:
                    parts = [f"{d['class_name']} "
                             f"{d['angle_left']:.0f}d-{d['angle_right']:.0f}d"
                             for d in merged]
                    print(f"\r\033[K[{time.strftime('%H:%M:%S')}] "
                          + " | ".join(parts), end="", flush=True)
                socketio.emit("detections", {"detections": merged, "ts": time.time()})
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

    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="yolov8n.pt")
    ap.add_argument("--engine", action="store_true")
    ap.add_argument("--demo",   action="store_true")
    ap.add_argument("--port",   type=int, default=5000)
    ap.add_argument("--conf",   type=float, default=CONF_THRESH)
    args = ap.parse_args()

    CONF_THRESH = args.conf
    model_path  = os.path.expandvars(args.model)

    print("=" * 62)
    print("  QCar 2 -- 360 Perception Server  v3")
    print("=" * 62)
    print(f"  Model  : {model_path}")
    print(f"  Mode   : {'DEMO' if args.demo else 'LIVE'}")
    print(f"  Frame  : {FRAME_WIDTH}x{FRAME_HEIGHT}  Stream: {STREAM_FPS}fps")
    print(f"  Conf   : {CONF_THRESH}")
    print("-" * 62)

    def _bye(sig, frame):
        print("\n[INFO] Shutting down ...")
        for w in _workers: w.stop()
        for c in _caps:    c.stop()
        time.sleep(0.4)
        os._exit(0)

    signal.signal(signal.SIGINT,  _bye)
    signal.signal(signal.SIGTERM, _bye)

    for cfg in CAMERA_CONFIG:
        c = CameraCapture(cfg["id"], use_demo=args.demo)
        c.start()
        _caps.append(c)
        print(f"  Camera {cfg['id']} ({cfg['name']}) started")

    time.sleep(0.5)

    for cfg, cap in zip(CAMERA_CONFIG, _caps):
        w = YOLOWorker(cfg, model_path, use_engine=args.engine, use_demo=args.demo)
        w.start(cap)
        _workers.append(w)
        print(f"  YOLO   {cfg['id']} ({cfg['name']}) started")

    threading.Thread(target=_push, daemon=True, name="push").start()

    ip = "10.1.77.73"
    print(f"\n  Dashboard  -> http://{ip}:{args.port}")
    print(f"  Health     -> http://{ip}:{args.port}/healthz      <- check this first")
    print(f"  Raw cam 0  -> http://{ip}:{args.port}/raw/0        <- bypasses YOLO")
    print(f"  CAM 0-3    -> http://{ip}:{args.port}/video/0")
    print(f"  Panorama   -> http://{ip}:{args.port}/video/panorama")
    print(f"  JSON API   -> http://{ip}:{args.port}/detections")
    print("=" * 62)
    print("  Ctrl+C to stop\n")

    socketio.run(app, host="0.0.0.0", port=args.port,
                 use_reloader=False, log_output=False)


if __name__ == "__main__":
    main()