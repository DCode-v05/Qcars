#!/usr/bin/env python3
"""
QCar 2 — 360° Perception Server  v5  (Quanser VideoCapture)

Uses quanser.multimedia.VideoCapture to access CSI cameras via
nvargus-daemon, since OpenCV is compiled without GStreamer and
cv2.VideoCapture cannot open CSI cameras by index on Jetson.

Architecture
------------
  ISPScheduler (1 thread)
    └─ round-robin: cap0.read() → cap1.read() → cap2.read() → cap3.read()
    └─ puts frames into per-camera queues

  YOLOWorker × 4 (4 threads)
    └─ pulls from camera queue → runs inference → updates annotated frame

  Flask MJPEG streams
    └─ always send last annotated frame (never block)
"""

import argparse
import math
import os
import queue
import signal
import threading
import time
import logging
from typing import List, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify
from flask_socketio import SocketIO
from quanser.multimedia import VideoCapture, ImageFormat, ImageDataType
from quanser.multimedia.exceptions import MediaError

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
YOLO_WIDTH     = 640
JPEG_QUALITY   = 75
CONF_THRESH    = 0.40
STREAM_FPS     = 8.0           # MJPEG push rate to browser
CAM_STAGGER_S  = 0.30          # delay between opening each camera (let ISP settle)
ISP_GRAB_DELAY = 0.04          # delay between sequential grabs (40ms)

# Resolution candidates — tried in order until one works per camera
# IMX219 modes first, then AR0144 / common V4L2 modes
VIDEO_MODES = [
    (820, 410, 120.0),   # IMX219 Mode 3 — confirmed working on QCar 2
    (820, 616, 60.0),    # IMX219 Mode 4
    (640, 480, 30.0),    # universal fallback
    (1280, 720, 60.0),   # common HD
    (1640, 820, 30.0),   # IMX219 half-res
    (1640, 1232, 30.0),  # IMX219 full-ish
]

CAMERA_CONFIG = [
    {"id": 0, "name": "RIGHT", "centre_deg":  90.0},
    {"id": 1, "name": "BACK",  "centre_deg": 180.0},
    {"id": 2, "name": "FRONT", "centre_deg":   0.0},
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
    return (world_ang(centre, px_to_offset(bx,         fw)),
            world_ang(centre, px_to_offset(bx + bw,    fw)),
            world_ang(centre, px_to_offset(bx + bw/2., fw)))

# ══════════════════════════════════════════════════════════════════════════════
# ISP Scheduler  —  THE KEY FIX
# One thread, sequential round-robin reads, stagger between cameras
# ══════════════════════════════════════════════════════════════════════════════
class ISPScheduler:
    """
    Grabs frames from all 4 CSI cameras via quanser.multimedia.VideoCapture.

    Only ONE camera is open at any given time (Jetson ISP limitation).
    Cycle: open cam → read several frames → stop/close → next cam.
    """

    READS_PER_CAM = 5   # frames to read before rotating to next camera

    def __init__(self, cam_ids: List[int], queues: List[queue.Queue]):
        self.cam_ids   = cam_ids
        self.queues    = queues
        self._running  = False
        self._thread: Optional[threading.Thread] = None
        self._good     = [0] * len(cam_ids)
        self._fail     = [0] * len(cam_ids)
        self._modes: List[Optional[tuple]] = [None] * len(cam_ids)
        self._bufs: List[Optional[np.ndarray]] = [None] * len(cam_ids)

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True,
                                         name="isp_sched")
        self._thread.start()

    def stop(self):
        self._running = False

    def stats(self):
        return list(zip(self._good, self._fail))

    def get_mode(self, idx):
        return self._modes[idx]

    # ── internal ──────────────────────────────────────────────────────────────
    def _try_open(self, cam_id, w, h, fps):
        """Open a camera using two-step pattern. Returns cap or None."""
        url = f"video://localhost:{cam_id}"
        cap = VideoCapture()
        try:
            cap.open(url, fps, w, h,
                     ImageFormat.ROW_MAJOR_INTERLEAVED_BGR,
                     ImageDataType.UINT8, [], 0)
            cap.start()
            return cap
        except MediaError as me:
            logging.warning(f"[ISP] cam {cam_id} open({w}x{h}@{fps}) "
                            f"failed: {me.get_error_message()}")
        except Exception as ex:
            logging.warning(f"[ISP] cam {cam_id} open({w}x{h}@{fps}) "
                            f"failed: {ex}")
        try:
            cap.close()
        except Exception:
            pass
        return None

    def _close_cap(self, cap):
        if cap is None:
            return
        try:
            cap.stop()
        except Exception:
            pass
        try:
            cap.close()
        except Exception:
            pass

    def _wait_first_frame(self, cap, buf):
        """Poll read() until a frame arrives (up to 1s)."""
        for _ in range(100):
            try:
                if cap.read(buf):
                    return True
            except Exception:
                return False
            time.sleep(0.01)
        return False

    def _probe_camera(self, cam_id: int, idx: int):
        """Try each VIDEO_MODE until one works. Closes cap after probe."""
        for attempt in range(3):
            for w, h, fps in VIDEO_MODES:
                if not self._running:
                    return
                cap = self._try_open(cam_id, w, h, fps)
                if cap is None:
                    time.sleep(0.05)
                    continue
                test_buf = np.zeros((h, w, 3), dtype=np.uint8)
                got = self._wait_first_frame(cap, test_buf)
                self._close_cap(cap)
                if got:
                    self._modes[idx] = (w, h, fps)
                    self._bufs[idx]  = np.zeros((h, w, 3), dtype=np.uint8)
                    logging.info(f"[ISP] cam {cam_id} probed OK: {w}x{h} @ {fps}fps")
                    return
                else:
                    logging.warning(f"[ISP] cam {cam_id} opened {w}x{h}@{fps} "
                                    f"but read failed")
            if attempt < 2:
                logging.warning(f"[ISP] cam {cam_id}: attempt {attempt+1} failed, "
                                f"retrying in 1s...")
                time.sleep(1.0)
        logging.error(f"[ISP] cam {cam_id}: ALL video modes failed after 3 attempts!")

    def _grab_camera(self, cam_id: int, idx: int, q: queue.Queue):
        """Open one camera, read several frames, close it, push last to queue."""
        mode = self._modes[idx]
        if mode is None:
            return
        w, h, fps = mode
        buf = self._bufs[idx]

        cap = self._try_open(cam_id, w, h, fps)
        if cap is None:
            self._fail[idx] += 1
            return

        got_any = False
        try:
            # Wait for first frame (ISP warmup)
            if not self._wait_first_frame(cap, buf):
                self._fail[idx] += 1
                return
            got_any = True
            self._good[idx] += 1

            # Read additional frames to get the freshest one
            for _ in range(self.READS_PER_CAM - 1):
                if not self._running:
                    break
                try:
                    if cap.read(buf):
                        self._good[idx] += 1
                    else:
                        self._fail[idx] += 1
                except Exception:
                    self._fail[idx] += 1
        finally:
            self._close_cap(cap)

        if got_any:
            frame = buf.copy()
            if q.full():
                try: q.get_nowait()
                except queue.Empty: pass
            try: q.put_nowait(frame)
            except queue.Full: pass

    def _loop(self):
        n = len(self.cam_ids)

        # Phase 1: probe each camera (one at a time, close after each)
        logging.info(f"[ISP] probing {n} cameras for supported video modes...")
        for i, cam_id in enumerate(self.cam_ids):
            if not self._running:
                return
            self._probe_camera(cam_id, i)
            time.sleep(CAM_STAGGER_S)

        resolved = sum(1 for m in self._modes if m is not None)
        logging.info(f"[ISP] {resolved}/{n} cameras resolved, starting capture loop")

        # Phase 2: round-robin — open ONE camera, grab frames, close, next
        idx = 0
        while self._running:
            if self._modes[idx] is not None:
                self._grab_camera(self.cam_ids[idx], idx, self.queues[idx])
                time.sleep(CAM_STAGGER_S)  # let ISP settle between cameras
            idx = (idx + 1) % n


# Default placeholder size (used before probe completes)
_PH_W, _PH_H = 820, 410

# ══════════════════════════════════════════════════════════════════════════════
# Per-camera frame store  (replaces CameraCapture)
# ══════════════════════════════════════════════════════════════════════════════
def _placeholder(cam_id):
    f = np.zeros((_PH_H, _PH_W, 3), dtype=np.uint8)
    cv2.putText(f, f"CAM {cam_id} — warming up",
                (20, _PH_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 180, 80), 2)
    return f


class CamStore:
    """Holds the latest annotated frame for one camera. Thread-safe."""
    def __init__(self, cam_id):
        self.cam_id   = cam_id
        self._frame   = _placeholder(cam_id)
        self._raw     = _placeholder(cam_id)
        self._lock    = threading.Lock()

    def put_raw(self, frame):
        with self._lock:
            self._raw = frame.copy()

    def put_annotated(self, frame):
        with self._lock:
            self._frame = frame.copy()

    def get_annotated(self):
        with self._lock:
            return self._frame.copy()

    def get_raw(self):
        with self._lock:
            return self._raw.copy()


# ══════════════════════════════════════════════════════════════════════════════
# YOLO Worker  (one per camera, pulls from ISP queue)
# ══════════════════════════════════════════════════════════════════════════════
class YOLOWorker:
    def __init__(self, cam_cfg, model_path, store: CamStore,
                 isp_queue: queue.Queue,
                 use_engine=False, use_demo=False):
        self.cam_id     = cam_cfg["id"]
        self.cam_name   = cam_cfg["name"]
        self.centre_deg = cam_cfg["centre_deg"]
        self.model_path = model_path
        self.store      = store
        self.isp_q      = isp_queue
        self.use_engine = use_engine
        self.use_demo   = use_demo
        self._model     = None
        self._dets      = []
        self._lock      = threading.Lock()
        self._running   = False
        self._demo_t    = 0.0

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True,
                         name=f"yolo{self.cam_id}").start()

    def stop(self):
        self._running = False

    def get_dets(self):
        with self._lock:
            return list(self._dets)

    def _load(self):
        if self.use_demo:
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            logging.info(f"[YOLO {self.cam_id}] loaded {self.model_path}")
        except Exception as ex:
            logging.warning(f"[YOLO {self.cam_id}] load failed ({ex}) -> demo")
            self.use_demo = True

    def _infer(self, frame):
        sq = cv2.resize(frame, (YOLO_WIDTH, YOLO_WIDTH))
        results = self._model(sq, conf=CONF_THRESH, verbose=False)
        fw, fh  = frame.shape[1], frame.shape[0]
        sx, sy  = fw / YOLO_WIDTH, fh / YOLO_WIDTH
        boxes = []
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                boxes.append((int(box.cls[0]), float(box.conf[0]),
                               x1*sx, y1*sy, (x2-x1)*sx, (y2-y1)*sy))
        return boxes

    def _demo_boxes(self):
        cx = int((math.sin(self._demo_t * 0.7 + self.cam_id) * 0.35 + 0.5)
                 * _PH_W)
        cy = _PH_H // 2
        self._demo_t += 1.0 / 15.0
        return [(0, 0.92, float(cx-55), float(cy-55), 110.0, 110.0)]

    def _annotate(self, frame, raw_boxes):
        fw, fh    = frame.shape[1], frame.shape[0]
        annotated = frame.copy()
        dets      = []

        for (cls_id, conf, bx, by, bw, bh) in raw_boxes:
            bx = max(0., min(bx, fw-1));  bw = min(bw, fw-bx)
            by = max(0., min(by, fh-1));  bh = min(bh, fh-by)
            if bw < 2 or bh < 2:
                continue

            al, ar, ac = box_angles(bx, by, bw, bh, fw, self.centre_deg)
            name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)
            col  = CLASS_COLOURS[cls_id % len(CLASS_COLOURS)]

            dets.append({
                "cam_id": self.cam_id, "cam_name": self.cam_name,
                "class_id": cls_id, "class_name": name,
                "confidence": round(conf, 3),
                "bbox_px": [round(bx),round(by),round(bw),round(bh)],
                "angle_left": round(al,1), "angle_right": round(ar,1),
                "angle_centre": round(ac,1),
            })

            x1,y1 = int(bx),int(by);  x2,y2 = int(bx+bw),int(by+bh)
            cv2.rectangle(annotated, (x1,y1),(x2,y2), col, 2)
            lbl = f"{name} {conf:.2f} {al:.0f}d-{ar:.0f}d"
            lw,lh = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            ty = max(lh+4, y1)
            cv2.rectangle(annotated,(x1,ty-lh-4),(x1+lw+4,ty),col,-1)
            cv2.putText(annotated, lbl,(x1+2,ty-2),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,0),1,cv2.LINE_AA)

        # HUD
        cv2.rectangle(annotated,(0,0),(fw,28),(0,0,0),-1)
        cv2.putText(annotated,
            f"CAM{self.cam_id} {self.cam_name}  "
            f"{self.centre_deg-H_FOV_DEG/2:.0f}d->{self.centre_deg+H_FOV_DEG/2:.0f}d"
            f"  [{len(dets)} obj]",
            (5,19), cv2.FONT_HERSHEY_SIMPLEX, 0.52,(0,255,128),1,cv2.LINE_AA)

        return annotated, dets

    def _loop(self):
        self._load()
        while self._running:
            try:
                if self.use_demo:
                    # pull from queue (demo frames are generated by ISPScheduler
                    # which also serves demo frames, but in demo mode we generate here)
                    frame = self._make_demo_frame()
                    raw_boxes = self._demo_boxes()
                    time.sleep(1.0 / 15.0)
                else:
                    try:
                        frame = self.isp_q.get(timeout=1.0)
                    except queue.Empty:
                        # No frame from ISP — keep showing last annotated
                        continue
                    self.store.put_raw(frame)
                    raw_boxes = self._infer(frame)

                annotated, dets = self._annotate(frame, raw_boxes)
                self.store.put_annotated(annotated)
                with self._lock:
                    self._dets = dets

            except Exception as ex:
                logging.error(f"[YOLO {self.cam_id}] {ex}")
                time.sleep(0.05)

    def _make_demo_frame(self):
        f = np.zeros((_PH_H, _PH_W, 3), dtype=np.uint8)
        colours = [(30,60,120),(30,120,60),(120,60,30),(80,30,120)]
        f[:] = colours[self.cam_id % 4]
        cx = int((math.sin(self._demo_t * 0.7 + self.cam_id)*0.35+0.5)*_PH_W)
        cy = _PH_H // 2
        cv2.circle(f, (cx,cy), 55, (0,255,180), -1)
        cv2.putText(f, f"CAM {self.cam_id}  DEMO",
                    (20,65), cv2.FONT_HERSHEY_SIMPLEX,1.4,(255,255,255),2)
        return f


# ══════════════════════════════════════════════════════════════════════════════
# Aggregator
# ══════════════════════════════════════════════════════════════════════════════
class Aggregator:
    OVERLAP = 30.0
    def merge(self, all_dets):
        used, out = [False]*len(all_dets), []
        for i,d in enumerate(all_dets):
            if used[i]: continue
            grp=[d]; used[i]=True
            for j in range(i+1,len(all_dets)):
                if used[j]: continue
                e=all_dets[j]
                if e["class_id"]!=d["class_id"]: continue
                if self._ov(d["angle_left"],d["angle_right"],
                            e["angle_left"],e["angle_right"])>=self.OVERLAP:
                    grp.append(e); used[j]=True
            out.append(max(grp, key=lambda x:x["confidence"]))
        out.sort(key=lambda x:x["angle_left"])
        return out

    @staticmethod
    def _ov(l1,r1,l2,r2):
        l1%=360;r1%=360;l2%=360;r2%=360
        if r1<l1: r1+=360
        if r2<l2: r2+=360
        return max(0.,min(r1,r2)-max(l1,l2))


# ══════════════════════════════════════════════════════════════════════════════
# Panorama
# ══════════════════════════════════════════════════════════════════════════════
def panorama(frames):
    TH=180; labels=["RIGHT","BACK","FRONT","LEFT"]; strips=[]
    for i,f in enumerate(frames):
        h,w=f.shape[:2]; nw=max(1,int(w*TH/h)); s=cv2.resize(f,(nw,TH))
        cv2.putText(s,labels[i],(5,TH-6),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,0),1)
        cv2.line(s,(nw-1,0),(nw-1,TH),(50,50,50),1); strips.append(s)
    return np.hstack(strips)


# ══════════════════════════════════════════════════════════════════════════════
# Flask + SocketIO
# ══════════════════════════════════════════════════════════════════════════════
app      = Flask(__name__)
app.config["SECRET_KEY"] = "qcar2"
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*",
                    ping_timeout=20, ping_interval=10)

_stores:  List[CamStore]   = []
_workers: List[YOLOWorker] = []
_isp:     Optional[ISPScheduler] = None
_agg      = Aggregator()
_SI       = 1.0 / STREAM_FPS


def _enc(frame):
    ok,buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None

def _stream(fn):
    while True:
        t0=time.time(); data=_enc(fn())
        if data:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+data+b"\r\n"
        rem=_SI-(time.time()-t0)
        if rem>0: time.sleep(rem)


@app.route("/video/<int:cid>")
def vid(cid):
    if cid<0 or cid>=len(_stores): return "not found",404
    s=_stores[cid]
    return Response(_stream(s.get_annotated),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/raw/<int:cid>")
def raw_vid(cid):
    if cid<0 or cid>=len(_stores): return "not found",404
    s=_stores[cid]
    return Response(_stream(s.get_raw),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video/panorama")
def vid_pano():
    def _p():
        if not _stores: return np.zeros((180,820,3),dtype=np.uint8)
        return panorama([s.get_annotated() for s in _stores])
    return Response(_stream(_p),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/detections")
def dets_route():
    all_d=[]
    for w in _workers: all_d.extend(w.get_dets())
    return jsonify(detections=_agg.merge(all_d), ts=time.time())

@app.route("/healthz")
def health():
    stats=[]
    if _isp:
        for (g,f),(cfg) in zip(_isp.stats(), CAMERA_CONFIG):
            stats.append({"cam":cfg["id"],"name":cfg["name"],"good":g,"fail":f})
    return jsonify(status="ok", cameras=stats, ts=time.time())

@app.route("/")
def index():
    here=os.path.dirname(os.path.abspath(__file__))
    html=os.path.join(here,"qcar360_dashboard.html")
    try: return open(html).read()
    except FileNotFoundError:
        return ("<h2 style='font-family:monospace;color:#0f0;background:#111;"
                "padding:2em'>qcar360_dashboard.html not found next to server.</h2>")

def _push():
    while True:
        try:
            if _workers:
                all_d=[]
                for w in _workers: all_d.extend(w.get_dets())
                merged=_agg.merge(all_d)
                if merged:
                    parts=[f"{d['class_name']} "
                           f"{d['angle_left']:.0f}d-{d['angle_right']:.0f}d"
                           for d in merged]
                    print(f"\r\033[K[{time.strftime('%H:%M:%S')}] "
                          +"|".join(parts), end="", flush=True)
                socketio.emit("detections",{"detections":merged,"ts":time.time()})
        except Exception as ex:
            logging.error(f"[push] {ex}")
        time.sleep(0.1)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global CONF_THRESH, _isp
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
    print("  QCar 2 -- 360 Perception Server  v5")
    print("=" * 62)
    print(f"  Model  : {model_path}")
    print(f"  Mode   : {'DEMO' if args.demo else 'LIVE — quanser.multimedia sequential scheduler'}")
    print(f"  Modes  : {len(VIDEO_MODES)} candidates (auto-probe per camera)")
    print(f"  Stream : {STREAM_FPS}fps   Conf: {CONF_THRESH}")
    print("-" * 62)

    def _bye(sig, frame):
        print("\n[INFO] Shutting down ...")
        if _isp: _isp.stop()
        for w in _workers: w.stop()
        time.sleep(0.5)
        os._exit(0)

    signal.signal(signal.SIGINT,  _bye)
    signal.signal(signal.SIGTERM, _bye)

    # Create per-camera frame stores and ISP queues
    isp_queues = [queue.Queue(maxsize=2) for _ in CAMERA_CONFIG]
    for cfg in CAMERA_CONFIG:
        _stores.append(CamStore(cfg["id"]))

    if not args.demo:
        # Quick sanity check: try opening cam 0 on the main thread first
        # Start ISP scheduler (one camera open at a time, round-robin)
        _isp = ISPScheduler([cfg["id"] for cfg in CAMERA_CONFIG], isp_queues)
        _isp.start()
        print("  ISP scheduler started (open-read-close, 1 camera at a time)")
    else:
        print("  Demo mode — ISP scheduler not used")

    # Start YOLO workers
    for cfg, store, q in zip(CAMERA_CONFIG, _stores, isp_queues):
        w = YOLOWorker(cfg, model_path, store, q,
                       use_engine=args.engine, use_demo=args.demo)
        w.start()
        _workers.append(w)
        print(f"  YOLO {cfg['id']} ({cfg['name']}) started")

    threading.Thread(target=_push, daemon=True, name="push").start()

    ip = "10.1.77.73"
    print(f"\n  Dashboard  -> http://{ip}:{args.port}")
    print(f"  Health     -> http://{ip}:{args.port}/healthz   <- check cam frame counts")
    print(f"  Raw cam 0  -> http://{ip}:{args.port}/raw/0     <- no YOLO overlay")
    print(f"  CAM 0-3    -> http://{ip}:{args.port}/video/0")
    print(f"  Panorama   -> http://{ip}:{args.port}/video/panorama")
    print(f"  JSON API   -> http://{ip}:{args.port}/detections")
    print("=" * 62)
    print("  Ctrl+C to stop\n")

    socketio.run(app, host="0.0.0.0", port=args.port,
                 use_reloader=False, log_output=False)


if __name__ == "__main__":
    main()