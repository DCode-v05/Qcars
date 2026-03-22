#!/usr/bin/env python3
"""
QCar 2 — 360 Perception Server  v6  (PAL QCarCameras)

Uses pal.products.qcar.QCarCameras (same as Final/) to access CSI cameras.
QCarCameras wraps Camera2D which handles VideoCapture lifecycle correctly.

Architecture
------------
  CameraReader (1 thread)
    └─ calls cameras.readAll() in a loop
    └─ copies each frame into per-camera queues

  YOLOProcessor (1 thread, 1 model)
    └─ round-robin pulls from camera queues → inference → updates annotated frame

  Flask snapshot endpoints (no persistent MJPEG connections)
    └─ /snapshot/N returns a single JPEG per request
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
from pal.products.qcar import QCarCameras

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
YOLO_WIDTH     = 640
JPEG_QUALITY   = 75
CONF_THRESH    = 0.40
CSI_WIDTH      = 820
CSI_HEIGHT     = 410
CSI_FPS        = 30             # matches Final/perception.py — PAL default
READ_DELAY     = 0.033          # ~30fps read rate

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
# Camera Reader — uses PAL QCarCameras (same approach as Final/perception.py)
# Cameras are opened ONCE on the main thread, then read continuously.
# ══════════════════════════════════════════════════════════════════════════════
class CameraReader:
    """
    Reads all 4 CSI cameras using QCarCameras (PAL layer).
    Cameras are initialized on the main thread via open().
    A background thread calls readAll() and pushes frames to queues.
    """

    def __init__(self, queues: List[queue.Queue]):
        self.queues    = queues
        self._cameras: Optional[QCarCameras] = None
        self._running  = False
        self._thread: Optional[threading.Thread] = None
        self._good     = [0] * 4
        self._fail     = [0] * 4

    def open(self):
        """Open cameras on the MAIN thread (required by PAL/nvargus)."""
        logging.info("[CAM] Opening QCarCameras (820x410 @ 30fps, all 4 enabled)...")
        self._cameras = QCarCameras(
            frameWidth=CSI_WIDTH,
            frameHeight=CSI_HEIGHT,
            frameRate=CSI_FPS,
            enableRight=True,
            enableBack=True,
            enableFront=True,
            enableLeft=True,
        )
        # Warmup: discard first few frames (ISP pipeline fill)
        logging.info("[CAM] Warming up cameras...")
        for _ in range(30):
            self._cameras.readAll()
            time.sleep(READ_DELAY)
        logging.info("[CAM] Cameras ready.")

    def start(self):
        """Start background reader thread."""
        if self._cameras is None:
            raise RuntimeError("Call open() before start()")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="cam_reader")
        self._thread.start()

    def stop(self):
        self._running = False

    def terminate(self):
        """Close all cameras."""
        if self._cameras:
            try:
                self._cameras.terminate()
            except Exception as e:
                logging.warning(f"[CAM] terminate error: {e}")

    def stats(self):
        return list(zip(self._good, self._fail))

    def _loop(self):
        logging.info("[CAM] Reader thread started.")
        while self._running:
            try:
                flags = self._cameras.readAll()
            except Exception as ex:
                logging.error(f"[CAM] readAll error: {ex}")
                time.sleep(0.1)
                continue

            for i, cam in enumerate(self._cameras.csi):
                if cam is None:
                    continue
                frame = cam.imageData.copy()
                if frame.max() > 0:  # valid frame (not all-black)
                    self._good[i] += 1
                    q = self.queues[i]
                    if q.full():
                        try: q.get_nowait()
                        except queue.Empty: pass
                    try: q.put_nowait(frame)
                    except queue.Full: pass
                else:
                    self._fail[i] += 1

            time.sleep(READ_DELAY)


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
class YOLOProcessor:
    """Single YOLO model, processes all cameras round-robin in ONE thread.
    Prevents GPU OOM / segfault from multiple concurrent CUDA contexts."""

    def __init__(self, cam_cfgs, model_path, stores: List[CamStore],
                 cam_queues: List[queue.Queue], use_demo=False):
        self.cam_cfgs   = cam_cfgs
        self.model_path = model_path
        self.stores     = stores
        self.cam_queues = cam_queues
        self.use_demo   = use_demo
        self._model     = None
        self._dets      = [[] for _ in cam_cfgs]  # per-camera detections
        self._lock      = threading.Lock()
        self._running   = False
        self._demo_t    = 0.0

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True,
                         name="yolo").start()

    def stop(self):
        self._running = False

    def get_all_dets(self):
        with self._lock:
            out = []
            for d in self._dets:
                out.extend(d)
            return out

    def _load(self):
        if self.use_demo:
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            logging.info(f"[YOLO] loaded {self.model_path}")
        except Exception as ex:
            logging.warning(f"[YOLO] load failed ({ex}) -> demo")
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

    def _annotate(self, frame, raw_boxes, cam_cfg):
        cam_id     = cam_cfg["id"]
        cam_name   = cam_cfg["name"]
        centre_deg = cam_cfg["centre_deg"]
        fw, fh     = frame.shape[1], frame.shape[0]
        annotated  = frame.copy()
        dets       = []

        for (cls_id, conf, bx, by, bw, bh) in raw_boxes:
            bx = max(0., min(bx, fw-1));  bw = min(bw, fw-bx)
            by = max(0., min(by, fh-1));  bh = min(bh, fh-by)
            if bw < 2 or bh < 2:
                continue

            al, ar, ac = box_angles(bx, by, bw, bh, fw, centre_deg)
            name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)
            col  = CLASS_COLOURS[cls_id % len(CLASS_COLOURS)]

            dets.append({
                "cam_id": cam_id, "cam_name": cam_name,
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
            f"CAM{cam_id} {cam_name}  "
            f"{centre_deg-H_FOV_DEG/2:.0f}d->{centre_deg+H_FOV_DEG/2:.0f}d"
            f"  [{len(dets)} obj]",
            (5,19), cv2.FONT_HERSHEY_SIMPLEX, 0.52,(0,255,128),1,cv2.LINE_AA)

        return annotated, dets

    def _loop(self):
        self._load()
        first_infer = True
        n = len(self.cam_cfgs)
        idx = 0

        while self._running:
            try:
                # Round-robin: try to get a frame from the next camera
                q     = self.cam_queues[idx]
                store = self.stores[idx]
                cfg   = self.cam_cfgs[idx]
                idx   = (idx + 1) % n

                try:
                    frame = q.get(timeout=0.1)
                except queue.Empty:
                    continue

                store.put_raw(frame)
                store.put_annotated(frame)  # show raw immediately

                if self.use_demo:
                    raw_boxes = []
                else:
                    if first_infer:
                        logging.info("[YOLO] first inference starting "
                                     "(may take 30-60s for TensorRT)...")
                    t0 = time.time()
                    raw_boxes = self._infer(frame)
                    if first_infer:
                        logging.info(f"[YOLO] first inference done "
                                     f"in {time.time()-t0:.1f}s")
                        first_infer = False

                annotated, dets = self._annotate(frame, raw_boxes, cfg)
                store.put_annotated(annotated)
                with self._lock:
                    self._dets[cfg["id"]] = dets

            except Exception as ex:
                logging.error(f"[YOLO] {ex}")
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

_stores:    List[CamStore]   = []
_yolo:      Optional[YOLOProcessor] = None
_reader:    Optional[CameraReader] = None
_agg        = Aggregator()


def _enc(frame):
    ok,buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None


@app.route("/snapshot/<int:cid>")
def snapshot(cid):
    if cid<0 or cid>=len(_stores): return "not found",404
    data = _enc(_stores[cid].get_annotated())
    if not data: return "encode error",500
    return Response(data, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})

@app.route("/snapshot/panorama")
def snapshot_pano():
    if not _stores:
        f = np.zeros((180,820,3),dtype=np.uint8)
    else:
        f = panorama([s.get_annotated() for s in _stores])
    data = _enc(f)
    if not data: return "encode error",500
    return Response(data, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})

@app.route("/raw/<int:cid>")
def raw_snap(cid):
    if cid<0 or cid>=len(_stores): return "not found",404
    data = _enc(_stores[cid].get_raw())
    if not data: return "encode error",500
    return Response(data, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})

@app.route("/detections")
def dets_route():
    all_d = _yolo.get_all_dets() if _yolo else []
    return jsonify(detections=_agg.merge(all_d), ts=time.time())

@app.route("/healthz")
def health():
    stats=[]
    if _reader:
        for (g,f),(cfg) in zip(_reader.stats(), CAMERA_CONFIG):
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
            if _yolo:
                all_d = _yolo.get_all_dets()
                merged = _agg.merge(all_d)
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
    global CONF_THRESH, _reader
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
    print("  QCar 2 -- 360 Perception Server  v6  (PAL QCarCameras)")
    print("=" * 62)
    print(f"  Model  : {model_path}")
    print(f"  Mode   : {'DEMO' if args.demo else 'LIVE — PAL QCarCameras'}")
    print(f"  Camera : {CSI_WIDTH}x{CSI_HEIGHT} @ {CSI_FPS}fps")
    print(f"  Conf   : {CONF_THRESH}")
    print("-" * 62)

    def _bye(sig, frame):
        print("\n[INFO] Shutting down ...")
        if _yolo: _yolo.stop()
        if _reader:
            _reader.stop()
            _reader.terminate()
        time.sleep(0.5)
        os._exit(0)

    signal.signal(signal.SIGINT,  _bye)
    signal.signal(signal.SIGTERM, _bye)

    # Create per-camera frame stores and queues
    cam_queues = [queue.Queue(maxsize=2) for _ in CAMERA_CONFIG]
    for cfg in CAMERA_CONFIG:
        _stores.append(CamStore(cfg["id"]))

    if not args.demo:
        # Open cameras on MAIN thread (required by PAL/nvargus)
        _reader = CameraReader(cam_queues)
        _reader.open()
        _reader.start()
        print("  Camera reader started (PAL QCarCameras, all 4 enabled)")
    else:
        print("  Demo mode -- cameras not used")

    # Start single YOLO processor (one model, all cameras round-robin)
    _yolo = YOLOProcessor(CAMERA_CONFIG, model_path, _stores, cam_queues,
                          use_demo=args.demo)
    _yolo.start()
    print("  YOLO processor started (single model, round-robin)")

    threading.Thread(target=_push, daemon=True, name="push").start()

    ip = "10.1.77.73"
    print(f"\n  Dashboard  -> http://{ip}:{args.port}")
    print(f"  Health     -> http://{ip}:{args.port}/healthz")
    print(f"  Snapshots  -> http://{ip}:{args.port}/snapshot/0..3")
    print(f"  Panorama   -> http://{ip}:{args.port}/snapshot/panorama")
    print(f"  JSON API   -> http://{ip}:{args.port}/detections")
    print("=" * 62)
    print("  Ctrl+C to stop\n")

    socketio.run(app, host="0.0.0.0", port=args.port,
                 use_reloader=False, log_output=False)


if __name__ == "__main__":
    main()