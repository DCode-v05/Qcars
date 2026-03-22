"""
dashboard.py — Minimal web dashboard for monitoring autonomous driving.
Optional: pass --dashboard to main.py to enable.
"""
import os
import threading
import time
import logging

import cv2
import numpy as np
from flask import Flask, Response, jsonify

import config as cfg


class CamStore:
    """Thread-safe frame store for one camera."""
    def __init__(self, cam_id):
        self.cam_id = cam_id
        h, w = cfg.CSI_HEIGHT, cfg.CSI_WIDTH
        self._frame = np.zeros((h, w, 3), dtype=np.uint8)
        self._raw   = np.zeros((h, w, 3), dtype=np.uint8)
        self._lock  = threading.Lock()

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


class Dashboard:
    """Flask web dashboard for camera feeds and status."""

    def __init__(self, stores):
        self._stores = stores
        self._app = Flask(__name__)
        self._lidar_data = None     # set by main loop
        self._nav_data = None       # set by main loop
        self._det_data = []         # set by main loop
        self._data_lock = threading.Lock()
        self._setup_routes()
        self._thread = None

    def update_lidar(self, distances, angles, valid):
        """Called from main loop to push LiDAR data."""
        with self._data_lock:
            pts = []
            for i in range(len(distances)):
                if valid[i] and distances[i] < cfg.LIDAR_MAX_M:
                    pts.append({'d': round(float(distances[i]), 3),
                                'a': round(float(angles[i]), 4)})
            self._lidar_data = pts

    def update_nav(self, detection, nav_result):
        """Called from main loop to push navigation state."""
        with self._data_lock:
            self._nav_data = {
                'zone': str(detection['zone']),
                'behaviour': str(detection['behaviour']),
                'distance_m': float(detection['distance_m']),
                'rear_min_m': float(detection['rear_min_m']),
                'has_path': bool(detection['has_path']),
                'best_path_angle': round(float(detection['best_path_angle']), 4),
                'best_gap_width': round(float(detection['best_gap_width']), 1),
                'drive_forward': bool(detection['drive_forward']),
                'obstacle_type': str(detection['obstacle_type']),
                'state': str(nav_result['state']),
                'throttle': float(nav_result['throttle']),
                'steering': round(float(nav_result['steering']), 4),
            }

    def update_detections(self, dets):
        """Called from main loop to push YOLO detections."""
        with self._data_lock:
            self._det_data = [d.to_dict() if hasattr(d, 'to_dict') else d
                              for d in dets]

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="dashboard")
        self._thread.start()

    def stop(self):
        pass  # daemon thread dies with process

    def _run(self):
        self._app.run(host="0.0.0.0", port=cfg.DASHBOARD_PORT,
                      threaded=True, use_reloader=False)

    def _enc(self, frame):
        ok, buf = cv2.imencode('.jpg', frame,
                               [cv2.IMWRITE_JPEG_QUALITY, cfg.JPEG_QUALITY])
        return buf.tobytes() if ok else None

    def _setup_routes(self):
        app = self._app
        stores = self._stores

        @app.route("/snapshot/<int:cid>")
        def snapshot(cid):
            if cid < 0 or cid >= len(stores):
                return "not found", 404
            data = self._enc(stores[cid].get_annotated())
            if not data:
                return "encode error", 500
            return Response(data, mimetype="image/jpeg",
                            headers={"Cache-Control": "no-store"})

        @app.route("/snapshot/panorama")
        def pano():
            strips = []
            labels = ["RIGHT", "BACK", "FRONT", "LEFT"]
            for i, s in enumerate(stores):
                f = s.get_annotated()
                h, w = f.shape[:2]
                nw = max(1, int(w * 180 / h))
                strip = cv2.resize(f, (nw, 180))
                cv2.putText(strip, labels[i], (5, 174),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                strips.append(strip)
            pano_img = np.hstack(strips) if strips else np.zeros((180, 820, 3), dtype=np.uint8)
            data = self._enc(pano_img)
            if not data:
                return "encode error", 500
            return Response(data, mimetype="image/jpeg",
                            headers={"Cache-Control": "no-store"})

        @app.route("/lidar")
        def lidar():
            with self._data_lock:
                pts = self._lidar_data or []
            return jsonify(points=pts, ts=time.time())

        @app.route("/nav")
        def nav():
            with self._data_lock:
                nd = self._nav_data or {}
            return jsonify(**nd, ts=time.time())

        @app.route("/detections")
        def detections():
            with self._data_lock:
                dets = list(self._det_data)
            return jsonify(detections=dets, ts=time.time())

        @app.route("/healthz")
        def health():
            return jsonify(status="ok", ts=time.time())

        @app.route("/")
        def index():
            here = os.path.dirname(os.path.abspath(__file__))
            html = os.path.join(here, "qcar360_dashboard.html")
            try:
                return open(html).read()
            except FileNotFoundError:
                return "<h2>Dashboard HTML not found</h2>", 404
