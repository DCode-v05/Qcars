"""
yolo_processor.py — Single YOLO model processes all 4 cameras round-robin.
Prevents GPU OOM from multiple CUDA contexts on Jetson.
"""
import math
import queue
import threading
import time
import logging

import cv2
import numpy as np

import config as cfg


def _bgr(cls_id):
    hue = int((cls_id * 137.508) % 180)
    hsv = np.uint8([[[hue, 220, 220]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


CLASS_COLOURS = [_bgr(i) for i in range(len(cfg.COCO_CLASSES))]

_tan_hfov = math.tan(math.radians(cfg.H_FOV_DEG / 2.0))


def px_to_world_angle(px, frame_w, centre_deg):
    """Convert pixel x-coordinate to world angle (degrees)."""
    offset = math.degrees(math.atan((px / frame_w - 0.5) * 2.0 * _tan_hfov))
    return (centre_deg + offset) % 360.0


class Detection:
    """Single YOLO detection with world-angle information."""
    __slots__ = ('cam_id', 'cam_name', 'class_id', 'class_name',
                 'confidence', 'bbox_px', 'angle_left', 'angle_right',
                 'angle_centre', 'obj_type')

    def __init__(self, cam_id, cam_name, cls_id, conf,
                 bx, by, bw, bh, fw, centre_deg):
        self.cam_id      = cam_id
        self.cam_name    = cam_name
        self.class_id    = cls_id
        self.class_name  = cfg.COCO_CLASSES[cls_id] if cls_id < len(cfg.COCO_CLASSES) else str(cls_id)
        self.confidence  = conf
        self.bbox_px     = (bx, by, bw, bh)
        self.angle_left  = px_to_world_angle(bx, fw, centre_deg)
        self.angle_right = px_to_world_angle(bx + bw, fw, centre_deg)
        self.angle_centre = px_to_world_angle(bx + bw / 2, fw, centre_deg)

        # Categorize object type
        if cls_id in cfg.PERSON_IDS:
            self.obj_type = 'PERSON'
        elif cls_id in cfg.MOVING_IDS:
            self.obj_type = 'MOVING'
        elif cls_id in cfg.STATIC_IDS:
            self.obj_type = 'STATIC'
        else:
            self.obj_type = 'UNKNOWN'

    def to_dict(self):
        return {
            'cam_id': self.cam_id, 'cam_name': self.cam_name,
            'class_id': self.class_id, 'class_name': self.class_name,
            'confidence': round(self.confidence, 3),
            'bbox_px': list(self.bbox_px),
            'angle_left': round(self.angle_left, 1),
            'angle_right': round(self.angle_right, 1),
            'angle_centre': round(self.angle_centre, 1),
            'obj_type': self.obj_type,
        }


class YOLOProcessor:
    """Single YOLO model, processes frames from all cameras in one thread."""

    def __init__(self, model_path, cam_queues, stores=None):
        self.model_path = model_path
        self.cam_queues = cam_queues
        self.stores     = stores       # optional CamStore list for dashboard
        self._model     = None
        self._dets      = [[] for _ in cfg.CAMERA_CONFIG]
        self._lock      = threading.Lock()
        self._running   = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True, name="yolo").start()

    def stop(self):
        self._running = False

    def get_detections(self) -> list:
        """All current detections across all cameras."""
        with self._lock:
            out = []
            for dets in self._dets:
                out.extend(dets)
            return out

    def get_camera_dets(self, cam_idx) -> list:
        with self._lock:
            return list(self._dets[cam_idx])

    def _load(self):
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            logging.info(f"[YOLO] loaded {self.model_path}")
        except Exception as ex:
            logging.error(f"[YOLO] load failed: {ex}")
            self._model = None

    def _infer(self, frame):
        if self._model is None:
            return []
        sq = cv2.resize(frame, (cfg.YOLO_WIDTH, cfg.YOLO_WIDTH))
        results = self._model(sq, conf=cfg.CONF_THRESH, verbose=False)
        fw, fh = frame.shape[1], frame.shape[0]
        sx, sy = fw / cfg.YOLO_WIDTH, fh / cfg.YOLO_WIDTH
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append((int(box.cls[0]), float(box.conf[0]),
                              x1 * sx, y1 * sy,
                              (x2 - x1) * sx, (y2 - y1) * sy))
        return boxes

    def _annotate(self, frame, detections):
        """Draw detection boxes on frame. Returns annotated frame."""
        annotated = frame.copy()
        for det in detections:
            bx, by, bw, bh = det.bbox_px
            x1, y1 = int(bx), int(by)
            x2, y2 = int(bx + bw), int(by + bh)
            col = CLASS_COLOURS[det.class_id % len(CLASS_COLOURS)]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
            lbl = f"{det.class_name} {det.confidence:.2f}"
            lw, lh = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            ty = max(lh + 4, y1)
            cv2.rectangle(annotated, (x1, ty - lh - 4), (x1 + lw + 4, ty), col, -1)
            cv2.putText(annotated, lbl, (x1 + 2, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        return annotated

    def _loop(self):
        self._load()
        first = True
        n = len(cfg.CAMERA_CONFIG)
        idx = 0

        while self._running:
            try:
                q   = self.cam_queues[idx]
                cam = cfg.CAMERA_CONFIG[idx]
                idx = (idx + 1) % n

                try:
                    frame = q.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Show raw frame immediately on dashboard
                if self.stores and idx < len(self.stores):
                    self.stores[cam["id"]].put_raw(frame)
                    self.stores[cam["id"]].put_annotated(frame)

                if first:
                    logging.info("[YOLO] first inference starting...")
                    t0 = time.time()

                raw_boxes = self._infer(frame)

                if first:
                    logging.info(f"[YOLO] first inference done in {time.time()-t0:.1f}s")
                    first = False

                # Build Detection objects
                fw = frame.shape[1]
                dets = []
                for cls_id, conf, bx, by, bw, bh in raw_boxes:
                    bx = max(0., min(bx, fw - 1))
                    bw = min(bw, fw - bx)
                    if bw < 2 or bh < 2:
                        continue
                    dets.append(Detection(
                        cam["id"], cam["name"], cls_id, conf,
                        bx, by, bw, bh, fw, cam["centre_deg"]
                    ))

                # Update annotated frame for dashboard
                if self.stores:
                    annotated = self._annotate(frame, dets)
                    self.stores[cam["id"]].put_annotated(annotated)

                with self._lock:
                    self._dets[cam["id"]] = dets

            except Exception as ex:
                logging.error(f"[YOLO] {ex}")
                time.sleep(0.05)
