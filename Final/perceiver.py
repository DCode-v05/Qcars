"""
perceiver.py  —  Phase 4: YOLO object classification for QCar 2
Runs YOLOv8s-seg on the front camera. Classifies obstacles as:
  PERSON  → car WAITS (people move)
  MOVING  → car WAITS (bikes, dogs — unpredictable)
  STATIC  → car STEERS AROUND (boxes, chairs — won't move)
  NONE    → nothing detected
"""
import time
import numpy as np
from pit.YOLO.nets import YOLOv8

# COCO class categorisation
PERSON_IDS = {0}
MOVING_IDS = {1, 2, 3, 16}   # bicycle, car, motorcycle, dog
STATIC_IDS = {
    24, 25, 26, 28, 32, 39, 41, 43, 44, 45,
    56, 57, 58, 59, 60, 62, 63, 67, 73, 74, 76, 79,
}
ALL_CLASSES = sorted(PERSON_IDS | MOVING_IDS | STATIC_IDS)

OBJ_PERSON = 'PERSON'
OBJ_MOVING = 'MOVING'
OBJ_STATIC = 'STATIC'
OBJ_NONE   = 'NONE'


def classify(class_id: int) -> str:
    cid = int(class_id)
    if cid in PERSON_IDS: return OBJ_PERSON
    if cid in MOVING_IDS: return OBJ_MOVING
    if cid in STATIC_IDS: return OBJ_STATIC
    return OBJ_NONE


class PerceiverConfig:
    IMAGE_WIDTH        = 640
    IMAGE_HEIGHT       = 480
    CONFIDENCE         = 0.35
    FORWARD_COL_CENTRE = 320
    FORWARD_COL_MARGIN = 220   # ±220px from centre = full useful width
    MAX_DIST_M         = 3.0
    MODEL_PATH         = None  # None = use default (auto-downloads)


def _empty():
    return {
        'nearest_type':    OBJ_NONE,
        'nearest_name':    '',
        'nearest_dist_m':  99.0,
        'nearest_conf':    0.0,
        'nearest_x':       0,
        'nearest_y':       0,
        'all_detections':  [],
        'annotated_frame': None,
        'yolo_fps':        0.0,
        'n_detections':    0,
    }


class Perceiver:
    """
    Wraps YOLOv8s-seg. Call perceive() every tick.

    Usage:
        perceiver = Perceiver(PerceiverConfig())
        perceiver.open()
        result = perceiver.perceive(bgr_frame, depth_frame)
        if result['nearest_type'] == OBJ_PERSON:
            # wait
    """

    def __init__(self, config: PerceiverConfig):
        self.cfg   = config
        self._yolo = None

    def open(self):
        print("  Loading YOLOv8s-seg (uses cached TensorRT engine)...")
        self._yolo = YOLOv8(
            imageWidth  = self.cfg.IMAGE_WIDTH,
            imageHeight = self.cfg.IMAGE_HEIGHT,
            modelPath   = self.cfg.MODEL_PATH,
        )
        print("  YOLOv8s-seg: OK")

    def perceive(self, bgr_frame: np.ndarray, depth_frame: np.ndarray) -> dict:
        result = _empty()
        if self._yolo is None:
            return result
        try:
            img = self._yolo.pre_process(bgr_frame)
            self._yolo.predict(
                img,
                classes    = ALL_CLASSES,
                confidence = self.cfg.CONFIDENCE,
                verbose    = False,
            )
            result['yolo_fps'] = float(self._yolo.FPS)

            n = len(self._yolo.objectsDetected)
            if n == 0:
                result['annotated_frame'] = img.copy()
                return result

            # post_processing with depth gives real distances per detection
            detections = self._yolo.post_processing(
                alignedDepth    = depth_frame,
                clippingDistance= self.cfg.MAX_DIST_M,
            )

            pred0    = self._yolo.predictions[0]
            class_ids= self._yolo.objectsDetected
            all_dets = []
            nearest  = None
            nearest_d= 99.0

            for i, obs in enumerate(detections):
                cid      = int(class_ids[i])
                obj_type = classify(cid)
                dist_m   = float(obs.distance) if obs.distance else 99.0
                conf     = float(pred0.boxes.conf.cpu().numpy()[i])
                cx       = int(obs.x)

                # Only report objects in the forward region
                if abs(cx - self.cfg.FORWARD_COL_CENTRE) > self.cfg.FORWARD_COL_MARGIN:
                    continue
                if dist_m > self.cfg.MAX_DIST_M:
                    continue

                det = {
                    'type': obj_type, 'name': obs.name, 'dist_m': dist_m,
                    'conf': conf, 'x': obs.x, 'y': obs.y, 'class_id': cid,
                }
                all_dets.append(det)
                if dist_m < nearest_d:
                    nearest_d = dist_m
                    nearest   = det

            result['all_detections'] = all_dets
            result['n_detections']   = len(all_dets)
            if nearest:
                result.update({
                    'nearest_type':   nearest['type'],
                    'nearest_name':   nearest['name'],
                    'nearest_dist_m': nearest['dist_m'],
                    'nearest_conf':   nearest['conf'],
                    'nearest_x':      nearest['x'],
                    'nearest_y':      nearest['y'],
                })
            result['annotated_frame'] = self._yolo.post_process_render(showFPS=True)

        except Exception as e:
            print(f"  [WARN] Perceiver error: {e}")
            result['annotated_frame'] = bgr_frame.copy()
        return result

    def close(self):
        self._yolo = None
        print("  YOLOv8: released")
