"""
perceiver.py  —  Phase 4: YOLO object classification for QCar 2
═══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE DOES
───────────────────
Runs YOLOv8s-seg on the front CSI camera frame every tick and classifies
detected objects into three categories that change how the car behaves:

  PERSON  → car STOPS and WAITS  (people move — don't steer around them)
  MOVING  → car STOPS and WAITS  (bikes, cars, dogs — unpredictable)
  STATIC  → car STEERS AROUND    (boxes, chairs, bottles — won't move)

WHY YOLO FOR THIS MISSION
──────────────────────────
  LiDAR sees: "obstacle at 1.2m"
  YOLO adds:  "that is a PERSON"
  Result: car waits patiently instead of trying to steer around a person.
  Without YOLO, a person and a cardboard box get identical treatment.

MODEL: YOLOv8s-seg
───────────────────
  Source:  https://quanserinc.box.com/shared/static/ce0gxomeg4b12wlcch9cmlh0376nditf.pt
  Classes: 80 COCO classes → we filter to 27 relevant ones
  Output:  bounding boxes + segmentation masks per detection
  Speed:   ~20-30fps on Jetson GPU (TensorRT engine)

VERIFIED API (from pit/YOLO/nets.py)
─────────────────────────────────────
  YOLOv8(imageWidth, imageHeight, modelPath=None)
  model.pre_process(bgr_frame)           → preprocessed img (in-place)
  model.predict(img, classes, confidence) → predictions[0]
  model.post_processing(alignedDepth)    → list of Obstacle objects
  model.post_process_render(showFPS)     → annotated BGR frame
  Obstacle: .name, .distance, .x, .y

DEPTH INTEGRATION
──────────────────
  We pass rs_depth_m (float32 metres) as alignedDepth to post_processing().
  YOLO then computes median depth under each segmentation mask.
  Result: each detection gets a real distance in metres, not just pixels.
  Note: rs_depth_m shape is (H,W,1) float32 — squeeze to (H,W) for YOLO.

HOW TO TEST
────────────
  python3 perceiver.py
  Point front camera at a person, a chair, a bottle.
  Confirm correct classification and distance in terminal output.
═══════════════════════════════════════════════════════════════════════════════
"""

import time
import numpy as np
import cv2

from pit.YOLO.nets import YOLOv8
from perception import SensorManager, Config as PerceptionConfig


# ═════════════════════════════════════════════════════════════════════════════
#  COCO CLASS CATEGORISATION
# ═════════════════════════════════════════════════════════════════════════════

# Classes where car STOPS and WAITS (entity will move on its own)
PERSON_CLASS_IDS = [0]                      # person

# Classes where car STOPS and WAITS (unpredictable movement)
MOVING_CLASS_IDS = [1, 2, 3, 16]           # bicycle, car, motorcycle, dog

# Classes where car STEERS AROUND (won't move by themselves)
STATIC_CLASS_IDS = [
    24, 25, 26, 28,                         # backpack, umbrella, handbag, suitcase
    32, 39, 41, 43, 44, 45,                 # sports ball, bottle, cup, knife, spoon, bowl
    56, 57, 58, 59, 60,                     # chair, couch, plant, bed, table
    62, 63, 67, 73, 74, 76, 79,             # tv, laptop, phone, book, clock, scissors, toothbrush
]

# All classes we ask YOLO to detect (ignore everything else)
ALL_MONITORED_CLASSES = PERSON_CLASS_IDS + MOVING_CLASS_IDS + STATIC_CLASS_IDS

# Human-readable category strings
OBJ_PERSON = 'PERSON'
OBJ_MOVING = 'MOVING'
OBJ_STATIC = 'STATIC'
OBJ_NONE   = 'NONE'

# COCO class names for display (subset)
COCO_NAMES = {
    0:'person', 1:'bicycle', 2:'car', 3:'motorcycle', 16:'dog',
    24:'backpack', 25:'umbrella', 26:'handbag', 28:'suitcase',
    32:'ball', 39:'bottle', 41:'cup', 43:'knife', 44:'spoon', 45:'bowl',
    56:'chair', 57:'couch', 58:'plant', 59:'bed', 60:'table',
    62:'tv', 63:'laptop', 67:'phone', 73:'book', 74:'clock',
    76:'scissors', 79:'toothbrush',
}

def classify_class_id(class_id: int) -> str:
    """Map a COCO class ID to our three behaviour categories."""
    cid = int(class_id)
    if cid in PERSON_CLASS_IDS:
        return OBJ_PERSON
    if cid in MOVING_CLASS_IDS:
        return OBJ_MOVING
    if cid in STATIC_CLASS_IDS:
        return OBJ_STATIC
    return OBJ_NONE


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════════════════════

class PerceiverConfig:

    IMAGE_WIDTH       = 640
    IMAGE_HEIGHT      = 480
    CSI_CAMERA_ID     = '2'   # front camera on this QCar is index 2

    # Detection confidence threshold (0–1). Lower = more detections but more
    # false positives. 0.35 is a good balance for indoor use.
    CONFIDENCE        = 0.35

    # NOTE: update PerceptionConfig.CSI_CAMERA_IDS = ["2"] in perception.py
    # OR override here - the standalone test will set it

    # Only report detections in the FORWARD region of the frame.
    # Objects at the extreme sides are not in our path.
    # Column range: centre ± FORWARD_COL_MARGIN pixels.
    # At 640px wide, centre=320. ±200 covers the full relevant path width.
    FORWARD_COL_CENTRE = 320
    FORWARD_COL_MARGIN = 200   # detections outside 120-520px are ignored

    # Maximum distance to report as an obstacle (metres).
    # Objects further than this are not yet a concern.
    MAX_OBSTACLE_DIST_M = 3.0

    # Minimum mask pixel count to trust a detection (filters tiny blips)
    MIN_MASK_PIXELS   = 200

    # Model path — None = use default (downloads automatically if missing)
    MODEL_PATH        = None


# ═════════════════════════════════════════════════════════════════════════════
#  PERCEPTION RESULT  —  returned every tick
# ═════════════════════════════════════════════════════════════════════════════

def _empty_perception() -> dict:
    return {
        # Nearest threat in the forward region
        'nearest_type':     OBJ_NONE,    # PERSON / MOVING / STATIC / NONE
        'nearest_name':     '',          # COCO class name e.g. 'person'
        'nearest_dist_m':   99.0,        # metres (from depth camera)
        'nearest_conf':     0.0,         # YOLO confidence 0–1
        'nearest_x':        0,           # bounding box top-left x (pixels)
        'nearest_y':        0,           # bounding box top-left y (pixels)

        # All detections in frame (for dashboard)
        'all_detections':   [],          # list of dicts

        # Annotated frame for dashboard
        'annotated_frame':  None,        # ndarray (H,W,3) uint8 BGR or None

        # Performance
        'yolo_fps':         0.0,
        'n_detections':     0,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  PERCEIVER
# ═════════════════════════════════════════════════════════════════════════════

class Perceiver:
    """
    Wraps YOLOv8s-seg. Runs on every camera frame and returns structured
    detection results with behaviour categories.

    Usage
    ─────
        perceiver = Perceiver(PerceiverConfig())
        perceiver.open()
        try:
            while running:
                data = sensors.read()
                result = perceiver.perceive(
                    data['csi_frames']['2'],
                    data['rs_depth_m']
                )
                if result['nearest_type'] == OBJ_PERSON:
                    # stop and wait
        finally:
            perceiver.close()
    """

    def __init__(self, config: PerceiverConfig):
        self.cfg   = config
        self._yolo = None

    def open(self):
        """Load YOLOv8 TensorRT engine. First run downloads + converts model (~2-3 min)."""
        print("  Loading YOLOv8s-seg...")
        print("  (First run: downloads model ~200MB and converts to TensorRT ~2 min)")
        self._yolo = YOLOv8(
            imageWidth  = self.cfg.IMAGE_WIDTH,
            imageHeight = self.cfg.IMAGE_HEIGHT,
            modelPath   = self.cfg.MODEL_PATH,
        )
        print("  YOLOv8s-seg: OK")

    def perceive(self, bgr_frame: np.ndarray,
                 depth_frame: np.ndarray) -> dict:
        """
        Run YOLO on one CSI camera frame with depth-based distance measurement.

        Parameters
        ──────────
        bgr_frame   : ndarray (H, W, 3) uint8  from Camera2D.imageData
        depth_frame : ndarray (H, W, 1) float32 metres from Camera3D
                      0.0 = invalid pixel

        Returns
        ───────
        dict — see _empty_perception() for all keys
        """
        result = _empty_perception()

        if self._yolo is None:
            return result

        try:
            # Step 1: pre-process (resize to model input size)
            img = self._yolo.pre_process(bgr_frame)

            # Step 2: YOLO inference — filter to our monitored classes only
            self._yolo.predict(
                img,
                classes    = ALL_MONITORED_CLASSES,
                confidence = self.cfg.CONFIDENCE,
                verbose    = False,
            )
            result['yolo_fps'] = float(self._yolo.FPS)

            n = len(self._yolo.objectsDetected)
            result['n_detections'] = n

            if n == 0:
                # No detections — still render clean frame
                result['annotated_frame'] = img.copy()
                return result

            # Step 3: post-processing with depth
            # depth_frame is (H,W,1) float32 metres — squeeze to (H,W)
            # YOLO post_processing expects depth as (H,W,1) or None
            depth_for_yolo = depth_frame  # already (H,W,1) float32

            detections = self._yolo.post_processing(
                alignedDepth    = depth_for_yolo,
                clippingDistance= self.cfg.MAX_OBSTACLE_DIST_M,
            )

            # Step 4: classify and filter detections
            all_dets   = []
            nearest    = None
            nearest_d  = 99.0

            pred0      = self._yolo.predictions[0]
            class_ids  = self._yolo.objectsDetected   # numpy array

            for i, obs in enumerate(detections):
                cid      = int(class_ids[i])
                obj_type = classify_class_id(cid)
                dist_m   = float(obs.distance) if obs.distance else 99.0
                conf     = float(pred0.boxes.conf.cpu().numpy()[i])
                cx       = int(obs.x)

                # Filter: only forward region
                if abs(cx - self.cfg.FORWARD_COL_CENTRE) > self.cfg.FORWARD_COL_MARGIN:
                    continue

                # Filter: only within max distance
                if dist_m > self.cfg.MAX_OBSTACLE_DIST_M:
                    continue

                det = {
                    'type':   obj_type,
                    'name':   obs.name,
                    'dist_m': dist_m,
                    'conf':   conf,
                    'x':      obs.x,
                    'y':      obs.y,
                    'class_id': cid,
                }
                all_dets.append(det)

                # Track the nearest threatening object
                if dist_m < nearest_d:
                    nearest_d = dist_m
                    nearest   = det

            result['all_detections'] = all_dets
            result['n_detections']   = len(all_dets)

            if nearest:
                result['nearest_type']   = nearest['type']
                result['nearest_name']   = nearest['name']
                result['nearest_dist_m'] = nearest['dist_m']
                result['nearest_conf']   = nearest['conf']
                result['nearest_x']      = nearest['x']
                result['nearest_y']      = nearest['y']

            # Step 5: render annotated frame for dashboard
            result['annotated_frame'] = self._yolo.post_process_render(
                showFPS = True
            )

        except Exception as e:
            print(f"  [WARN] Perceiver error: {e}")
            result['annotated_frame'] = bgr_frame.copy()

        return result

    def close(self):
        """Release YOLO model. GPU memory freed automatically."""
        self._yolo = None
        print("  YOLOv8: released")


# ═════════════════════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ═════════════════════════════════════════════════════════════════════════════

def main():
    p_cfg = PerceptionConfig()
    c_cfg = PerceiverConfig()

    sensors   = SensorManager(p_cfg)
    perceiver = Perceiver(c_cfg)

    PRINT_INTERVAL_S = 0.25
    TEST_DURATION_S  = 30.0

    print("═" * 62)
    print("  QCar 2 — Phase 4: Perceiver (YOLO) Test")
    print("  Point front camera at objects.")
    print("  Person → PERSON  |  Chair/bottle → STATIC  |  Bike → MOVING")
    print(f"  Running {TEST_DURATION_S:.0f}s. Ctrl+C to stop.")
    print("═" * 62)

    sensors.open()
    perceiver.open()

    try:
        start_t    = time.perf_counter()
        tick_start = start_t
        last_print = 0.0

        while True:
            elapsed = time.perf_counter() - start_t
            if elapsed >= TEST_DURATION_S:
                break

            data   = sensors.read()
            result = perceiver.perceive(
                data['csi_frames'][c_cfg.CSI_CAMERA_ID],
                data['rs_depth_m'],
            )

            if elapsed - last_print >= PRINT_INTERVAL_S:
                typ   = result['nearest_type']
                name  = result['nearest_name']
                dist  = result['nearest_dist_m']
                conf  = result['nearest_conf']
                n     = result['n_detections']
                fps   = result['yolo_fps']

                icon = {'PERSON':'👤','MOVING':'🚴','STATIC':'📦','NONE':'  '}
                type_icon = {'PERSON':'!','MOVING':'~','STATIC':'+','NONE':' '}[typ]

                print(
                    f"[{elapsed:5.1f}s] "
                    f"[{type_icon} {typ:<6}]  "
                    f"nearest={name:<12}  "
                    f"dist={dist:5.2f}m  "
                    f"conf={conf:.2f}  "
                    f"n={n}  "
                    f"fps={fps:.0f}"
                )

                # Print all detections if more than one
                if n > 1:
                    for d in result['all_detections']:
                        print(f"         → {d['type']:<6} {d['name']:<12} "
                              f"{d['dist_m']:.2f}m  conf={d['conf']:.2f}")

                last_print = elapsed

            elapsed_tick = time.perf_counter() - tick_start
            sleep_time   = p_cfg.LOOP_DT - elapsed_tick
            if sleep_time > 0:
                time.sleep(sleep_time)
            tick_start = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

    finally:
        perceiver.close()
        sensors.close()

    print("\n" + "═" * 62)
    print("  PHASE 4 CHECKLIST:")
    checks = [
        "Person in frame       → type=PERSON, name='person'",
        "Chair/bottle in frame → type=STATIC, correct name",
        "Bicycle in frame      → type=MOVING",
        "Nothing in frame      → type=NONE, n=0",
        "Distance reading      → dist_m matches real distance (±0.3m)",
        "FPS                   → yolo_fps > 10 (target >20)",
    ]
    for c in checks:
        print(f"  [ ] {c}")
    print("═" * 62)


if __name__ == "__main__":
    main()
