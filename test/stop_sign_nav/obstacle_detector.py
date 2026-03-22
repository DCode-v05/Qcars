"""
obstacle_detector.py  —  Phase 3+4: Fused obstacle detection
═══════════════════════════════════════════════════════════════════════════════
SENSOR FUSION ARCHITECTURE
───────────────────────────
  LiDAR    → IS there an obstacle? HOW FAR? (fast, reliable, all conditions)
  RealSense → WHICH SIDE has clear space? HOW WIDE is the gap?
  YOLO     → WHAT IS IT? (person=wait, static=steer-around)

COMBINED RESULT EVERY TICK
───────────────────────────
  zone          CLEAR / WARN / STOP
  distance_m    float  metres to nearest obstacle
  avoid_side    'left' or 'right'
  obstacle_type NONE / PERSON / MOVING / STATIC
  behaviour     NAVIGATE / WAIT / AVOID / EMERGENCY_STOP
                (the FSM uses this directly)

BEHAVIOUR RULES
────────────────
  zone=CLEAR  → NAVIGATE regardless of type
  zone=WARN + type=PERSON  → WAIT (person will move)
  zone=WARN + type=MOVING  → WAIT (unpredictable)
  zone=WARN + type=STATIC  → AVOID (steer around it)
  zone=WARN + type=NONE    → AVOID (LiDAR sees it, YOLO can't classify it)
  zone=STOP   → EMERGENCY_STOP (too close regardless of type)
═══════════════════════════════════════════════════════════════════════════════
"""

import time
import numpy as np

from perception  import SensorManager, Config as PerceptionConfig
from perceiver   import Perceiver, PerceiverConfig, OBJ_PERSON, OBJ_MOVING, OBJ_STATIC, OBJ_NONE


# ═════════════════════════════════════════════════════════════════════════════
#  BEHAVIOUR CONSTANTS  (used by state_machine.py)
# ═════════════════════════════════════════════════════════════════════════════

BEHAVIOUR_NAVIGATE        = 'NAVIGATE'
BEHAVIOUR_WAIT            = 'WAIT'
BEHAVIOUR_AVOID           = 'AVOID'
BEHAVIOUR_EMERGENCY_STOP  = 'EMERGENCY_STOP'

ZONE_CLEAR = 'CLEAR'
ZONE_WARN  = 'WARN'
ZONE_STOP  = 'STOP'


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════════════════════

class DetectorConfig:

    # LiDAR zones
    WARN_DISTANCE_M    = 1.5
    STOP_DISTANCE_M    = 0.4

    # LiDAR forward sector: ±40° around 0 (forward)
    FORWARD_HALF_ANGLE = np.radians(40)
    MIN_SECTOR_READINGS= 3
    DISTANCE_SMOOTH_N  = 3

    # RealSense depth strip for avoidance decision
    DEPTH_ROW_START    = 160
    DEPTH_ROW_END      = 320
    DEPTH_MAX_M        = 3.0
    DEPTH_MIN_VALID_PX = 80

    # Gap finder: scan columns in WARN_DISTANCE_M wide strip
    # A "gap" is a contiguous column range where depth > GAP_MIN_DEPTH_M
    GAP_MIN_DEPTH_M    = 0.8    # column must have at least this much clearance
    GAP_MIN_WIDTH_PX   = 60     # minimum gap width in pixels to be usable

    # Dashboard polar plot
    PLOT_MAX_RANGE_M   = 4.0
    DASHBOARD_PORT     = 5000


# ═════════════════════════════════════════════════════════════════════════════
#  EMPTY RESULT
# ═════════════════════════════════════════════════════════════════════════════

def _empty_result() -> dict:
    return {
        # Core outputs for state_machine.py
        'zone':             ZONE_CLEAR,
        'distance_m':       99.0,
        'avoid_side':       'left',
        'obstacle_type':    OBJ_NONE,
        'behaviour':        BEHAVIOUR_NAVIGATE,

        # Detail for dashboard
        'sector_min_m':     99.0,
        'left_count':       0,
        'right_count':      0,
        'left_clear_m':     99.0,
        'right_clear_m':    99.0,
        'gap_side':         'left',
        'gap_width_px':     0,
        'yolo_name':        '',
        'yolo_conf':        0.0,
        'yolo_fps':         0.0,
        'n_yolo_dets':      0,
        'new_lidar_scan':   False,
        'all_distances':    np.array([]),
        'all_angles':       np.array([]),
        'all_valid':        np.array([], dtype=bool),
        'annotated_frame':  None,
        'battery_v':        0.0,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  OBSTACLE DETECTOR
# ═════════════════════════════════════════════════════════════════════════════

class ObstacleDetector:
    """
    Fuses LiDAR + RealSense + YOLO into one detection result per tick.

    Usage
    ─────
        detector = ObstacleDetector(DetectorConfig(), Perceiver(PerceiverConfig()))
        ...
        result = detector.detect(sensor_data)
        if result['behaviour'] == BEHAVIOUR_WAIT:
            # person in path — hold position
    """

    def __init__(self, config: DetectorConfig, perceiver: Perceiver):
        self.cfg        = config
        self.perceiver  = perceiver
        self._dist_hist = []
        self._last      = _empty_result()

    def detect(self, sensor_data: dict) -> dict:
        new_scan = sensor_data.get('lidar_new_scan', False)

        # ── LiDAR ─────────────────────────────────────────────────────────
        if new_scan:
            lidar = self._analyse_lidar(
                sensor_data['lidar_distances'],
                sensor_data['lidar_angles'],
                sensor_data['lidar_valid'],
            )
        else:
            lidar = {
                'zone':        self._last['zone'],
                'distance_m':  self._last['distance_m'],
                'sector_min':  self._last['sector_min_m'],
                'left_count':  self._last['left_count'],
                'right_count': self._last['right_count'],
            }

        # ── RealSense depth ────────────────────────────────────────────────
        depth = self._analyse_depth(sensor_data['rs_depth_m'])

        # ── YOLO (runs every tick on GPU — fast enough) ────────────────────
        perception = self.perceiver.perceive(
            sensor_data['csi_frames']['2'],
            sensor_data['rs_depth_m'],
        )

        # ── Combine all three ──────────────────────────────────────────────
        result = self._combine(lidar, depth, perception, new_scan)

        # Attach raw data for dashboard
        result['all_distances']   = sensor_data['lidar_distances'].copy()
        result['all_angles']      = sensor_data['lidar_angles'].copy()
        result['all_valid']       = sensor_data['lidar_valid'].copy()
        result['annotated_frame'] = perception['annotated_frame']
        result['battery_v']       = sensor_data['battery_voltage']

        self._last = result
        return result

    # ──────────────────────────────────────────────────────────────────────
    #  LIDAR
    # ──────────────────────────────────────────────────────────────────────

    def _analyse_lidar(self, distances, angles, valid) -> dict:
        cfg = self.cfg
        if len(distances) == 0 or valid.sum() < cfg.MIN_SECTOR_READINGS:
            return {'zone': ZONE_CLEAR, 'distance_m': 99.0,
                    'sector_min': 99.0, 'left_count': 0, 'right_count': 0}

        in_left   = (angles >= 0) & (angles <= cfg.FORWARD_HALF_ANGLE)
        in_right  = (angles >= (2*np.pi - cfg.FORWARD_HALF_ANGLE))
        in_sector = (in_left | in_right) & valid

        if in_sector.sum() < cfg.MIN_SECTOR_READINGS:
            return {'zone': ZONE_CLEAR, 'distance_m': 99.0,
                    'sector_min': 99.0, 'left_count': 0, 'right_count': 0}

        sector_d = distances[in_sector]
        sector_min = float(sector_d.min())

        self._dist_hist.append(sector_min)
        if len(self._dist_hist) > cfg.DISTANCE_SMOOTH_N:
            self._dist_hist.pop(0)
        smoothed = float(np.mean(self._dist_hist))

        lc = int((distances[in_left  & valid] < cfg.WARN_DISTANCE_M).sum())
        rc = int((distances[in_right & valid] < cfg.WARN_DISTANCE_M).sum())

        if smoothed <= cfg.STOP_DISTANCE_M:
            zone = ZONE_STOP
        elif smoothed <= cfg.WARN_DISTANCE_M:
            zone = ZONE_WARN
        else:
            zone = ZONE_CLEAR

        return {
            'zone': zone, 'distance_m': smoothed,
            'sector_min': sector_min,
            'left_count': lc, 'right_count': rc,
        }

    # ──────────────────────────────────────────────────────────────────────
    #  REALSENSE DEPTH — gap finder
    # ──────────────────────────────────────────────────────────────────────

    def _analyse_depth(self, depth_frame: np.ndarray) -> dict:
        """
        Find the widest clear gap (column range with depth > GAP_MIN_DEPTH_M)
        in a horizontal strip of the depth frame.

        Returns side to steer toward and gap width in pixels.
        """
        cfg   = self.cfg
        strip = depth_frame[cfg.DEPTH_ROW_START:cfg.DEPTH_ROW_END, :, 0]
        W     = strip.shape[1]
        mid   = W // 2

        # Column-wise mean depth (ignore zeros = invalid pixels)
        col_means = np.where(strip > 0, strip, np.nan)
        col_means = np.nanmean(col_means, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)

        # Simple left/right mean for avoidance direction
        left_mean  = float(col_means[:mid].mean()) if col_means[:mid].mean() > 0 else 99.0
        right_mean = float(col_means[mid:].mean()) if col_means[mid:].mean() > 0 else 99.0

        # Gap finder: find widest contiguous clear column range
        clear_cols = col_means > cfg.GAP_MIN_DEPTH_M
        best_gap_start, best_gap_len = 0, 0
        cur_start, cur_len = 0, 0
        for i, is_clear in enumerate(clear_cols):
            if is_clear:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_gap_len:
                    best_gap_len   = cur_len
                    best_gap_start = cur_start
            else:
                cur_len = 0

        gap_centre = best_gap_start + best_gap_len // 2
        gap_side   = 'left' if gap_centre < mid else 'right'

        # Fallback to simple left/right if no gap found
        if best_gap_len < cfg.GAP_MIN_WIDTH_PX:
            gap_side = 'left' if left_mean >= right_mean else 'right'

        return {
            'left_clear_m':  left_mean,
            'right_clear_m': right_mean,
            'gap_side':      gap_side,
            'gap_width_px':  best_gap_len,
        }

    # ──────────────────────────────────────────────────────────────────────
    #  COMBINE
    # ──────────────────────────────────────────────────────────────────────

    def _combine(self, lidar: dict, depth: dict,
                 perception: dict, new_scan: bool) -> dict:
        """
        Fuse three sensor results into one behaviour decision.

        Priority order:
          1. LiDAR zone determines urgency (fast, reliable)
          2. YOLO type determines WAIT vs AVOID behaviour
          3. RealSense gap determines avoidance direction
        """
        zone      = lidar['zone']
        dist      = lidar['distance_m']
        obj_type  = perception['nearest_type']
        avoid_side= depth['gap_side']

        # Behaviour decision
        if zone == ZONE_STOP:
            behaviour = BEHAVIOUR_EMERGENCY_STOP

        elif zone == ZONE_WARN:
            if obj_type in (OBJ_PERSON, OBJ_MOVING):
                behaviour = BEHAVIOUR_WAIT    # entity will move — wait
            else:
                behaviour = BEHAVIOUR_AVOID   # static or unknown — steer around

        else:  # ZONE_CLEAR
            behaviour = BEHAVIOUR_NAVIGATE

        return {
            'zone':           zone,
            'distance_m':     dist,
            'avoid_side':     avoid_side,
            'obstacle_type':  obj_type,
            'behaviour':      behaviour,
            'sector_min_m':   lidar['sector_min'],
            'left_count':     lidar['left_count'],
            'right_count':    lidar['right_count'],
            'left_clear_m':   depth['left_clear_m'],
            'right_clear_m':  depth['right_clear_m'],
            'gap_side':       depth['gap_side'],
            'gap_width_px':   depth['gap_width_px'],
            'yolo_name':      perception['nearest_name'],
            'yolo_conf':      perception['nearest_conf'],
            'yolo_fps':       perception['yolo_fps'],
            'n_yolo_dets':    perception['n_detections'],
            'new_lidar_scan': new_scan,
        }
