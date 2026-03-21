"""
obstacle_detector.py  —  Phase 3+4: Sensor fusion for obstacle detection
Fuses LiDAR + RealSense + YOLO into one behaviour decision per tick.

  LiDAR    → IS there an obstacle? HOW FAR?   (primary — fast, reliable)
  RealSense → WHICH SIDE has clear space?      (avoidance direction)
  YOLO     → WHAT IS IT?                       (person=wait, static=steer)

Fixes applied:
  - Depth frame validity check (handles RealSense dropout)
  - Basic LiDAR-angle ↔ YOLO-column correlation for multi-obstacle scenes
  - Shared constants from constants.py

Behaviour output (used directly by state_machine.py):
  NAVIGATE        → path clear, drive to goal
  WAIT            → person/moving obstacle, hold position
  AVOID           → static obstacle, steer around it
  EMERGENCY_STOP  → too close (<0.4m), halt immediately
"""
import numpy as np
from perceiver import (Perceiver, PerceiverConfig,
                       OBJ_PERSON, OBJ_MOVING, OBJ_STATIC, OBJ_NONE)

# Zone constants
ZONE_CLEAR = 'CLEAR'
ZONE_WARN  = 'WARN'
ZONE_STOP  = 'STOP'

# Behaviour constants (used by state_machine.py)
BEHAVIOUR_NAVIGATE       = 'NAVIGATE'
BEHAVIOUR_WAIT           = 'WAIT'
BEHAVIOUR_AVOID          = 'AVOID'
BEHAVIOUR_EMERGENCY_STOP = 'EMERGENCY_STOP'


class DetectorConfig:
    WARN_DISTANCE_M     = 1.5
    STOP_DISTANCE_M     = 0.4
    FORWARD_HALF_ANGLE  = np.radians(40)
    MIN_SECTOR_READINGS = 3
    DISTANCE_SMOOTH_N   = 3
    DEPTH_ROW_START     = 160
    DEPTH_ROW_END       = 320
    DEPTH_MAX_M         = 3.0
    DEPTH_MIN_VALID_PX  = 80
    GAP_MIN_DEPTH_M     = 0.8
    GAP_MIN_WIDTH_PX    = 60
    # LiDAR-to-camera correlation
    CAMERA_FOV_RAD      = np.radians(90)    # approximate horizontal FOV of front camera
    CAMERA_WIDTH_PX     = 640
    LIDAR_YOLO_ANGLE_TOL = np.radians(10)   # angular tolerance for matching


def _empty_result():
    return {
        'zone':            ZONE_CLEAR,
        'distance_m':      99.0,
        'avoid_side':      'left',
        'obstacle_type':   OBJ_NONE,
        'behaviour':       BEHAVIOUR_NAVIGATE,
        'sector_min_m':    99.0,
        'left_count':      0,
        'right_count':     0,
        'left_clear_m':    99.0,
        'right_clear_m':   99.0,
        'gap_side':        'left',
        'gap_width_px':    0,
        'yolo_name':       '',
        'yolo_conf':       0.0,
        'yolo_fps':        0.0,
        'n_yolo_dets':     0,
        'new_lidar_scan':  False,
        'all_distances':   np.array([]),
        'all_angles':      np.array([]),
        'all_valid':       np.array([], dtype=bool),
        'annotated_frame': None,
        'battery_v':       0.0,
    }


class ObstacleDetector:
    """
    Usage:
        perceiver = Perceiver(PerceiverConfig())
        detector  = ObstacleDetector(DetectorConfig(), perceiver)
        result = detector.detect(sensor_data)
        if result['behaviour'] == BEHAVIOUR_WAIT:
            pass  # person in path
    """

    def __init__(self, config: DetectorConfig, perceiver: Perceiver):
        self.cfg       = config
        self.perceiver = perceiver
        self._dist_hist= []
        self._last     = _empty_result()

    def detect(self, sensor_data: dict) -> dict:
        new_scan = sensor_data.get('lidar_new_scan', False)

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

        depth = self._analyse_depth(sensor_data['rs_depth_m'])

        perception = self.perceiver.perceive(
            sensor_data['csi_front'],
            sensor_data['rs_depth_m'],
        )

        # Correlate YOLO detection with LiDAR angle if we have both
        correlated_type = self._correlate_lidar_yolo(
            sensor_data['lidar_distances'],
            sensor_data['lidar_angles'],
            sensor_data['lidar_valid'],
            perception,
        )

        result = self._combine(lidar, depth, perception, new_scan, correlated_type)
        result['all_distances']   = sensor_data['lidar_distances'].copy() \
                                    if len(sensor_data['lidar_distances']) > 0 \
                                    else np.array([])
        result['all_angles']      = sensor_data['lidar_angles'].copy() \
                                    if len(sensor_data['lidar_angles']) > 0 \
                                    else np.array([])
        result['all_valid']       = sensor_data['lidar_valid'].copy() \
                                    if len(sensor_data['lidar_valid']) > 0 \
                                    else np.array([], dtype=bool)
        result['annotated_frame'] = perception['annotated_frame']
        result['battery_v']       = sensor_data['battery_voltage']
        self._last = result
        return result

    def _analyse_lidar(self, distances, angles, valid) -> dict:
        cfg = self.cfg
        if len(distances) == 0 or valid.sum() < cfg.MIN_SECTOR_READINGS:
            return {'zone': ZONE_CLEAR, 'distance_m': 99.0,
                    'sector_min': 99.0, 'left_count': 0, 'right_count': 0}

        in_left   = (angles >= 0) & (angles <= cfg.FORWARD_HALF_ANGLE)
        in_right  = angles >= (2 * np.pi - cfg.FORWARD_HALF_ANGLE)
        in_sector = (in_left | in_right) & valid

        if in_sector.sum() < cfg.MIN_SECTOR_READINGS:
            return {'zone': ZONE_CLEAR, 'distance_m': 99.0,
                    'sector_min': 99.0, 'left_count': 0, 'right_count': 0}

        sector_d   = distances[in_sector]
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

        return {'zone': zone, 'distance_m': smoothed,
                'sector_min': sector_min, 'left_count': lc, 'right_count': rc}

    def _analyse_depth(self, depth_frame: np.ndarray) -> dict:
        cfg   = self.cfg

        # Depth validity check — handle RealSense dropout or warmup
        if depth_frame is None or depth_frame.size == 0:
            return {'left_clear_m': 99.0, 'right_clear_m': 99.0,
                    'gap_side': 'left', 'gap_width_px': 0}

        strip = depth_frame[cfg.DEPTH_ROW_START:cfg.DEPTH_ROW_END, :, 0]
        valid_count = int((strip > 0).sum())

        # If too few valid pixels, depth is unreliable — default to safe
        if valid_count < cfg.DEPTH_MIN_VALID_PX:
            return {'left_clear_m': 99.0, 'right_clear_m': 99.0,
                    'gap_side': 'left', 'gap_width_px': 0}

        W   = strip.shape[1]
        mid = W // 2

        col_means = np.where(strip > 0, strip, np.nan)
        col_means = np.nanmean(col_means, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)

        lm = float(col_means[:mid][col_means[:mid] > 0].mean()) \
             if (col_means[:mid] > 0).any() else 99.0
        rm = float(col_means[mid:][col_means[mid:] > 0].mean()) \
             if (col_means[mid:] > 0).any() else 99.0

        # Gap finder: widest contiguous clear column range
        clear = col_means > cfg.GAP_MIN_DEPTH_M
        best_start, best_len = 0, 0
        cur_start, cur_len   = 0, 0
        for i, is_clear in enumerate(clear):
            if is_clear:
                if cur_len == 0: cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_len   = cur_len
                    best_start = cur_start
            else:
                cur_len = 0

        gap_centre = best_start + best_len // 2
        if best_len >= cfg.GAP_MIN_WIDTH_PX:
            gap_side = 'left' if gap_centre < mid else 'right'
        else:
            gap_side = 'left' if lm >= rm else 'right'

        return {'left_clear_m': lm, 'right_clear_m': rm,
                'gap_side': gap_side, 'gap_width_px': best_len}

    def _correlate_lidar_yolo(self, distances, angles, valid, perception) -> str:
        """
        Basic spatial correlation: map the nearest YOLO detection's image column
        to a LiDAR angle and check if a LiDAR hit exists at that angle.
        Returns the YOLO object type if correlation succeeds, else OBJ_NONE.

        This prevents misattribution when multiple objects are in the scene
        (e.g. person on the left, box on the right — LiDAR sees the box
        but YOLO's 'nearest' is the person).
        """
        cfg = self.cfg
        nearest_type = perception.get('nearest_type', OBJ_NONE)
        if nearest_type == OBJ_NONE:
            return OBJ_NONE
        if len(distances) == 0 or valid.sum() == 0:
            return nearest_type  # can't correlate, trust YOLO

        nearest_x = perception.get('nearest_x', cfg.CAMERA_WIDTH_PX // 2)
        # Map image column to angle: centre=0, left=+angle, right=-angle
        col_offset = float(nearest_x - cfg.CAMERA_WIDTH_PX // 2)
        yolo_angle = -(col_offset / (cfg.CAMERA_WIDTH_PX / 2)) * (cfg.CAMERA_FOV_RAD / 2)

        # Normalise to [0, 2π] to match LiDAR convention
        if yolo_angle < 0:
            yolo_angle += 2 * np.pi

        # Check if any valid LiDAR reading falls near this angle
        # within the forward sector and within warning distance
        fwd_left  = (angles >= 0) & (angles <= cfg.FORWARD_HALF_ANGLE)
        fwd_right = angles >= (2 * np.pi - cfg.FORWARD_HALF_ANGLE)
        in_sector = (fwd_left | fwd_right) & valid

        if in_sector.sum() == 0:
            return nearest_type

        sector_angles = angles[in_sector]
        sector_dists  = distances[in_sector]

        # Angular distance (handling wrap-around)
        diff = np.abs(sector_angles - yolo_angle)
        diff = np.minimum(diff, 2 * np.pi - diff)

        close_mask = diff < cfg.LIDAR_YOLO_ANGLE_TOL
        if close_mask.sum() > 0 and sector_dists[close_mask].min() < cfg.WARN_DISTANCE_M:
            # LiDAR confirms obstacle at YOLO's angular position
            return nearest_type

        # No LiDAR confirmation at YOLO's position — the LiDAR obstacle
        # may be a different object than what YOLO identified as nearest.
        # Fall back to generic classification.
        return OBJ_STATIC

    def _combine(self, lidar, depth, perception, new_scan, correlated_type=None) -> dict:
        zone     = lidar['zone']
        dist     = lidar['distance_m']
        # Use correlated type if available, otherwise fall back to YOLO nearest
        obj_type = correlated_type if correlated_type else perception['nearest_type']
        avoid    = depth['gap_side']

        if zone == ZONE_STOP:
            behaviour = BEHAVIOUR_EMERGENCY_STOP
        elif zone == ZONE_WARN:
            if obj_type in (OBJ_PERSON, OBJ_MOVING):
                behaviour = BEHAVIOUR_WAIT
            else:
                behaviour = BEHAVIOUR_AVOID
        else:
            behaviour = BEHAVIOUR_NAVIGATE

        return {
            'zone':           zone,
            'distance_m':     dist,
            'avoid_side':     avoid,
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
