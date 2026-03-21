"""
obstacle_detector.py  —  Sensor fusion for obstacle detection
Fuses LiDAR + RealSense + YOLO into one behaviour decision per tick.

  LiDAR    → IS there an obstacle? HOW FAR? FRONT AND REAR.
  RealSense → WHICH SIDE has clear space?
  YOLO     → WHAT IS IT? (person=wait, static=steer)

KEY FIX: Full 360-degree LiDAR awareness.
  - Forward sector (±40°): primary obstacle detection
  - Rear sector (180°±40°): reverse safety check
  - Returns front_blocked, rear_clear for autonomous direction decisions
"""
import numpy as np
import warnings
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
    # Forward sector
    WARN_DISTANCE_M     = 1.5
    STOP_DISTANCE_M     = 0.4
    FORWARD_HALF_ANGLE  = np.radians(40)
    MIN_SECTOR_READINGS = 3
    DISTANCE_SMOOTH_N   = 3

    # Rear sector
    REAR_HALF_ANGLE     = np.radians(40)
    REAR_CLEAR_M        = 0.5     # rear is "clear" if min distance > this

    # Depth camera
    DEPTH_ROW_START     = 160
    DEPTH_ROW_END       = 320
    DEPTH_MAX_M         = 3.0
    DEPTH_MIN_VALID_PX  = 80
    GAP_MIN_DEPTH_M     = 0.8
    GAP_MIN_WIDTH_PX    = 60

    # LiDAR-to-camera correlation
    CAMERA_FOV_RAD      = np.radians(90)
    CAMERA_WIDTH_PX     = 640
    LIDAR_YOLO_ANGLE_TOL = np.radians(12)


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
        # Rear awareness
        'rear_min_m':      99.0,
        'rear_clear':      True,
        'front_blocked':   False,
    }


class ObstacleDetector:

    def __init__(self, config: DetectorConfig, perceiver: Perceiver):
        self.cfg       = config
        self.perceiver = perceiver
        self._dist_hist= []
        self._last     = _empty_result()

    def detect(self, sensor_data: dict) -> dict:
        new_scan = sensor_data.get('lidar_new_scan', False)
        distances = sensor_data['lidar_distances']
        angles    = sensor_data['lidar_angles']
        valid     = sensor_data['lidar_valid']

        if new_scan:
            lidar_fwd = self._analyse_lidar_forward(distances, angles, valid)
            lidar_rear = self._analyse_lidar_rear(distances, angles, valid)
        else:
            lidar_fwd = {
                'zone':        self._last['zone'],
                'distance_m':  self._last['distance_m'],
                'sector_min':  self._last['sector_min_m'],
                'left_count':  self._last['left_count'],
                'right_count': self._last['right_count'],
            }
            lidar_rear = {
                'rear_min_m':  self._last['rear_min_m'],
                'rear_clear':  self._last['rear_clear'],
            }

        depth = self._analyse_depth(sensor_data['rs_depth_m'])

        perception = self.perceiver.perceive(
            sensor_data['csi_front'],
            sensor_data['rs_depth_m'],
        )

        # Correlate YOLO with LiDAR for accurate classification
        correlated_type = self._correlate_lidar_yolo(
            distances, angles, valid, perception,
        )

        front_blocked = lidar_fwd['zone'] in (ZONE_STOP, ZONE_WARN)
        result = self._combine(lidar_fwd, depth, perception, new_scan, correlated_type)

        # Add rear awareness
        result['rear_min_m']    = lidar_rear['rear_min_m']
        result['rear_clear']    = lidar_rear['rear_clear']
        result['front_blocked'] = front_blocked

        # Add raw LiDAR data for dashboard
        result['all_distances']   = distances.copy() if len(distances) > 0 else np.array([])
        result['all_angles']      = angles.copy() if len(angles) > 0 else np.array([])
        result['all_valid']       = valid.copy() if len(valid) > 0 else np.array([], dtype=bool)
        result['annotated_frame'] = perception['annotated_frame']
        result['battery_v']       = sensor_data['battery_voltage']

        self._last = result
        return result

    def _analyse_lidar_forward(self, distances, angles, valid) -> dict:
        """Forward sector: ±FORWARD_HALF_ANGLE from 0° (front of car)."""
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

    def _analyse_lidar_rear(self, distances, angles, valid) -> dict:
        """Rear sector: ±REAR_HALF_ANGLE from π (back of car)."""
        cfg = self.cfg
        if len(distances) == 0 or valid.sum() < 2:
            return {'rear_min_m': 99.0, 'rear_clear': True}

        # Rear is around π radians (180°)
        rear_lo = np.pi - cfg.REAR_HALF_ANGLE
        rear_hi = np.pi + cfg.REAR_HALF_ANGLE
        in_rear = (angles >= rear_lo) & (angles <= rear_hi) & valid

        if in_rear.sum() < 2:
            return {'rear_min_m': 99.0, 'rear_clear': True}

        rear_d   = distances[in_rear]
        rear_min = float(rear_d.min())

        return {
            'rear_min_m': rear_min,
            'rear_clear': rear_min > cfg.REAR_CLEAR_M,
        }

    def _analyse_depth(self, depth_frame: np.ndarray) -> dict:
        cfg = self.cfg

        # Depth validity check
        if depth_frame is None or depth_frame.size == 0:
            return {'left_clear_m': 99.0, 'right_clear_m': 99.0,
                    'gap_side': 'left', 'gap_width_px': 0}

        strip = depth_frame[cfg.DEPTH_ROW_START:cfg.DEPTH_ROW_END, :, 0]
        valid_count = int((strip > 0).sum())
        if valid_count < cfg.DEPTH_MIN_VALID_PX:
            return {'left_clear_m': 99.0, 'right_clear_m': 99.0,
                    'gap_side': 'left', 'gap_width_px': 0}

        W   = strip.shape[1]
        mid = W // 2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_means = np.where(strip > 0, strip, np.nan)
            col_means = np.nanmean(col_means, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)

        left_valid  = col_means[:mid][col_means[:mid] > 0]
        right_valid = col_means[mid:][col_means[mid:] > 0]
        lm = float(left_valid.mean())  if len(left_valid)  > 0 else 99.0
        rm = float(right_valid.mean()) if len(right_valid) > 0 else 99.0

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
        """Map nearest YOLO detection column to LiDAR angle for verification."""
        cfg = self.cfg
        nearest_type = perception.get('nearest_type', OBJ_NONE)
        if nearest_type == OBJ_NONE:
            return OBJ_NONE
        if len(distances) == 0 or valid.sum() == 0:
            return nearest_type

        nearest_x = perception.get('nearest_x', cfg.CAMERA_WIDTH_PX // 2)
        col_offset = float(nearest_x - cfg.CAMERA_WIDTH_PX // 2)
        yolo_angle = -(col_offset / (cfg.CAMERA_WIDTH_PX / 2)) * (cfg.CAMERA_FOV_RAD / 2)
        if yolo_angle < 0:
            yolo_angle += 2 * np.pi

        fwd_left  = (angles >= 0) & (angles <= cfg.FORWARD_HALF_ANGLE)
        fwd_right = angles >= (2 * np.pi - cfg.FORWARD_HALF_ANGLE)
        in_sector = (fwd_left | fwd_right) & valid
        if in_sector.sum() == 0:
            return nearest_type

        sector_angles = angles[in_sector]
        sector_dists  = distances[in_sector]

        diff = np.abs(sector_angles - yolo_angle)
        diff = np.minimum(diff, 2 * np.pi - diff)

        close_mask = diff < cfg.LIDAR_YOLO_ANGLE_TOL
        if close_mask.sum() > 0 and sector_dists[close_mask].min() < cfg.WARN_DISTANCE_M:
            return nearest_type

        # LiDAR doesn't confirm YOLO at that angle — treat as generic static
        return OBJ_STATIC

    def _combine(self, lidar, depth, perception, new_scan, correlated_type=None) -> dict:
        zone     = lidar['zone']
        dist     = lidar['distance_m']
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
