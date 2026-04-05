"""
obstacle_detector.py  —  Sensor fusion + LiDAR path planner for QCar 2

Core algorithm: Vector Field Histogram (VFH) gap finder.
  1. Divide 360° LiDAR scan into angular bins (10° each = 36 bins)
  2. For each bin, record minimum distance
  3. Find contiguous "open" gaps (bins where min_dist > threshold)
  4. Score each gap: width × depth, biased toward goal direction
  5. Return: best_path_angle, steer_cmd, drive_forward

Decision logic:
  - Best gap is forward-ish (±90°) → drive forward, steer toward gap
  - Best gap is behind (>90° from fwd) AND front blocked → reverse toward gap
  - No gap found → STOP

Also fuses: RealSense depth (gap confirmation) + YOLO (person=wait).
"""
import numpy as np
import warnings
from perceiver import (Perceiver, PerceiverConfig,
                       OBJ_PERSON, OBJ_MOVING, OBJ_STATIC, OBJ_NONE)

# Zone constants
ZONE_CLEAR = 'CLEAR'
ZONE_WARN  = 'WARN'
ZONE_STOP  = 'STOP'

# Behaviour constants
BEHAVIOUR_NAVIGATE       = 'NAVIGATE'
BEHAVIOUR_WAIT           = 'WAIT'
BEHAVIOUR_AVOID          = 'AVOID'
BEHAVIOUR_EMERGENCY_STOP = 'EMERGENCY_STOP'


class DetectorConfig:
    # Forward sector (for zone classification)
    WARN_DISTANCE_M     = 1.5
    STOP_DISTANCE_M     = 0.4
    FORWARD_HALF_ANGLE  = np.radians(40)
    MIN_SECTOR_READINGS = 3
    DISTANCE_SMOOTH_N   = 3

    # Rear sector
    REAR_HALF_ANGLE     = np.radians(40)
    REAR_CLEAR_M        = 0.5

    # VFH path planner
    NUM_BINS            = 36        # 360° / 36 = 10° per bin
    BIN_WIDTH_RAD       = 2 * np.pi / 36
    GAP_THRESHOLD_M     = 0.6      # bin is "open" if min_dist > this
    MIN_GAP_BINS        = 3         # gap must be >= 3 bins wide (30°)
    GOAL_BIAS_WEIGHT    = 2.0       # extra score for gaps near goal direction
    PATH_MAX_RANGE_M    = 4.0       # clip LiDAR beyond this for planning

    # Depth camera
    DEPTH_ROW_START     = 160
    DEPTH_ROW_END       = 320
    DEPTH_MAX_M         = 3.0
    DEPTH_MIN_VALID_PX  = 80
    GAP_MIN_DEPTH_M     = 0.8
    GAP_MIN_WIDTH_PX    = 60

    # LiDAR-to-camera correlation
    CAMERA_FOV_RAD       = np.radians(90)
    CAMERA_WIDTH_PX      = 640
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
        # Path planner output
        'has_path':        True,
        'best_path_angle': 0.0,
        'path_steer':      0.0,
        'drive_forward':   True,
        'best_gap_width_deg': 0.0,
        'best_gap_depth_m':  99.0,
    }


class ObstacleDetector:

    def __init__(self, config: DetectorConfig, perceiver: Perceiver):
        self.cfg        = config
        self.perceiver  = perceiver
        self._dist_hist = []
        self._last      = _empty_result()
        self._goal_heading = 0.0    # set by observer each tick

    def set_goal_heading(self, heading_rad: float):
        """Set current heading toward goal (from navigator). Used to bias path selection."""
        self._goal_heading = heading_rad

    def detect(self, sensor_data: dict) -> dict:
        new_scan  = sensor_data.get('lidar_new_scan', False)
        distances = sensor_data['lidar_distances']
        angles    = sensor_data['lidar_angles']
        valid     = sensor_data['lidar_valid']

        if new_scan and len(distances) > 0 and valid.sum() > 0:
            lidar_fwd  = self._analyse_lidar_forward(distances, angles, valid)
            lidar_rear = self._analyse_lidar_rear(distances, angles, valid)
            path_plan  = self._find_best_path(distances, angles, valid)
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
            path_plan = {
                'has_path':          self._last.get('has_path', True),
                'best_path_angle':   self._last.get('best_path_angle', 0.0),
                'path_steer':        self._last.get('path_steer', 0.0),
                'drive_forward':     self._last.get('drive_forward', True),
                'best_gap_width_deg':self._last.get('best_gap_width_deg', 0.0),
                'best_gap_depth_m':  self._last.get('best_gap_depth_m', 99.0),
            }

        depth = self._analyse_depth(sensor_data['rs_depth_m'])

        perception = self.perceiver.perceive(
            sensor_data['csi_front'],
            sensor_data['rs_depth_m'],
        )

        correlated_type = self._correlate_lidar_yolo(
            distances, angles, valid, perception,
        )

        front_blocked = lidar_fwd['zone'] in (ZONE_STOP, ZONE_WARN)
        result = self._combine(lidar_fwd, depth, perception, new_scan, correlated_type)

        # Rear awareness
        result['rear_min_m']    = lidar_rear['rear_min_m']
        result['rear_clear']    = lidar_rear['rear_clear']
        result['front_blocked'] = front_blocked

        # Path planner output
        result['has_path']          = path_plan['has_path']
        result['best_path_angle']   = path_plan['best_path_angle']
        result['path_steer']        = path_plan['path_steer']
        result['drive_forward']     = path_plan['drive_forward']
        result['best_gap_width_deg']= path_plan['best_gap_width_deg']
        result['best_gap_depth_m']  = path_plan['best_gap_depth_m']

        # Override avoid_side from path planner (more accurate than depth alone)
        if path_plan['has_path']:
            angle = path_plan['best_path_angle']
            # Normalize to [-π, π]: positive = left, negative = right
            if angle > np.pi:
                angle -= 2 * np.pi
            result['avoid_side'] = 'left' if angle >= 0 else 'right'

        # Raw LiDAR for dashboard
        result['all_distances']   = distances.copy() if len(distances) > 0 else np.array([])
        result['all_angles']      = angles.copy() if len(angles) > 0 else np.array([])
        result['all_valid']       = valid.copy() if len(valid) > 0 else np.array([], dtype=bool)
        result['annotated_frame'] = perception['annotated_frame']
        result['battery_v']       = sensor_data['battery_voltage']

        self._last = result
        return result

    # ── VFH Path Planner ─────────────────────────────────────────────────────

    def _find_best_path(self, distances, angles, valid) -> dict:
        """
        Vector Field Histogram: scan 360° LiDAR → find best navigable gap.

        Returns:
            has_path:          bool   — any navigable gap found
            best_path_angle:   float  — center angle of best gap (rad, 0=fwd)
            path_steer:        float  — steering command toward gap (rad)
            drive_forward:     bool   — True=go forward, False=reverse needed
            best_gap_width_deg:float  — width of chosen gap in degrees
            best_gap_depth_m:  float  — average depth of chosen gap
        """
        cfg = self.cfg
        no_path = {
            'has_path': False, 'best_path_angle': 0.0, 'path_steer': 0.0,
            'drive_forward': True, 'best_gap_width_deg': 0.0, 'best_gap_depth_m': 0.0,
        }

        if len(distances) == 0 or valid.sum() < 5:
            return {**no_path, 'has_path': True}  # no data = assume clear

        # Step 1: Build angular histogram
        # Bins: 0..NUM_BINS-1, each covers BIN_WIDTH_RAD
        n_bins = cfg.NUM_BINS
        bin_min = np.full(n_bins, cfg.PATH_MAX_RANGE_M)  # max distance = "clear"
        bin_count = np.zeros(n_bins, dtype=int)

        v_mask = valid & (distances > 0.05) & (distances < cfg.PATH_MAX_RANGE_M)
        v_dist = distances[v_mask]
        v_ang  = angles[v_mask] % (2 * np.pi)
        bin_idx = (v_ang / cfg.BIN_WIDTH_RAD).astype(int) % n_bins

        for i in range(len(v_dist)):
            b = bin_idx[i]
            bin_count[b] += 1
            if v_dist[i] < bin_min[b]:
                bin_min[b] = v_dist[i]

        # Bins with no readings → assume clear (no obstacle detected there)
        # (LiDAR might not have full coverage in some angles)

        # Step 2: Mark open/blocked bins
        is_open = bin_min > cfg.GAP_THRESHOLD_M

        # Step 3: Find contiguous gaps (circular — wraps around)
        # Double the array to handle wrap-around
        doubled = np.concatenate([is_open, is_open])
        gaps = []
        i = 0
        while i < 2 * n_bins:
            if doubled[i]:
                start = i
                while i < 2 * n_bins and doubled[i]:
                    i += 1
                length = i - start
                # Only keep gaps that don't span more than full circle
                if length <= n_bins:
                    gaps.append((start % n_bins, length))
            else:
                i += 1

        # Deduplicate gaps that appear twice due to doubling
        seen = set()
        unique_gaps = []
        for start, length in gaps:
            key = (start, length)
            if key not in seen and length >= cfg.MIN_GAP_BINS:
                seen.add(key)
                unique_gaps.append((start, length))

        if not unique_gaps:
            return no_path

        # Step 4: Score each gap
        goal_heading = self._goal_heading
        if goal_heading < 0:
            goal_heading += 2 * np.pi

        best_score = -1.0
        best_gap   = None

        for start, length in unique_gaps:
            # Center angle of gap
            center_bin = (start + length / 2.0) % n_bins
            center_angle = center_bin * cfg.BIN_WIDTH_RAD

            # Average depth of bins in this gap
            gap_bins = [(start + j) % n_bins for j in range(length)]
            gap_depths = bin_min[gap_bins]
            avg_depth = float(np.mean(gap_depths))

            # Width in degrees
            width_deg = length * np.degrees(cfg.BIN_WIDTH_RAD)

            # Angular distance from goal direction
            angle_diff = abs(center_angle - goal_heading)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff

            # Score: prefer wide gaps, deep gaps, near goal
            # Normalize: width [0-360] → [0-1], depth [0-4] → [0-1]
            width_score = min(width_deg / 180.0, 1.0)
            depth_score = min(avg_depth / cfg.PATH_MAX_RANGE_M, 1.0)
            goal_score  = 1.0 - (angle_diff / np.pi)  # 1.0 = toward goal, 0.0 = opposite

            score = (width_score * 0.3 +
                     depth_score * 0.3 +
                     goal_score  * cfg.GOAL_BIAS_WEIGHT * 0.4)

            if score > best_score:
                best_score = score
                best_gap   = (center_angle, width_deg, avg_depth)

        if best_gap is None:
            return no_path

        best_angle, gap_width_deg, gap_depth = best_gap

        # Step 5: Convert best angle to steering command
        # Normalize angle to [-π, π] where 0 = forward
        steer_angle = best_angle
        if steer_angle > np.pi:
            steer_angle -= 2 * np.pi

        # Is the best path forward or behind us?
        is_forward = abs(steer_angle) <= (np.pi / 2 + 0.2)  # ~110° tolerance

        # Steering command: proportional to angle offset, clamped to ±π/6
        max_steer = np.pi / 6
        if is_forward:
            # Steer toward gap: positive steer_angle = left = positive steering
            path_steer = float(np.clip(steer_angle * 0.8, -max_steer, max_steer))
        else:
            # Gap is behind us → need to reverse
            # While reversing, steering is inverted: steer toward where rear should go
            # If gap is at +150° (rear-left), reverse and steer right to swing rear left
            rear_angle = steer_angle - np.sign(steer_angle) * np.pi
            path_steer = float(np.clip(rear_angle * 0.8, -max_steer, max_steer))

        return {
            'has_path':           True,
            'best_path_angle':    best_angle,
            'path_steer':         path_steer,
            'drive_forward':      is_forward,
            'best_gap_width_deg': gap_width_deg,
            'best_gap_depth_m':   gap_depth,
        }

    # ── Forward / Rear sector analysis ────────────────────────────────────────

    def _analyse_lidar_forward(self, distances, angles, valid) -> dict:
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
        cfg = self.cfg
        if len(distances) == 0 or valid.sum() < 2:
            return {'rear_min_m': 99.0, 'rear_clear': True}

        rear_lo = np.pi - cfg.REAR_HALF_ANGLE
        rear_hi = np.pi + cfg.REAR_HALF_ANGLE
        in_rear = (angles >= rear_lo) & (angles <= rear_hi) & valid

        if in_rear.sum() < 2:
            return {'rear_min_m': 99.0, 'rear_clear': True}

        rear_d   = distances[in_rear]
        rear_min = float(rear_d.min())
        return {'rear_min_m': rear_min, 'rear_clear': rear_min > cfg.REAR_CLEAR_M}

    # ── Depth camera ──────────────────────────────────────────────────────────

    def _analyse_depth(self, depth_frame: np.ndarray) -> dict:
        cfg = self.cfg
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

    # ── YOLO correlation ──────────────────────────────────────────────────────

    def _correlate_lidar_yolo(self, distances, angles, valid, perception) -> str:
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

        return OBJ_STATIC

    # ── Combine all sources ───────────────────────────────────────────────────

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
