"""
obstacle_detector.py — VFH path planner + LiDAR + YOLO fusion.
Adapted from Final/obstacle_detector.py for 360-degree camera setup.

Uses Vector Field Histogram to find the best gap in LiDAR data,
then correlates with YOLO detections from all 4 cameras to classify
obstacles as PERSON/MOVING/STATIC.
"""
import math
import numpy as np
import logging
from collections import deque

import config as cfg


class ObstacleDetector:
    """
    360-degree obstacle detection using LiDAR + YOLO fusion.

    Output dict:
        zone:            'CLEAR' / 'WARN' / 'STOP'
        behaviour:       'NAVIGATE' / 'AVOID' / 'WAIT' / 'EMERGENCY_STOP'
        distance_m:      smoothed forward distance (metres)
        rear_min_m:      minimum rear distance (metres)
        has_path:        True if VFH found a usable gap
        best_path_angle: centre angle of chosen gap (radians, 0=front)
        best_gap_width:  gap width (degrees)
        best_gap_depth:  gap depth (metres)
        path_steer:      steering command to follow gap (-0.5 .. +0.5)
        drive_forward:   True=forward, False=reverse
        obstacle_type:   'NONE' / 'PERSON' / 'MOVING' / 'STATIC'
        avoid_side:      'LEFT' / 'RIGHT' / 'NONE'
    """

    def __init__(self):
        self._histogram = np.full(cfg.VFH_NUM_BINS, cfg.VFH_PLAN_RANGE_M)
        self._front_history = deque(maxlen=cfg.SMOOTH_WINDOW)
        self._goal_heading = 0.0   # radians, body frame

    def set_goal_heading(self, heading_rad):
        """Set desired heading for VFH goal bias (body frame, 0=front)."""
        self._goal_heading = heading_rad

    def detect(self, sensor_data, yolo_dets=None):
        """
        Main detection call. Returns detection result dict.

        Args:
            sensor_data: dict from SensorManager.read()
            yolo_dets:   list of Detection objects from YOLOProcessor
        """
        distances = sensor_data['lidar_distances']
        angles    = sensor_data['lidar_angles']
        valid     = sensor_data['lidar_valid']

        # 1. Build VFH histogram from LiDAR
        self._build_histogram(distances, angles, valid)

        # 2. Analyse forward and rear sectors
        front_dist = self._sector_min(distances, angles, valid,
                                      0.0, cfg.FRONT_SECTOR_DEG)
        rear_dist  = self._sector_min(distances, angles, valid,
                                      np.pi, cfg.REAR_SECTOR_DEG)

        # Smooth forward distance
        if front_dist < cfg.LIDAR_MAX_M:
            self._front_history.append(front_dist)
        smoothed = float(np.mean(self._front_history)) if self._front_history else front_dist

        # 3. Determine zone
        if smoothed > cfg.ZONE_CLEAR_M:
            zone = 'CLEAR'
        elif smoothed > cfg.ZONE_WARN_M:
            zone = 'WARN'
        else:
            zone = 'STOP'

        # 4. Find best gap using VFH
        gaps = self._find_gaps()
        best_gap = self._score_gaps(gaps)

        has_path       = best_gap is not None
        path_angle     = 0.0
        gap_width_deg  = 0.0
        gap_depth      = 0.0
        path_steer     = 0.0
        drive_forward  = True

        if has_path:
            path_angle    = best_gap['angle']
            gap_width_deg = best_gap['width_deg']
            gap_depth     = best_gap['depth']
            path_steer, drive_forward = self._compute_steering(path_angle)

        # 5. YOLO correlation — classify obstacle type
        obstacle_type = 'NONE'
        if yolo_dets and zone != 'CLEAR':
            obstacle_type = self._correlate_yolo(yolo_dets, distances, angles, valid)

        # 6. Determine behaviour
        if obstacle_type == 'PERSON' and zone != 'CLEAR':
            behaviour = 'WAIT'
        elif zone == 'STOP':
            behaviour = 'EMERGENCY_STOP'
        elif zone == 'WARN':
            behaviour = 'AVOID'
        else:
            behaviour = 'NAVIGATE'

        # 7. Determine avoid side from path angle
        if has_path and behaviour in ('AVOID', 'EMERGENCY_STOP'):
            norm_angle = path_angle
            if norm_angle > np.pi:
                norm_angle -= 2 * np.pi
            avoid_side = 'LEFT' if norm_angle > 0 else 'RIGHT'
        else:
            avoid_side = 'NONE'

        return {
            'zone':            zone,
            'behaviour':       behaviour,
            'distance_m':      round(smoothed, 3),
            'rear_min_m':      round(rear_dist, 3),
            'has_path':        has_path,
            'best_path_angle': path_angle,
            'best_gap_width':  gap_width_deg,
            'best_gap_depth':  gap_depth,
            'path_steer':      round(path_steer, 4),
            'drive_forward':   drive_forward,
            'obstacle_type':   obstacle_type,
            'avoid_side':      avoid_side,
        }

    # ── VFH Histogram ─────────────────────────────────────────────────────────

    def _build_histogram(self, distances, angles, valid):
        """Bin LiDAR readings into angular histogram."""
        self._histogram[:] = cfg.VFH_PLAN_RANGE_M

        if len(distances) == 0:
            return

        for i in range(len(distances)):
            if not valid[i]:
                continue
            d = distances[i]
            if d > cfg.VFH_PLAN_RANGE_M:
                continue
            a = angles[i] % (2 * np.pi)
            bin_idx = int(a / (2 * np.pi) * cfg.VFH_NUM_BINS) % cfg.VFH_NUM_BINS
            if d < self._histogram[bin_idx]:
                self._histogram[bin_idx] = d

    def _sector_min(self, distances, angles, valid,
                    centre_rad, half_width_deg):
        """Find minimum distance in a sector."""
        if len(distances) == 0:
            return cfg.LIDAR_MAX_M

        hw = np.radians(half_width_deg)
        mins = []
        for i in range(len(distances)):
            if not valid[i]:
                continue
            a = angles[i] % (2 * np.pi)
            diff = abs(a - centre_rad)
            if diff > np.pi:
                diff = 2 * np.pi - diff
            if diff <= hw:
                mins.append(distances[i])

        return min(mins) if mins else cfg.LIDAR_MAX_M

    # ── Gap Finding ───────────────────────────────────────────────────────────

    def _find_gaps(self):
        """Find contiguous open gaps in the histogram."""
        n = cfg.VFH_NUM_BINS
        is_open = self._histogram > cfg.VFH_GAP_THRESH_M

        # Find contiguous sequences of open bins (circular)
        gaps = []
        visited = [False] * n

        for start in range(n):
            if not is_open[start] or visited[start]:
                continue
            # Walk forward through contiguous open bins
            length = 0
            depth_sum = 0.0
            idx = start
            while is_open[idx % n] and length < n:
                visited[idx % n] = True
                depth_sum += self._histogram[idx % n]
                length += 1
                idx += 1

            if length >= cfg.VFH_MIN_GAP_BINS:
                centre_bin = (start + length / 2) % n
                centre_angle = centre_bin * (2 * np.pi / n)
                gaps.append({
                    'start_bin':  start,
                    'length':     length,
                    'width_deg':  length * cfg.VFH_BIN_WIDTH_DEG,
                    'depth':      depth_sum / length,
                    'angle':      centre_angle,
                })

        return gaps

    def _score_gaps(self, gaps):
        """Score gaps by width, depth, and alignment with goal heading."""
        if not gaps:
            return None

        best = None
        best_score = -1.0

        for gap in gaps:
            # Width score: prefer wider gaps, saturate at 180°
            w_score = min(gap['width_deg'] / 180.0, 1.0)

            # Depth score: prefer deeper gaps
            d_score = min(gap['depth'] / cfg.VFH_PLAN_RANGE_M, 1.0)

            # Goal alignment: prefer gaps toward goal heading
            angle_diff = abs(gap['angle'] - self._goal_heading)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            g_score = 1.0 - (angle_diff / np.pi)  # 1.0 = perfect alignment

            score = 0.3 * w_score + 0.3 * d_score + 0.4 * g_score * cfg.VFH_GOAL_BIAS

            if score > best_score:
                best_score = score
                best = gap

        return best

    # ── Steering ──────────────────────────────────────────────────────────────

    def _compute_steering(self, path_angle_rad):
        """Convert gap angle to steering command + forward/reverse decision."""
        # Normalize to [-pi, pi]
        a = path_angle_rad
        if a > np.pi:
            a -= 2 * np.pi

        # If gap is roughly ahead (±110°): drive forward
        if abs(a) <= np.radians(110):
            steer = cfg.STEERING_GAIN * a
            steer = max(-cfg.MAX_STEERING_RAD, min(steer, cfg.MAX_STEERING_RAD))
            return float(steer), True
        else:
            # Gap is behind: reverse, invert steering to swing rear toward gap
            rear_angle = a - np.pi if a > 0 else a + np.pi
            steer = cfg.STEERING_GAIN * rear_angle
            steer = max(-cfg.MAX_STEERING_RAD, min(steer, cfg.MAX_STEERING_RAD))
            return float(steer), False

    # ── YOLO Correlation ──────────────────────────────────────────────────────

    def _correlate_yolo(self, yolo_dets, distances, angles, valid):
        """
        Match YOLO detections with LiDAR readings.
        Returns the most critical obstacle type found.
        """
        has_person = False
        has_moving = False

        for det in yolo_dets:
            det_angle_rad = np.radians(det.angle_centre)

            # Check if any LiDAR reading within ±YOLO_CORR_ANGLE confirms
            # an obstacle at this angle within WARN distance
            hw = np.radians(cfg.YOLO_CORR_ANGLE)
            for i in range(len(distances)):
                if not valid[i]:
                    continue
                a = angles[i] % (2 * np.pi)
                diff = abs(a - det_angle_rad)
                if diff > np.pi:
                    diff = 2 * np.pi - diff
                if diff <= hw and distances[i] <= cfg.ZONE_CLEAR_M:
                    if det.obj_type == 'PERSON':
                        has_person = True
                    elif det.obj_type == 'MOVING':
                        has_moving = True
                    break

        if has_person:
            return 'PERSON'
        elif has_moving:
            return 'MOVING'
        else:
            return 'STATIC'
