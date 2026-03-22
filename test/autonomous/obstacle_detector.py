"""
obstacle_detector.py — Vehicle-aware VFH path planner + 360° LiDAR + YOLO fusion.

Key improvements over basic VFH:
  1. Obstacles inflated by car half-width (car can't fit through narrow gaps)
  2. Gap passability checked against turning radius at that distance
  3. YOLO detections injected INTO the histogram (not just correlated after)
  4. Side-clearance checks for safe turning
  5. Steering scaled by obstacle proximity (gentler near obstacles)
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
        zone, behaviour, distance_m, rear_min_m,
        left_min_m, right_min_m,
        has_path, best_path_angle, best_gap_width, best_gap_depth,
        path_steer, drive_forward,
        obstacle_type, avoid_side
    """

    def __init__(self):
        self._histogram = np.full(cfg.VFH_NUM_BINS, cfg.VFH_PLAN_RANGE_M)
        self._front_history = deque(maxlen=cfg.SMOOTH_WINDOW)
        self._goal_heading = 0.0

    def set_goal_heading(self, heading_rad):
        self._goal_heading = heading_rad

    def detect(self, sensor_data, yolo_dets=None):
        distances = sensor_data['lidar_distances']
        angles    = sensor_data['lidar_angles']
        valid     = sensor_data['lidar_valid']

        # 1. Build VFH histogram from LiDAR (with obstacle inflation)
        self._build_histogram(distances, angles, valid)

        # 2. Inject YOLO detections into histogram as virtual obstacles
        if yolo_dets:
            self._inject_yolo(yolo_dets, distances, angles, valid)

        # 3. Analyse all sectors (front, rear, left, right)
        front_dist = self._sector_min_hist(0.0, cfg.FRONT_SECTOR_DEG)
        rear_dist  = self._sector_min_hist(np.pi, cfg.REAR_SECTOR_DEG)
        left_dist  = self._sector_min_hist(np.pi * 1.5, cfg.SIDE_SECTOR_DEG)
        right_dist = self._sector_min_hist(np.pi * 0.5, cfg.SIDE_SECTOR_DEG)

        # Smooth forward distance
        if front_dist < cfg.LIDAR_MAX_M:
            self._front_history.append(front_dist)
        smoothed = float(np.mean(self._front_history)) if self._front_history else front_dist

        # 4. Determine zone
        if smoothed > cfg.ZONE_CLEAR_M:
            zone = 'CLEAR'
        elif smoothed > cfg.ZONE_WARN_M:
            zone = 'WARN'
        else:
            zone = 'STOP'

        # 5. Find best gap (vehicle-aware)
        gaps = self._find_gaps()
        gaps = self._filter_passable_gaps(gaps)
        best_gap = self._score_gaps(gaps, left_dist, right_dist)

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
            path_steer, drive_forward = self._compute_steering(
                path_angle, smoothed, left_dist, right_dist)

        # 6. YOLO obstacle classification
        obstacle_type = 'NONE'
        if yolo_dets and zone != 'CLEAR':
            obstacle_type = self._classify_yolo(yolo_dets, distances, angles, valid)

        # 7. Determine behaviour
        if obstacle_type == 'PERSON' and zone != 'CLEAR':
            behaviour = 'WAIT'
        elif zone == 'STOP':
            behaviour = 'EMERGENCY_STOP'
        elif zone == 'WARN':
            behaviour = 'AVOID'
        else:
            behaviour = 'NAVIGATE'

        # 8. Avoid side from path + side clearance
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
            'left_min_m':      round(left_dist, 3),
            'right_min_m':     round(right_dist, 3),
            'has_path':        has_path,
            'best_path_angle': path_angle,
            'best_gap_width':  gap_width_deg,
            'best_gap_depth':  gap_depth,
            'path_steer':      round(path_steer, 4),
            'drive_forward':   drive_forward,
            'obstacle_type':   obstacle_type,
            'avoid_side':      avoid_side,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  VFH HISTOGRAM — obstacle inflation by car width
    # ══════════════════════════════════════════════════════════════════════════

    def _build_histogram(self, distances, angles, valid):
        """Build histogram with obstacle inflation for car width."""
        self._histogram[:] = cfg.VFH_PLAN_RANGE_M
        n_bins = cfg.VFH_NUM_BINS

        if len(distances) == 0:
            return

        for i in range(len(distances)):
            if not valid[i]:
                continue
            d = distances[i]
            if d > cfg.VFH_PLAN_RANGE_M:
                continue

            a = angles[i] % (2 * np.pi)
            centre_bin = int(a / (2 * np.pi) * n_bins) % n_bins

            # Inflate obstacle: at distance d, car half-width subtends
            # an angle of arctan(half_width / d)
            if d > 0.05:
                inflate_rad = math.atan(cfg.OBSTACLE_INFLATE_M / d)
                inflate_bins = max(1, int(math.ceil(
                    inflate_rad / cfg.VFH_BIN_WIDTH_RAD)))
            else:
                inflate_bins = 3  # very close — block wide sector

            # Mark centre bin + neighbouring bins
            for offset in range(-inflate_bins, inflate_bins + 1):
                b = (centre_bin + offset) % n_bins
                if d < self._histogram[b]:
                    self._histogram[b] = d

    def _sector_min_hist(self, centre_rad, half_width_deg):
        """Get minimum distance in a sector from the histogram."""
        n_bins = cfg.VFH_NUM_BINS
        hw_bins = int(half_width_deg / cfg.VFH_BIN_WIDTH_DEG)
        centre_bin = int((centre_rad / (2 * np.pi)) * n_bins) % n_bins

        min_d = cfg.VFH_PLAN_RANGE_M
        for offset in range(-hw_bins, hw_bins + 1):
            b = (centre_bin + offset) % n_bins
            if self._histogram[b] < min_d:
                min_d = self._histogram[b]
        return min_d

    # ══════════════════════════════════════════════════════════════════════════
    #  YOLO INJECTION — camera detections become histogram obstacles
    # ══════════════════════════════════════════════════════════════════════════

    def _inject_yolo(self, yolo_dets, distances, angles, valid):
        """Inject YOLO detections into the VFH histogram.
        If LiDAR confirms an object nearby, use LiDAR distance.
        Otherwise, inject at YOLO_INJECT_DIST_M as a conservative estimate."""
        n_bins = cfg.VFH_NUM_BINS

        for det in yolo_dets:
            det_angle_rad = np.radians(det.angle_centre) % (2 * np.pi)
            det_left_rad  = np.radians(det.angle_left) % (2 * np.pi)
            det_right_rad = np.radians(det.angle_right) % (2 * np.pi)

            # Find closest LiDAR reading within the detection's angular span
            lidar_dist = None
            hw = np.radians(cfg.YOLO_CORR_ANGLE)
            for i in range(len(distances)):
                if not valid[i]:
                    continue
                a = angles[i] % (2 * np.pi)
                diff = abs(a - det_angle_rad)
                if diff > np.pi:
                    diff = 2 * np.pi - diff
                if diff <= hw and distances[i] <= cfg.ZONE_CLEAR_M:
                    if lidar_dist is None or distances[i] < lidar_dist:
                        lidar_dist = float(distances[i])

            inject_dist = lidar_dist if lidar_dist else cfg.YOLO_INJECT_DIST_M

            # Inject into histogram bins covering the detection's angular span
            bin_left  = int((det_left_rad / (2 * np.pi)) * n_bins) % n_bins
            bin_right = int((det_right_rad / (2 * np.pi)) * n_bins) % n_bins

            # Handle wrap-around
            b = bin_left
            for _ in range(n_bins):
                if inject_dist < self._histogram[b]:
                    self._histogram[b] = inject_dist
                if b == bin_right:
                    break
                b = (b + 1) % n_bins

    # ══════════════════════════════════════════════════════════════════════════
    #  GAP FINDING — vehicle-aware passability
    # ══════════════════════════════════════════════════════════════════════════

    def _find_gaps(self):
        """Find contiguous open gaps in the histogram."""
        n = cfg.VFH_NUM_BINS
        is_open = self._histogram > cfg.VFH_GAP_THRESH_M

        gaps = []
        visited = [False] * n

        for start in range(n):
            if not is_open[start] or visited[start]:
                continue
            length = 0
            depth_sum = 0.0
            depth_min = cfg.VFH_PLAN_RANGE_M
            idx = start
            while is_open[idx % n] and length < n:
                visited[idx % n] = True
                d = self._histogram[idx % n]
                depth_sum += d
                depth_min = min(depth_min, d)
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
                    'depth_min':  depth_min,
                    'angle':      centre_angle,
                })

        return gaps

    def _filter_passable_gaps(self, gaps):
        """Remove gaps the car can't physically fit through."""
        passable = []
        for gap in gaps:
            # At the gap's minimum depth, check if angular width is enough
            # for the car body to pass through
            d = gap['depth_min']
            if d < 0.1:
                continue
            # Width needed: 2 * arctan(car_half_width / depth)
            needed_rad = 2 * math.atan(cfg.CAR_HALF_WIDTH_M / d)
            needed_deg = math.degrees(needed_rad)
            if gap['width_deg'] >= needed_deg:
                passable.append(gap)
        return passable

    def _score_gaps(self, gaps, left_dist, right_dist):
        """Score gaps by width, depth, goal alignment, and turning safety."""
        if not gaps:
            return None

        best = None
        best_score = -1.0

        for gap in gaps:
            # Width score (wider = better, saturate at 90°)
            w_score = min(gap['width_deg'] / 90.0, 1.0)

            # Depth score (deeper = better)
            d_score = min(gap['depth'] / cfg.VFH_PLAN_RANGE_M, 1.0)

            # Goal alignment (prefer gaps toward desired heading)
            angle_diff = abs(gap['angle'] - self._goal_heading)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            g_score = 1.0 - (angle_diff / np.pi)

            # Turn penalty: penalise gaps requiring sharp turns
            # (gap angle far from straight ahead)
            gap_angle_norm = gap['angle']
            if gap_angle_norm > np.pi:
                gap_angle_norm -= 2 * np.pi
            turn_severity = abs(gap_angle_norm) / np.pi  # 0=straight, 1=behind
            t_penalty = 1.0 - cfg.VFH_TURN_PENALTY * turn_severity

            # Side clearance bonus: if turning left, prefer more left clearance
            side_bonus = 0.0
            if gap_angle_norm > 0.1 and left_dist > cfg.SIDE_CLEAR_M:
                side_bonus = 0.1
            elif gap_angle_norm < -0.1 and right_dist > cfg.SIDE_CLEAR_M:
                side_bonus = 0.1

            score = (0.25 * w_score +
                     0.25 * d_score +
                     0.35 * g_score * cfg.VFH_GOAL_BIAS +
                     0.15 * t_penalty +
                     side_bonus)

            if score > best_score:
                best_score = score
                best = gap

        return best

    # ══════════════════════════════════════════════════════════════════════════
    #  STEERING — vehicle geometry aware
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_steering(self, path_angle_rad, front_dist,
                          left_dist, right_dist):
        """Convert gap angle to steering command with vehicle constraints."""
        # Normalize to [-pi, pi]
        a = path_angle_rad
        if a > np.pi:
            a -= 2 * np.pi

        # Decide forward vs reverse
        if abs(a) <= np.radians(110):
            forward = True
            steer_angle = a
        else:
            forward = False
            steer_angle = a - np.pi if a > 0 else a + np.pi

        # Base steering command
        steer = cfg.STEERING_GAIN * steer_angle

        # Scale down steering near obstacles (gentler turns when close)
        proximity = min(front_dist, 2.0) / 2.0  # 0..1 (0=very close)
        proximity_scale = 0.5 + 0.5 * proximity  # 0.5..1.0
        steer *= proximity_scale

        # Check side clearance: don't steer into a wall
        if steer > 0.05 and right_dist < cfg.SIDE_CLEAR_M:
            # Turning right but right side is blocked
            steer = min(steer, 0.05)
        elif steer < -0.05 and left_dist < cfg.SIDE_CLEAR_M:
            # Turning left but left side is blocked
            steer = max(steer, -0.05)

        # Clamp to max steering angle
        steer = max(-cfg.MAX_STEERING_RAD, min(steer, cfg.MAX_STEERING_RAD))

        return float(steer), forward

    # ══════════════════════════════════════════════════════════════════════════
    #  YOLO CLASSIFICATION
    # ══════════════════════════════════════════════════════════════════════════

    def _classify_yolo(self, yolo_dets, distances, angles, valid):
        """Classify the most critical obstacle type from YOLO + LiDAR."""
        has_person = False
        has_moving = False

        for det in yolo_dets:
            det_angle_rad = np.radians(det.angle_centre) % (2 * np.pi)
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
