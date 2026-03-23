"""
obstacle_detector.py — VFH histogram + DWA trajectory planner.

Builds a polar histogram from LiDAR + RealSense depth + YOLO detections,
then uses DWA (Dynamic Window Approach) to sample Ackermann trajectories
and pick the best collision-free path.

The car NEVER fully stops. DWA always finds the best available trajectory.
"""
import math
import numpy as np
import logging
from collections import deque

import config as cfg
from dwa_planner import DWAPlanner


class ObstacleDetector:

    def __init__(self):
        self._histogram = np.full(cfg.VFH_NUM_BINS, cfg.VFH_PLAN_RANGE_M)
        self._front_history = deque(maxlen=cfg.SMOOTH_WINDOW)
        self._goal_heading = 0.0
        self._dwa = DWAPlanner()
        self._last_steering = 0.0

    def set_goal_heading(self, heading_rad):
        self._goal_heading = heading_rad

    def detect(self, sensor_data, yolo_dets=None):
        distances = sensor_data['lidar_distances']
        angles    = sensor_data['lidar_angles']
        valid     = sensor_data['lidar_valid']
        depth_m   = sensor_data.get('rs_depth_m')

        # 1. Build VFH histogram from LiDAR
        self._build_histogram(distances, angles, valid)

        # 2. Inject RealSense depth obstacles (catches low objects)
        if depth_m is not None:
            self._inject_depth(depth_m)

        # 3. Inject YOLO detections
        if yolo_dets:
            self._inject_yolo(yolo_dets, distances, angles, valid)

        # 4. Sector analysis
        front_dist = self._sector_min_hist(0.0, cfg.FRONT_SECTOR_DEG)
        rear_dist  = self._sector_min_hist(np.pi, cfg.REAR_SECTOR_DEG)
        left_dist  = self._sector_min_hist(np.pi * 1.5, cfg.SIDE_SECTOR_DEG)
        right_dist = self._sector_min_hist(np.pi * 0.5, cfg.SIDE_SECTOR_DEG)

        # Smooth forward distance
        if front_dist < cfg.LIDAR_MAX_M:
            self._front_history.append(front_dist)
        smoothed = float(np.mean(self._front_history)) if self._front_history else front_dist

        # 5. Zone (informational only — does NOT cause stopping)
        if smoothed > cfg.ZONE_CLEAR_M:
            zone = 'CLEAR'
        elif smoothed > cfg.ZONE_WARN_M:
            zone = 'WARN'
        else:
            zone = 'STOP'

        # 6. DWA trajectory planning — replaces VFH gap-finding
        dwa_result = self._dwa.plan(
            histogram=self._histogram,
            current_steering=self._last_steering,
            goal_heading=self._goal_heading,
        )
        path_steer    = dwa_result['steering']
        drive_forward = dwa_result['drive_forward']
        path_angle    = dwa_result['best_angle']
        gap_width_deg = dwa_result['gap_width_deg']
        gap_depth     = dwa_result['gap_depth']
        self._last_steering = path_steer

        # 7. YOLO classification
        obstacle_type = 'NONE'
        if yolo_dets and zone != 'CLEAR':
            obstacle_type = self._classify_yolo(yolo_dets, distances, angles, valid)

        # 8. Behaviour
        if obstacle_type == 'PERSON' and smoothed < 0.5:
            behaviour = 'SLOW'
        elif zone == 'STOP':
            behaviour = 'AVOID'
        elif zone == 'WARN':
            behaviour = 'AVOID'
        else:
            behaviour = 'NAVIGATE'

        # Avoid side (for LED indicators)
        norm_angle = path_angle
        if norm_angle > np.pi:
            norm_angle -= 2 * np.pi
        avoid_side = 'LEFT' if norm_angle > 0 else 'RIGHT'

        return {
            'zone':            zone,
            'behaviour':       behaviour,
            'distance_m':      round(smoothed, 3),
            'rear_min_m':      round(rear_dist, 3),
            'left_min_m':      round(left_dist, 3),
            'right_min_m':     round(right_dist, 3),
            'has_path':        True,
            'best_path_angle': path_angle,
            'best_gap_width':  gap_width_deg,
            'best_gap_depth':  gap_depth,
            'path_steer':      round(path_steer, 4),
            'drive_forward':   drive_forward,
            'obstacle_type':   obstacle_type,
            'avoid_side':      avoid_side,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  HISTOGRAM — distance-weighted, inflated by car width
    # ══════════════════════════════════════════════════════════════════════════

    def _build_histogram(self, distances, angles, valid):
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

            # Inflate minimally — DWA trajectory sim handles car width
            if d > 0.05:
                inflate_rad = math.atan(cfg.OBSTACLE_INFLATE_M / d)
                inflate_bins = max(1, int(math.ceil(
                    inflate_rad / cfg.VFH_BIN_WIDTH_RAD)))
                inflate_bins = min(inflate_bins, 2)
            else:
                inflate_bins = 1

            for offset in range(-inflate_bins, inflate_bins + 1):
                b = (centre_bin + offset) % n_bins
                if d < self._histogram[b]:
                    self._histogram[b] = d

    def _sector_min_hist(self, centre_rad, half_width_deg):
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
    #  REALSENSE DEPTH — catches low obstacles LiDAR misses
    # ══════════════════════════════════════════════════════════════════════════

    def _inject_depth(self, depth_m):
        if depth_m is None or depth_m.size == 0:
            return

        dm = np.squeeze(depth_m)
        if dm.ndim != 2:
            return

        h, w = dm.shape
        roi_top = min(cfg.DEPTH_ROI_TOP, h - 1)
        roi_bot = min(cfg.DEPTH_ROI_BOT, h)
        roi = dm[roi_top:roi_bot, :]

        n_cols = 12
        col_width = w // n_cols
        half_fov = cfg.DEPTH_SECTOR_DEG / 2.0
        n_bins = cfg.VFH_NUM_BINS

        for c in range(n_cols):
            col_start = c * col_width
            col_end   = min(col_start + col_width, w)
            col_data  = roi[:, col_start:col_end]

            valid_mask = col_data > 0.01
            if not valid_mask.any():
                continue

            min_depth = float(col_data[valid_mask].min())

            if min_depth < cfg.DEPTH_WARN_M:
                frac = (c + 0.5) / n_cols
                angle_deg = half_fov - frac * cfg.DEPTH_SECTOR_DEG
                angle_rad = np.radians(angle_deg) % (2 * np.pi)
                b = int(angle_rad / (2 * np.pi) * n_bins) % n_bins

                if min_depth < self._histogram[b]:
                    self._histogram[b] = min_depth
                    for off in [-1, 1]:
                        bb = (b + off) % n_bins
                        if min_depth < self._histogram[bb]:
                            self._histogram[bb] = min_depth

    # ══════════════════════════════════════════════════════════════════════════
    #  YOLO INJECTION
    # ══════════════════════════════════════════════════════════════════════════

    def _inject_yolo(self, yolo_dets, distances, angles, valid):
        n_bins = cfg.VFH_NUM_BINS

        for det in yolo_dets:
            det_angle_rad = np.radians(det.angle_centre) % (2 * np.pi)

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

            if lidar_dist is None:
                continue

            inject_dist = lidar_dist

            det_left  = np.radians(det.angle_left) % (2 * np.pi)
            det_right = np.radians(det.angle_right) % (2 * np.pi)
            b_left  = int((det_left / (2 * np.pi)) * n_bins) % n_bins
            b_right = int((det_right / (2 * np.pi)) * n_bins) % n_bins

            b = b_left
            for _ in range(n_bins):
                if inject_dist < self._histogram[b]:
                    self._histogram[b] = inject_dist
                if b == b_right:
                    break
                b = (b + 1) % n_bins

    # ══════════════════════════════════════════════════════════════════════════
    #  YOLO CLASSIFICATION
    # ══════════════════════════════════════════════════════════════════════════

    def _classify_yolo(self, yolo_dets, distances, angles, valid):
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
        return 'STATIC'
