"""
obstacle_detector.py — Distance-aware VFH + RealSense depth + YOLO fusion.

KEY DESIGN: The car NEVER fully stops. Always finds the best available path,
even if it's tight. Uses distance to decide speed, not whether to stop.

Fixes:
  1. Distance-weighted histogram — far obstacles don't block as much
  2. ALWAYS finds a path — fallback to widest/deepest gap or max-distance bin
  3. RealSense depth catches low obstacles LiDAR misses
  4. YOLO detections injected with LiDAR-confirmed distance
  5. Proper distance-based steering scaling
"""
import math
import numpy as np
import logging
from collections import deque

import config as cfg


class ObstacleDetector:

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

        # 6. ALWAYS find a path — never return has_path=False
        gaps = self._find_gaps()
        gaps = self._filter_passable_gaps(gaps)
        best_gap = self._score_gaps(gaps, left_dist, right_dist)

        # FALLBACK: if no passable gap, find the single best direction
        if best_gap is None:
            best_gap = self._fallback_direction()

        path_angle    = best_gap['angle']
        gap_width_deg = best_gap['width_deg']
        gap_depth     = best_gap['depth']
        path_steer, drive_forward = self._compute_steering(
            path_angle, smoothed, left_dist, right_dist)

        # 7. YOLO classification
        obstacle_type = 'NONE'
        if yolo_dets and zone != 'CLEAR':
            obstacle_type = self._classify_yolo(yolo_dets, distances, angles, valid)

        # 8. Behaviour (simplified — no more EMERGENCY_STOP causing full stop)
        if obstacle_type == 'PERSON' and smoothed < 0.5:
            behaviour = 'SLOW'     # slow down near person, don't stop
        elif zone == 'STOP':
            behaviour = 'AVOID'    # tight space — steer away, keep moving
        elif zone == 'WARN':
            behaviour = 'AVOID'
        else:
            behaviour = 'NAVIGATE'

        # Avoid side
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
            'has_path':        True,  # always True now
            'best_path_angle': path_angle,
            'best_gap_width':  gap_width_deg,
            'best_gap_depth':  gap_depth,
            'path_steer':      round(path_steer, 4),
            'drive_forward':   drive_forward,
            'obstacle_type':   obstacle_type,
            'avoid_side':      avoid_side,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  HISTOGRAM — distance-weighted, inflated
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

            # Inflate by car width at this distance
            if d > 0.05:
                inflate_rad = math.atan(cfg.OBSTACLE_INFLATE_M / d)
                inflate_bins = max(1, int(math.ceil(
                    inflate_rad / cfg.VFH_BIN_WIDTH_RAD)))
                # Cap inflation — don't let a single far point block everything
                inflate_bins = min(inflate_bins, 5)
            else:
                inflate_bins = 3

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
        """Scan RealSense depth for close obstacles in the bottom portion
        of the image (where low/ground-level objects appear)."""
        if depth_m is None or depth_m.size == 0:
            return

        # Squeeze to 2D if needed
        dm = np.squeeze(depth_m)
        if dm.ndim != 2:
            return

        h, w = dm.shape
        roi_top = min(cfg.DEPTH_ROI_TOP, h - 1)
        roi_bot = min(cfg.DEPTH_ROI_BOT, h)
        roi = dm[roi_top:roi_bot, :]

        # Divide ROI into columns, each mapped to a front-sector angle
        n_cols = 12  # divide into 12 angular slices
        col_width = w // n_cols
        half_fov = cfg.DEPTH_SECTOR_DEG / 2.0

        n_bins = cfg.VFH_NUM_BINS
        for c in range(n_cols):
            col_start = c * col_width
            col_end   = min(col_start + col_width, w)
            col_data  = roi[:, col_start:col_end]

            # Get valid (non-zero) depth readings
            valid_mask = col_data > 0.01
            if not valid_mask.any():
                continue

            min_depth = float(col_data[valid_mask].min())

            if min_depth < cfg.DEPTH_WARN_M:
                # Map column to angle: leftmost col = +half_fov, rightmost = -half_fov
                frac = (c + 0.5) / n_cols
                angle_deg = half_fov - frac * cfg.DEPTH_SECTOR_DEG
                angle_rad = np.radians(angle_deg) % (2 * np.pi)
                b = int(angle_rad / (2 * np.pi) * n_bins) % n_bins

                # Inject as obstacle if closer than what LiDAR sees
                if min_depth < self._histogram[b]:
                    self._histogram[b] = min_depth
                    # Inflate by 1 bin on each side
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

            # Find LiDAR-confirmed distance
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

            # Only inject if LiDAR confirms something nearby
            # Don't inject far-away YOLO detections as close obstacles
            if lidar_dist is None:
                continue

            inject_dist = lidar_dist

            # Inject into histogram
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
    #  GAP FINDING — always finds something
    # ══════════════════════════════════════════════════════════════════════════

    def _find_gaps(self):
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
                # KEY: pick the best STEERING ANGLE within the gap.
                # Don't blindly use the gap center — find the point in the
                # gap closest to forward (0 rad) to bias toward driving forward.
                start_angle = start * (2 * np.pi / n)
                end_angle = ((start + length) % n) * (2 * np.pi / n)

                target_angle = self._best_angle_in_gap(
                    start, length, n)

                gaps.append({
                    'start_bin':  start,
                    'length':     length,
                    'width_deg':  length * cfg.VFH_BIN_WIDTH_DEG,
                    'depth':      depth_sum / length,
                    'depth_min':  depth_min,
                    'angle':      target_angle,
                })

        return gaps

    def _best_angle_in_gap(self, start_bin, length, n):
        """Find the angle within the gap closest to forward (0 rad).
        For wide gaps, this prevents always targeting the gap center
        when the center is behind but the edges are reachable forward."""
        # Forward is bin 0 (angle 0)
        forward_bin = 0

        # Check if forward (bin 0) is inside this gap
        end_bin = (start_bin + length) % n
        if length >= n:
            # Gap covers everything — go forward
            return 0.0

        # Check if forward_bin is in the gap
        in_gap = False
        for i in range(length):
            if (start_bin + i) % n == forward_bin:
                in_gap = True
                break

        if in_gap:
            return 0.0  # forward is open!

        # Forward not in gap — find the gap edge closest to forward
        left_edge_bin = start_bin % n
        right_edge_bin = (start_bin + length - 1) % n

        left_edge_angle = left_edge_bin * (2 * np.pi / n)
        right_edge_angle = right_edge_bin * (2 * np.pi / n)

        # Normalize to [-pi, pi]
        def norm(a):
            while a > np.pi: a -= 2 * np.pi
            while a < -np.pi: a += 2 * np.pi
            return a

        left_diff = abs(norm(left_edge_angle))
        right_diff = abs(norm(right_edge_angle))

        if left_diff < right_diff:
            return left_edge_angle
        else:
            return right_edge_angle

    def _filter_passable_gaps(self, gaps):
        passable = []
        for gap in gaps:
            d = gap['depth_min']
            if d < 0.1:
                continue
            needed_rad = 2 * math.atan(cfg.CAR_HALF_WIDTH_M / d)
            needed_deg = math.degrees(needed_rad)
            if gap['width_deg'] >= needed_deg:
                passable.append(gap)
        return passable

    def _fallback_direction(self):
        """When no passable gap exists, pick the direction with most space.
        Strongly prefer forward-facing bins — only pick rear as last resort."""
        n = cfg.VFH_NUM_BINS

        # Score each bin: distance + forward bias
        best_bin = 0
        best_score = -1.0
        for b in range(n):
            d = float(self._histogram[b])
            angle_b = b * (2 * np.pi / n)
            norm_a = angle_b if angle_b <= np.pi else angle_b - 2 * np.pi
            # Forward bias: bins near 0 get a bonus
            fwd_bias = 1.0 - abs(norm_a) / np.pi  # 1.0 at front, 0.0 at rear
            score = d + fwd_bias * 0.5  # 0.5m equivalent bonus for forward
            if score > best_score:
                best_score = score
                best_bin = b

        best_dist = float(self._histogram[best_bin])
        angle = best_bin * (2 * np.pi / n)

        # Try to build a small "virtual gap" around the best bin
        length = 1
        for off in range(1, 5):
            left_b  = (best_bin - off) % n
            right_b = (best_bin + off) % n
            if (self._histogram[left_b] > cfg.VFH_GAP_THRESH_M * 0.5 or
                self._histogram[right_b] > cfg.VFH_GAP_THRESH_M * 0.5):
                length += 1
            else:
                break

        return {
            'start_bin':  best_bin,
            'length':     max(length, 2),
            'width_deg':  max(length, 2) * cfg.VFH_BIN_WIDTH_DEG,
            'depth':      best_dist,
            'depth_min':  best_dist,
            'angle':      angle,
        }

    def _score_gaps(self, gaps, left_dist, right_dist):
        if not gaps:
            return None

        best = None
        best_score = -1.0

        for gap in gaps:
            w_score = min(gap['width_deg'] / 90.0, 1.0)
            d_score = min(gap['depth'] / cfg.VFH_PLAN_RANGE_M, 1.0)

            # Goal alignment
            angle_diff = abs(gap['angle'] - self._goal_heading)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            g_score = 1.0 - (angle_diff / np.pi)

            # Forward preference: heavily reward gaps reachable without reversing
            gap_norm = gap['angle']
            if gap_norm > np.pi:
                gap_norm -= 2 * np.pi
            abs_angle = abs(gap_norm)

            # Tiered forward bonus: strongly prefer straight, then side, penalize rear
            if abs_angle <= np.radians(30):
                fwd_bonus = 1.0    # straight ahead — best
            elif abs_angle <= np.radians(90):
                fwd_bonus = 0.7    # side gap — good
            elif abs_angle <= np.radians(120):
                fwd_bonus = 0.4    # wide side — acceptable
            else:
                fwd_bonus = 0.0    # behind — last resort

            # Turn severity: prefer gaps closer to straight ahead
            turn_severity = abs(gap_norm) / np.pi
            t_penalty = 1.0 - cfg.VFH_TURN_PENALTY * turn_severity

            # Side clearance bonus
            side_bonus = 0.0
            if gap_norm > 0.1 and left_dist > cfg.SIDE_CLEAR_M:
                side_bonus = 0.1
            elif gap_norm < -0.1 and right_dist > cfg.SIDE_CLEAR_M:
                side_bonus = 0.1

            score = (0.20 * w_score +
                     0.20 * d_score +
                     0.25 * g_score * cfg.VFH_GOAL_BIAS +
                     0.15 * t_penalty +
                     fwd_bonus +
                     side_bonus)

            if score > best_score:
                best_score = score
                best = gap

        return best

    # ══════════════════════════════════════════════════════════════════════════
    #  STEERING — distance-aware, uses surrounding space
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_steering(self, path_angle_rad, front_dist,
                          left_dist, right_dist):
        """Compute forward steering toward the best gap.
        Returns (steer, drive_forward).
        drive_forward=False ONLY when ALL forward paths are truly blocked."""
        a = path_angle_rad
        if a > np.pi:
            a -= 2 * np.pi

        # Is the best gap reachable going forward? (within ±120° of front)
        forward = abs(a) <= np.radians(120)

        if forward:
            steer = cfg.STEERING_GAIN * a

            # URGENCY scaling: steer HARDER when close to obstacles, not softer.
            # When front_dist is small, we MUST turn aggressively.
            if front_dist < cfg.ZONE_WARN_M:
                # Very close — full urgency, steer as hard as needed
                urgency = 1.3
            elif front_dist < cfg.ZONE_CLEAR_M:
                # Moderate distance — proportional urgency
                urgency = 0.8 + 0.5 * (1.0 - front_dist / cfg.ZONE_CLEAR_M)
            else:
                # Clear path — gentle steering
                urgency = 0.8
            steer *= urgency

            # Side clearance: only limit steering if about to scrape a wall
            if steer > 0.05 and right_dist < cfg.SIDE_CLEAR_M:
                steer = min(steer, 0.05)
            elif steer < -0.05 and left_dist < cfg.SIDE_CLEAR_M:
                steer = max(steer, -0.05)
        else:
            # Best gap is behind — report this to navigator.
            # Provide a forward steering suggestion toward the SIDE with more space
            # (navigator will decide whether to actually reverse)
            if left_dist > right_dist:
                steer = cfg.MAX_STEERING_RAD * 0.8
            else:
                steer = -cfg.MAX_STEERING_RAD * 0.8

        steer = max(-cfg.MAX_STEERING_RAD, min(steer, cfg.MAX_STEERING_RAD))
        return float(steer), forward

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
