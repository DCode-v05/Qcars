"""
navigator.py — DWA-driven navigation with anti-oscillation & crash detection.

KEY FIX: Direction commitment — once FWD or REV is chosen, the car HOLDS that
direction for at least DIR_COMMIT_TICKS before allowing a change. This prevents
the rapid FWD↔REV oscillation that was the main problem.

If front is blocked: reverse for a full RECOVERY_DURATION_S (2.5s) to create
enough space, THEN try turning. No half-measures.

States:
  DRIVING     — forward, steering from DWA
  REVERSING   — DWA chose reverse, committed for minimum ticks
  RECOVERING  — oscillation/crash/stuck, forced 2.5s reverse+turn escape
  MANUAL_FWD  — user pressed 'd'
  MANUAL_REV  — user pressed 'r'
"""
import time
import numpy as np
from collections import deque

import config as cfg


# IMU thresholds
CRASH_ACCEL_G      = 1.2       # acceleration spike > 1.2g excess = impact
CRASH_COOLDOWN_S   = 2.5       # min time between crash detections
GRAVITY_MPS2       = 9.81

# Oscillation: if LiDAR front distance barely changes while direction flips
OSCILLATION_WINDOW = 20        # track last 20 ticks
OSCILLATION_FLIPS  = 4         # 4+ direction changes in window = oscillating


class Navigator:

    def __init__(self):
        self.state            = 'DRIVING'
        self._last_steer      = 0.0
        self._manual_mode     = None
        self._crash_time      = 0.0
        self._crash_reversing = False
        # Direction commitment: hold chosen direction for minimum ticks
        self._committed_fwd   = True
        self._commit_counter  = 0
        # Stuck detection (max-steer lock)
        self._stuck_counter   = 0
        self._recovery_start  = 0.0
        self._recovery_steer  = 0.0
        # Heading tracking
        self._heading_rad     = 0.0
        # Oscillation detection
        self._dir_history     = deque(maxlen=OSCILLATION_WINDOW)
        self._last_dir_fwd    = True
        self._oscillation_detected = False
        # IMU for crash
        self._accel_history   = deque(maxlen=5)
        # LiDAR history for oscillation (front dist not changing = stuck)
        self._front_history   = deque(maxlen=OSCILLATION_WINDOW)

    def reset(self):
        self.state = 'DRIVING'
        self._last_steer = 0.0
        self._manual_mode = None
        self._crash_reversing = False
        self._committed_fwd = True
        self._commit_counter = cfg.DIR_COMMIT_TICKS  # commit initial direction
        self._stuck_counter = 0
        self._heading_rad = 0.0
        self._dir_history.clear()
        self._last_dir_fwd = True
        self._oscillation_detected = False
        self._accel_history.clear()
        self._front_history.clear()

    def set_manual(self, mode):
        self._manual_mode = mode

    def update(self, detection, dt, imu_accel=None, imu_gyro=None):
        path_steer = detection['path_steer']
        drive_fwd  = detection['drive_forward']
        front_dist = detection['distance_m']
        rear_dist  = detection['rear_min_m']
        left_dist  = detection.get('left_min_m', 9.0)
        right_dist = detection.get('right_min_m', 9.0)
        now        = time.perf_counter()

        # ── Heading integration ───────────────────────────────────────────
        if imu_gyro is not None and dt > 0:
            yaw_rate = float(imu_gyro[2]) - cfg.GYRO_BIAS_Z
            self._heading_rad += yaw_rate * dt

        # ── IMU crash detection (peak-based) ──────────────────────────────
        crash_detected = False
        if imu_accel is not None:
            accel_mag = float(np.linalg.norm(imu_accel))
            self._accel_history.append(accel_mag)
            if len(self._accel_history) >= 2:
                peak = max(self._accel_history)
                excess_g = abs(peak - GRAVITY_MPS2) / GRAVITY_MPS2
                if excess_g > CRASH_ACCEL_G:
                    if now - self._crash_time > CRASH_COOLDOWN_S:
                        crash_detected = True
                        self._crash_time = now
                        self._crash_reversing = True
                        self._accel_history.clear()

        if self._crash_reversing:
            if now - self._crash_time > 1.5:
                self._crash_reversing = False

        # ── Oscillation detection ─────────────────────────────────────────
        # Track direction flips AND check if front distance is stuck
        self._oscillation_detected = False
        dir_changed = (drive_fwd != self._last_dir_fwd)
        self._dir_history.append(1 if dir_changed else 0)
        self._last_dir_fwd = drive_fwd
        self._front_history.append(front_dist)

        flip_count = sum(self._dir_history)
        # Also check if front distance is barely changing (car not making progress)
        front_stuck = False
        if len(self._front_history) >= 10:
            front_range = max(self._front_history) - min(self._front_history)
            front_stuck = front_range < 0.08  # less than 8cm change = not moving

        is_oscillating = (flip_count >= OSCILLATION_FLIPS) or \
                         (flip_count >= 3 and front_stuck)

        # ── Direction commitment ──────────────────────────────────────────
        # Once a direction is chosen, hold it for DIR_COMMIT_TICKS before
        # allowing a switch. This is the KEY anti-oscillation mechanism.
        self._commit_counter = max(0, self._commit_counter - 1)

        if self._commit_counter == 0:
            # Allowed to change direction
            if drive_fwd != self._committed_fwd:
                self._committed_fwd = drive_fwd
                self._commit_counter = cfg.DIR_COMMIT_TICKS
        # Use committed direction, not raw DWA output
        effective_fwd = self._committed_fwd

        # ── State machine ─────────────────────────────────────────────────
        if self._manual_mode == 'd':
            self.state = 'MANUAL_FWD'
            self._stuck_counter = 0
        elif self._manual_mode == 'r':
            self.state = 'MANUAL_REV'
            self._stuck_counter = 0
        elif self._crash_reversing:
            # Crash → immediate recovery
            self.state = 'RECOVERING'
            self._recovery_start = self._crash_time
            if left_dist > right_dist:
                self._recovery_steer = cfg.MAX_STEERING_RAD * 0.9
            else:
                self._recovery_steer = -cfg.MAX_STEERING_RAD * 0.9
            self._stuck_counter = 0
            self._commit_counter = 0
        elif self.state == 'RECOVERING':
            # Continue recovery until duration expires
            if now - self._recovery_start > cfg.RECOVERY_DURATION_S:
                self.state = 'DRIVING'
                self._stuck_counter = 0
                self._committed_fwd = True
                self._commit_counter = cfg.DIR_COMMIT_TICKS
                self._dir_history.clear()
                self._front_history.clear()
        elif is_oscillating and self.state not in ('MANUAL_FWD', 'MANUAL_REV'):
            # Oscillation detected → force a committed escape
            self._oscillation_detected = True
            self.state = 'RECOVERING'
            self._recovery_start = now
            if left_dist > right_dist:
                self._recovery_steer = cfg.MAX_STEERING_RAD * 0.9
            else:
                self._recovery_steer = -cfg.MAX_STEERING_RAD * 0.9
            self._dir_history.clear()
            self._front_history.clear()
            self._stuck_counter = 0
            self._commit_counter = 0
        else:
            # ── Normal autonomous: use committed direction ────────────────
            if effective_fwd:
                self.state = 'DRIVING'
            else:
                self.state = 'REVERSING'

            # ── Stuck detection (max-steer for too long) ──────────────────
            if self.state == 'DRIVING':
                if abs(path_steer) > cfg.MAX_STEERING_RAD * cfg.STUCK_STEER_FRAC:
                    self._stuck_counter += 1
                else:
                    self._stuck_counter = max(0, self._stuck_counter - 2)

                if self._stuck_counter >= cfg.STUCK_THRESHOLD:
                    self.state = 'RECOVERING'
                    self._recovery_start = now
                    self._recovery_steer = -np.sign(path_steer) * cfg.MAX_STEERING_RAD * 0.9
                    self._stuck_counter = 0
                    self._commit_counter = 0
            else:
                self._stuck_counter = max(0, self._stuck_counter - 1)

        # ── Compute throttle + steering ───────────────────────────────────

        if self.state == 'DRIVING':
            throttle = cfg.THROTTLE
            if front_dist > cfg.ZONE_CLEAR_M:
                steering = self._smooth_steer(path_steer * 0.7, dt)
            else:
                steering = self._smooth_steer(path_steer, dt)

        elif self.state == 'REVERSING':
            throttle = -cfg.THROTTLE
            steering = self._smooth_steer(path_steer, dt)

        elif self.state == 'RECOVERING':
            throttle = -cfg.THROTTLE
            steering = self._smooth_steer(self._recovery_steer, dt)

        elif self.state == 'MANUAL_FWD':
            throttle = cfg.THROTTLE
            steering = self._smooth_steer(path_steer, dt)

        elif self.state == 'MANUAL_REV':
            throttle = -cfg.THROTTLE
            steering = self._smooth_steer(path_steer, dt)

        else:
            throttle = cfg.THROTTLE
            steering = self._smooth_steer(path_steer, dt)

        leds = self._get_leds(detection.get('avoid_side', 'NONE'))

        return {
            'throttle':              throttle,
            'steering':              steering,
            'state':                 self.state,
            'leds':                  leds,
            'crash_detected':        crash_detected,
            'oscillation_detected':  self._oscillation_detected,
            'heading_rad':           self._heading_rad,
        }

    def _smooth_steer(self, target, dt):
        max_delta = cfg.MAX_STEER_RATE * dt
        delta = target - self._last_steer
        delta = max(-max_delta, min(delta, max_delta))
        self._last_steer += delta
        self._last_steer = max(-cfg.MAX_STEERING_RAD,
                               min(self._last_steer, cfg.MAX_STEERING_RAD))
        return self._last_steer

    def _get_leds(self, avoid_side):
        leds = np.zeros(8, dtype=np.float64)
        leds[6] = leds[7] = 1.0  # headlights

        if self.state in ('REVERSING', 'RECOVERING'):
            leds[4] = 1.0
        if self.state == 'RECOVERING':
            leds[0] = leds[1] = 1.0  # hazard
            leds[2] = leds[3] = 1.0
        elif self.state in ('MANUAL_FWD', 'MANUAL_REV'):
            leds[0] = leds[1] = 1.0
            leds[2] = leds[3] = 1.0
            if self.state == 'MANUAL_REV':
                leds[4] = 1.0
        elif avoid_side == 'LEFT':
            leds[0] = leds[1] = 1.0
        elif avoid_side == 'RIGHT':
            leds[2] = leds[3] = 1.0

        return leds
