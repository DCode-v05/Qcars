"""
navigator.py — DWA-driven navigation with oscillation + crash detection.

States:
  DRIVING     — forward, steering from DWA
  REVERSING   — DWA chose reverse trajectory
  RECOVERING  — oscillation/stuck detected, forced escape maneuver
  MANUAL_FWD  — user pressed 'd'
  MANUAL_REV  — user pressed 'r'
"""
import time
import numpy as np
from collections import deque

import config as cfg


# IMU thresholds
CRASH_ACCEL_G      = 1.2       # acceleration spike > 1.2g excess = impact
CRASH_COOLDOWN_S   = 2.0       # min time between crash detections
GRAVITY_MPS2       = 9.81

# Oscillation detection: if direction flips > N times in M seconds → stuck
OSCILLATION_WINDOW = 30        # track last 30 ticks (~1s at 30Hz)
OSCILLATION_FLIPS  = 6         # 6+ direction changes in window = oscillating


class Navigator:

    def __init__(self):
        self.state           = 'DRIVING'
        self._last_steer     = 0.0
        self._manual_mode    = None
        self._crash_time     = 0.0
        self._crash_reversing = False
        # Stuck detection (max-steer lock)
        self._stuck_counter  = 0
        self._recovery_start = 0.0
        self._recovery_steer = 0.0
        # Heading tracking (gyroscope yaw integration)
        self._heading_rad    = 0.0
        # Oscillation detection (forward/reverse flipping)
        self._dir_history    = deque(maxlen=OSCILLATION_WINDOW)
        self._last_dir_fwd   = True
        self._oscillation_detected = False
        # IMU smoothing for crash detection
        self._accel_history  = deque(maxlen=5)

    def reset(self):
        self.state = 'DRIVING'
        self._last_steer = 0.0
        self._manual_mode = None
        self._crash_reversing = False
        self._stuck_counter = 0
        self._heading_rad = 0.0
        self._dir_history.clear()
        self._last_dir_fwd = True
        self._oscillation_detected = False
        self._accel_history.clear()

    def set_manual(self, mode):
        """Called from key listener: 'd'=forward, 'r'=reverse, None=auto."""
        self._manual_mode = mode

    def update(self, detection, dt, imu_accel=None, imu_gyro=None):
        """
        Returns dict with throttle, steering, state, leds, heading_rad,
        crash_detected, oscillation_detected.
        """
        path_steer = detection['path_steer']
        drive_fwd  = detection['drive_forward']
        front_dist = detection['distance_m']
        rear_dist  = detection['rear_min_m']
        left_dist  = detection.get('left_min_m', 9.0)
        right_dist = detection.get('right_min_m', 9.0)
        now        = time.perf_counter()

        # ── Heading integration from gyroscope ────────────────────────────
        if imu_gyro is not None and dt > 0:
            yaw_rate = float(imu_gyro[2]) - cfg.GYRO_BIAS_Z
            self._heading_rad += yaw_rate * dt

        # ── IMU crash / impact detection ──────────────────────────────────
        crash_detected = False
        if imu_accel is not None:
            accel_mag = float(np.linalg.norm(imu_accel))
            self._accel_history.append(accel_mag)

            # Check peak of recent readings (max, not average — catches spikes)
            if len(self._accel_history) >= 2:
                peak_accel = max(self._accel_history)
                excess_g = abs(peak_accel - GRAVITY_MPS2) / GRAVITY_MPS2
                if excess_g > CRASH_ACCEL_G:
                    if now - self._crash_time > CRASH_COOLDOWN_S:
                        crash_detected = True
                        self._crash_time = now
                        self._crash_reversing = True
                        self._accel_history.clear()

        if self._crash_reversing:
            if now - self._crash_time > 1.5:
                self._crash_reversing = False

        # ── Oscillation detection (forward/reverse flipping) ──────────────
        # Track direction changes: if the car flips FWD↔REV too rapidly,
        # it's oscillating and needs a committed escape maneuver.
        self._oscillation_detected = False
        dir_changed = (drive_fwd != self._last_dir_fwd)
        self._dir_history.append(1 if dir_changed else 0)
        self._last_dir_fwd = drive_fwd

        flip_count = sum(self._dir_history)
        if flip_count >= OSCILLATION_FLIPS and self.state not in ('RECOVERING', 'MANUAL_FWD', 'MANUAL_REV'):
            self._oscillation_detected = True
            # Force a committed escape: pick the side with more space
            self.state = 'RECOVERING'
            self._recovery_start = now
            if left_dist > right_dist:
                self._recovery_steer = cfg.MAX_STEERING_RAD * 0.8
            else:
                self._recovery_steer = -cfg.MAX_STEERING_RAD * 0.8
            self._dir_history.clear()
            self._stuck_counter = 0

        # ── Manual override ───────────────────────────────────────────────
        if self._manual_mode == 'd':
            self.state = 'MANUAL_FWD'
            self._stuck_counter = 0
        elif self._manual_mode == 'r':
            self.state = 'MANUAL_REV'
            self._stuck_counter = 0
        elif self._crash_reversing:
            self.state = 'RECOVERING'
            self._recovery_start = self._crash_time
            # Reverse away from whatever was hit — steer toward most space
            if left_dist > right_dist:
                self._recovery_steer = cfg.MAX_STEERING_RAD * 0.8
            else:
                self._recovery_steer = -cfg.MAX_STEERING_RAD * 0.8
            self._stuck_counter = 0
        elif self.state == 'RECOVERING':
            # ── Continue recovery until duration expires ───────────────────
            elapsed_rec = now - self._recovery_start
            if elapsed_rec > cfg.RECOVERY_DURATION_S:
                self.state = 'DRIVING'
                self._stuck_counter = 0
        elif not self._oscillation_detected:
            # ── Autonomous decision: trust DWA ────────────────────────────
            if drive_fwd:
                self.state = 'DRIVING'
            else:
                self.state = 'REVERSING'

            # ── Stuck detection (max-steer lock for too long) ─────────────
            if self.state == 'DRIVING':
                if abs(path_steer) > cfg.MAX_STEERING_RAD * cfg.STUCK_STEER_FRAC:
                    self._stuck_counter += 1
                else:
                    self._stuck_counter = max(0, self._stuck_counter - 2)

                if self._stuck_counter >= cfg.STUCK_THRESHOLD:
                    self.state = 'RECOVERING'
                    self._recovery_start = now
                    self._recovery_steer = -np.sign(path_steer) * cfg.MAX_STEERING_RAD * 0.8
                    self._stuck_counter = 0
            else:
                self._stuck_counter = max(0, self._stuck_counter - 1)

        # ── Compute throttle + steering ───────────────────────────────────

        if self.state == 'DRIVING':
            throttle = cfg.THROTTLE
            if front_dist > cfg.ZONE_CLEAR_M:
                blended = path_steer * 0.7
                steering = self._smooth_steer(blended, dt)
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
            leds[4] = 1.0  # brake/reverse lights
        if self.state == 'RECOVERING':
            # Hazard flash = recovery indicator
            leds[0] = leds[1] = 1.0
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
