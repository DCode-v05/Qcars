"""
navigator.py — DWA-driven navigation with stuck detection & recovery.

The DWA planner (in obstacle_detector) selects the best trajectory including
forward/reverse direction. This navigator:
  1. Trusts DWA's forward/reverse decision
  2. Detects when the car is STUCK (max steering lock for too long)
  3. Triggers RECOVERY (reverse + opposite turn) to escape
  4. Handles manual overrides and IMU crash detection

States:
  DRIVING     — forward, steering from DWA
  REVERSING   — DWA chose reverse trajectory
  RECOVERING  — stuck detected, reversing with opposite turn to escape
  MANUAL_FWD  — user pressed 'd'
  MANUAL_REV  — user pressed 'r'
"""
import time
import numpy as np

import config as cfg


# IMU crash detection thresholds
CRASH_ACCEL_G     = 2.5
CRASH_COOLDOWN_S  = 2.0
GRAVITY_MPS2      = 9.81


class Navigator:

    def __init__(self):
        self.state           = 'DRIVING'
        self._last_steer     = 0.0
        self._manual_mode    = None
        self._crash_time     = 0.0
        self._crash_reversing = False
        # Stuck detection
        self._stuck_counter  = 0
        self._recovery_start = 0.0
        self._recovery_steer = 0.0
        # Heading tracking (integrated from gyroscope yaw)
        self._heading_rad    = 0.0   # accumulated yaw, 0 = initial heading

    def reset(self):
        self.state = 'DRIVING'
        self._last_steer = 0.0
        self._manual_mode = None
        self._crash_reversing = False
        self._stuck_counter = 0
        self._heading_rad = 0.0

    def set_manual(self, mode):
        """Called from key listener: 'd'=forward, 'r'=reverse, None=auto."""
        self._manual_mode = mode

    def update(self, detection, dt, imu_accel=None, imu_gyro=None):
        """
        Returns dict with throttle, steering, state, leds, heading_rad.
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
            yaw_rate = float(imu_gyro[2]) - cfg.GYRO_BIAS_Z  # z-axis = yaw
            self._heading_rad += yaw_rate * dt

        # ── IMU crash detection ───────────────────────────────────────────
        crash_detected = False
        if imu_accel is not None:
            accel_mag = np.linalg.norm(imu_accel)
            excess = abs(accel_mag - GRAVITY_MPS2)
            if excess > CRASH_ACCEL_G * GRAVITY_MPS2:
                if now - self._crash_time > CRASH_COOLDOWN_S:
                    crash_detected = True
                    self._crash_time = now
                    self._crash_reversing = True

        if self._crash_reversing:
            if now - self._crash_time > 1.0:
                self._crash_reversing = False

        # ── Manual override ───────────────────────────────────────────────
        if self._manual_mode == 'd':
            self.state = 'MANUAL_FWD'
            self._stuck_counter = 0
        elif self._manual_mode == 'r':
            self.state = 'MANUAL_REV'
            self._stuck_counter = 0
        elif self._crash_reversing:
            self.state = 'REVERSING'
            self._stuck_counter = 0
        elif self.state == 'RECOVERING':
            # ── Continue recovery until duration expires ───────────────────
            elapsed = now - self._recovery_start
            if elapsed > cfg.RECOVERY_DURATION_S:
                self.state = 'DRIVING'
                self._stuck_counter = 0
            # else stay in RECOVERING
        else:
            # ── Autonomous decision: trust DWA ────────────────────────────
            if drive_fwd:
                self.state = 'DRIVING'
            else:
                self.state = 'REVERSING'

            # ── Stuck detection ───────────────────────────────────────────
            # If driving forward with near-max steering for too long,
            # the car is likely stuck against an obstacle it can't turn around.
            if self.state == 'DRIVING':
                if abs(path_steer) > cfg.MAX_STEERING_RAD * cfg.STUCK_STEER_FRAC:
                    self._stuck_counter += 1
                else:
                    # Decay counter when steering is moderate
                    self._stuck_counter = max(0, self._stuck_counter - 2)

                if self._stuck_counter >= cfg.STUCK_THRESHOLD:
                    # Trigger recovery: reverse + opposite turn direction
                    self.state = 'RECOVERING'
                    self._recovery_start = now
                    self._recovery_steer = -np.sign(path_steer) * cfg.MAX_STEERING_RAD * 0.8
                    self._stuck_counter = 0
            else:
                # Not driving forward — reset stuck counter
                self._stuck_counter = max(0, self._stuck_counter - 1)

        # ── Compute throttle + steering ───────────────────────────────────

        if self.state == 'DRIVING':
            throttle = cfg.THROTTLE
            if front_dist > cfg.ZONE_CLEAR_M:
                # Clear ahead — blend toward straight
                blended = path_steer * 0.7
                steering = self._smooth_steer(blended, dt)
            else:
                steering = self._smooth_steer(path_steer, dt)

        elif self.state == 'REVERSING':
            throttle = -cfg.THROTTLE
            # DWA already picked the best reverse steering angle
            steering = self._smooth_steer(path_steer, dt)

        elif self.state == 'RECOVERING':
            throttle = -cfg.THROTTLE
            # Steer in the OPPOSITE direction to escape the stuck spot
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
            'throttle':       throttle,
            'steering':       steering,
            'state':          self.state,
            'leds':           leds,
            'crash_detected': crash_detected,
            'heading_rad':    self._heading_rad,
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
            leds[0] = leds[1] = 1.0  # hazard = recovery indicator
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
