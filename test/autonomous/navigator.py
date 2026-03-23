"""
navigator.py — Forward-biased navigation with manual override & IMU crash detection.

RULES:
  1. ALWAYS prefer forward. Only reverse when ALL forward paths are blocked.
  2. No timers — car never stops on its own (only 'q' or Ctrl+C).
  3. 'd' = force DRIVE (forward), 'r' = force REVERSE (manual override).
  4. IMU spike = crash detected → immediately reverse away.

States:
  DRIVING     — forward, steering toward best path
  REVERSING   — all forward paths blocked OR crash detected, backing up with turn
  MANUAL_FWD  — user pressed 'd', forced forward
  MANUAL_REV  — user pressed 'r', forced reverse
"""
import time
import numpy as np

import config as cfg


# IMU crash detection thresholds
CRASH_ACCEL_G     = 2.5     # acceleration spike > 2.5g = crash
CRASH_COOLDOWN_S  = 2.0     # after crash reverse, wait before allowing another
GRAVITY_MPS2      = 9.81


class Navigator:

    def __init__(self):
        self.state           = 'DRIVING'
        self._last_steer     = 0.0
        self._manual_mode    = None    # 'd' or 'r' or None
        self._crash_time     = 0.0
        self._crash_reversing = False

    def reset(self):
        self.state = 'DRIVING'
        self._last_steer = 0.0
        self._manual_mode = None
        self._crash_reversing = False

    def set_manual(self, mode):
        """Called from key listener: 'd'=forward, 'r'=reverse, None=auto."""
        self._manual_mode = mode

    def update(self, detection, dt, imu_accel=None):
        """
        Returns dict with throttle, steering, state, leds.

        Args:
            detection: dict from ObstacleDetector
            dt: time step
            imu_accel: numpy array [ax, ay, az] in m/s^2 (optional)
        """
        path_steer = detection['path_steer']
        drive_fwd  = detection['drive_forward']
        front_dist = detection['distance_m']
        rear_dist  = detection['rear_min_m']
        left_dist  = detection.get('left_min_m', 9.0)
        right_dist = detection.get('right_min_m', 9.0)
        now        = time.perf_counter()

        # ── IMU crash detection ───────────────────────────────────────────
        crash_detected = False
        if imu_accel is not None:
            accel_mag = np.linalg.norm(imu_accel)
            # Remove gravity, check for spike
            excess = abs(accel_mag - GRAVITY_MPS2)
            if excess > CRASH_ACCEL_G * GRAVITY_MPS2:
                if now - self._crash_time > CRASH_COOLDOWN_S:
                    crash_detected = True
                    self._crash_time = now
                    self._crash_reversing = True

        # Crash reverse: back away for a bit
        if self._crash_reversing:
            if now - self._crash_time > 1.0:
                self._crash_reversing = False

        # ── Manual override ───────────────────────────────────────────────
        if self._manual_mode == 'd':
            self.state = 'MANUAL_FWD'
        elif self._manual_mode == 'r':
            self.state = 'MANUAL_REV'
        elif self._crash_reversing:
            self.state = 'REVERSING'
        else:
            # ── Autonomous decision ───────────────────────────────────────
            # PRIORITY: Always prefer FORWARD + STEERING over reversing.
            # The car should steer left/right around obstacles first.
            # Only reverse when ALL forward paths are truly blocked.
            #
            # Priority:
            #   1. VFH found forward gap → DRIVING (steer around)
            #   2. Front not critically close → DRIVING
            #   3. All forward blocked, rear clear → REVERSING
            #   4. Boxed in → DRIVING with hard steer

            if drive_fwd:
                # VFH found a forward-reachable gap — always drive forward.
                # Even if something is close on one side, VFH found a way around.
                self.state = 'DRIVING'
            elif front_dist > cfg.ZONE_WARN_M:
                # Front isn't critical — keep driving forward
                self.state = 'DRIVING'
            elif rear_dist > cfg.REAR_CLEAR_M:
                # All forward paths blocked AND front is close — reverse
                self.state = 'REVERSING'
            else:
                # Boxed in — creep forward with hard steer
                self.state = 'DRIVING'

        # ── Compute throttle + steering ───────────────────────────────────
        # RULE: Throttle is ALWAYS 0.10 (forward or reverse). Never reduce.

        if self.state == 'DRIVING':
            throttle = cfg.THROTTLE
            # Straightening: if front is clear (no obstacle nearby),
            # blend steering toward 0 (straight) to drive in a line
            if front_dist > cfg.ZONE_CLEAR_M:
                # Clear ahead — steer toward straight (0°)
                straight_blend = 0.3  # 30% toward straight each tick
                blended = path_steer * (1.0 - straight_blend)
                steering = self._smooth_steer(blended, dt)
            else:
                steering = self._smooth_steer(path_steer, dt)

        elif self.state == 'REVERSING':
            throttle = -cfg.THROTTLE
            # Steer toward side with more space while reversing
            if left_dist > right_dist:
                rev_steer = cfg.MAX_STEERING_RAD * 0.8
            elif right_dist > left_dist:
                rev_steer = -cfg.MAX_STEERING_RAD * 0.8
            else:
                rev_steer = cfg.MAX_STEERING_RAD * 0.6 if path_steer > 0 else -cfg.MAX_STEERING_RAD * 0.6
            steering = self._smooth_steer(rev_steer, dt)

        elif self.state == 'MANUAL_FWD':
            throttle = cfg.THROTTLE
            steering = self._smooth_steer(path_steer, dt)

        elif self.state == 'MANUAL_REV':
            throttle = -cfg.THROTTLE
            steering = self._smooth_steer(path_steer, dt)

        else:
            throttle = cfg.THROTTLE
            steering = self._smooth_steer(path_steer, dt)

        # NO distance-based throttle reduction — always full 0.10

        leds = self._get_leds(detection.get('avoid_side', 'NONE'))

        return {
            'throttle':       throttle,
            'steering':       steering,
            'state':          self.state,
            'leds':           leds,
            'crash_detected': crash_detected,
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

        if self.state == 'REVERSING':
            leds[4] = 1.0  # brake
        elif self.state in ('MANUAL_FWD', 'MANUAL_REV'):
            leds[0] = leds[1] = 1.0  # hazard = manual mode indicator
            leds[2] = leds[3] = 1.0
            if self.state == 'MANUAL_REV':
                leds[4] = 1.0
        elif avoid_side == 'LEFT':
            leds[0] = leds[1] = 1.0
        elif avoid_side == 'RIGHT':
            leds[2] = leds[3] = 1.0

        return leds
