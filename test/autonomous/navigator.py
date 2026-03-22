"""
navigator.py — Steering + throttle decisions based on obstacle detection.
Always drives at THROTTLE (0.1) — forward or reverse based on VFH path.

States:
  NAVIGATING  — path clear, drive toward goal
  AVOIDING    — obstacle in WARN zone, follow VFH gap
  WAITING     — person detected, hold position
  STOPPED     — no path / truly stuck
  REVERSING   — front blocked, reverse through rear gap
"""
import time
import numpy as np
import logging

import config as cfg


class Navigator:
    """
    Simple state machine: always 0.1 throttle, steer based on VFH path.
    """

    def __init__(self):
        self.state           = 'NAVIGATING'
        self._last_steer     = 0.0
        self._reverse_start  = 0.0
        self._wait_start     = 0.0
        self._stuck_start    = 0.0

    def reset(self):
        self.state = 'NAVIGATING'
        self._last_steer = 0.0

    def update(self, detection, dt):
        """
        Returns dict with throttle, steering, state, leds.

        Args:
            detection: dict from ObstacleDetector.detect()
            dt: time step
        """
        zone       = detection['zone']
        behaviour  = detection['behaviour']
        has_path   = detection['has_path']
        path_steer = detection['path_steer']
        drive_fwd  = detection['drive_forward']
        rear_clear = detection['rear_min_m'] > cfg.REAR_CLEAR_M
        left_clear = detection.get('left_min_m', 9.0) > cfg.SIDE_CLEAR_M
        right_clear = detection.get('right_min_m', 9.0) > cfg.SIDE_CLEAR_M
        now        = time.perf_counter()

        # ── State transitions ─────────────────────────────────────────────

        if self.state == 'REVERSING':
            if now - self._reverse_start > cfg.REVERSE_TIMEOUT_S:
                self.state = 'NAVIGATING'
            elif not rear_clear:
                self.state = 'STOPPED'
                self._stuck_start = now

        if self.state == 'STOPPED':
            if now - self._stuck_start > cfg.STUCK_RETRY_S:
                self.state = 'NAVIGATING'

        if self.state == 'WAITING':
            if now - self._wait_start > cfg.WAIT_TIMEOUT_S:
                self.state = 'NAVIGATING'
            elif behaviour != 'WAIT':
                self.state = 'NAVIGATING'

        # Fresh decision (not locked in REVERSING/STOPPED/WAITING)
        if self.state in ('NAVIGATING', 'AVOIDING'):
            if behaviour == 'WAIT':
                self.state = 'WAITING'
                self._wait_start = now
            elif behaviour == 'EMERGENCY_STOP':
                if has_path and not drive_fwd and rear_clear:
                    self.state = 'REVERSING'
                    self._reverse_start = now
                elif has_path and drive_fwd:
                    self.state = 'AVOIDING'
                elif rear_clear:
                    self.state = 'REVERSING'
                    self._reverse_start = now
                else:
                    self.state = 'STOPPED'
                    self._stuck_start = now
            elif behaviour == 'AVOID':
                self.state = 'AVOIDING'
            else:
                self.state = 'NAVIGATING'

        # ── Compute throttle + steering ───────────────────────────────────

        if self.state == 'NAVIGATING':
            throttle = cfg.THROTTLE
            steering = self._smooth_steer(path_steer if has_path else 0.0, dt)

        elif self.state == 'AVOIDING':
            throttle = cfg.THROTTLE
            steering = self._smooth_steer(path_steer, dt)

        elif self.state == 'REVERSING':
            throttle = -cfg.THROTTLE
            steering = self._smooth_steer(path_steer, dt)

        elif self.state == 'WAITING':
            throttle = 0.0
            steering = self._last_steer  # hold current steering

        elif self.state == 'STOPPED':
            throttle = 0.0
            steering = 0.0

        else:
            throttle = 0.0
            steering = 0.0

        # ── LEDs ──────────────────────────────────────────────────────────
        leds = self._get_leds(detection.get('avoid_side', 'NONE'))

        return {
            'throttle': throttle,
            'steering': steering,
            'state':    self.state,
            'leds':     leds,
        }

    def _smooth_steer(self, target, dt):
        """Rate-limit steering changes for smooth driving."""
        max_delta = cfg.MAX_STEER_RATE * dt
        delta = target - self._last_steer
        delta = max(-max_delta, min(delta, max_delta))
        self._last_steer += delta
        self._last_steer = max(-cfg.MAX_STEERING_RAD,
                               min(self._last_steer, cfg.MAX_STEERING_RAD))
        return self._last_steer

    def _get_leds(self, avoid_side):
        """LED array: [L-ind, L-ind2, R-ind, R-ind2, brake, spare, head-L, head-R]"""
        leds = np.zeros(8, dtype=np.float64)

        if self.state == 'NAVIGATING':
            leds[6] = leds[7] = 1.0  # headlights

        elif self.state == 'AVOIDING':
            leds[6] = leds[7] = 1.0
            if avoid_side == 'LEFT':
                leds[0] = leds[1] = 1.0
            elif avoid_side == 'RIGHT':
                leds[2] = leds[3] = 1.0

        elif self.state == 'WAITING':
            leds[0] = leds[1] = 1.0  # hazard
            leds[2] = leds[3] = 1.0

        elif self.state == 'REVERSING':
            leds[4] = 1.0  # brake
            leds[6] = leds[7] = 1.0

        elif self.state == 'STOPPED':
            leds[4] = 1.0  # brake
            leds[6] = leds[7] = 1.0

        return leds
