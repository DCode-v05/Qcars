"""
navigator.py — Steering + throttle decisions. Car NEVER fully stops.

States:
  NAVIGATING  — path clear, full throttle
  AVOIDING    — obstacle nearby, steer around, full throttle
  CREEPING    — very tight space, reduced throttle but still moving
  REVERSING   — best path is behind, reverse
  SLOWING     — person detected nearby, slow down briefly
"""
import time
import numpy as np
import logging

import config as cfg


class Navigator:
    """
    The car ALWAYS moves. Throttle is either:
      - cfg.THROTTLE (0.10) for normal driving
      - cfg.CREEP_THROTTLE (0.04) for tight spaces / near person
      - -cfg.THROTTLE (-0.10) for reversing
    Never zero.
    """

    def __init__(self):
        self.state           = 'NAVIGATING'
        self._last_steer     = 0.0
        self._reverse_start  = 0.0
        self._reverse_end    = 0.0   # cooldown: don't re-enter reverse immediately
        self._slow_start     = 0.0

    def reset(self):
        self.state = 'NAVIGATING'
        self._last_steer = 0.0
        self._reverse_end = 0.0

    def update(self, detection, dt):
        zone       = detection['zone']
        behaviour  = detection['behaviour']
        path_steer = detection['path_steer']
        drive_fwd  = detection['drive_forward']
        front_dist = detection['distance_m']
        rear_clear = detection['rear_min_m'] > cfg.REAR_CLEAR_M
        now        = time.perf_counter()

        # ── State transitions ─────────────────────────────────────────────

        if self.state == 'REVERSING':
            if now - self._reverse_start > cfg.REVERSE_TIMEOUT_S:
                # After reversing, force CREEPING forward with turn
                # (don't jump back to REVERSING immediately)
                self.state = 'CREEPING'
                self._reverse_end = now
            elif not rear_clear:
                self.state = 'CREEPING'
                self._reverse_end = now

        if self.state == 'SLOWING':
            if now - self._slow_start > cfg.PERSON_PAUSE_S:
                self.state = 'NAVIGATING'
            elif behaviour != 'SLOW':
                self.state = 'NAVIGATING'

        # Fresh decisions for non-locked states
        # Reverse cooldown: after reversing, must creep forward for 1.5s
        # before allowing another reverse
        reverse_cooldown = (now - self._reverse_end) < 1.5

        if self.state in ('NAVIGATING', 'AVOIDING', 'CREEPING'):
            if behaviour == 'SLOW':
                self.state = 'SLOWING'
                self._slow_start = now
            elif zone == 'STOP' and not drive_fwd and rear_clear and not reverse_cooldown:
                # Front blocked, path behind, and not in cooldown — reverse
                self.state = 'REVERSING'
                self._reverse_start = now
            elif zone == 'STOP':
                # Very tight OR in reverse cooldown — creep forward with turn
                self.state = 'CREEPING'
            elif zone == 'WARN':
                self.state = 'AVOIDING'
            else:
                self.state = 'NAVIGATING'

        # ── Compute throttle + steering ───────────────────────────────────

        if self.state == 'NAVIGATING':
            throttle = cfg.THROTTLE
            steering = self._smooth_steer(path_steer, dt)

        elif self.state == 'AVOIDING':
            throttle = cfg.THROTTLE
            steering = self._smooth_steer(path_steer, dt)

        elif self.state == 'CREEPING':
            # Still moving! Just slower for tight spaces
            throttle = cfg.CREEP_THROTTLE
            steering = self._smooth_steer(path_steer, dt)

        elif self.state == 'REVERSING':
            throttle = -cfg.THROTTLE
            steering = self._smooth_steer(path_steer, dt)

        elif self.state == 'SLOWING':
            # Near person — slow creep, still steering
            throttle = cfg.CREEP_THROTTLE
            steering = self._smooth_steer(path_steer, dt)

        else:
            throttle = cfg.CREEP_THROTTLE
            steering = self._smooth_steer(path_steer, dt)

        # ── Throttle scaling by distance ──────────────────────────────────
        # Reduce throttle proportionally when close to obstacles
        if throttle > 0 and front_dist < cfg.ZONE_CLEAR_M:
            dist_ratio = max(front_dist, 0.1) / cfg.ZONE_CLEAR_M
            scaled = cfg.CREEP_THROTTLE + (cfg.THROTTLE - cfg.CREEP_THROTTLE) * dist_ratio
            throttle = min(throttle, scaled)

        leds = self._get_leds(detection.get('avoid_side', 'NONE'))

        return {
            'throttle': throttle,
            'steering': steering,
            'state':    self.state,
            'leds':     leds,
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
        leds[6] = leds[7] = 1.0  # headlights always on

        if self.state == 'AVOIDING' or self.state == 'CREEPING':
            if avoid_side == 'LEFT':
                leds[0] = leds[1] = 1.0
            elif avoid_side == 'RIGHT':
                leds[2] = leds[3] = 1.0

        elif self.state == 'REVERSING':
            leds[4] = 1.0  # brake

        elif self.state == 'SLOWING':
            leds[0] = leds[1] = 1.0  # hazard
            leds[2] = leds[3] = 1.0

        return leds
