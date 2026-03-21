"""
state_machine.py  —  Decision brain for QCar 2

States:
  IDLE          → waiting, motors off
  NAVIGATING    → driving to goal at cruise speed
  AVOIDING      → static obstacle, steer around it at reduced speed
  WAITING       → person/moving obstacle, hold position
  STOPPED       → too close (<0.4m) AND rear blocked, full halt
  REVERSING     → front blocked + rear clear, back up immediately
  ARRIVED       → reached goal, stop permanently

KEY LOGIC:
  Front blocked + rear clear  → REVERSE immediately (no delay)
  Front blocked + rear blocked → STOP (wait for clearance)
  Front clear                 → NAVIGATE / AVOID as normal
  Avoidance steering is proportional to proximity (softer at range)
"""
import numpy as np
from hal.utilities.control import PID
from constants import TACH_TO_MPS, BATTERY_LOW_VOLTAGE, BATTERY_CRIT_VOLTAGE
from obstacle_detector import (
    ZONE_CLEAR, ZONE_WARN, ZONE_STOP,
    BEHAVIOUR_NAVIGATE, BEHAVIOUR_WAIT,
    BEHAVIOUR_AVOID, BEHAVIOUR_EMERGENCY_STOP,
)
from lights import (
    STATE_IDLE, STATE_NAVIGATING, STATE_AVOIDING,
    STATE_WAITING, STATE_STOPPED, STATE_ARRIVED, STATE_REVERSING,
)


class StateMachineConfig:
    # Speed
    V_REF_NAVIGATE  = 0.2    # m/s cruise speed
    V_REF_AVOID     = 0.1    # m/s during avoidance
    V_REF_REVERSE   = 0.1    # m/s during reverse
    MAX_THROTTLE    = 0.10   # absolute throttle ceiling
    MAX_STEERING    = np.pi / 6   # ±30°

    # Speed PID
    K_SPEED_P = 0.10
    K_SPEED_I = 1.0
    K_SPEED_D = 0.0

    # Avoidance steering — proportional to proximity
    AVOID_STEER_MIN  = 0.15   # steer gain at warn distance (1.5m)
    AVOID_STEER_MAX  = 0.45   # steer gain at stop distance (0.4m)

    # Waiting timeout
    WAIT_TIMEOUT_S  = 8.0

    # Reverse duration: back up for N seconds then re-evaluate
    REVERSE_DURATION_S = 1.2


class StateMachine:

    def __init__(self, config: StateMachineConfig):
        self.cfg   = config
        self.state = STATE_IDLE
        self._speed_pid = PID(
            Kp      = config.K_SPEED_P,
            Ki      = config.K_SPEED_I,
            Kd      = config.K_SPEED_D,
            uLimits = (-config.MAX_THROTTLE, config.MAX_THROTTLE),
        )
        self._wait_timer    = 0.0
        self._reverse_timer = 0.0
        self._reverse_side  = 'left'
        self._battery_warned = False

    def reset(self):
        self.state = STATE_NAVIGATING
        self._speed_pid.reset()
        self._wait_timer    = 0.0
        self._reverse_timer = 0.0
        self._battery_warned = False
        print(f"[StateMachine] Reset → {self.state}")

    def update(self, detection: dict, nav_result: dict,
               sensor_data: dict, dt: float) -> dict:

        behaviour   = detection['behaviour']
        avoid_side  = detection['avoid_side']
        arrived     = nav_result['arrived']
        nav_steer   = nav_result['steering_cmd']
        front_dist  = detection['distance_m']

        # Rear awareness from LiDAR
        rear_clear    = detection.get('rear_clear', True)
        front_blocked = detection.get('front_blocked', False)

        # Speed from motor tachometer
        current_speed = float(sensor_data['motor_speed']) * TACH_TO_MPS

        # Battery check
        battery_v  = float(sensor_data['battery_voltage'])
        battery_ok = True
        if battery_v > 1.0:
            if battery_v < BATTERY_CRIT_VOLTAGE:
                if self.state not in (STATE_ARRIVED, STATE_IDLE):
                    print(f"  [FSM] CRITICAL: Battery {battery_v:.1f}V — forcing stop!")
                    self.state = STATE_ARRIVED
                    battery_ok = False
            elif battery_v < BATTERY_LOW_VOLTAGE and not self._battery_warned:
                print(f"  [FSM] WARNING: Battery low {battery_v:.1f}V")
                self._battery_warned = True

        # ── State transitions ─────────────────────────────────────────────

        if self.state == STATE_ARRIVED:
            pass  # terminal

        elif self.state == STATE_REVERSING:
            self._reverse_timer += dt
            if self._reverse_timer >= self.cfg.REVERSE_DURATION_S:
                print(f"  [FSM] Reverse done ({self._reverse_timer:.1f}s) → NAVIGATING")
                self._reverse_timer = 0.0
                self.state = STATE_NAVIGATING
            elif not rear_clear:
                # Rear became blocked while reversing — stop
                print(f"  [FSM] Rear blocked during reverse → STOPPED")
                self._reverse_timer = 0.0
                self.state = STATE_STOPPED

        elif arrived:
            self.state = STATE_ARRIVED

        elif behaviour == BEHAVIOUR_EMERGENCY_STOP:
            # Front too close — decide: reverse or stop
            if rear_clear:
                if self.state != STATE_REVERSING:
                    print(f"  [FSM] Front STOP zone + rear clear → REVERSING")
                    self._reverse_timer = 0.0
                    self._reverse_side  = avoid_side
                self.state = STATE_REVERSING
            else:
                self.state = STATE_STOPPED

        elif behaviour == BEHAVIOUR_WAIT:
            self._wait_timer += dt
            if self.state != STATE_WAITING:
                self.state = STATE_WAITING
                self._wait_timer = 0.0
                print(f"  [FSM] WAITING — type={detection['obstacle_type']}")
            if self._wait_timer >= self.cfg.WAIT_TIMEOUT_S:
                print(f"  [FSM] Wait timeout {self._wait_timer:.1f}s — resuming")
                self._wait_timer = 0.0
                self.state = STATE_NAVIGATING

        elif behaviour == BEHAVIOUR_AVOID:
            self.state = STATE_AVOIDING

        elif behaviour == BEHAVIOUR_NAVIGATE:
            if self.state not in (STATE_NAVIGATING, STATE_ARRIVED):
                print(f"  [FSM] Path clear → NAVIGATING")
            self.state = STATE_NAVIGATING
            self._wait_timer = 0.0

        # ── Compute throttle + steering ───────────────────────────────────

        if self.state == STATE_NAVIGATING:
            v_ref    = self.cfg.V_REF_NAVIGATE
            throttle = self._speed_pid.update(r=v_ref, y=current_speed, dt=dt)
            steering = nav_steer

        elif self.state == STATE_AVOIDING:
            v_ref    = self.cfg.V_REF_AVOID
            throttle = self._speed_pid.update(r=v_ref, y=current_speed, dt=dt)

            # Proportional avoidance: steer harder when closer
            proximity = np.clip(
                (1.5 - front_dist) / (1.5 - 0.4), 0.0, 1.0
            )
            avoid_gain = (self.cfg.AVOID_STEER_MIN +
                          proximity * (self.cfg.AVOID_STEER_MAX - self.cfg.AVOID_STEER_MIN))
            avoid_offset = avoid_gain * (1.0 if avoid_side == 'left' else -1.0)
            steering = float(np.clip(
                nav_steer + avoid_offset,
                -self.cfg.MAX_STEERING, self.cfg.MAX_STEERING
            ))

        elif self.state == STATE_REVERSING:
            v_ref    = -self.cfg.V_REF_REVERSE
            throttle = self._speed_pid.update(r=v_ref, y=current_speed, dt=dt)
            # Steer opposite while reversing
            reverse_steer = -0.3 * (1.0 if self._reverse_side == 'left' else -1.0)
            steering = float(np.clip(reverse_steer,
                                     -self.cfg.MAX_STEERING, self.cfg.MAX_STEERING))

        elif self.state in (STATE_WAITING, STATE_STOPPED, STATE_IDLE, STATE_ARRIVED):
            throttle = 0.0
            steering = 0.0
            self._speed_pid.reset()

        else:
            throttle = 0.0
            steering = 0.0

        return {
            'state':      self.state,
            'throttle':   float(np.clip(throttle, -self.cfg.MAX_THROTTLE,
                                                    self.cfg.MAX_THROTTLE)),
            'steering':   float(np.clip(steering, -self.cfg.MAX_STEERING,
                                                    self.cfg.MAX_STEERING)),
            'v_ref':      v_ref if self.state in (STATE_NAVIGATING, STATE_AVOIDING, STATE_REVERSING) else 0.0,
            'speed_mps':  current_speed,
            'battery_ok': battery_ok,
        }
