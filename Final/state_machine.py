"""
state_machine.py  —  Decision brain for QCar 2

Uses LiDAR path planner output to decide WHAT to do:
  - Path planner says "go forward, steer X" → NAVIGATING/AVOIDING with that steer
  - Path planner says "reverse, steer X"    → REVERSING with that steer
  - Path planner says "no path"             → STOPPED
  - YOLO sees person                        → WAITING

States:
  IDLE          → motors off
  NAVIGATING    → driving toward goal, path clear
  AVOIDING      → obstacle in warn zone, steering around via path planner
  WAITING       → person detected, hold position
  STOPPED       → no path found or rear blocked, full halt
  REVERSING     → front blocked, best gap is behind, backing up
  ARRIVED       → reached goal
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
    V_REF_AVOID     = 0.12   # m/s during obstacle avoidance
    V_REF_REVERSE   = 0.10   # m/s during reverse
    MAX_THROTTLE    = 0.10   # absolute throttle ceiling
    MAX_STEERING    = np.pi / 6   # ±30°

    # Speed PID
    K_SPEED_P = 0.10
    K_SPEED_I = 1.0
    K_SPEED_D = 0.0

    # Waiting timeout
    WAIT_TIMEOUT_S     = 8.0

    # Reverse: max duration before re-evaluating
    REVERSE_MAX_S      = 2.0

    # Stopped: if stuck for this long, try reverse anyway
    STUCK_TIMEOUT_S    = 3.0


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
        self._wait_timer     = 0.0
        self._reverse_timer  = 0.0
        self._stuck_timer    = 0.0
        self._path_steer     = 0.0   # from path planner
        self._battery_warned = False

    def reset(self):
        self.state = STATE_NAVIGATING
        self._speed_pid.reset()
        self._wait_timer     = 0.0
        self._reverse_timer  = 0.0
        self._stuck_timer    = 0.0
        self._battery_warned = False
        print(f"[StateMachine] Reset → {self.state}")

    def update(self, detection: dict, nav_result: dict,
               sensor_data: dict, dt: float) -> dict:

        behaviour     = detection['behaviour']
        arrived       = nav_result['arrived']
        nav_steer     = nav_result['steering_cmd']
        front_dist    = detection['distance_m']

        # Path planner outputs
        has_path      = detection.get('has_path', True)
        path_steer    = detection.get('path_steer', 0.0)
        drive_forward = detection.get('drive_forward', True)
        rear_clear    = detection.get('rear_clear', True)
        front_blocked = detection.get('front_blocked', False)

        # Speed from motor tachometer (corrected conversion)
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

        elif arrived:
            self.state = STATE_ARRIVED

        elif self.state == STATE_REVERSING:
            # Continue reversing until timer expires or rear becomes blocked
            self._reverse_timer += dt
            if self._reverse_timer >= self.cfg.REVERSE_MAX_S:
                print(f"  [FSM] Reverse done ({self._reverse_timer:.1f}s) → NAVIGATING")
                self._reverse_timer = 0.0
                self._stuck_timer   = 0.0
                self.state = STATE_NAVIGATING
            elif not rear_clear:
                print(f"  [FSM] Rear blocked during reverse → STOPPED")
                self._reverse_timer = 0.0
                self.state = STATE_STOPPED

        elif behaviour == BEHAVIOUR_WAIT:
            # Person/moving obstacle — hold position
            self._wait_timer += dt
            if self.state != STATE_WAITING:
                self.state = STATE_WAITING
                self._wait_timer = 0.0
                print(f"  [FSM] WAITING — type={detection['obstacle_type']}")
            if self._wait_timer >= self.cfg.WAIT_TIMEOUT_S:
                print(f"  [FSM] Wait timeout {self._wait_timer:.1f}s — resuming")
                self._wait_timer = 0.0
                self.state = STATE_NAVIGATING

        elif behaviour == BEHAVIOUR_EMERGENCY_STOP:
            # Front too close — use path planner to decide
            if has_path and not drive_forward and rear_clear:
                # Best gap is behind us — reverse
                if self.state != STATE_REVERSING:
                    print(f"  [FSM] Front STOP + best path behind → REVERSING")
                    self._reverse_timer = 0.0
                    self._path_steer = path_steer
                self.state = STATE_REVERSING
            elif has_path and drive_forward:
                # Path planner found a forward escape route — try avoiding
                self.state = STATE_AVOIDING
                self._path_steer = path_steer
            elif rear_clear:
                # No great path but rear is clear — just back up
                if self.state != STATE_REVERSING:
                    print(f"  [FSM] Front STOP + rear clear → REVERSING (fallback)")
                    self._reverse_timer = 0.0
                    self._path_steer = 0.0
                self.state = STATE_REVERSING
            else:
                # Truly stuck — front and rear blocked
                self._stuck_timer += dt
                if self._stuck_timer >= self.cfg.STUCK_TIMEOUT_S and rear_clear:
                    print(f"  [FSM] Stuck {self._stuck_timer:.1f}s → trying REVERSE")
                    self._stuck_timer = 0.0
                    self._reverse_timer = 0.0
                    self._path_steer = 0.0
                    self.state = STATE_REVERSING
                else:
                    self.state = STATE_STOPPED

        elif behaviour == BEHAVIOUR_AVOID:
            # Obstacle in warn zone — steer using path planner
            self.state = STATE_AVOIDING
            self._path_steer = path_steer
            self._stuck_timer = 0.0

        elif behaviour == BEHAVIOUR_NAVIGATE:
            # Path clear — normal driving
            if self.state not in (STATE_NAVIGATING, STATE_ARRIVED):
                print(f"  [FSM] Path clear → NAVIGATING")
            self.state = STATE_NAVIGATING
            self._wait_timer  = 0.0
            self._stuck_timer = 0.0

        # ── Compute throttle + steering ───────────────────────────────────

        v_ref    = 0.0
        throttle = 0.0
        steering = 0.0

        if self.state == STATE_NAVIGATING:
            v_ref    = self.cfg.V_REF_NAVIGATE
            throttle = self._speed_pid.update(r=v_ref, y=current_speed, dt=dt)
            # Use navigator heading PID when path is clear
            steering = nav_steer

        elif self.state == STATE_AVOIDING:
            v_ref    = self.cfg.V_REF_AVOID
            throttle = self._speed_pid.update(r=v_ref, y=current_speed, dt=dt)
            # Use path planner steering — it already points toward the best gap
            steering = self._path_steer

        elif self.state == STATE_REVERSING:
            v_ref    = -self.cfg.V_REF_REVERSE
            throttle = self._speed_pid.update(r=v_ref, y=current_speed, dt=dt)
            # Path planner already computed reverse steering
            steering = self._path_steer

        elif self.state in (STATE_WAITING, STATE_STOPPED, STATE_IDLE, STATE_ARRIVED):
            throttle = 0.0
            steering = 0.0
            self._speed_pid.reset()

        return {
            'state':      self.state,
            'throttle':   float(np.clip(throttle, -self.cfg.MAX_THROTTLE,
                                                    self.cfg.MAX_THROTTLE)),
            'steering':   float(np.clip(steering, -self.cfg.MAX_STEERING,
                                                    self.cfg.MAX_STEERING)),
            'v_ref':      v_ref,
            'speed_mps':  current_speed,
            'battery_ok': battery_ok,
        }
