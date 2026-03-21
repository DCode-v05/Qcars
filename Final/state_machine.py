"""
state_machine.py  —  Phase 7: Decision brain for QCar 2
Takes obstacle detection + navigation info → decides state + throttle + steering.

States:
  IDLE          → waiting, motors off
  NAVIGATING    → driving straight to goal at V_REF speed
  AVOIDING      → static obstacle, steer around it at half speed
  WAITING       → person/moving obstacle, hold position
  STOPPED       → too close (<0.4m), emergency halt
  REVERSING     → stuck too long in STOPPED, reverse to clear obstacle
  ARRIVED       → reached goal, stop permanently

Fixes applied:
  - Speed conversion uses TACH_TO_MPS (not CPS_TO_MPS) for motorTach input
  - REVERSING state added with timeout-triggered back-up manoeuvre
  - V_REF_NAVIGATE capped to be achievable within MAX_THROTTLE
  - Battery voltage monitoring with auto-stop on critical low
  - Shared constants from constants.py
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
    V_REF_NAVIGATE  = 0.2    # m/s — achievable within MAX_THROTTLE range
    V_REF_AVOID     = 0.1    # m/s during obstacle avoidance
    V_REF_REVERSE   = 0.1    # m/s during reverse manoeuvre
    MAX_THROTTLE    = 0.15   # absolute throttle ceiling (safe for testing)
    MAX_STEERING    = np.pi / 6   # ±30°

    # Speed PID (Quanser values: Kp=0.1, Ki=1.0)
    K_SPEED_P = 0.10
    K_SPEED_I = 1.0
    K_SPEED_D = 0.0

    # Avoidance steering gain (added on top of navigator steering)
    AVOID_STEER_GAIN = 0.4   # multiplied by avoid_side ±1

    # Waiting timeout: if person doesn't move in N seconds, try to navigate
    WAIT_TIMEOUT_S  = 8.0

    # Stopped timeout: if stuck in STOPPED for N seconds, try reversing
    STOP_TIMEOUT_S  = 3.0

    # Reverse duration: back up for N seconds then try to navigate
    REVERSE_DURATION_S = 1.5


class StateMachine:
    """
    Usage:
        sm = StateMachine(StateMachineConfig())
        sm.reset()
        result = sm.update(detection, nav_result, sensor_data, dt)
        throttle = result['throttle']
        steering = result['steering']
        state    = result['state']
    """

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
        self._stop_timer    = 0.0
        self._reverse_timer = 0.0
        self._reverse_side  = 'left'
        self._battery_warned = False

    def reset(self):
        self.state = STATE_NAVIGATING   # start driving immediately
        self._speed_pid.reset()
        self._wait_timer    = 0.0
        self._stop_timer    = 0.0
        self._reverse_timer = 0.0
        self._battery_warned = False
        print(f"[StateMachine] Reset → {self.state}")

    def update(self, detection: dict, nav_result: dict,
               sensor_data: dict, dt: float) -> dict:
        """
        One FSM tick. Returns throttle, steering, state, battery_ok.

        Parameters:
            detection   dict  from ObstacleDetector.detect()
            nav_result  dict  from Navigator.update()
            sensor_data dict  from SensorManager.read()
            dt          float seconds since last tick
        """
        behaviour = detection['behaviour']
        avoid_side= detection['avoid_side']
        arrived   = nav_result['arrived']
        dist_left = nav_result['distance_remaining']
        nav_steer = nav_result['steering_cmd']

        # Fix #5: Use TACH_TO_MPS for motorTach (rad/s) → wheel m/s
        current_speed = float(sensor_data['motor_speed']) * TACH_TO_MPS

        # Fix #18: Battery voltage monitoring
        battery_v  = float(sensor_data['battery_voltage'])
        battery_ok = True
        if battery_v > 1.0:    # valid reading (0 = sensor not ready)
            if battery_v < BATTERY_CRIT_VOLTAGE:
                if self.state not in (STATE_ARRIVED, STATE_IDLE):
                    print(f"  [FSM] CRITICAL: Battery {battery_v:.1f}V < {BATTERY_CRIT_VOLTAGE}V — forcing stop!")
                    self.state = STATE_ARRIVED   # terminal — won't resume
                    battery_ok = False
            elif battery_v < BATTERY_LOW_VOLTAGE and not self._battery_warned:
                print(f"  [FSM] WARNING: Battery low {battery_v:.1f}V")
                self._battery_warned = True

        # ── State transitions ─────────────────────────────────────────────

        if self.state == STATE_ARRIVED:
            # Terminal state — stay here
            pass

        elif self.state == STATE_REVERSING:
            # Continue reversing until duration expires
            self._reverse_timer += dt
            if self._reverse_timer >= self.cfg.REVERSE_DURATION_S:
                print(f"  [FSM] Reverse complete ({self._reverse_timer:.1f}s) → NAVIGATING")
                self._reverse_timer = 0.0
                self._stop_timer    = 0.0
                self.state = STATE_NAVIGATING

        elif arrived:
            self.state = STATE_ARRIVED

        elif behaviour == BEHAVIOUR_EMERGENCY_STOP:
            if self.state != STATE_STOPPED:
                self._stop_timer = 0.0
            self._stop_timer += dt
            self.state = STATE_STOPPED

            # Fix #6: If stuck in STOPPED too long, try reversing
            if self._stop_timer >= self.cfg.STOP_TIMEOUT_S:
                print(f"  [FSM] Stuck in STOPPED for {self._stop_timer:.1f}s → REVERSING")
                self._reverse_timer = 0.0
                self._reverse_side  = avoid_side
                self.state = STATE_REVERSING

        elif behaviour == BEHAVIOUR_WAIT:
            self._stop_timer = 0.0
            self._wait_timer += dt
            if self.state != STATE_WAITING:
                self.state = STATE_WAITING
                self._wait_timer = 0.0
                print(f"  [FSM] WAITING — obstacle type={detection['obstacle_type']}")
            if self._wait_timer >= self.cfg.WAIT_TIMEOUT_S:
                print(f"  [FSM] Wait timeout after {self._wait_timer:.1f}s — resuming")
                self._wait_timer = 0.0
                self.state = STATE_NAVIGATING

        elif behaviour == BEHAVIOUR_AVOID:
            self._stop_timer = 0.0
            self.state = STATE_AVOIDING

        elif behaviour == BEHAVIOUR_NAVIGATE:
            self._stop_timer = 0.0
            if self.state not in (STATE_NAVIGATING, STATE_ARRIVED):
                print(f"  [FSM] Path clear → NAVIGATING")
            self.state = STATE_NAVIGATING
            self._wait_timer = 0.0

        # ── Compute throttle + steering based on state ────────────────────

        if self.state == STATE_NAVIGATING:
            v_ref    = self.cfg.V_REF_NAVIGATE
            throttle = self._speed_pid.update(r=v_ref, y=current_speed, dt=dt)
            steering = nav_steer

        elif self.state == STATE_AVOIDING:
            v_ref    = self.cfg.V_REF_AVOID
            throttle = self._speed_pid.update(r=v_ref, y=current_speed, dt=dt)
            avoid_offset = self.cfg.AVOID_STEER_GAIN * (1.0 if avoid_side == 'left' else -1.0)
            steering = float(np.clip(
                nav_steer + avoid_offset,
                -self.cfg.MAX_STEERING, self.cfg.MAX_STEERING
            ))

        elif self.state == STATE_REVERSING:
            # Reverse slowly, steer away from the obstacle
            v_ref    = -self.cfg.V_REF_REVERSE   # negative = backwards
            throttle = self._speed_pid.update(r=v_ref, y=current_speed, dt=dt)
            # Steer opposite to avoid_side while reversing
            reverse_steer = -self.cfg.AVOID_STEER_GAIN * (1.0 if self._reverse_side == 'left' else -1.0)
            steering = float(np.clip(reverse_steer,
                                     -self.cfg.MAX_STEERING, self.cfg.MAX_STEERING))

        elif self.state in (STATE_WAITING, STATE_STOPPED, STATE_IDLE, STATE_ARRIVED):
            throttle = 0.0
            steering = 0.0
            self._speed_pid.reset()

        else:
            throttle = 0.0
            steering = 0.0

        v_ref_out = self.cfg.V_REF_NAVIGATE if self.state == STATE_NAVIGATING \
                    else self.cfg.V_REF_AVOID if self.state == STATE_AVOIDING \
                    else -self.cfg.V_REF_REVERSE if self.state == STATE_REVERSING \
                    else 0.0

        return {
            'state':      self.state,
            'throttle':   float(np.clip(throttle, -self.cfg.MAX_THROTTLE,
                                                    self.cfg.MAX_THROTTLE)),
            'steering':   float(np.clip(steering, -self.cfg.MAX_STEERING,
                                                    self.cfg.MAX_STEERING)),
            'v_ref':      v_ref_out,
            'speed_mps':  current_speed,
            'battery_ok': battery_ok,
        }
