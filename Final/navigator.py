"""
navigator.py  —  Phase 5: Heading and distance to goal
Goal: drive 2.0 metres forward from start position then stop.
Uses EKF pose from estimation.py. No sensors needed directly.
"""
import numpy as np
from pal.utilities.math import wrap_to_pi


class NavigatorConfig:
    GOAL_DISTANCE_M  = 2.0    # metres forward from start
    GOAL_RADIUS_M    = 0.15   # within this = ARRIVED
    # PID for heading correction
    K_HEADING_P      = 1.2
    K_HEADING_I      = 0.0
    K_HEADING_D      = 0.05
    MAX_STEERING_RAD = np.pi / 6   # ±30°


class Navigator:
    """
    Computes steering correction to keep the car going straight to its goal.

    At start: goal is 2m directly ahead (in the direction the car is facing).
    Goal pose is fixed at initialisation — car drives to [x_goal, y_goal].

    Usage:
        nav = Navigator(NavigatorConfig())
        nav.reset(initial_pose)   # call once at start
        nav_result = nav.update(current_pose, dt)
        steering = nav_result['steering_cmd']
        arrived  = nav_result['arrived']
    """

    def __init__(self, config: NavigatorConfig):
        self.cfg       = config
        self._goal_x   = 0.0
        self._goal_y   = 0.0
        self._ei       = 0.0   # heading error integral
        self._prev_e   = None

    def reset(self, initial_pose: np.ndarray):
        """
        Set the goal 2m forward from the car's current pose.
        Call once before the main loop starts.
        """
        x0, y0, theta0 = float(initial_pose[0]), float(initial_pose[1]), float(initial_pose[2])
        self._goal_x = x0 + self.cfg.GOAL_DISTANCE_M * np.cos(theta0)
        self._goal_y = y0 + self.cfg.GOAL_DISTANCE_M * np.sin(theta0)
        self._ei     = 0.0
        self._prev_e = None
        print(f"[Navigator] Goal set: ({self._goal_x:.3f}, {self._goal_y:.3f})  "
              f"distance={self.cfg.GOAL_DISTANCE_M:.1f}m")

    def update(self, pose: np.ndarray, dt: float) -> dict:
        """
        Compute navigation outputs from current pose.

        Returns:
            distance_remaining  float   metres to goal
            heading_error       float   radians (positive = need to turn left)
            steering_cmd        float   radians (-π/6 to +π/6)
            arrived             bool    True when within GOAL_RADIUS_M
        """
        x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])

        # Vector to goal
        dx = self._goal_x - x
        dy = self._goal_y - y
        dist = float(np.sqrt(dx**2 + dy**2))

        # Arrived check
        if dist <= self.cfg.GOAL_RADIUS_M:
            return {
                'distance_remaining': dist,
                'heading_error':      0.0,
                'steering_cmd':       0.0,
                'arrived':            True,
                'goal_x':             self._goal_x,
                'goal_y':             self._goal_y,
            }

        # Desired heading toward goal
        desired_heading = float(np.arctan2(dy, dx))
        heading_error   = float(wrap_to_pi(desired_heading - theta))

        # PD controller for steering
        if dt > 0:
            self._ei += heading_error * dt
        de = 0.0
        if self._prev_e is not None and dt > 0.001:
            de = (heading_error - self._prev_e) / dt
        self._prev_e = heading_error

        steering = (self.cfg.K_HEADING_P * heading_error +
                    self.cfg.K_HEADING_I * self._ei +
                    self.cfg.K_HEADING_D * de)
        steering = float(np.clip(steering,
                                 -self.cfg.MAX_STEERING_RAD,
                                  self.cfg.MAX_STEERING_RAD))

        return {
            'distance_remaining': dist,
            'heading_error':      heading_error,
            'steering_cmd':       steering,
            'arrived':            False,
            'goal_x':             self._goal_x,
            'goal_y':             self._goal_y,
        }
