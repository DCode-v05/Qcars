"""
localizer.py — Dead reckoning position tracking for QCar 2.

Integrates motor tachometer (speed) and gyroscope (yaw rate) to estimate
the car's (x, y, theta) pose in world frame. Start position = (0, 0, 0).

Drift is acceptable for short indoor runs (<30s, <10m).
"""
import math
import numpy as np

import config as cfg


class Localizer:

    def __init__(self):
        self.x     = 0.0
        self.y     = 0.0
        self.theta = 0.0  # heading in radians, 0 = initial forward
        self._prev_speed = 0.0

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self._prev_speed = 0.0

    def update(self, sensor_data, dt, throttle_sign=1.0):
        """Update pose from sensor data.

        Args:
            sensor_data: dict from SensorManager.read()
            dt: time step in seconds
            throttle_sign: +1.0 for forward, -1.0 for reverse
        Returns:
            (x, y, theta) tuple
        """
        if dt <= 0:
            return self.pose

        # Speed from motor tachometer
        motor_speed = sensor_data.get('motor_speed', 0.0)
        v = abs(motor_speed) * cfg.TACH_TO_MPS * throttle_sign

        # Yaw rate from gyroscope (z-axis), remove bias
        gyro = sensor_data.get('gyroscope')
        if gyro is not None:
            yaw_rate = float(gyro[2]) - cfg.GYRO_BIAS_Z
        else:
            yaw_rate = 0.0

        # Integrate heading
        self.theta += yaw_rate * dt

        # Integrate position (mid-point method for better accuracy)
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt

        self._prev_speed = v
        return self.pose

    @property
    def pose(self):
        return (self.x, self.y, self.theta)

    @property
    def speed(self):
        return self._prev_speed
