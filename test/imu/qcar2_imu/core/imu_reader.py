import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataclasses import dataclass
from quanser.hardware import HIL
from quanser.common import GenericError
from config.imu_config import (
    BOARD_TYPE, BOARD_ID,
    ALL_IMU_CHANNELS,
    GYRO_MAX_RADS, ACCEL_MAX_MS2
)


@dataclass
class IMUData:
    # Gyroscope (rad/s)
    gyro_x: float
    gyro_y: float
    gyro_z: float
    # Accelerometer (m/s²)
    accel_x: float
    accel_y: float
    accel_z: float
    # Magnetometer (Tesla)
    mag_x: float
    mag_y: float
    mag_z: float

    def as_array(self):
        return np.array([
            self.gyro_x,  self.gyro_y,  self.gyro_z,
            self.accel_x, self.accel_y, self.accel_z,
            self.mag_x,   self.mag_y,   self.mag_z
        ], dtype=np.float64)

    def is_valid(self):
        return (
            abs(self.gyro_x)  <= GYRO_MAX_RADS and
            abs(self.gyro_y)  <= GYRO_MAX_RADS and
            abs(self.gyro_z)  <= GYRO_MAX_RADS and
            abs(self.accel_x) <= ACCEL_MAX_MS2 and
            abs(self.accel_y) <= ACCEL_MAX_MS2 and
            abs(self.accel_z) <= ACCEL_MAX_MS2
        )

    def __str__(self):
        return (
            f"Gyro  (rad/s): X={self.gyro_x:+7.4f}  "
            f"Y={self.gyro_y:+7.4f}  Z={self.gyro_z:+7.4f}\n"
            f"Accel (m/s²):  X={self.accel_x:+7.4f}  "
            f"Y={self.accel_y:+7.4f}  Z={self.accel_z:+7.4f}\n"
            f"Mag   (µT):    X={self.mag_x*1e6:+7.2f}  "
            f"Y={self.mag_y*1e6:+7.2f}  Z={self.mag_z*1e6:+7.2f}"
        )


class IMUReader:
    def __init__(self):
        self._card   = None
        self._buffer = np.zeros(len(ALL_IMU_CHANNELS), dtype=np.float64)
        self._n_ch   = len(ALL_IMU_CHANNELS)

    def open(self):
        try:
            self._card = HIL(BOARD_TYPE, BOARD_ID)
            print(f"[IMUReader] HIL opened: {BOARD_TYPE}/{BOARD_ID}")
        except GenericError as e:
            raise RuntimeError(
                f"[IMUReader] Failed to open HIL board.\n"
                f"  Check: systemctl status quanser_license_manager\n"
                f"  Error: {e}"
            )

    def read(self):
        if self._card is None:
            raise RuntimeError("[IMUReader] Call open() before read().")
        self._card.read_other(ALL_IMU_CHANNELS, self._n_ch, self._buffer)
        return IMUData(
            gyro_x  = self._buffer[0],
            gyro_y  = self._buffer[1],
            gyro_z  = self._buffer[2],
            accel_x = self._buffer[3],
            accel_y = self._buffer[4],
            accel_z = self._buffer[5],
            mag_x   = self._buffer[6],
            mag_y   = self._buffer[7],
            mag_z   = self._buffer[8],
        )

    def close(self):
        if self._card is not None:
            self._card.close()
            self._card = None
            print("[IMUReader] HIL closed.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
