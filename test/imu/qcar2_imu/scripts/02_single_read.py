import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core.imu_reader import IMUReader
from core.imu_fusion import MadgwickFilter
from config.imu_config import SAMPLE_RATE_HZ, DT

WARMUP_SECONDS = 2.0

def main():
    filt = MadgwickFilter(sample_rate_hz=SAMPLE_RATE_HZ, beta=0.1)
    print(f"Warming up filter for {WARMUP_SECONDS}s...")

    with IMUReader() as imu:
        t_start = time.monotonic()
        t_next  = t_start

        while (time.monotonic() - t_start) < WARMUP_SECONDS:
            d = imu.read()
            filt.update(
                np.array([d.gyro_x,  d.gyro_y,  d.gyro_z]),
                np.array([d.accel_x, d.accel_y, d.accel_z]),
                np.array([d.mag_x,   d.mag_y,   d.mag_z])
            )
            t_next += DT
            while time.monotonic() < t_next:
                pass

        d = imu.read()
        filt.update(
            np.array([d.gyro_x,  d.gyro_y,  d.gyro_z]),
            np.array([d.accel_x, d.accel_y, d.accel_z]),
            np.array([d.mag_x,   d.mag_y,   d.mag_z])
        )
        angles = filt.get_euler_angles()

        print("\n=== RAW 9-AXIS DATA ===")
        print(d)
        print("\n=== FUSED ORIENTATION ===")
        print(f"  Roll  : {angles['roll_deg']:+7.2f} deg  (lean left/right)")
        print(f"  Pitch : {angles['pitch_deg']:+7.2f} deg  (nose up/down)")
        print(f"  Yaw   : {angles['yaw_deg']:+7.2f} deg  (compass heading)")

if __name__ == "__main__":
    main()
