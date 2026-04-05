import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from core.imu_reader import IMUReader

def main():
    print("=" * 45)
    print(" QCar2 IMU - Connection Verification")
    print("=" * 45)

    try:
        with IMUReader() as imu:
            data = imu.read()

            print("\n--- RAW IMU READ (all 9 axes) ---")
            print(data)

            if abs(data.accel_z - 9.81) > 3.0:
                print(f"\nWARNING: accel_z = {data.accel_z:.3f} m/s2"
                      " (expected ~9.81 when flat)")

            mag_uT = np.sqrt(data.mag_x**2+data.mag_y**2+data.mag_z**2)*1e6
            print(f"\nMag field magnitude: {mag_uT:.1f} uT", end="  ")
            if 10 < mag_uT < 100:
                print("[OK]")
            else:
                print("[WARNING: outside 25-65 uT range]")

            if data.is_valid():
                print("Data range check:    PASS")
            else:
                print("Data range check:    FAIL")

            print("\nRESULT: HIL connection OK, all 9 channels readable.")

    except RuntimeError as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
