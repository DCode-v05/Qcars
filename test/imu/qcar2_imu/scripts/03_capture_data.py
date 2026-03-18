import sys, os, time, csv, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime
import numpy as np
from core.imu_reader import IMUReader
from core.imu_fusion import MadgwickFilter
from config.imu_config import SAMPLE_RATE_HZ, DT, IMU_LABELS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(data_dir, f"imu_{ts}.csv")

    header        = ["timestamp_s"] + IMU_LABELS + ["roll_deg","pitch_deg","yaw_deg"]
    filt          = MadgwickFilter(sample_rate_hz=SAMPLE_RATE_HZ, beta=0.05)
    total_samples = int(args.duration * SAMPLE_RATE_HZ)
    written       = 0

    print(f"Capturing {args.duration}s @ {SAMPLE_RATE_HZ}Hz")
    print(f"Output: {out}")
    print("Ctrl+C to stop early.\n")

    with open(out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        try:
            with IMUReader() as imu:
                t_start = time.monotonic()
                t_next  = t_start

                for i in range(total_samples):
                    d = imu.read()
                    filt.update(
                        np.array([d.gyro_x,  d.gyro_y,  d.gyro_z]),
                        np.array([d.accel_x, d.accel_y, d.accel_z]),
                        np.array([d.mag_x,   d.mag_y,   d.mag_z])
                    )
                    ang     = filt.get_euler_angles()
                    elapsed = time.monotonic() - t_start

                    writer.writerow([
                        f"{elapsed:.4f}",
                        f"{d.gyro_x:.6f}",  f"{d.gyro_y:.6f}",  f"{d.gyro_z:.6f}",
                        f"{d.accel_x:.6f}", f"{d.accel_y:.6f}", f"{d.accel_z:.6f}",
                        f"{d.mag_x:.9f}",   f"{d.mag_y:.9f}",   f"{d.mag_z:.9f}",
                        f"{ang['roll_deg']:.4f}",
                        f"{ang['pitch_deg']:.4f}",
                        f"{ang['yaw_deg']:.4f}",
                    ])
                    written += 1

                    if i % 50 == 0:
                        print(f"\r[{elapsed:5.1f}s] "
                              f"Yaw={ang['yaw_deg']:+6.1f}deg  "
                              f"Pitch={ang['pitch_deg']:+5.1f}deg  "
                              f"Roll={ang['roll_deg']:+5.1f}deg  "
                              f"({written} samples)",
                              end='', flush=True)

                    t_next += DT
                    while time.monotonic() < t_next:
                        pass

        except KeyboardInterrupt:
            print("\nStopped early.")

    print(f"\nDone. {written} samples saved to:\n  {out}")

if __name__ == "__main__":
    main()
