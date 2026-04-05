import sys, os, time
sys.path.insert(0, '/home/nvidia/Documents/Quanser/libraries/python')

import numpy as np
from pal.products.qcar import QCar
from core.imu_fusion import MadgwickFilter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config.imu_config import SAMPLE_RATE_HZ, DT, STEER_MAX, STEER_MIN

# ── Tune these ─────────────────────────────────────────────
THROTTLE = 0.05    # 5% — start safe
RUN_TIME = 5.0     # seconds
KP       = 0.04
KI       = 0.001
KD       = 0.008
WARMUP_S = 2.0


class HeadingPID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral   = 0.0
        self.prev_error = 0.0

    def update(self, yaw_deg, dt):
        error           = -yaw_deg
        self.integral  += error * dt
        self.integral   = float(np.clip(self.integral, -10.0, 10.0))
        derivative      = (error - self.prev_error) / dt
        self.prev_error = error
        out = self.kp*error + self.ki*self.integral + self.kd*derivative
        return float(np.clip(out, STEER_MIN, STEER_MAX))


def main():
    pid  = HeadingPID(KP, KI, KD)
    filt = MadgwickFilter(sample_rate_hz=SAMPLE_RATE_HZ, beta=0.05)
    LEDs = np.array([0, 0, 0, 0, 0, 0, 1, 1])  # headlights on

    print("=" * 50)
    print(" QCar2 - IMU Heading-Hold Drive (PAL)")
    print(f" Throttle={THROTTLE:.0%}  Duration={RUN_TIME}s")
    print(" Ctrl+C = emergency stop")
    print("=" * 50)

    with QCar(readMode=0, frequency=SAMPLE_RATE_HZ) as car:
        try:
            # ── Phase 1: Warmup ────────────────────────────────
            print(f"\n[1/3] Warmup {WARMUP_S}s — keep car still...")
            t_next = time.monotonic()
            t_end  = t_next + WARMUP_S

            while time.monotonic() < t_end:
                car.read()
                filt.update(
                    car.gyroscope.copy(),
                    car.accelerometer.copy(),
                    np.zeros(3, dtype=np.float64)
                )
                t_next += DT
                while time.monotonic() < t_next:
                    pass

            ang = filt.get_euler_angles()
            print(f"    Initial yaw: {ang['yaw_deg']:+.2f} deg")

            # ── Phase 2: Drive ─────────────────────────────────
            print(f"[2/3] Driving {RUN_TIME}s...")
            t_start = time.monotonic()
            t_next  = t_start

            while (time.monotonic() - t_start) < RUN_TIME:
                car.read()

                filt.update(
                    car.gyroscope.copy(),
                    car.accelerometer.copy(),
                    np.zeros(3, dtype=np.float64)
                )
                ang   = filt.get_euler_angles()
                steer = pid.update(ang['yaw_deg'], DT)

                car.write(THROTTLE, steer, LEDs)

                elapsed = time.monotonic() - t_start
                print(f"\r[{elapsed:4.1f}s] "
                      f"Yaw={ang['yaw_deg']:+5.1f}deg  "
                      f"Steer={steer:+.3f}  "
                      f"Throttle={THROTTLE:.2f}  "
                      f"Bat={car.batteryVoltage:.1f}V",
                      end='', flush=True)

                t_next += DT
                while time.monotonic() < t_next:
                    pass

        except KeyboardInterrupt:
            print("\n\nEmergency stop.")

        finally:
            # QCar __exit__ calls terminate() which zeroes motors
            print("\n[3/3] Stopping motors...")


if __name__ == "__main__":
    main()
