"""
estimation.py  —  Phase 2: EKF pose estimation for QCar 2
Tracks WHERE the car is using encoder + IMU gyroscope fusion.

Fixes applied:
  - Gyroscope used for heading prediction (more accurate than bicycle model alone)
  - Gyro bias correction applied
  - Correction step available for external measurements (SLAM, scan-matching)
  - Shared constants from constants.py (no duplication)
"""
import time
import numpy as np
from hal.utilities.estimation import EKF
from pal.utilities.math       import wrap_to_pi
from constants import (
    WHEEL_RADIUS, WHEEL_BASE, PIN_TO_SPUR_RATIO,
    ENCODER_CPR, ENCODER_PPR, CPS_TO_MPS, GYRO_BIAS_Z,
)


class EstimationConfig:
    INITIAL_POSE       = np.zeros((3, 1))
    INITIAL_COVARIANCE = np.eye(3) * 0.01
    Q_x     = 0.01
    Q_y     = 0.01
    Q_theta = 0.005       # tighter — gyro gives good heading
    R_x     = 0.10        # measurement noise (for correction step)
    R_y     = 0.10
    R_theta = 0.05


def _predict_f(x_hat, u, dt):
    """
    State prediction using encoder speed + gyroscope heading rate.
    u = [v_wheel_mps, gyro_z_corrected_rad_s]
    """
    x, y, theta = float(x_hat[0]), float(x_hat[1]), float(x_hat[2])
    v     = float(u[0])     # wheel speed in m/s (from encoder)
    omega = float(u[1])     # heading rate in rad/s (from gyroscope, bias-corrected)

    x_new     = x     + v * np.cos(theta) * dt
    y_new     = y     + v * np.sin(theta) * dt
    theta_new = theta + omega * dt
    theta_new = float(wrap_to_pi(theta_new))
    return np.array([[x_new], [y_new], [theta_new]])


def _predict_J_f(x_hat, u, dt):
    """Jacobian of prediction model w.r.t. state."""
    theta = float(x_hat[2])
    v     = float(u[0])
    F = np.eye(3)
    F[0, 2] = -v * np.sin(theta) * dt
    F[1, 2] = +v * np.cos(theta) * dt
    return F


class PoseEstimator:
    """
    EKF-based pose estimator with encoder + gyroscope fusion.

    Usage:
        estimator = PoseEstimator(EstimationConfig())
        estimator.reset()
        pose = estimator.update(sensor_data, dt)
        x, y, theta = pose
    """

    def __init__(self, config: EstimationConfig):
        self.cfg           = config
        self._ekf          = None
        self._last_encoder = None

    def reset(self, initial_pose=None):
        cfg = self.cfg
        x0  = initial_pose if initial_pose is not None else cfg.INITIAL_POSE
        x0  = np.array(x0, dtype=float).reshape((3, 1))
        self._ekf = EKF(
            x_0 = x0,
            P_0 = cfg.INITIAL_COVARIANCE.copy(),
            Q   = np.diagflat([cfg.Q_x, cfg.Q_y, cfg.Q_theta]),
            R   = np.diagflat([cfg.R_x, cfg.R_y, cfg.R_theta]),
            f   = _predict_f,
            J_f = _predict_J_f,
            C   = np.eye(3),
        )
        self._last_encoder = None
        print(f"[PoseEstimator] EKF initialised (encoder+gyro).  "
              f"x={float(x0[0]):.3f}  y={float(x0[1]):.3f}  θ={float(x0[2]):.3f}")

    def update(self, sensor_data: dict, dt: float) -> np.ndarray:
        """
        Predict step using encoder speed + gyroscope heading rate.
        Returns current pose [x, y, theta].
        """
        v_wheel = self._encoder_to_speed(sensor_data, dt)
        gyro_z  = float(sensor_data['gyroscope'][2]) - GYRO_BIAS_Z

        u = [v_wheel, gyro_z]
        if dt > 0:
            self._ekf.predict(u, dt)

        pose = self._ekf.x_hat.flatten()
        pose[2] = float(wrap_to_pi(pose[2]))
        return pose

    def correct(self, measured_pose):
        """
        EKF correction step — call when an external measurement is available
        (e.g. scan-matching, GPS, or visual landmark).
        measured_pose: [x, y, theta]
        """
        y = np.array(measured_pose, dtype=float).reshape((3, 1))
        self._ekf.correct(y)
        self._ekf.x_hat[2, 0] = float(wrap_to_pi(self._ekf.x_hat[2, 0]))

    @property
    def pose(self):
        p = self._ekf.x_hat.flatten().copy()
        p[2] = float(wrap_to_pi(p[2]))
        return p

    @property
    def uncertainty(self):
        return np.sqrt(np.diag(self._ekf.P))

    def _encoder_to_speed(self, sensor_data, dt):
        current = sensor_data['motor_encoder']
        if self._last_encoder is None or dt <= 0:
            self._last_encoder = current
            return 0.0
        delta = current - self._last_encoder
        self._last_encoder = current
        return float(delta / dt) * CPS_TO_MPS


# ── Standalone test ───────────────────────────────────────────────────────────

def main():
    from perception import SensorManager, Config as PerceptionConfig

    p_cfg     = PerceptionConfig()
    e_cfg     = EstimationConfig()
    sensors   = SensorManager(p_cfg)
    estimator = PoseEstimator(e_cfg)

    print("═"*60)
    print("  QCar 2 — estimation.py standalone test")
    print("  Push car 1m forward → x should increase ~1.0")
    print("  Rotate 90° CCW → θ should increase ~1.57 rad")
    print("═"*60)

    sensors.open()
    estimator.reset()

    try:
        start  = time.perf_counter()
        tick_t = start
        last_t = start
        last_print = 0.0

        while True:
            elapsed = time.perf_counter() - start
            if elapsed >= 20.0:
                break

            data = sensors.read()
            now  = time.perf_counter()
            dt   = now - last_t
            last_t = now

            pose = estimator.update(data, dt)

            if elapsed - last_print >= 0.5:
                x, y, theta = pose
                ux, uy, ut  = estimator.uncertainty
                speed = data['motor_speed']
                enc   = data['motor_encoder']
                gyro  = data['gyroscope'][2]
                print(
                    f"[{elapsed:5.1f}s]  "
                    f"x={x:+.4f}m  y={y:+.4f}m  θ={theta:+.4f}rad ({np.degrees(theta):+.1f}°)"
                    f"  σx={ux:.3f}  speed={speed:+.4f}  enc={enc}  gyro_z={gyro:+.4f}"
                )
                last_print = elapsed

            elapsed_tick = time.perf_counter() - tick_t
            sleep = p_cfg.LOOP_DT - elapsed_tick
            if sleep > 0:
                time.sleep(sleep)
            tick_t = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        sensors.close()

if __name__ == "__main__":
    main()
