"""
estimation.py  —  Phase 2: Pose estimation for the QCar 2
═══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE DOES
───────────────────
Takes the raw sensor dict from perception.py and produces a smooth, fused
estimate of the car's pose every tick:

    pose = [x (m),  y (m),  θ (rad)]
           forward  left    heading (0 = facing +x, CCW positive)

Uses an Extended Kalman Filter (EKF) from hal.utilities.estimation.
Motion model  = kinematic bicycle model  (encoder speed + steering angle)
No GPS/LiDAR correction in Phase 2 — pure dead-reckoning.
Corrections from LiDAR scan-matching are added in Phase 3+.

CLASSES USED  (verified from real source files)
───────────────────────────────────────────────
  EKF               hal/utilities/estimation.py   — nonlinear predict+correct
  wrap_to_pi        pal/utilities/math.py          — angle normalisation

PHYSICAL CONSTANTS  (from pal/products/qcar_config.py)
───────────────────────────────────────────────────────
  WHEEL_RADIUS      = 0.066/2  = 0.033 m
  WHEEL_BASE        = 0.256 m   (front-to-rear axle)  — lab uses 0.257, we use config
  PIN_TO_SPUR_RATIO = (13×19)/(70×37) = 0.09545...
  ENCODER_CPR       = 720 counts/rev  (from qcar.py line 140)
  ENCODER_PPR       = 4              (quadrature: 4 pulses per count)
  CPS_TO_MPS        = 1/(CPR×PPR) × PIN_TO_SPUR × 2π × WHEEL_RADIUS

KINEMATIC BICYCLE MODEL
───────────────────────
  State  x = [x_pos, y_pos, θ]ᵀ
  Input  u = [v (m/s), δ (rad)]   v=wheel speed, δ=steering angle

  ẋ = v · cos(θ)
  ẏ = v · sin(θ)
  θ̇ = v · tan(δ) / L

  Discretised (Euler, small dt):
  x_{k+1} = x_k + v·cos(θ_k)·dt
  y_{k+1} = y_k + v·sin(θ_k)·dt
  θ_{k+1} = θ_k + v·tan(δ)·dt / L   (then wrapped to [-π, π])

JACOBIAN  (needed by EKF nonlinear predict)
───────────────────────────────────────────
  F = ∂f/∂x  =  I + dt · [[0,  0,  -v·sin(θ)],
                            [0,  0,  +v·cos(θ)],
                            [0,  0,  0        ]]

HOW TO RUN (standalone test)
──────────────────────────────
  cd ~/test/stop_sign_nav
  python3 estimation.py

  Drives the estimator with live sensor data for 15 s.
  Prints pose every 0.5 s. Push the car by hand and watch x/y change.

PASS CRITERIA BEFORE PHASE 3
──────────────────────────────
  [ ] Push car ~1 m forward  → x increases by ~1.0
  [ ] Rotate car ~90° CCW    → θ increases by ~π/2 (≈ 1.57 rad)
  [ ] Stationary: pose drift  < 0.02 m over 10 s
  [ ] Loop timing mean        < 50 ms
═══════════════════════════════════════════════════════════════════════════════
"""

import time
import numpy as np

from hal.utilities.estimation import EKF
from pal.utilities.math       import wrap_to_pi

# perception.py lives in the same folder — import SensorManager from it
from perception import SensorManager, Config as PerceptionConfig


# ═════════════════════════════════════════════════════════════════════════════
#  PHYSICAL CONSTANTS  —  verified from qcar_config.py
# ═════════════════════════════════════════════════════════════════════════════

WHEEL_RADIUS      = 0.066 / 2           # metres   (= 0.033 m)
WHEEL_BASE        = 0.256               # metres   front-to-rear axle
PIN_TO_SPUR_RATIO = (13.0 * 19.0) / (70.0 * 37.0)   # = 0.09545...
ENCODER_CPR       = 720                 # counts per revolution (qcar.py line 140)
ENCODER_PPR       = 4                   # quadrature pulses per count

# Converts motor shaft counts-per-second → wheel linear speed (m/s)
# Formula from qcar.py line 149:
#   CPS_TO_MPS = (1/(CPR*PPR)) * PIN_TO_SPUR * 2*pi * WHEEL_RADIUS
CPS_TO_MPS = (1.0 / (ENCODER_CPR * ENCODER_PPR)) * PIN_TO_SPUR_RATIO \
             * 2.0 * np.pi * WHEEL_RADIUS


# ═════════════════════════════════════════════════════════════════════════════
#  EKF CONFIG  —  tune these if drift is too high
# ═════════════════════════════════════════════════════════════════════════════

class EstimationConfig:

    # Initial pose [x, y, θ]  — car starts at origin, facing +x
    INITIAL_POSE        = np.zeros((3, 1))
    INITIAL_COVARIANCE  = np.eye(3) * 0.01     # small: we're confident at start

    # Process noise Q — how much we distrust the motion model each tick
    # Larger = estimator reacts faster to real motion but noisier
    # Units: [m², m², rad²] per tick
    Q_x     = 0.01
    Q_y     = 0.01
    Q_theta = 0.01

    # Measurement noise R — how much we distrust the encoder speed reading
    # Only one measurement: wheel speed (m/s)
    # Larger = estimator trusts encoder less, relies more on model prediction
    R_speed = 0.05

    # Steering angle — QCar 2 has no steering encoder.
    # We use the last commanded steering angle from the control layer.
    # In Phase 2 (no driving) this is 0. Phase 3+ will pass the real value.
    DEFAULT_STEERING_RAD = 0.0

    # Gyroscope — used to cross-check heading rate (not in EKF yet, Phase 3+)
    # Z-axis gyro drift bias (rad/s) — from Phase 1 data: mean ≈ 0.024 rad/s
    GYRO_BIAS_Z = 0.024   # subtract this before using gyro for heading


# ═════════════════════════════════════════════════════════════════════════════
#  BICYCLE MODEL FUNCTIONS  —  passed to EKF as f and J_f
# ═════════════════════════════════════════════════════════════════════════════

def bicycle_f(x_hat: np.ndarray, u: list, dt: float) -> np.ndarray:
    """
    Kinematic bicycle model — state transition function f(x, u, dt).

    Parameters
    ──────────
    x_hat : ndarray (3,1)   current state estimate [x, y, θ]
    u     : [v, δ]          v = wheel speed (m/s), δ = steering angle (rad)
    dt    : float           time step (seconds)

    Returns
    ───────
    ndarray (3,1)  predicted next state
    """
    x, y, theta = float(x_hat[0]), float(x_hat[1]), float(x_hat[2])
    v     = float(u[0])
    delta = float(u[1])

    x_new     = x     + v * np.cos(theta) * dt
    y_new     = y     + v * np.sin(theta) * dt
    theta_new = theta + v * np.tan(delta) / WHEEL_BASE * dt

    # Wrap heading to [-π, π] to prevent unbounded growth
    theta_new = float(wrap_to_pi(theta_new))

    return np.array([[x_new], [y_new], [theta_new]])


def bicycle_J_f(x_hat: np.ndarray, u: list, dt: float) -> np.ndarray:
    """
    Jacobian of the bicycle model  ∂f/∂x  —  needed by EKF predict step.

    F = I + dt · [[0,  0,  -v·sin(θ)],
                  [0,  0,  +v·cos(θ)],
                  [0,  0,   0       ]]

    Parameters  (same signature as bicycle_f)
    Returns ndarray (3,3)
    """
    theta = float(x_hat[2])
    v     = float(u[0])

    F = np.eye(3)
    F[0, 2] = -v * np.sin(theta) * dt
    F[1, 2] = +v * np.cos(theta) * dt
    # F[2, 2] remains 1 (theta depends on itself, not on x or y)
    return F


# ═════════════════════════════════════════════════════════════════════════════
#  POSE ESTIMATOR
# ═════════════════════════════════════════════════════════════════════════════

class PoseEstimator:
    """
    Wraps the HAL EKF with QCar-specific motion model and sensor conversion.

    Usage
    ─────
        estimator = PoseEstimator(EstimationConfig())
        estimator.reset()
        ...
        # Each loop tick:
        pose = estimator.update(sensor_data, dt, steering_rad=0.0)
        x, y, theta = pose
    """

    def __init__(self, config: EstimationConfig):
        self.cfg  = config
        self._ekf = None
        self._last_encoder = None   # for delta-encoder speed calculation
        self._last_t       = None

    def reset(self, initial_pose: np.ndarray = None):
        """
        Initialise (or re-initialise) the EKF.
        Call once before the loop, or whenever you want to reset the origin.

        initial_pose : ndarray (3,) or (3,1)  [x, y, θ]
                       None → uses EstimationConfig.INITIAL_POSE
        """
        cfg = self.cfg
        x0  = initial_pose if initial_pose is not None else cfg.INITIAL_POSE
        x0  = np.array(x0, dtype=float).reshape((3, 1))

        # Measurement model:
        # We measure wheel speed (m/s) — a linear function of state?
        # Actually speed is a control input, not a state measurement.
        # For Phase 2 dead-reckoning we have NO external measurement
        # corrections (no GPS, no LiDAR yet).
        #
        # EKF with only prediction (no correction) is identical to
        # simple odometry integration — but we keep the EKF structure
        # so Phase 3+ can add corrections with zero structural changes.
        #
        # We use a linear measurement model:  y = C · x + noise
        # C = identity  (we could "measure" x, y, θ directly from GPS later)
        # For now we never call .correct() — the EKF just predicts.

        self._ekf = EKF(
            x_0 = x0,
            P_0 = cfg.INITIAL_COVARIANCE,
            Q   = np.diagflat([cfg.Q_x, cfg.Q_y, cfg.Q_theta]),
            R   = np.diagflat([cfg.R_speed, cfg.R_speed, cfg.R_speed]),
            f   = bicycle_f,
            J_f = bicycle_J_f,
            C   = np.eye(3),    # identity output matrix (for future corrections)
        )

        self._last_encoder = None
        self._last_t       = None
        print("[PoseEstimator] EKF initialised.  "
              f"Initial pose: x={float(x0[0]):.3f}  "
              f"y={float(x0[1]):.3f}  "
              f"θ={float(x0[2]):.3f} rad")

    def update(self,
               sensor_data: dict,
               dt: float,
               steering_rad: float = 0.0) -> np.ndarray:
        """
        Run one EKF predict step using the latest sensor data.

        Parameters
        ──────────
        sensor_data  : dict from SensorManager.read()
        dt           : float  time since last update (seconds)
        steering_rad : float  current steering angle (rad)
                       0.0 in Phase 2 (car stationary / no control yet)
                       Phase 3+ passes the real commanded steering angle

        Returns
        ───────
        ndarray (3,)  current pose estimate [x (m), y (m), θ (rad)]
        """
        # ── Convert encoder counts → wheel speed (m/s) ────────────────────
        v_wheel = self._encoder_to_speed(sensor_data, dt)

        # ── Build control input vector ────────────────────────────────────
        # u = [wheel_speed (m/s),  steering_angle (rad)]
        u = [v_wheel, steering_rad]

        # ── EKF predict step ──────────────────────────────────────────────
        # This calls bicycle_f and bicycle_J_f internally.
        # predict(u, dt) → updates self._ekf.x_hat and self._ekf.P
        if dt > 0:
            self._ekf.predict(u, dt)

        # ── Return flat pose array ────────────────────────────────────────
        pose = self._ekf.x_hat.flatten()
        # Ensure heading stays in [-π, π] after every step
        pose[2] = float(wrap_to_pi(pose[2]))
        return pose

    def correct_with_pose(self, measured_pose: np.ndarray):
        """
        EKF correction step using a direct pose measurement.
        Not used in Phase 2. Called in Phase 3+ when LiDAR scan-matching
        provides a reliable [x, y, θ] estimate.

        measured_pose : ndarray (3,)  [x, y, θ]
        """
        y = np.array(measured_pose, dtype=float).reshape((3, 1))
        self._ekf.correct(y)
        # Re-wrap heading after correction
        self._ekf.x_hat[2, 0] = float(wrap_to_pi(self._ekf.x_hat[2, 0]))

    @property
    def pose(self) -> np.ndarray:
        """Current pose estimate as flat (3,) array [x, y, θ]."""
        p = self._ekf.x_hat.flatten().copy()
        p[2] = float(wrap_to_pi(p[2]))
        return p

    @property
    def covariance(self) -> np.ndarray:
        """Current 3×3 covariance matrix P."""
        return self._ekf.P.copy()

    @property
    def uncertainty(self) -> np.ndarray:
        """Diagonal of P as (3,) — position and heading uncertainty (std dev)."""
        return np.sqrt(np.diag(self._ekf.P))

    # ── PRIVATE ───────────────────────────────────────────────────────────

    def _encoder_to_speed(self, sensor_data: dict, dt: float) -> float:
        """
        Convert cumulative motorEncoder counts to wheel linear speed (m/s).

        The encoder counts are CUMULATIVE — they increase monotonically.
        Speed = (delta_counts / dt) × CPS_TO_MPS

        Why delta counts and not motorTach directly?
        motorTach is already in rad/s (motor shaft). We could use it, but
        deriving speed from encoder deltas is more robust because:
          1. It averages over the full dt interval rather than one sample
          2. It's less sensitive to the HIL read rate jitter
          3. It matches how the Quanser labs compute speed internally

        Both approaches work — this is the more accurate one.
        """
        current_counts = sensor_data['motor_encoder']

        if self._last_encoder is None or dt <= 0:
            # First tick — no delta available yet, assume stationary
            self._last_encoder = current_counts
            return 0.0

        delta_counts = current_counts - self._last_encoder
        self._last_encoder = current_counts

        # Counts per second → metres per second
        speed_mps = (delta_counts / dt) * CPS_TO_MPS
        return float(speed_mps)


# ═════════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC PRINTER
# ═════════════════════════════════════════════════════════════════════════════

def print_pose_summary(pose: np.ndarray,
                       uncertainty: np.ndarray,
                       sensor_data: dict,
                       elapsed: float,
                       v_wheel: float = 0.0):
    """Print a one-block pose snapshot."""
    x, y, theta = pose
    sx, sy, st  = uncertainty
    accel = sensor_data['accelerometer']
    gyro  = sensor_data['gyroscope']

    # Gyro z — corrected for bias
    gyro_z_corrected = gyro[2] - EstimationConfig.GYRO_BIAS_Z

    print(
        f"\n[{elapsed:6.2f}s] ─────────────────────────────────────────────\n"
        f"  Pose    : x={x:+.4f} m   y={y:+.4f} m   θ={theta:+.4f} rad "
        f"({np.degrees(theta):+.1f}°)\n"
        f"  Uncert  : σx={sx:.4f}   σy={sy:.4f}   σθ={st:.4f}\n"
        f"  Encoder : speed={sensor_data['motor_speed']:+.4f} rad/s  "
        f"counts={sensor_data['motor_encoder']}\n"
        f"  IMU     : accel_z={accel[2]:+.3f} m/s²  "
        f"gyro_z={gyro[2]:+.5f} rad/s  "
        f"(corrected: {gyro_z_corrected:+.5f})\n"
        f"  Battery : {sensor_data['battery_voltage']:.2f} V"
    )


def print_final_report(pose_history: list, dt_history: list):
    """Statistics after the test run."""
    if len(pose_history) < 2:
        print("Not enough data.")
        return

    poses = np.array(pose_history)
    dts   = np.array(dt_history)

    x_range   = poses[:, 0].max() - poses[:, 0].min()
    y_range   = poses[:, 1].max() - poses[:, 1].min()
    th_range  = poses[:, 2].max() - poses[:, 2].min()
    final     = poses[-1]

    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  PHASE 2 FINAL REPORT  —  {len(poses)} ticks")
    print(sep)
    print(f"\n  Final pose  : x={final[0]:+.4f} m  y={final[1]:+.4f} m  "
          f"θ={final[2]:+.4f} rad ({np.degrees(final[2]):+.1f}°)")
    print(f"  Pose range  : Δx={x_range:.4f} m  Δy={y_range:.4f} m  "
          f"Δθ={th_range:.4f} rad")
    print(f"\n  Loop timing : mean={dts.mean()*1000:.1f} ms  "
          f"std={dts.std()*1000:.2f} ms  "
          f"max={dts.max()*1000:.1f} ms")

    print(f"\n{sep}")
    print("  CHECKLIST:")

    stationary_drift = np.sqrt(final[0]**2 + final[1]**2)
    rows = [
        ("PASS" if stationary_drift < 0.05 else "WARN",
         f"stationary drift < 0.05 m  (got {stationary_drift:.4f} m)"),
        ("PASS" if dts.mean() < 0.05 else "WARN",
         f"loop mean < 50 ms  (got {dts.mean()*1000:.1f} ms)"),
        ("NOTE",
         "push car 1 m → x should increase ~1.0 (manual test)"),
        ("NOTE",
         "rotate car 90° CCW → θ should increase ~1.57 rad (manual test)"),
    ]
    for status, text in rows:
        icon = "✓" if status == "PASS" else ("!" if status == "NOTE" else "?")
        print(f"  [{status}] {icon}  {text}")
    print(sep)

    all_auto = all(r[0] in ("PASS", "NOTE") for r in rows)
    if all_auto:
        print("\n  ✓ Auto-checks passed.")
        print("  Run the manual push/rotate tests above to confirm odometry.")
        print("  If those pass → ready for Phase 3 (lane following + control)\n")
    else:
        print("\n  Some checks need attention — see above.\n")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN  —  15-second live test
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # Reuse Phase 1 sensor manager — identical interface
    p_cfg   = PerceptionConfig()
    sensors = SensorManager(p_cfg)
    e_cfg   = EstimationConfig()
    estimator = PoseEstimator(e_cfg)

    TEST_DURATION_S  = 15.0
    PRINT_INTERVAL_S = 0.5

    pose_history = []
    dt_history   = []
    last_print   = 0.0

    print("═" * 60)
    print("  QCar 2 — Phase 2: Pose Estimation Test")
    print(f"  Running {TEST_DURATION_S:.0f}s after warm-up. Ctrl+C to stop early.")
    print("  Car should be STATIONARY. Drift < 0.05 m over 15 s = PASS.")
    print("═" * 60)

    sensors.open()
    estimator.reset()

    try:
        start_t    = time.perf_counter()
        tick_start = start_t
        last_t     = start_t

        while True:
            elapsed = time.perf_counter() - start_t
            if elapsed >= TEST_DURATION_S:
                break

            # ── Read sensors ──────────────────────────────────────────────
            data = sensors.read()

            # ── Compute dt ────────────────────────────────────────────────
            now = time.perf_counter()
            dt  = now - last_t
            last_t = now

            # ── Update pose estimate ──────────────────────────────────────
            # steering_rad=0.0 because the car is stationary in Phase 2
            # Phase 3+ replaces this with the real commanded steering angle
            pose = estimator.update(
                sensor_data  = data,
                dt           = dt,
                steering_rad = EstimationConfig.DEFAULT_STEERING_RAD,
            )

            pose_history.append(pose.copy())
            dt_history.append(dt)

            # ── Print summary ─────────────────────────────────────────────
            if elapsed - last_print >= PRINT_INTERVAL_S:
                print_pose_summary(
                    pose        = pose,
                    uncertainty = estimator.uncertainty,
                    sensor_data = data,
                    elapsed     = elapsed,
                )
                last_print = elapsed

            # ── Soft real-time pacing ─────────────────────────────────────
            elapsed_tick = time.perf_counter() - tick_start
            sleep_time   = p_cfg.LOOP_DT - elapsed_tick
            if sleep_time > 0:
                time.sleep(sleep_time)
            tick_start = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

    finally:
        sensors.close()

    print_final_report(pose_history, dt_history)


if __name__ == "__main__":
    main()
