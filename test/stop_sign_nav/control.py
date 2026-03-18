"""
control.py  —  Phase 3: Lane following + speed control for the QCar 2
═══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE DOES
───────────────────
Makes the car drive itself:
  1. LaneNet reads the front CSI camera → finds lane markings
  2. Computes lateral offset  (how far left/right of lane centre)
  3. Steering PID corrects heading to stay centred
  4. Speed PID holds a constant forward speed

This is the first phase where the car actually moves autonomously.

CLASSES USED  (verified from real source files)
───────────────────────────────────────────────
  LaneNet    pit/LaneNet/nets.py      — TensorRT lane segmentation
  PID        hal/utilities/control.py — PID(Kp,Ki,Kd,uLimits), update(r,y,dt)
  QCar       pal/products/qcar.py     — read_write_std(throttle, steering)

VERIFIED API (from actual source)
──────────────────────────────────
  LaneNet(imageHeight, imageWidth, rowUpperBound)
  myLaneNet.pre_process(bgr_frame)      → imgTensor
  myLaneNet.predict(imgTensor)          → (binaryPred 256×512 uint8,
                                           instancePred 256×512×3 float)
  myLaneNet.post_process()              → lane_mask (256×512×3 uint8)
  myLaneNet.render(showFPS=True)        → annotated BGR frame

  PID.update(r, y, dt)   r=setpoint, y=measured  →  control output
    Confirmed: e = r - y  (NOT e = y - r)

  QCar.read_write_std(throttle, steering)
    throttle: −0.3 to +0.3   (Quanser maxThrottle = 0.3)
    steering: −π/6 to +π/6   (≈ ±0.524 rad)

LATERAL OFFSET CALCULATION
───────────────────────────
  LaneNet has NO built-in lateral offset method.
  We compute it from binaryPred (256×512 white pixels = lane markings):
    1. Take the bottom third of binaryPred (rows 170–255) — closest to car
    2. Find centroid column of all lane pixels
    3. Subtract image centre (256) → offset in pixels
    4. Convert: pixels × (lane_width_m / lane_width_px) → metres
       lane_width_px ≈ 200 px at 512 width (calibrate if needed)
       lane_width_m  = 0.37 m (QCar lane marking separation)

HOW TO RUN
──────────
  cd ~/test/stop_sign_nav
  python3 control.py

  Car will drive forward in its lane. Press Ctrl+C to stop.
  ALWAYS have someone ready to catch the car or press Ctrl+C.

SAFETY
──────
  MAX_THROTTLE   = 0.10   (very slow for first test — increase gradually)
  EMERGENCY_STOP = Ctrl+C → sets throttle=0 steering=0 in finally block
  If car drifts badly → reduce K_STEERING, check lane markings are visible

PASS CRITERIA
─────────────
  [ ] Car drives forward without veering off lane
  [ ] Speed stays close to V_REF (±0.05 m/s)
  [ ] Lateral offset printed stays within ±0.05 m
  [ ] No crashes in a 10-second run
═══════════════════════════════════════════════════════════════════════════════
"""

import time
import numpy as np
import cv2

from pit.LaneNet.nets      import LaneNet
from hal.utilities.control import PID
from perception            import SensorManager, Config as PerceptionConfig
from estimation            import PoseEstimator, EstimationConfig


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG  —  ALL tunable values in one place
# ═════════════════════════════════════════════════════════════════════════════

class ControlConfig:

    # ── Safety ───────────────────────────────────────────────────────────────
    # Start very slow. Once confirmed working, increase to 0.2 then 0.3.
    MAX_THROTTLE        = 0.10      # absolute ceiling on throttle output
    MAX_STEERING        = np.pi/6   # ±30° — matches Quanser hardware limit

    # ── Speed controller (PI) ─────────────────────────────────────────────
    # Quanser lab values: Kp=0.1, Ki=1.0 — confirmed from qcar_functions.py
    V_REF               = 0.3       # target speed m/s (slow for Phase 3)
    K_SPEED_P           = 0.10
    K_SPEED_I           = 1.0
    K_SPEED_D           = 0.0

    # ── Steering controller (P only for lane centring) ─────────────────────
    # Positive offset (drifted right) → negative steering (steer left)
    # Start with small K_STEERING. Too large = oscillation.
    K_STEERING_P        = 0.8
    K_STEERING_I        = 0.0
    K_STEERING_D        = 0.0

    # ── LaneNet ───────────────────────────────────────────────────────────
    IMAGE_WIDTH         = 640
    IMAGE_HEIGHT        = 480
    # rowUpperBound: rows above this are ignored (sky, bonnet)
    # 228 is the Quanser example value — matches their training data
    ROW_UPPER_BOUND     = 228

    # Lateral offset calibration
    # binaryPred is 512px wide. Lane markings span ~200px at this width.
    # Measured lane width on SDCS track: ~0.37m between marking centres.
    LANE_WIDTH_PX       = 200.0     # pixels between left and right marking
    LANE_WIDTH_M        = 0.37      # metres — measure on your actual track

    # Bottom-third row range for centroid calculation (closest to car)
    CENTROID_ROW_START  = 170       # out of 256 (LaneNet output height)
    CENTROID_ROW_END    = 256

    # Minimum lane pixels needed to trust the offset reading
    MIN_LANE_PIXELS     = 50

    # ── Loop ─────────────────────────────────────────────────────────────
    LOOP_RATE_HZ        = 30
    LOOP_DT             = 1.0 / 30

    # ── Test duration ────────────────────────────────────────────────────
    RUN_DURATION_S      = 10.0      # stop after this many seconds
    PRINT_INTERVAL_S    = 0.25      # print status every 250 ms


# ═════════════════════════════════════════════════════════════════════════════
#  LANE DETECTOR
#  Wraps LaneNet and computes lateral offset from binaryPred pixel centroids
# ═════════════════════════════════════════════════════════════════════════════

class LaneDetector:
    """
    Wraps LaneNet. Each tick:
      1. pre_process(frame) → imgTensor
      2. predict(imgTensor) → binaryPred, instancePred
      3. compute_lateral_offset(binaryPred) → metres

    lateral_offset > 0  →  car is to the RIGHT of lane centre → steer LEFT
    lateral_offset < 0  →  car is to the LEFT  of lane centre → steer RIGHT
    """

    def __init__(self, config: ControlConfig):
        self.cfg   = config
        self._net  = None
        self._px_to_m = (config.LANE_WIDTH_M / config.LANE_WIDTH_PX)

    def open(self):
        """Load LaneNet TensorRT engine. Takes ~2s on first run."""
        print("  Loading LaneNet TensorRT engine...")
        self._net = LaneNet(
            imageHeight  = self.cfg.IMAGE_HEIGHT,
            imageWidth   = self.cfg.IMAGE_WIDTH,
            rowUpperBound= self.cfg.ROW_UPPER_BOUND,
        )
        print("  LaneNet: OK")

    def detect(self, bgr_frame: np.ndarray) -> dict:
        """
        Run lane detection on one BGR frame.

        Parameters
        ──────────
        bgr_frame : ndarray (H, W, 3) uint8  from Camera2D.imageData

        Returns
        ───────
        dict:
          lateral_offset_m   float   metres (+ve = drifted right)
          lane_pixels        int     number of valid lane pixels found
          lane_detected      bool    True if enough pixels for reliable offset
          binary_pred        ndarray (256, 512) uint8  raw lane mask
          annotated_frame    ndarray (H, W, 3) uint8   visualisation
        """
        # Step 1: pre-process (resize to 256×512, normalise)
        img_tensor = self._net.pre_process(bgr_frame)

        # Step 2: TensorRT inference
        # Returns: binaryPred (256×512 uint8, 255=lane), instancePred (float)
        binary_pred, instance_pred = self._net.predict(img_tensor)

        # Step 3: compute lateral offset from binary_pred centroid
        offset_m, n_pixels = self._compute_offset(binary_pred)

        # Step 4: render annotated frame for debugging
        annotated = self._net.render(showFPS=True)

        return {
            'lateral_offset_m':  offset_m,
            'lane_pixels':       n_pixels,
            'lane_detected':     n_pixels >= self.cfg.MIN_LANE_PIXELS,
            'binary_pred':       binary_pred,
            'annotated_frame':   annotated,
        }

    def _compute_offset(self, binary_pred: np.ndarray):
        """
        Compute lateral offset from the centroid of lane pixels
        in the bottom third of the binaryPred image.

        binaryPred is 256 rows × 512 cols.
        Image centre column = 256.
        Positive column offset = car is right of lane centre.

        Returns (offset_metres, n_pixels)
        """
        # Crop to bottom rows — closest part of road, most reliable
        roi = binary_pred[
            self.cfg.CENTROID_ROW_START : self.cfg.CENTROID_ROW_END,
            :
        ]

        # Find all lane pixels in ROI
        lane_rows, lane_cols = np.where(roi == 255)
        n_pixels = len(lane_cols)

        if n_pixels < self.cfg.MIN_LANE_PIXELS:
            # Not enough lane visible — return zero (hold current steering)
            return 0.0, n_pixels

        # Column centroid of all lane pixels
        centroid_col = float(lane_cols.mean())

        # Image centre column (LaneNet output is 512px wide)
        image_centre = 512.0 / 2.0   # = 256.0

        # Positive = centroid is right of centre = car drifted right
        offset_px = centroid_col - image_centre

        # Convert pixels → metres
        offset_m = offset_px * self._px_to_m

        return float(offset_m), n_pixels

    def terminate(self):
        """No explicit close needed for LaneNet — GPU memory freed by GC."""
        self._net = None
        print("  LaneNet: released")


# ═════════════════════════════════════════════════════════════════════════════
#  DRIVE CONTROLLER
#  Speed PID + Steering PID wired to QCar actuators
# ═════════════════════════════════════════════════════════════════════════════

class DriveController:
    """
    Two independent PID controllers:
      speed_pid   : throttle = PID(V_REF, motorTach_mps, dt)
      steering_pid: steering = PID(0.0,   lateral_offset_m, dt)
                    setpoint=0 means "be centred in the lane"

    Both use hal.utilities.control.PID confirmed API:
      PID.update(r, y, dt)  where r=setpoint, y=measured, e=r-y
    """

    def __init__(self, config: ControlConfig):
        self.cfg = config

        self.speed_pid = PID(
            Kp      = config.K_SPEED_P,
            Ki      = config.K_SPEED_I,
            Kd      = config.K_SPEED_D,
            uLimits = (-config.MAX_THROTTLE, config.MAX_THROTTLE),
        )

        self.steering_pid = PID(
            Kp      = config.K_STEERING_P,
            Ki      = config.K_STEERING_I,
            Kd      = config.K_STEERING_D,
            uLimits = (-config.MAX_STEERING, config.MAX_STEERING),
        )

    def reset(self):
        self.speed_pid.reset()
        self.steering_pid.reset()

    def update(self,
               current_speed_mps: float,
               lateral_offset_m: float,
               lane_detected: bool,
               dt: float) -> tuple:
        """
        Compute throttle and steering commands.

        Parameters
        ──────────
        current_speed_mps : float   current wheel speed in m/s
                            (convert motorTach rad/s via CPS_TO_MPS)
        lateral_offset_m  : float   metres right of lane centre
        lane_detected     : bool    if False, hold current steering
        dt                : float   seconds since last update

        Returns
        ───────
        (throttle, steering)
          throttle : float  −MAX to +MAX  (clamped by PID uLimits)
          steering : float  −π/6 to +π/6 (clamped by PID uLimits)
        """
        # Speed PID: error = V_REF - current_speed
        throttle = self.speed_pid.update(
            r  = self.cfg.V_REF,
            y  = current_speed_mps,
            dt = dt,
        )

        # Steering PID: error = 0 - lateral_offset (setpoint = centred)
        # If lane not detected, send 0 steering (go straight)
        if lane_detected:
            steering = self.steering_pid.update(
                r  = 0.0,              # setpoint: be at lane centre
                y  = lateral_offset_m, # measured: how far right we are
                dt = dt,
            )
        else:
            steering = 0.0
            self.steering_pid.reset()  # reset integrator to avoid windup

        return float(throttle), float(steering)


# ═════════════════════════════════════════════════════════════════════════════
#  MOTOR TACH → WHEEL SPEED CONVERSION
#  Same constants as estimation.py — verified from qcar_config.py
# ═════════════════════════════════════════════════════════════════════════════

_WHEEL_RADIUS      = 0.066 / 2
_PIN_TO_SPUR_RATIO = (13.0 * 19.0) / (70.0 * 37.0)
_ENCODER_CPR       = 720
_ENCODER_PPR       = 4
_TACH_TO_MPS = _PIN_TO_SPUR_RATIO * _WHEEL_RADIUS
# motorTach is in rad/s of the MOTOR SHAFT.
# wheel_speed_mps = motorTach_rad_s * PIN_TO_SPUR * WHEEL_RADIUS


def tach_to_mps(motor_tach_rad_s: float) -> float:
    """Convert motor shaft rad/s → wheel linear speed m/s."""
    return float(motor_tach_rad_s) * _TACH_TO_MPS


# ═════════════════════════════════════════════════════════════════════════════
#  STATUS PRINTER
# ═════════════════════════════════════════════════════════════════════════════

def print_status(elapsed: float,
                 throttle: float,
                 steering: float,
                 speed_mps: float,
                 offset_m: float,
                 lane_px: int,
                 pose: np.ndarray):
    x, y, theta = pose
    print(
        f"[{elapsed:6.2f}s] "
        f"throttle={throttle:+.3f}  steer={steering:+.4f}rad  "
        f"speed={speed_mps:+.3f}m/s  "
        f"offset={offset_m:+.4f}m  "
        f"lane_px={lane_px:4d}  "
        f"pose=({x:+.3f},{y:+.3f},{np.degrees(theta):+.1f}°)"
    )


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN  —  autonomous lane following loop
# ═════════════════════════════════════════════════════════════════════════════

def main():
    c_cfg   = ControlConfig()
    p_cfg   = PerceptionConfig()
    e_cfg   = EstimationConfig()

    # ── Open all subsystems ───────────────────────────────────────────────
    sensors   = SensorManager(p_cfg)
    estimator = PoseEstimator(e_cfg)
    detector  = LaneDetector(c_cfg)
    controller= DriveController(c_cfg)

    print("═" * 60)
    print("  QCar 2 — Phase 3: Lane Following")
    print(f"  Speed target: {c_cfg.V_REF} m/s  "
          f"Max throttle: {c_cfg.MAX_THROTTLE}")
    print(f"  Duration: {c_cfg.RUN_DURATION_S:.0f}s. Ctrl+C to stop.")
    print("  SAFETY: keep hand near car for first run.")
    print("═" * 60)

    sensors.open()          # opens QCar, CSI cameras, RealSense, LiDAR
    detector.open()         # loads LaneNet TensorRT engine
    estimator.reset()       # initialise EKF at origin
    controller.reset()      # zero PID integrators

    # Retrieve the QCar object from SensorManager so we can write to it
    # SensorManager._qcar is the QCar instance opened in _open_qcar()
    qcar = sensors._qcar

    throttle = 0.0
    steering = 0.0

    try:
        start_t    = time.perf_counter()
        tick_start = start_t
        last_t     = start_t
        last_print = 0.0

        while True:
            elapsed = time.perf_counter() - start_t
            if elapsed >= c_cfg.RUN_DURATION_S:
                break

            # ── 1. Read all sensors ───────────────────────────────────────
            data = sensors.read()

            # ── 2. Compute dt ─────────────────────────────────────────────
            now = time.perf_counter()
            dt  = max(now - last_t, 1e-4)   # guard against zero dt
            last_t = now

            # ── 3. Update pose estimate ───────────────────────────────────
            pose = estimator.update(
                sensor_data  = data,
                dt           = dt,
                steering_rad = steering,    # feed last steering command back
            )

            # ── 4. Lane detection ─────────────────────────────────────────
            # Use front CSI camera (id "0")
            front_frame = data['csi_frames']['0']
            lane = detector.detect(front_frame)

            # ── 5. Convert motor tach → wheel speed ───────────────────────
            speed_mps = tach_to_mps(data['motor_speed'])

            # ── 6. Compute control commands ───────────────────────────────
            throttle, steering = controller.update(
                current_speed_mps = speed_mps,
                lateral_offset_m  = lane['lateral_offset_m'],
                lane_detected     = lane['lane_detected'],
                dt                = dt,
            )

            # ── 7. Send commands to hardware ──────────────────────────────
            # read_write_std reads sensors AND writes commands atomically.
            # This is the call that actually drives the car.
            qcar.read_write_std(
                throttle = throttle,
                steering = steering,
            )

            # ── 8. Print status ───────────────────────────────────────────
            if elapsed - last_print >= c_cfg.PRINT_INTERVAL_S:
                print_status(
                    elapsed   = elapsed,
                    throttle  = throttle,
                    steering  = steering,
                    speed_mps = speed_mps,
                    offset_m  = lane['lateral_offset_m'],
                    lane_px   = lane['lane_pixels'],
                    pose      = pose,
                )
                last_print = elapsed

            # ── 9. Soft real-time pacing ──────────────────────────────────
            elapsed_tick = time.perf_counter() - tick_start
            sleep_time   = c_cfg.LOOP_DT - elapsed_tick
            if sleep_time > 0:
                time.sleep(sleep_time)
            tick_start = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Ctrl+C).")

    finally:
        # ── SAFETY: always zero actuators before closing ──────────────────
        print("\n[SAFETY] Zeroing throttle and steering...")
        try:
            qcar.read_write_std(throttle=0.0, steering=0.0)
            qcar.read_write_std(throttle=0.0, steering=0.0)  # send twice
        except Exception:
            pass

        detector.terminate()
        sensors.close()
        print("Stopped cleanly.")


# ═════════════════════════════════════════════════════════════════════════════
#  TUNING GUIDE  —  read before first run
# ═════════════════════════════════════════════════════════════════════════════
"""
STEP-BY-STEP FIRST RUN:

1. Place car on track with lane markings clearly visible.
   The front CSI camera (id="0") must see the lane markings.

2. Run with MAX_THROTTLE=0.10, V_REF=0.3 first.
   Watch 'lane_px' in the printout — should be > 100 regularly.
   Watch 'offset' — should hover near 0.0 when centred.

3. IF lane_px is always small (< 50):
   - rowUpperBound might be wrong. Try ROW_UPPER_BOUND=150 or 200.
   - Lighting might be poor. Try moving to better-lit area.
   - LaneNet model might not match your track markings.

4. IF car oscillates left-right:
   - Reduce K_STEERING_P (try 0.4, then 0.2)
   - Add K_STEERING_D (try 0.05) to dampen oscillation

5. IF car doesn't respond to lane:
   - Increase K_STEERING_P (try 1.2, then 1.5)

6. IF speed is unstable:
   - Reduce K_SPEED_I (try 0.5) to slow the integrator
   - Reduce V_REF (try 0.2)

7. Once stable, increase MAX_THROTTLE → 0.15, then V_REF → 0.5
"""

if __name__ == "__main__":
    main()
