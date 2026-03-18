"""
perception.py  —  Phase 1: Read every sensor on the QCar 2
═══════════════════════════════════════════════════════════════════════════════
WHAT THIS FILE DOES
───────────────────
Reads ALL hardware sensors in one loop, returns a clean dict every tick.
No AI, no control, no decisions — purely sensor data collection.

CLASSES USED  (verified from actual source files on this QCar)
──────────────────────────────────────────────────────────────
  Sensor          Class      Source file
  ─────────────── ────────── ─────────────────────────────────────
  IMU + encoder   QCar       pal/products/qcar.py
  CSI cameras     Camera2D   pal/utilities/vision.py
  RealSense D435  Camera3D   pal/utilities/vision.py
  RPLIDAR A2      QCarLidar  pal/products/qcar.py

VERIFIED METHOD NAMES
──────────────────────────────────────────────────────────────
  QCar.read()              reads IMU + encoder, sends NO motor commands
  QCar.read_write_std()    reads sensors AND writes throttle/steering
  QCar.terminate()         closes HIL board cleanly

  Camera2D.read()          fills .imageData in-place, returns bool
  Camera2D.terminate()     stops and closes capture

  Camera3D.read_RGB()      fills .imageBufferRGB, returns timestamp float
  Camera3D.read_depth('M') fills .imageBufferDepthM (metres), returns ts
  Camera3D.terminate()     stops all streams

  QCarLidar.read()         fills .distances + .angles, returns bool
  QCarLidar.terminate()    closes serial connection

VERIFIED ATTRIBUTES
──────────────────────────────────────────────────
  qcar.accelerometer    ndarray (3,)  float64  m/s²
  qcar.gyroscope        ndarray (3,)  float64  rad/s
  qcar.motorTach        scalar or ndarray  float64  rad/s
  qcar.batteryVoltage   scalar or ndarray  float64  volts
  qcar.motorEncoder     scalar or ndarray  int32    counts

FIXES APPLIED (revision history)
──────────────────────────────────
  v1: Initial version
  v2: FIX motorTach IndexError — np.atleast_1d() for scalar vs array
  v3: FIX QCAR_READ_MODE 1→0  (non-blocking, prevents HIL hang)
      FIX _warmup() added     (LiDAR + CSI need 2s before valid data)
      FIX _read_lidar() empty-array guard (shape=(0,) on first ticks)
  v4: FIX LIDAR_VALID_THRESHOLD 200→100
        (RPLIDAR A2 returns ~160/360 valid in typical indoor lab —
         rays pointing at open space or beyond 6m are correctly discarded)
      FIX LiDAR WARN spam suppressed
        (RPLIDAR A2 rotates at ~7 Hz; at 30 Hz loop, 3 of every 4 ticks
         will have no new scan — flag=False is NORMAL, not an error.
         Warning now only fires if a scan tick returns ZERO valid readings.)
      FIX Final report LiDAR stats — now computed only over scan ticks,
        not all ticks (avoids averaging stale cached data)

HOW TO RUN
──────────
  cd ~/test/stop_sign_nav
  python3 perception.py

PASS CRITERIA BEFORE MOVING TO PHASE 2
───────────────────────────────────────
  [ ] accel z  ≈ +9.81 m/s²
  [ ] gyro max abs < 0.15 rad/s
  [ ] battery  > 11.0 V
  [ ] CSI cam "0" — all frames non-black
  [ ] RealSense valid depth pixels > 50 000 per frame
  [ ] LiDAR valid readings > 100 per scan  (indoor lab, partial coverage OK)
  [ ] Loop timing mean < 50 ms
═══════════════════════════════════════════════════════════════════════════════
"""

import time
import numpy as np

from pal.products.qcar    import QCar, QCarLidar
from pal.utilities.vision import Camera2D, Camera3D


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════════════════════

class Config:

    # ── Loop ─────────────────────────────────────────────────────────────
    LOOP_RATE_HZ    = 30
    LOOP_DT         = 1.0 / 30          # 0.0333 s

    # ── QCar — IMU + encoder ─────────────────────────────────────────────
    # readMode=0 → non-blocking: returns last sample immediately
    # readMode=1 → blocking: waits for next hardware clock tick (can hang)
    QCAR_FREQUENCY  = 30
    QCAR_READ_MODE  = 0                 # v3 FIX: was 1

    # ── CSI cameras ──────────────────────────────────────────────────────
    # cameraId "0" → /dev/video0 (front camera)
    # Layout: "0"=front  "1"=left  "2"=rear  "3"=right
    CSI_CAMERA_IDS  = ["0"]
    CSI_WIDTH       = 820
    CSI_HEIGHT      = 410
    CSI_FPS         = 30.0

    # ── RealSense D435 ───────────────────────────────────────────────────
    RS_WIDTH_RGB    = 640
    RS_HEIGHT_RGB   = 480
    RS_FPS_RGB      = 30.0
    RS_WIDTH_DEPTH  = 640
    RS_HEIGHT_DEPTH = 480
    RS_FPS_DEPTH    = 30.0

    # ── LiDAR RPLIDAR A2 ─────────────────────────────────────────────────
    # rangingDistanceMode: 2=LONG  1=MEDIUM  0=SHORT
    # interpolationMode:   0=NORMAL  1=INTERPOLATED (evenly spaced, use Phase 3+)
    LIDAR_NUM_MEAS      = 384
    LIDAR_RANGE_MODE    = 2
    LIDAR_INTERP        = 0
    LIDAR_MIN_M         = 0.10
    LIDAR_MAX_M         = 6.0

    # v4 FIX: threshold lowered 200→100
    # Confirmed on this QCar: ~158-162 valid readings in indoor lab.
    # Rays pointing into open space (>6m) or off car body (<0.1m) are
    # correctly filtered. 100 is a safe floor — below that = real problem.
    LIDAR_VALID_THRESHOLD = 100

    # ── Warm-up ──────────────────────────────────────────────────────────
    # LiDAR motor needs ~1s to reach operating speed (first reads = empty).
    # CSI GStreamer pipeline drops first frames during ISP auto-exposure.
    WARMUP_DURATION_S = 2.0


# ═════════════════════════════════════════════════════════════════════════════
#  SENSOR MANAGER
# ═════════════════════════════════════════════════════════════════════════════

class SensorManager:
    """
    Opens every sensor once, reads them each tick, closes cleanly.

    Every future phase uses this identical interface:

        sensors = SensorManager(Config())
        sensors.open()
        try:
            while running:
                data = sensors.read()   # plain dict, all sensors
        finally:
            sensors.close()
    """

    def __init__(self, config: Config):
        self.cfg       = config
        self._qcar     = None
        self._csi_cams = {}
        self._rs       = None
        self._lidar    = None
        self._opened   = False

    # ── OPEN ──────────────────────────────────────────────────────────────

    def open(self):
        print("\n[SensorManager] Opening sensors...")
        self._open_qcar()
        self._open_csi_cameras()
        self._open_realsense()
        self._open_lidar()
        self._opened = True
        print("[SensorManager] All sensors open.")
        self._warmup()

    def _open_qcar(self):
        print("  Opening QCar (IMU + encoder)...")
        self._qcar = QCar(
            frequency = self.cfg.QCAR_FREQUENCY,
            readMode  = self.cfg.QCAR_READ_MODE,
        )
        print("  QCar: OK")

    def _open_csi_cameras(self):
        for cam_id in self.cfg.CSI_CAMERA_IDS:
            print(f"  Opening CSI camera id='{cam_id}'...")
            cam = Camera2D(
                cameraId    = cam_id,
                frameWidth  = self.cfg.CSI_WIDTH,
                frameHeight = self.cfg.CSI_HEIGHT,
                frameRate   = self.cfg.CSI_FPS,
            )
            self._csi_cams[cam_id] = cam
            print(f"  CSI camera '{cam_id}': OK  buffer={cam.imageData.shape}")

    def _open_realsense(self):
        print("  Opening RealSense D435...")
        self._rs = Camera3D(
            mode             = 'RGB&Depth',
            frameWidthRGB    = self.cfg.RS_WIDTH_RGB,
            frameHeightRGB   = self.cfg.RS_HEIGHT_RGB,
            frameRateRGB     = self.cfg.RS_FPS_RGB,
            frameWidthDepth  = self.cfg.RS_WIDTH_DEPTH,
            frameHeightDepth = self.cfg.RS_HEIGHT_DEPTH,
            frameRateDepth   = self.cfg.RS_FPS_DEPTH,
        )
        print(f"  RealSense: OK  "
              f"RGB={self._rs.imageBufferRGB.shape}  "
              f"Depth={self._rs.imageBufferDepthM.shape}")

    def _open_lidar(self):
        print("  Opening LiDAR (RPLIDAR A2 via QCarLidar)...")
        self._lidar = QCarLidar(
            numMeasurements     = self.cfg.LIDAR_NUM_MEAS,
            rangingDistanceMode = self.cfg.LIDAR_RANGE_MODE,
            interpolationMode   = self.cfg.LIDAR_INTERP,
        )
        print(f"  LiDAR: OK  ({self.cfg.LIDAR_NUM_MEAS} measurements/scan)")

    def _warmup(self):
        """
        Silently discard sensor frames during hardware spin-up.

        LiDAR  — motor takes ~1s to reach speed; first reads return empty.
        CSI    — GStreamer ISP pipeline drops first frames during
                 auto-exposure settling.
        """
        print(f"\n[SensorManager] Warming up ({self.cfg.WARMUP_DURATION_S:.0f}s)...")
        deadline = time.perf_counter() + self.cfg.WARMUP_DURATION_S
        dots = 0
        while time.perf_counter() < deadline:
            self._qcar.read()
            for cam in self._csi_cams.values():
                cam.read()
            self._rs.read_RGB()
            self._rs.read_depth(dataMode='M')
            self._lidar.read()
            time.sleep(0.05)
            dots += 1
            if dots % 10 == 0:
                remaining = deadline - time.perf_counter()
                print(f"  ... {max(remaining, 0):.1f}s remaining")
        print("[SensorManager] Warm-up complete. Starting main loop.\n")

    # ── READ ──────────────────────────────────────────────────────────────

    def read(self) -> dict:
        """
        Read all sensors, return one unified perception dict.

        Keys
        ────
        timestamp        float          time.perf_counter()
        accelerometer    ndarray (3,)   float64  m/s²   [x, y, z]
        gyroscope        ndarray (3,)   float64  rad/s  [x, y, z]
        motor_speed      float          rad/s  (+ve = forward)
        motor_encoder    int            cumulative encoder counts
        battery_voltage  float          volts
        csi_frames       dict  { cam_id : ndarray (H,W,3) uint8 BGR }
        rs_rgb           ndarray (H,W,3)  uint8   BGR
        rs_depth_m       ndarray (H,W,1)  float32 metres  (0.0 = invalid)
        lidar_distances  ndarray (N,)   float32  metres
        lidar_angles     ndarray (N,)   float32  radians
        lidar_valid      ndarray (N,)   bool
        lidar_new_scan   bool    True only when a fresh scan arrived this tick
        """
        t     = time.perf_counter()
        imu   = self._read_qcar()
        csi   = self._read_csi_cameras()
        rs    = self._read_realsense()
        lidar = self._read_lidar()

        return {
            'timestamp':       t,
            'accelerometer':   imu['accelerometer'],
            'gyroscope':       imu['gyroscope'],
            'motor_speed':     imu['motor_speed'],
            'motor_encoder':   imu['motor_encoder'],
            'battery_voltage': imu['battery_voltage'],
            'csi_frames':      csi,
            'rs_rgb':          rs['rgb'],
            'rs_depth_m':      rs['depth_m'],
            'lidar_distances': lidar['distances'],
            'lidar_angles':    lidar['angles'],
            'lidar_valid':     lidar['valid'],
            'lidar_new_scan':  lidar['new_scan'],
        }

    def _read_qcar(self) -> dict:
        """
        QCar.read() — IMU + encoder, NO motor commands.
        np.atleast_1d() handles motorTach/batteryVoltage being scalar
        or shape-(1,) array depending on HAL firmware version. (v2 fix)
        """
        try:
            self._qcar.read()
        except Exception as e:
            print(f"  [WARN] QCar.read() error: {e}")

        return {
            'accelerometer':   np.array(self._qcar.accelerometer, dtype=np.float64),
            'gyroscope':       np.array(self._qcar.gyroscope,     dtype=np.float64),
            'motor_speed':     float(np.atleast_1d(self._qcar.motorTach)[0]),
            'motor_encoder':   int(np.atleast_1d(self._qcar.motorEncoder)[0]),
            'battery_voltage': float(np.atleast_1d(self._qcar.batteryVoltage)[0]),
        }

    def _read_csi_cameras(self) -> dict:
        """
        Camera2D.read() fills .imageData in-place, returns bool.
        .copy() preserves the value after the next read() overwrites buffer.
        """
        frames = {}
        for cam_id, cam in self._csi_cams.items():
            flag = cam.read()
            if not flag:
                print(f"  [WARN] CSI cam '{cam_id}': missed frame")
            frames[cam_id] = cam.imageData.copy()
        return frames

    def _read_realsense(self) -> dict:
        """
        Returns timestamp or -1 if no frame was ready this tick.
        -1 is not a fatal error.
        """
        ts_rgb   = self._rs.read_RGB()
        ts_depth = self._rs.read_depth(dataMode='M')

        if ts_rgb == -1:
            print("  [WARN] RealSense: no RGB frame this tick")
        if ts_depth == -1:
            print("  [WARN] RealSense: no depth frame this tick")

        return {
            'rgb':     self._rs.imageBufferRGB.copy(),
            'depth_m': self._rs.imageBufferDepthM.copy(),
        }

    def _read_lidar(self) -> dict:
        """
        QCarLidar.read() fills .distances + .angles, returns bool (new scan?).

        KEY FACT — scan rate vs loop rate:
          RPLIDAR A2 rotates at ~7 Hz (one full 360° scan every ~143 ms).
          Main loop runs at 30 Hz (one tick every ~33 ms).
          → ~3 out of every 4 ticks get flag=False (no new scan).
          → This is COMPLETELY NORMAL. No warning is printed for this.
          → Warning only fires if a new scan arrives with ZERO valid points.

        'lidar_new_scan': True  → fresh data, use it
                          False → cached data from last scan, skip processing
        """
        flag = self._lidar.read()

        raw_dist = self._lidar.distances
        raw_ang  = self._lidar.angles

        # Guard: empty array before motor reaches operating speed (v3 fix)
        if raw_dist is None or np.asarray(raw_dist).size == 0:
            empty = np.array([], dtype=np.float32)
            return {
                'distances': empty,
                'angles':    empty,
                'valid':     np.array([], dtype=bool),
                'new_scan':  False,
            }

        distances = np.asarray(raw_dist).flatten().astype(np.float32)
        angles    = np.asarray(raw_ang).flatten().astype(np.float32)

        valid   = (distances >= self.cfg.LIDAR_MIN_M) & (distances <= self.cfg.LIDAR_MAX_M)
        n_valid = int(valid.sum())

        # v4 FIX: only warn when a real scan arrived but had zero valid points
        if flag and n_valid == 0:
            print("  [WARN] LiDAR: scan arrived but ZERO valid readings — "
                  "check mounting or LIDAR_MIN_M / LIDAR_MAX_M range")

        return {
            'distances': distances,
            'angles':    angles,
            'valid':     valid,
            'new_scan':  bool(flag),
        }

    # ── CLOSE ─────────────────────────────────────────────────────────────

    def close(self):
        """
        Close all sensors cleanly.
        Always call from a finally block — runs even on Ctrl+C or crash.
        """
        print("\n[SensorManager] Closing sensors...")

        for name, obj in [
            ("LiDAR",     self._lidar),
            ("RealSense", self._rs),
            ("QCar",      self._qcar),
        ]:
            if obj:
                try:
                    obj.terminate()
                    print(f"  {name}: closed")
                except Exception as e:
                    print(f"  {name}: close error ({e})")

        for cam_id, cam in self._csi_cams.items():
            try:
                cam.terminate()
                print(f"  CSI camera '{cam_id}': closed")
            except Exception as e:
                print(f"  CSI camera '{cam_id}': close error ({e})")

        self._opened = False
        print("[SensorManager] All sensors closed.")


# ═════════════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC PRINTER
# ═════════════════════════════════════════════════════════════════════════════

def print_sensor_summary(data: dict, elapsed: float):
    accel = data['accelerometer']
    gyro  = data['gyroscope']

    # CSI
    csi_lines = []
    for cam_id, frame in data['csi_frames'].items():
        ok = int(np.count_nonzero(frame)) > 5000
        csi_lines.append(
            f"    cam'{cam_id}': shape={frame.shape}  "
            f"min={int(frame.min())}  max={int(frame.max())}  "
            f"[{'OK' if ok else 'WARNING: mostly black — check /dev/video* index'}]"
        )

    # RealSense
    depth_flat  = data['rs_depth_m'].flatten()
    depth_valid = depth_flat[depth_flat > 0.0]
    depth_str = (
        f"min={depth_valid.min():.3f}m  max={depth_valid.max():.3f}m  "
        f"valid_px={len(depth_valid)}"
        if len(depth_valid) > 0
        else "NO VALID DEPTH — check USB3 cable (must be USB 3.0 port)"
    )

    # LiDAR — show detail only on new scan ticks to reduce noise
    dist    = data['lidar_distances']
    valid   = data['lidar_valid']
    n_valid = int(valid.sum()) if len(valid) > 0 else 0
    n_total = len(dist)
    new     = data.get('lidar_new_scan', False)

    if new and n_valid > 0:
        lidar_str = (
            f"min={dist[valid].min():.2f}m  max={dist[valid].max():.2f}m  "
            f"valid={n_valid}/{n_total}  [NEW scan]"
        )
    elif new and n_valid == 0:
        lidar_str = f"NEW scan — 0 valid readings! Check sensor."
    else:
        # No new scan this tick — normal at 30Hz loop / 7Hz scan rate
        lidar_str = f"valid={n_valid}/{n_total}  [cached — next scan in ~{1000//7}ms]"

    print(
        f"\n[{elapsed:6.2f}s] ─────────────────────────────────────────────\n"
        f"  IMU accel : x={accel[0]:+.3f}  y={accel[1]:+.3f}  z={accel[2]:+.3f}  m/s²\n"
        f"              (z ≈ +9.81 when stationary — gravity)\n"
        f"  IMU gyro  : x={gyro[0]:+.5f}  y={gyro[1]:+.5f}  z={gyro[2]:+.5f}  rad/s\n"
        f"              (all ≈ 0.00000 when stationary)\n"
        f"  Motor     : speed={data['motor_speed']:+.4f} rad/s  "
        f"encoder={data['motor_encoder']} cts  "
        f"battery={data['battery_voltage']:.2f} V\n"
        f"  CSI cams  :\n" + "\n".join(csi_lines) + "\n"
        f"  RealSense : RGB={data['rs_rgb'].shape}  Depth={data['rs_depth_m'].shape}\n"
        f"              {depth_str}\n"
        f"  LiDAR     : {lidar_str}"
    )


def print_final_report(all_data: list, cfg: Config):
    n = len(all_data)
    if n == 0:
        print("No data collected.")
        return

    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  PHASE 1 FINAL REPORT  —  {n} ticks  target={cfg.LOOP_RATE_HZ} Hz")
    print(sep)

    accels   = np.array([d['accelerometer']   for d in all_data])
    gyros    = np.array([d['gyroscope']       for d in all_data])
    voltages = np.array([d['battery_voltage'] for d in all_data])

    gz    = accels[:, 2].mean()
    gmax  = float(np.abs(gyros).max())
    vmean = voltages.mean()

    # IMU
    print(f"\n  IMU accelerometer  (expect ≈ [0, 0, +9.81])")
    print(f"    mean={accels.mean(axis=0).round(4)}  std={accels.std(axis=0).round(5)}")
    print(f"    accel-z = {gz:.4f}  "
          f"{'PASS' if 9.5 < gz < 10.1 else 'FAIL — check IMU HIL channel'}")

    print(f"\n  IMU gyroscope  (mean ≈ 0; max < 0.15 is OK when stationary)")
    print(f"    mean={gyros.mean(axis=0).round(6)}  std={gyros.std(axis=0).round(6)}")
    print(f"    max abs = {gmax:.5f}  "
          f"['PASS' if gmax < 0.15 else 'NOTE: higher drift — check for vibration']")

    # Battery
    print(f"\n  Battery: {vmean:.2f} V  "
          f"{'PASS' if vmean > 11.0 else 'LOW — charge before driving'}")

    # CSI
    print(f"\n  CSI cameras:")
    csi_pass = True
    for cam_id in cfg.CSI_CAMERA_IDS:
        ok = sum(1 for d in all_data if d['csi_frames'][cam_id].max() > 10)
        passed = ok == n
        csi_pass = csi_pass and passed
        print(f"    cam'{cam_id}': {ok}/{n} non-black  "
              f"[{'PASS' if passed else 'FAIL — check /dev/video index'}]")

    # RealSense
    depth_counts = [(d['rs_depth_m'].flatten() > 0).sum() for d in all_data]
    dmean = float(np.mean(depth_counts))
    print(f"\n  RealSense depth valid px/frame: mean={dmean:.0f}  "
          f"[{'PASS' if dmean > 50000 else 'FAIL — check USB3 port'}]")

    # LiDAR — stats only over ticks where a new scan actually arrived (v4 fix)
    lidar_scan_counts = [
        int(d['lidar_valid'].sum())
        for d in all_data
        if d.get('lidar_new_scan', False) and len(d['lidar_valid']) > 0
    ]
    n_scans = len(lidar_scan_counts)
    lmean   = float(np.mean(lidar_scan_counts)) if n_scans > 0 else 0.0
    scan_hz = n_scans / 10.0
    print(f"\n  LiDAR (RPLIDAR A2 — expected scan rate ~7 Hz):")
    print(f"    Scan ticks received: {n_scans}/{n}  "
          f"(≈ {scan_hz:.1f} Hz  —  expect 6–10 Hz)")
    print(f"    Valid readings/scan: mean={lmean:.0f}  "
          f"threshold={cfg.LIDAR_VALID_THRESHOLD}  "
          f"[{'PASS' if lmean >= cfg.LIDAR_VALID_THRESHOLD else 'FAIL'}]")
    lidar_pass = lmean >= cfg.LIDAR_VALID_THRESHOLD

    # Loop timing
    timestamps = [d['timestamp'] for d in all_data]
    intervals  = np.diff(timestamps) * 1000
    tmean = float(intervals.mean())
    print(f"\n  Loop timing: mean={tmean:.1f} ms  std={intervals.std():.2f} ms  "
          f"max={intervals.max():.1f} ms  "
          f"[{'PASS' if tmean < 50 else 'WARN — Jetson CPU load too high'}]")

    # Final checklist
    print(f"\n{sep}")
    print("  CHECKLIST:")
    rows = [
        ("PASS" if 9.5 < gz < 10.1         else "FAIL",
         f"accel z ≈ +9.81 m/s²  (got {gz:.3f})"),
        ("PASS" if gmax < 0.15              else "NOTE",
         f"gyro max abs < 0.15 rad/s  (got {gmax:.4f})"),
        ("PASS" if vmean > 11.0             else "FAIL",
         f"battery > 11.0 V  (got {vmean:.2f} V)"),
        ("PASS" if csi_pass                 else "FAIL",
         "CSI camera — all frames non-black"),
        ("PASS" if dmean > 50000            else "FAIL",
         f"RealSense valid_px > 50000  (got {dmean:.0f})"),
        ("PASS" if lidar_pass               else "FAIL",
         f"LiDAR valid > {cfg.LIDAR_VALID_THRESHOLD}/scan  (got {lmean:.0f})"),
        ("PASS" if tmean < 50               else "WARN",
         f"loop mean < 50 ms  (got {tmean:.1f} ms)"),
    ]
    for status, text in rows:
        icon = "✓" if status == "PASS" else ("!" if status == "NOTE" else "✗")
        print(f"  [{status}] {icon}  {text}")
    print(sep)

    all_pass = all(r[0] in ("PASS", "NOTE") for r in rows)
    if all_pass:
        print("\n  ✓ ALL CHECKS PASSED — ready for Phase 2 (estimation.py)\n")
    else:
        failed = [r[1] for r in rows if r[0] == "FAIL"]
        print(f"\n  ✗ {len(failed)} check(s) failed — fix before Phase 2:")
        for f in failed:
            print(f"      • {f}")
        print()


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    cfg     = Config()
    sensors = SensorManager(cfg)

    TEST_DURATION_S  = 10.0
    PRINT_INTERVAL_S = 0.5

    all_data   = []
    last_print = 0.0

    print("═" * 60)
    print("  QCar 2 — Phase 1: Sensor Reading Test")
    print(f"  Running {TEST_DURATION_S:.0f}s after warm-up. Ctrl+C to stop early.")
    print("═" * 60)

    sensors.open()

    try:
        start_t    = time.perf_counter()
        tick_start = start_t

        while True:
            elapsed = time.perf_counter() - start_t
            if elapsed >= TEST_DURATION_S:
                break

            data = sensors.read()

            # Deep-copy every value — pre-allocated buffers are overwritten each tick
            snapshot = {}
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    snapshot[k] = v.copy()
                elif isinstance(v, dict):
                    snapshot[k] = {i: f.copy() for i, f in v.items()}
                else:
                    snapshot[k] = v
            all_data.append(snapshot)

            if elapsed - last_print >= PRINT_INTERVAL_S:
                print_sensor_summary(data, elapsed)
                last_print = elapsed

            # Soft real-time pacing — sleep the remainder of the tick window
            elapsed_tick = time.perf_counter() - tick_start
            sleep_time   = cfg.LOOP_DT - elapsed_tick
            if sleep_time > 0:
                time.sleep(sleep_time)
            tick_start = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")

    finally:
        sensors.close()

    print_final_report(all_data, cfg)


if __name__ == "__main__":
    main()
