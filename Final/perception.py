"""
perception.py  —  Phase 1: All sensor reading for QCar 2
KEY FIX v5: Uses QCarCameras (not raw Camera2D) — fixes missed frame errors.
QCarCameras uses QCAR_CONFIG['csiFront'] automatically, no hardcoded index.
RULE: Only ONE process may run this at a time.
"""
import time
import numpy as np
from pal.products.qcar    import QCar, QCarLidar, QCarCameras
from pal.utilities.vision import Camera3D


class Config:
    LOOP_RATE_HZ    = 30
    LOOP_DT         = 1.0 / 30
    QCAR_FREQUENCY  = 500
    QCAR_READ_MODE  = 0
    CSI_WIDTH       = 820
    CSI_HEIGHT      = 410
    CSI_FPS         = 30.0
    RS_WIDTH        = 640
    RS_HEIGHT       = 480
    RS_FPS          = 30.0
    LIDAR_NUM_MEAS      = 384
    LIDAR_RANGE_MODE    = 2
    LIDAR_INTERP        = 0
    LIDAR_MIN_M         = 0.10
    LIDAR_MAX_M         = 6.0
    LIDAR_VALID_THRESHOLD = 100
    WARMUP_DURATION_S = 2.0


class SensorManager:
    """
    Opens every sensor once. Reads all each tick. Closes cleanly.

    Sensor dict keys:
        timestamp        float
        accelerometer    ndarray (3,)    float64  m/s^2
        gyroscope        ndarray (3,)    float64  rad/s
        motor_speed      float           rad/s
        motor_encoder    int             cumulative counts
        battery_voltage  float           volts
        csi_front        ndarray (H,W,3) uint8 BGR
        csi_left         ndarray (H,W,3) uint8 BGR (zeros if disabled)
        csi_right        ndarray (H,W,3) uint8 BGR (zeros if disabled)
        csi_back         ndarray (H,W,3) uint8 BGR (zeros if disabled)
        rs_rgb           ndarray (H,W,3) uint8 BGR
        rs_depth_m       ndarray (H,W,1) float32 metres (0=invalid)
        lidar_distances  ndarray (N,)    float32 metres
        lidar_angles     ndarray (N,)    float32 radians
        lidar_valid      ndarray (N,)    bool
        lidar_new_scan   bool
    """

    def __init__(self, config: Config):
        self.cfg      = config
        self._qcar    = None
        self._cameras = None
        self._rs      = None
        self._lidar   = None
        H, W = config.CSI_HEIGHT, config.CSI_WIDTH
        self._blank   = np.zeros((H, W, 3), dtype=np.uint8)
        self._steering_offset = 0.0   # calibration offset (set after open)

    @property
    def qcar(self):
        """Public access to QCar hardware for read_write_std calls."""
        return self._qcar

    def write_command(self, throttle: float, steering: float, leds: np.ndarray):
        """
        Send throttle + steering + LEDs to QCar hardware.
        Applies steering calibration offset automatically.
        """
        corrected_steering = steering + self._steering_offset
        self._qcar.read_write_std(
            throttle=throttle,
            steering=corrected_steering,
            LEDs=leds,
        )

    def open(self):
        print("\n[SensorManager] Opening sensors...")
        self._open_qcar()
        self._open_realsense()
        self._open_lidar()
        self._open_cameras()   # cameras last — avoids GStreamer timeout
        print("[SensorManager] All sensors open.")
        self._warmup()

    def _open_qcar(self):
        print("  Opening QCar (IMU + encoder)...")
        self._qcar = QCar(frequency=self.cfg.QCAR_FREQUENCY,
                          readMode=self.cfg.QCAR_READ_MODE)
        # Load steering calibration offset if available
        try:
            from pal.products.qcar import QCAR_CONFIG
            self._steering_offset = float(QCAR_CONFIG.get('steeringOffset', 0.0))
            if self._steering_offset != 0.0:
                print(f"  Steering calibration offset: {self._steering_offset:+.4f} rad")
        except Exception:
            self._steering_offset = 0.0
        print("  QCar: OK")

    def _open_realsense(self):
        print("  Opening RealSense D435...")
        self._rs = Camera3D(
            mode='RGB&Depth',
            frameWidthRGB=self.cfg.RS_WIDTH, frameHeightRGB=self.cfg.RS_HEIGHT,
            frameRateRGB=self.cfg.RS_FPS,
            frameWidthDepth=self.cfg.RS_WIDTH, frameHeightDepth=self.cfg.RS_HEIGHT,
            frameRateDepth=self.cfg.RS_FPS,
        )
        print(f"  RealSense: OK  RGB={self._rs.imageBufferRGB.shape}  "
              f"Depth={self._rs.imageBufferDepthM.shape}")

    def _open_lidar(self):
        print("  Opening LiDAR (RPLIDAR A2)...")
        self._lidar = QCarLidar(
            numMeasurements=self.cfg.LIDAR_NUM_MEAS,
            rangingDistanceMode=self.cfg.LIDAR_RANGE_MODE,
            interpolationMode=self.cfg.LIDAR_INTERP,
        )
        print(f"  LiDAR: OK  ({self.cfg.LIDAR_NUM_MEAS} meas/scan)")

    def _open_cameras(self):
        print("  Opening CSI cameras (QCarCameras)...")
        self._cameras = QCarCameras(
            frameWidth=self.cfg.CSI_WIDTH,
            frameHeight=self.cfg.CSI_HEIGHT,
            frameRate=self.cfg.CSI_FPS,
            enableFront=True,
            enableLeft=False,
            enableRight=False,
            enableBack=False,
        )
        print(f"  CSI front: OK  "
              f"buffer={self._cameras.csiFront.imageData.shape}")

    def _warmup(self):
        print(f"\n[SensorManager] Warming up {self.cfg.WARMUP_DURATION_S:.0f}s...")
        deadline = time.perf_counter() + self.cfg.WARMUP_DURATION_S
        ticks = 0
        while time.perf_counter() < deadline:
            self._qcar.read()
            self._cameras.readAll()
            self._rs.read_RGB()
            self._rs.read_depth(dataMode='M')
            self._lidar.read()
            time.sleep(self.cfg.LOOP_DT)
            ticks += 1
            if ticks % 15 == 0:
                print(f"  ... {max(deadline - time.perf_counter(), 0):.1f}s remaining")
        print("[SensorManager] Warm-up complete.\n")

    def read(self) -> dict:
        t     = time.perf_counter()
        imu   = self._read_qcar()
        csi   = self._read_cameras()
        rs    = self._read_realsense()
        lidar = self._read_lidar()
        return {
            'timestamp':       t,
            'accelerometer':   imu['accelerometer'],
            'gyroscope':       imu['gyroscope'],
            'motor_speed':     imu['motor_speed'],
            'motor_encoder':   imu['motor_encoder'],
            'battery_voltage': imu['battery_voltage'],
            'csi_front':       csi['front'],
            'csi_left':        csi['left'],
            'csi_right':       csi['right'],
            'csi_back':        csi['back'],
            'rs_rgb':          rs['rgb'],
            'rs_depth_m':      rs['depth_m'],
            'lidar_distances': lidar['distances'],
            'lidar_angles':    lidar['angles'],
            'lidar_valid':     lidar['valid'],
            'lidar_new_scan':  lidar['new_scan'],
        }

    def _read_qcar(self) -> dict:
        try:
            self._qcar.read()
        except Exception as e:
            print(f"  [WARN] QCar read error: {e}")
        return {
            'accelerometer':   np.array(self._qcar.accelerometer, dtype=np.float64),
            'gyroscope':       np.array(self._qcar.gyroscope,     dtype=np.float64),
            'motor_speed':     float(np.atleast_1d(self._qcar.motorTach)[0]),
            'motor_encoder':   int(np.atleast_1d(self._qcar.motorEncoder)[0]),
            'battery_voltage': float(np.atleast_1d(self._qcar.batteryVoltage)[0]),
        }

    def _read_cameras(self) -> dict:
        self._cameras.readAll()
        def f(cam):
            return cam.imageData.copy() if cam is not None else self._blank.copy()
        return {
            'front': f(self._cameras.csiFront),
            'left':  f(self._cameras.csiLeft),
            'right': f(self._cameras.csiRight),
            'back':  f(self._cameras.csiBack),
        }

    def _read_realsense(self) -> dict:
        self._rs.read_RGB()
        self._rs.read_depth(dataMode='M')
        return {
            'rgb':     self._rs.imageBufferRGB.copy(),
            'depth_m': self._rs.imageBufferDepthM.copy(),
        }

    def _read_lidar(self) -> dict:
        flag     = self._lidar.read()
        raw_dist = self._lidar.distances
        raw_ang  = self._lidar.angles
        if raw_dist is None or np.asarray(raw_dist).size == 0:
            empty = np.array([], dtype=np.float32)
            return {'distances': empty, 'angles': empty,
                    'valid': np.array([], dtype=bool), 'new_scan': False}
        distances = np.asarray(raw_dist).flatten().astype(np.float32)
        angles    = np.asarray(raw_ang).flatten().astype(np.float32)
        valid     = ((distances >= self.cfg.LIDAR_MIN_M) &
                     (distances <= self.cfg.LIDAR_MAX_M))
        return {'distances': distances, 'angles': angles,
                'valid': valid, 'new_scan': bool(flag)}

    def close(self):
        print("\n[SensorManager] Closing sensors...")
        for name, obj, method in [
            ('LiDAR',     self._lidar,   'terminate'),
            ('RealSense', self._rs,      'terminate'),
            ('QCar',      self._qcar,    'terminate'),
        ]:
            if obj:
                try:
                    getattr(obj, method)()
                    print(f"  {name}: closed")
                except Exception as e:
                    print(f"  {name} close error: {e}")
        if self._cameras:
            try:
                for cam in self._cameras.csi:
                    if cam is not None:
                        cam.terminate()
                print("  CSI cameras: closed")
            except Exception as e:
                print(f"  CSI camera close error: {e}")
        print("[SensorManager] All sensors closed.")


# ── Standalone test ───────────────────────────────────────────────────────────

def main():
    cfg     = Config()
    sensors = SensorManager(cfg)
    print("═"*60)
    print("  QCar 2 — perception.py standalone test")
    print("  Ctrl+C to stop early.")
    print("═"*60)
    sensors.open()
    try:
        start = time.perf_counter()
        tick_t = start
        last_print = 0.0
        ticks = 0
        while True:
            elapsed = time.perf_counter() - start
            if elapsed >= 15.0:
                break
            data = sensors.read()
            ticks += 1
            if elapsed - last_print >= 1.0:
                accel    = data['accelerometer']
                front    = data['csi_front']
                depth    = data['rs_depth_m'].flatten()
                valid_px = int((depth > 0).sum())
                lv       = int(data['lidar_valid'].sum())
                print(f"\n[{elapsed:5.1f}s] ticks={ticks}")
                print(f"  accel_z  = {accel[2]:+.3f} m/s²  (expect ~+9.81)")
                print(f"  battery  = {data['battery_voltage']:.2f} V")
                print(f"  camera   max={int(front.max())}  {'OK' if front.max()>10 else 'BLACK!'}")
                print(f"  depth    valid_px={valid_px}  {'OK' if valid_px>50000 else 'LOW'}")
                print(f"  lidar    valid={lv}  new_scan={data['lidar_new_scan']}")
                last_print = elapsed
            elapsed_tick = time.perf_counter() - tick_t
            sleep = cfg.LOOP_DT - elapsed_tick
            if sleep > 0:
                time.sleep(sleep)
            tick_t = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        sensors.close()

if __name__ == "__main__":
    main()
