"""
sensors.py — Sensor reading for QCar 2 autonomous driving.
Opens QCar (IMU/encoder), QCarCameras (4 CSI), QCarLidar, RealSense D435.
"""
import time
import numpy as np
import logging
from pal.products.qcar import QCar, QCarLidar, QCarCameras
from pal.utilities.vision import Camera3D

import config as cfg


class SensorManager:
    """
    Opens all sensors once. Reads all each tick. Closes cleanly.

    Output dict keys:
        timestamp, accelerometer, gyroscope, motor_speed, battery_voltage,
        csi_frames (list of 4 ndarrays),
        rs_rgb (ndarray BGR), rs_depth_m (ndarray float32 metres),
        lidar_distances, lidar_angles, lidar_valid, lidar_new_scan
    """

    def __init__(self):
        self._qcar    = None
        self._cameras = None
        self._rs      = None
        self._lidar   = None
        self._blank_csi = np.zeros((cfg.CSI_HEIGHT, cfg.CSI_WIDTH, 3), dtype=np.uint8)
        self._blank_rs  = np.zeros((cfg.RS_HEIGHT, cfg.RS_WIDTH, 3), dtype=np.uint8)
        self._blank_depth = np.zeros((cfg.RS_HEIGHT, cfg.RS_WIDTH, 1), dtype=np.float32)
        self._steering_offset = 0.0

    def open(self):
        logging.info("[Sensors] Opening hardware...")
        self._open_qcar()
        self._open_realsense()
        self._open_lidar()
        self._open_cameras()
        self._warmup()
        logging.info("[Sensors] All sensors ready.")

    def _open_qcar(self):
        logging.info("  Opening QCar (IMU + encoder)...")
        self._qcar = QCar(frequency=cfg.QCAR_FREQUENCY,
                          readMode=cfg.QCAR_READ_MODE)
        try:
            from pal.products.qcar import QCAR_CONFIG
            self._steering_offset = float(QCAR_CONFIG.get('steeringOffset', 0.0))
            if self._steering_offset != 0.0:
                logging.info(f"  Steering offset: {self._steering_offset:+.4f} rad")
        except Exception:
            self._steering_offset = 0.0
        logging.info("  QCar: OK")

    def _open_realsense(self):
        logging.info("  Opening RealSense D435 (RGB+Depth)...")
        try:
            self._rs = Camera3D(
                mode='RGB&Depth',
                frameWidthRGB=cfg.RS_WIDTH, frameHeightRGB=cfg.RS_HEIGHT,
                frameRateRGB=cfg.RS_FPS,
                frameWidthDepth=cfg.RS_WIDTH, frameHeightDepth=cfg.RS_HEIGHT,
                frameRateDepth=cfg.RS_FPS,
            )
            logging.info(f"  RealSense: OK  RGB={self._rs.imageBufferRGB.shape}  "
                         f"Depth={self._rs.imageBufferDepthM.shape}")
        except Exception as e:
            logging.warning(f"  RealSense open failed: {e} — running without depth")
            self._rs = None

    def _open_lidar(self):
        logging.info("  Opening LiDAR (RPLIDAR A2)...")
        self._lidar = QCarLidar(
            numMeasurements=cfg.LIDAR_NUM_MEAS,
            rangingDistanceMode=cfg.LIDAR_RANGE_MODE,
            interpolationMode=cfg.LIDAR_INTERP,
        )
        logging.info(f"  LiDAR: OK ({cfg.LIDAR_NUM_MEAS} meas/scan)")

    def _open_cameras(self):
        logging.info("  Opening CSI cameras (QCarCameras, all 4)...")
        self._cameras = QCarCameras(
            frameWidth=cfg.CSI_WIDTH,
            frameHeight=cfg.CSI_HEIGHT,
            frameRate=cfg.CSI_FPS,
            enableRight=True,
            enableBack=True,
            enableFront=True,
            enableLeft=True,
        )
        logging.info("  CSI cameras: OK (4 enabled)")

    def _warmup(self):
        logging.info(f"  Warming up sensors ({cfg.WARMUP_S:.0f}s)...")
        deadline = time.perf_counter() + cfg.WARMUP_S
        while time.perf_counter() < deadline:
            self._qcar.read()
            self._cameras.readAll()
            if self._rs:
                self._rs.read_RGB()
                self._rs.read_depth(dataMode='M')
            self._lidar.read()
            time.sleep(cfg.LOOP_DT)

    def read(self) -> dict:
        t = time.perf_counter()
        imu   = self._read_qcar()
        csi   = self._read_cameras()
        rs    = self._read_realsense()
        lidar = self._read_lidar()
        return {
            'timestamp':       t,
            'accelerometer':   imu['accelerometer'],
            'gyroscope':       imu['gyroscope'],
            'motor_speed':     imu['motor_speed'],
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
        try:
            self._qcar.read()
        except Exception as e:
            logging.warning(f"  [WARN] QCar read: {e}")
        return {
            'accelerometer': np.array(self._qcar.accelerometer, dtype=np.float64),
            'gyroscope':     np.array(self._qcar.gyroscope, dtype=np.float64),
            'motor_speed':   float(np.atleast_1d(self._qcar.motorTach)[0]),
            'battery_voltage': float(np.atleast_1d(self._qcar.batteryVoltage)[0]),
        }

    def _read_cameras(self) -> list:
        self._cameras.readAll()
        frames = []
        for cam in self._cameras.csi:
            if cam is not None:
                frames.append(cam.imageData.copy())
            else:
                frames.append(self._blank_csi.copy())
        return frames

    def _read_realsense(self) -> dict:
        if self._rs is None:
            return {'rgb': self._blank_rs.copy(),
                    'depth_m': self._blank_depth.copy()}
        try:
            self._rs.read_RGB()
            self._rs.read_depth(dataMode='M')
            return {
                'rgb':     self._rs.imageBufferRGB.copy(),
                'depth_m': self._rs.imageBufferDepthM.copy(),
            }
        except Exception as e:
            logging.warning(f"  [WARN] RealSense read: {e}")
            return {'rgb': self._blank_rs.copy(),
                    'depth_m': self._blank_depth.copy()}

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

        # CRITICAL: Rotate LiDAR frame so 0 rad = car's actual front.
        # The LiDAR's 0° points backward; car front is at LIDAR_FRONT_DEG (180°).
        rotation = np.radians(cfg.LIDAR_FRONT_DEG)
        angles = (angles - rotation) % (2 * np.pi)

        valid = (distances >= cfg.LIDAR_MIN_M) & (distances <= cfg.LIDAR_MAX_M)
        return {'distances': distances, 'angles': angles,
                'valid': valid, 'new_scan': bool(flag)}

    def write_command(self, throttle: float, steering: float,
                      leds: np.ndarray = None):
        if leds is None:
            leds = np.zeros(8, dtype=np.float64)
        corrected = steering + self._steering_offset
        self._qcar.read_write_std(
            throttle=throttle,
            steering=corrected,
            LEDs=leds,
        )

    def close(self):
        logging.info("[Sensors] Closing...")
        for name, obj, method in [
            ('LiDAR',     self._lidar,   'terminate'),
            ('RealSense', self._rs,      'terminate'),
            ('QCar',      self._qcar,    'terminate'),
        ]:
            if obj:
                try:
                    getattr(obj, method)()
                    logging.info(f"  {name}: closed")
                except Exception as e:
                    logging.warning(f"  {name} close: {e}")
        if self._cameras:
            try:
                for cam in self._cameras.csi:
                    if cam is not None:
                        cam.terminate()
                logging.info("  CSI cameras: closed")
            except Exception as e:
                logging.warning(f"  CSI close: {e}")
        logging.info("[Sensors] Shutdown complete.")
