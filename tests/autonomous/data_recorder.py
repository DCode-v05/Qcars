"""
data_recorder.py — Records all sensor data to files for debugging.

Creates a timestamped folder with:
  recordings/
    YYYYMMDD_HHMMSS/
      csi_front.avi          — front camera video
      csi_right.avi          — right camera video
      csi_back.avi           — back camera video
      csi_left.avi           — left camera video
      realsense_rgb.avi      — RealSense RGB video
      realsense_depth/       — depth frames as .npy files
        depth_000001.npy
        ...
      lidar.csv              — timestamp, distances, angles per scan
      imu.csv                — timestamp, accel_x/y/z, gyro_x/y/z
      wheels.csv             — timestamp, motor_speed, battery_voltage,
                               throttle_cmd, steering_cmd
      nav.csv                — timestamp, state, x, y, theta, front_dist,
                               rear_dist, drive_forward, path_steer

Enable with: python main.py --record
"""
import os
import csv
import time
import threading
import logging

import cv2
import numpy as np

import config as cfg


CAM_NAMES = ["right", "back", "front", "left"]  # matches CAMERA_CONFIG order


class DataRecorder:

    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "recordings")
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._dir = os.path.join(base_dir, ts)
        os.makedirs(self._dir, exist_ok=True)

        self._depth_dir = os.path.join(self._dir, "realsense_depth")
        os.makedirs(self._depth_dir, exist_ok=True)

        # Video writers for CSI cameras
        self._csi_writers = []
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        for name in CAM_NAMES:
            path = os.path.join(self._dir, f"csi_{name}.avi")
            w = cv2.VideoWriter(path, fourcc, cfg.CSI_FPS,
                                (cfg.CSI_WIDTH, cfg.CSI_HEIGHT))
            self._csi_writers.append(w)

        # RealSense RGB video
        self._rs_writer = cv2.VideoWriter(
            os.path.join(self._dir, "realsense_rgb.avi"),
            fourcc, cfg.RS_FPS, (cfg.RS_WIDTH, cfg.RS_HEIGHT))

        # CSV files
        self._lidar_file = open(os.path.join(self._dir, "lidar.csv"), 'w',
                                newline='')
        self._lidar_csv = csv.writer(self._lidar_file)
        self._lidar_csv.writerow(['timestamp', 'num_points',
                                  'distances', 'angles'])

        self._imu_file = open(os.path.join(self._dir, "imu.csv"), 'w',
                              newline='')
        self._imu_csv = csv.writer(self._imu_file)
        self._imu_csv.writerow(['timestamp', 'accel_x', 'accel_y', 'accel_z',
                                'gyro_x', 'gyro_y', 'gyro_z'])

        self._wheels_file = open(os.path.join(self._dir, "wheels.csv"), 'w',
                                 newline='')
        self._wheels_csv = csv.writer(self._wheels_file)
        self._wheels_csv.writerow(['timestamp', 'motor_speed',
                                   'battery_voltage',
                                   'throttle_cmd', 'steering_cmd'])

        self._nav_file = open(os.path.join(self._dir, "nav.csv"), 'w',
                              newline='')
        self._nav_csv = csv.writer(self._nav_file)
        self._nav_csv.writerow(['timestamp', 'state', 'pos_x', 'pos_y',
                                'heading', 'front_dist', 'rear_dist',
                                'drive_forward', 'path_steer', 'zone',
                                'obstacle_type'])

        self._frame_count = 0
        self._lock = threading.Lock()
        self._closed = False

        logging.info(f"[Recorder] Saving to {self._dir}")

    def record_tick(self, sensor_data, detection=None, nav_result=None,
                    pose=None, throttle_cmd=0.0, steering_cmd=0.0):
        """Record one tick of all sensor data. Call from main loop."""
        if self._closed:
            return

        with self._lock:
            ts = sensor_data.get('timestamp', time.time())
            self._frame_count += 1

            # ── CSI camera frames → video ──
            csi_frames = sensor_data.get('csi_frames', [])
            for i, frame in enumerate(csi_frames):
                if i < len(self._csi_writers) and frame is not None:
                    self._csi_writers[i].write(frame)

            # ── RealSense RGB → video ──
            rs_rgb = sensor_data.get('rs_rgb')
            if rs_rgb is not None:
                self._rs_writer.write(rs_rgb)

            # ── RealSense depth → .npy (every 5th frame to save space) ──
            if self._frame_count % 5 == 0:
                rs_depth = sensor_data.get('rs_depth_m')
                if rs_depth is not None:
                    fname = f"depth_{self._frame_count:06d}.npy"
                    np.save(os.path.join(self._depth_dir, fname), rs_depth)

            # ── LiDAR → CSV ──
            distances = sensor_data.get('lidar_distances', np.array([]))
            angles = sensor_data.get('lidar_angles', np.array([]))
            if len(distances) > 0:
                dist_str = ';'.join(f'{d:.3f}' for d in distances)
                ang_str = ';'.join(f'{a:.4f}' for a in angles)
                self._lidar_csv.writerow([
                    f'{ts:.6f}', len(distances), dist_str, ang_str
                ])

            # ── IMU → CSV ──
            accel = sensor_data.get('accelerometer', np.zeros(3))
            gyro = sensor_data.get('gyroscope', np.zeros(3))
            self._imu_csv.writerow([
                f'{ts:.6f}',
                f'{accel[0]:.6f}', f'{accel[1]:.6f}', f'{accel[2]:.6f}',
                f'{gyro[0]:.6f}', f'{gyro[1]:.6f}', f'{gyro[2]:.6f}',
            ])

            # ── Wheels → CSV ──
            motor_speed = sensor_data.get('motor_speed', 0.0)
            battery = sensor_data.get('battery_voltage', 0.0)
            self._wheels_csv.writerow([
                f'{ts:.6f}',
                f'{motor_speed:.6f}', f'{battery:.3f}',
                f'{throttle_cmd:.4f}', f'{steering_cmd:.4f}',
            ])

            # ── Navigation state → CSV ──
            if detection and nav_result:
                px = pose[0] if pose else 0.0
                py = pose[1] if pose else 0.0
                pth = pose[2] if pose else 0.0
                self._nav_csv.writerow([
                    f'{ts:.6f}',
                    nav_result.get('state', ''),
                    f'{px:.4f}', f'{py:.4f}', f'{pth:.4f}',
                    f'{detection.get("distance_m", 0):.3f}',
                    f'{detection.get("rear_min_m", 0):.3f}',
                    detection.get('drive_forward', ''),
                    f'{detection.get("path_steer", 0):.4f}',
                    detection.get('zone', ''),
                    detection.get('obstacle_type', ''),
                ])

    def close(self):
        """Flush and close all files/writers."""
        if self._closed:
            return
        self._closed = True

        with self._lock:
            for w in self._csi_writers:
                try:
                    w.release()
                except Exception:
                    pass
            try:
                self._rs_writer.release()
            except Exception:
                pass

            for f in [self._lidar_file, self._imu_file,
                      self._wheels_file, self._nav_file]:
                try:
                    f.flush()
                    f.close()
                except Exception:
                    pass

        logging.info(f"[Recorder] Saved {self._frame_count} frames to {self._dir}")

    @property
    def output_dir(self):
        return self._dir

    @property
    def frame_count(self):
        return self._frame_count
