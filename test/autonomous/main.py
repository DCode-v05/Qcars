#!/usr/bin/env python3
"""
main.py — QCar 2 Autonomous Obstacle Avoidance (360-degree)

Controls:
    q       — quit (only way to stop)
    Ctrl+C  — quit
    d       — force DRIVE (forward) mode
    r       — force REVERSE mode
    a       — back to AUTO mode

USAGE:
    cd ~/Qcars/test/autonomous
    python main.py --model $MODELSPATH/yolov8n.pt --dashboard
"""
import argparse
import os
import queue
import select
import signal
import sys
import termios
import threading
import tty
import time
import logging

import numpy as np

import config as cfg
from sensors import SensorManager
from yolo_processor import YOLOProcessor
from obstacle_detector import ObstacleDetector
from navigator import Navigator


_shutdown = False
_nav_ref = None  # reference to navigator for key commands


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True
    print(f"\n[main] Signal {signum} — shutting down...")


def _key_listener():
    """Background thread: handles keyboard input.
    q = quit, d = drive forward, r = reverse, a = auto."""
    global _shutdown
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while not _shutdown:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = sys.stdin.read(1).lower()
                if ch == 'q':
                    _shutdown = True
                    print("\n[main] 'q' pressed — shutting down...")
                    break
                elif ch == 'd':
                    if _nav_ref:
                        _nav_ref.set_manual('d')
                    print("\r[KEY] MANUAL FORWARD (press 'a' for auto)     ", end="")
                elif ch == 'r':
                    if _nav_ref:
                        _nav_ref.set_manual('r')
                    print("\r[KEY] MANUAL REVERSE (press 'a' for auto)     ", end="")
                elif ch == 'a':
                    if _nav_ref:
                        _nav_ref.set_manual(None)
                    print("\r[KEY] AUTO MODE                               ", end="")
    except Exception:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def main():
    global _shutdown, _nav_ref

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="QCar 2 Autonomous Driving")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="YOLO model path")
    ap.add_argument("--no-yolo", action="store_true",
                    help="Disable YOLO (LiDAR-only)")
    ap.add_argument("--dashboard", action="store_true",
                    help="Enable web dashboard on port 5000")
    args = ap.parse_args()

    model_path = os.path.expandvars(args.model)

    signal.signal(signal.SIGINT, _signal_handler)
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
    except (OSError, AttributeError):
        pass

    print("=" * 62)
    print("  QCar 2 — Autonomous Obstacle Avoidance (360)")
    print("=" * 62)
    print(f"  Throttle : {cfg.THROTTLE} (constant)")
    print(f"  YOLO     : {'OFF' if args.no_yolo else model_path}")
    print(f"  Cameras  : 4 CSI + RealSense D435")
    print(f"  LiDAR    : {cfg.LIDAR_NUM_MEAS} measurements")
    print(f"  Dashboard: {'ON' if args.dashboard else 'OFF'}")
    print("-" * 62)
    print("  Controls: q=quit  d=drive  r=reverse  a=auto")
    print("-" * 62)

    # ── Build modules ─────────────────────────────────────────────────────
    sensors  = SensorManager()
    detector = ObstacleDetector()
    nav      = Navigator()
    _nav_ref = nav  # expose to key listener

    cam_queues = [queue.Queue(maxsize=2) for _ in cfg.CAMERA_CONFIG]

    dash = None
    stores = None
    if args.dashboard:
        from dashboard import Dashboard, CamStore
        stores = [CamStore(c["id"]) for c in cfg.CAMERA_CONFIG]
        dash = Dashboard(stores)

    yolo = None
    if not args.no_yolo:
        yolo = YOLOProcessor(model_path, cam_queues, stores=stores)

    # ── Open hardware ─────────────────────────────────────────────────────
    sensors.open()

    if yolo:
        yolo.start()
        logging.info("[main] YOLO processor started")

    if dash:
        dash.start()
        logging.info(f"[main] Dashboard on http://0.0.0.0:{cfg.DASHBOARD_PORT}")

    nav.reset()

    # ── Timing ────────────────────────────────────────────────────────────
    start_t    = time.perf_counter()
    last_t     = start_t
    last_print = 0.0
    tick_count = 0

    # Start key listener
    threading.Thread(target=_key_listener, daemon=True, name="keys").start()

    print(f"\n[main] Running. Press 'q' to stop.\n")

    try:
        while not _shutdown:
            tick_start = time.perf_counter()
            elapsed    = tick_start - start_t

            # ── 1. Read sensors ───────────────────────────────────────────
            data = sensors.read()
            tick_count += 1

            # ── 2. Compute dt ─────────────────────────────────────────────
            now = time.perf_counter()
            dt  = max(now - last_t, 1e-4)
            last_t = now

            # ── 3. Push frames to YOLO queues ─────────────────────────────
            frames = data['csi_frames']
            for i, frame in enumerate(frames):
                q = cam_queues[i]
                if q.full():
                    try: q.get_nowait()
                    except queue.Empty: pass
                try: q.put_nowait(frame)
                except queue.Full: pass

            # ── 4. Battery check ──────────────────────────────────────────
            batt = data['battery_voltage']
            if batt < cfg.BATTERY_CRIT_V and batt > 1.0:
                print(f"\n[SAFETY] Battery critical: {batt:.2f}V — stopping.")
                break

            # ── 5. Obstacle detection (LiDAR + YOLO + depth) ─────────────
            yolo_dets = yolo.get_detections() if yolo else []
            detection = detector.detect(data, yolo_dets)

            # ── 6. Navigate (with IMU for crash detection) ────────────────
            imu_accel = data.get('accelerometer')
            nav_result = nav.update(detection, dt, imu_accel=imu_accel)
            throttle   = nav_result['throttle']
            steering   = nav_result['steering']
            state      = nav_result['state']
            leds       = nav_result['leds']

            if nav_result.get('crash_detected'):
                print(f"\n[IMU] CRASH DETECTED — reversing!")

            # ── 7. Send to hardware ───────────────────────────────────────
            sensors.write_command(throttle=throttle, steering=steering,
                                 leds=leds)

            # ── 8. Push data to dashboard ─────────────────────────────────
            if dash:
                dash.update_lidar(data['lidar_distances'],
                                  data['lidar_angles'],
                                  data['lidar_valid'])
                dash.update_nav(detection, nav_result)
                dash.update_detections(yolo_dets)
                dash.update_realsense(data['rs_rgb'], data['rs_depth_m'])

            # ── 9. Terminal print every 0.5s ──────────────────────────────
            if elapsed - last_print >= 0.5:
                zone     = detection['zone']
                dist     = detection['distance_m']
                rear     = detection['rear_min_m']
                gap_w    = detection['best_gap_width']
                obs_type = detection['obstacle_type']
                fwd_str  = "FWD" if detection['drive_forward'] else "REV"
                path_deg = np.degrees(detection['best_path_angle'])
                if path_deg > 180:
                    path_deg -= 360

                mode = ""
                if state.startswith("MANUAL"):
                    mode = " [MANUAL]"

                print(
                    f"[{elapsed:5.1f}s] {state:<11}{mode} "
                    f"zone={zone:<5} dist={dist:.2f}m rear={rear:.2f}m "
                    f"path={fwd_str} {path_deg:+.0f}deg gap={gap_w:.0f}deg "
                    f"obs={obs_type:<7} "
                    f"thr={throttle:+.2f} str={steering:+.3f}"
                )
                last_print = elapsed

            # ── 10. Pacing ────────────────────────────────────────────────
            tick_elapsed = time.perf_counter() - tick_start
            sleep_time = cfg.LOOP_DT - tick_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[main] Stopped by Ctrl+C.")

    finally:
        print("\n[SAFETY] Zeroing throttle and steering...")
        try:
            zero_leds = np.zeros(8, dtype=np.float64)
            sensors.write_command(throttle=0.0, steering=0.0, leds=zero_leds)
            time.sleep(0.1)
            sensors.write_command(throttle=0.0, steering=0.0, leds=zero_leds)
        except Exception as e:
            print(f"  Warning: {e}")

        if yolo:
            yolo.stop()
        if dash:
            dash.stop()
        sensors.close()

        total = time.perf_counter() - start_t
        hz = tick_count / max(total, 1)
        print(f"\n  Run time: {total:.1f}s")
        print(f"  Ticks:    {tick_count} ({hz:.1f} Hz)")
        print("[main] Shutdown complete.")


if __name__ == "__main__":
    main()
