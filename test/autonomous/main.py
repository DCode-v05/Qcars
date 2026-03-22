#!/usr/bin/env python3
"""
main.py — QCar 2 Autonomous Obstacle Avoidance (360-degree)

USAGE:
    cd ~/Qcars/test/autonomous
    python main.py --model $MODELSPATH/yolov8n.pt

Architecture:
    sensors.py          → QCar + LiDAR + 4 CSI cameras
    yolo_processor.py   → single YOLO model, 4-camera round-robin (1 thread)
    obstacle_detector.py→ VFH path planner + LiDAR + YOLO fusion
    navigator.py        → steering/throttle decisions (always 0.1)
    dashboard.py        → optional web UI

Pipeline per tick:
    1. Read sensors (QCar + LiDAR + cameras)
    2. Push camera frames to YOLO queue
    3. Obstacle detection (LiDAR VFH + YOLO detections)
    4. Navigate: choose steering based on best path
    5. Send throttle + steering to car
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


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True
    print(f"\n[main] Signal {signum} — shutting down...")


def _key_listener():
    """Background thread: sets _shutdown when 'q' is pressed."""
    global _shutdown
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while not _shutdown:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = sys.stdin.read(1)
                if ch.lower() == 'q':
                    _shutdown = True
                    print("\n[main] 'q' pressed — shutting down...")
                    break
    except Exception:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def main():
    global _shutdown

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="QCar 2 Autonomous Driving")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="YOLO model path (default: yolov8n.pt)")
    ap.add_argument("--timeout", type=float, default=0,
                    help="Optional timeout in seconds (0=run until 'q')")
    ap.add_argument("--no-yolo", action="store_true",
                    help="Disable YOLO (LiDAR-only obstacle avoidance)")
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
    print(f"  Cameras  : 4 CSI ({cfg.CSI_WIDTH}x{cfg.CSI_HEIGHT} @ {cfg.CSI_FPS}fps)")
    print(f"  LiDAR    : {cfg.LIDAR_NUM_MEAS} measurements")
    print(f"  Timeout  : {args.timeout:.0f}s")
    print(f"  Dashboard: {'ON' if args.dashboard else 'OFF'}")
    print("-" * 62)

    # ── Build modules ─────────────────────────────────────────────────────
    sensors  = SensorManager()
    detector = ObstacleDetector()
    nav      = Navigator()

    # Camera frame queues (for YOLO processor)
    cam_queues = [queue.Queue(maxsize=2) for _ in cfg.CAMERA_CONFIG]

    # Dashboard (optional)
    dash = None
    stores = None
    if args.dashboard:
        from dashboard import Dashboard, CamStore
        stores = [CamStore(c["id"]) for c in cfg.CAMERA_CONFIG]
        dash = Dashboard(stores)

    # YOLO processor
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

    # Start 'q' key listener
    threading.Thread(target=_key_listener, daemon=True, name="keylistener").start()

    print(f"\n[main] Running. Press 'q' to stop.\n")

    try:
        while not _shutdown:
            tick_start = time.perf_counter()
            elapsed    = tick_start - start_t

            if args.timeout > 0 and elapsed >= args.timeout:
                print(f"\n[main] Timeout after {elapsed:.1f}s.")
                break

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

            # ── 5. Obstacle detection (LiDAR + YOLO) ─────────────────────
            yolo_dets = yolo.get_detections() if yolo else []
            detection = detector.detect(data, yolo_dets)

            # ── 6. Navigate ───────────────────────────────────────────────
            nav_result = nav.update(detection, dt)
            throttle   = nav_result['throttle']
            steering   = nav_result['steering']
            state      = nav_result['state']
            leds       = nav_result['leds']

            # ── 7. Send to hardware ───────────────────────────────────────
            sensors.write_command(throttle=throttle, steering=steering,
                                 leds=leds)

            # ── 7b. Push data to dashboard ────────────────────────────────
            if dash:
                dash.update_lidar(data['lidar_distances'],
                                  data['lidar_angles'],
                                  data['lidar_valid'])
                dash.update_nav(detection, nav_result)
                dash.update_detections(yolo_dets)

            # ── 8. Terminal print every 0.5s ──────────────────────────────
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

                print(
                    f"[{elapsed:5.1f}s] {state:<11} "
                    f"zone={zone:<5} dist={dist:.2f}m rear={rear:.2f}m "
                    f"path={fwd_str} {path_deg:+.0f}deg gap={gap_w:.0f}deg "
                    f"obs={obs_type:<7} "
                    f"thr={throttle:+.2f} str={steering:+.3f}"
                )
                last_print = elapsed

            # ── 9. Pacing ─────────────────────────────────────────────────
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
