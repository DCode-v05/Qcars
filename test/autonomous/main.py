#!/usr/bin/env python3
"""
main.py — QCar 2 Autonomous Navigation: Point A to Point B

Builds an occupancy grid from LiDAR, uses A* to find global path,
DWA for local obstacle avoidance along the path.

Controls:
    q       — quit
    Ctrl+C  — quit
    d       — force DRIVE (forward) mode
    r       — force REVERSE mode
    a       — back to AUTO mode

USAGE:
    python main.py --model $MODELSPATH/yolov8n.pt --dashboard --goal-x 2.0 --goal-y 0.0
"""
import argparse
import atexit
import math
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
from localizer import Localizer
from occupancy_grid import OccupancyGrid
from path_planner import PathPlanner


_shutdown = threading.Event()
_nav_ref = None


def _signal_handler(signum, frame):
    _shutdown.set()
    print(f"\n[main] Signal {signum} -- shutting down...")


def _key_listener():
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while not _shutdown.is_set():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = sys.stdin.read(1).lower()
                if ch == 'q':
                    _shutdown.set()
                    print("\n[main] 'q' pressed -- shutting down...")
                    break
                elif ch == 'd' and _nav_ref:
                    _nav_ref.set_manual('d')
                    print("\r[KEY] MANUAL FORWARD         ", end="")
                elif ch == 'r' and _nav_ref:
                    _nav_ref.set_manual('r')
                    print("\r[KEY] MANUAL REVERSE         ", end="")
                elif ch == 'a' and _nav_ref:
                    _nav_ref.set_manual(None)
                    print("\r[KEY] AUTO MODE              ", end="")
    except Exception:
        pass
    finally:
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except Exception:
            pass


def _force_shutdown(sensors, yolo, dash):
    try:
        zero_leds = np.zeros(8, dtype=np.float64)
        sensors.write_command(throttle=0.0, steering=0.0, leds=zero_leds)
    except Exception:
        pass
    try:
        sensors.close()
    except Exception:
        pass
    if yolo:
        try:
            yolo.stop()
        except Exception:
            pass


def main():
    global _nav_ref

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="QCar 2 A-to-B Navigation")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO model path")
    ap.add_argument("--no-yolo", action="store_true", help="Disable YOLO")
    ap.add_argument("--dashboard", action="store_true", help="Web dashboard")
    ap.add_argument("--goal-x", type=float, default=None,
                    help="Goal X in meters (from start)")
    ap.add_argument("--goal-y", type=float, default=None,
                    help="Goal Y in meters (from start)")
    ap.add_argument("--no-goal", action="store_true",
                    help="Reactive mode only (no A* path planning)")
    args = ap.parse_args()

    model_path = os.path.expandvars(args.model)

    signal.signal(signal.SIGINT, _signal_handler)
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
    except (OSError, AttributeError):
        pass

    # Determine goal
    has_goal = not args.no_goal and args.goal_x is not None
    goal_x = args.goal_x if args.goal_x is not None else 0.0
    goal_y = args.goal_y if args.goal_y is not None else 0.0

    print("=" * 62)
    print("  QCar 2 -- Point A to Point B Navigation")
    print("=" * 62)
    print(f"  Throttle : {cfg.THROTTLE}")
    print(f"  YOLO     : {'OFF' if args.no_yolo else model_path}")
    print(f"  LiDAR    : {cfg.LIDAR_NUM_MEAS} meas, range mode {cfg.LIDAR_RANGE_MODE}")
    print(f"  Dashboard: {'ON' if args.dashboard else 'OFF'}")
    if has_goal:
        print(f"  Goal     : ({goal_x:.2f}, {goal_y:.2f}) m")
    else:
        print(f"  Goal     : NONE (reactive mode)")
    print("-" * 62)
    print("  Controls: q=quit  d=drive  r=reverse  a=auto")
    print("-" * 62)

    # ── Build modules ─────────────────────────────────────────────────────
    sensors   = SensorManager()
    detector  = ObstacleDetector()
    nav       = Navigator()
    localizer = Localizer()
    grid      = OccupancyGrid()
    planner   = PathPlanner(grid)
    _nav_ref  = nav

    if has_goal:
        planner.set_goal(goal_x, goal_y)
        logging.info(f"[main] Goal set: ({goal_x:.2f}, {goal_y:.2f})")

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
    atexit.register(_force_shutdown, sensors, yolo, dash)

    if yolo:
        yolo.start()
        logging.info("[main] YOLO processor started")

    if dash:
        dash.start()
        logging.info(f"[main] Dashboard on http://0.0.0.0:{cfg.DASHBOARD_PORT}")

    nav.reset()
    localizer.reset()
    grid.reset()

    # ── Timing ────────────────────────────────────────────────────────────
    start_t       = time.perf_counter()
    last_t        = start_t
    last_print    = 0.0
    last_grid_t   = 0.0
    last_replan_t = 0.0
    tick_count    = 0
    goal_reached  = False
    last_throttle = cfg.THROTTLE

    threading.Thread(target=_key_listener, daemon=True, name="keys").start()
    print(f"\n[main] Running. Press 'q' to stop.\n")

    try:
        while not _shutdown.is_set():
            tick_start = time.perf_counter()
            elapsed    = tick_start - start_t

            # ── 1. Read sensors ───────────────────────────────────────────
            data = sensors.read()
            tick_count += 1

            if _shutdown.is_set():
                break

            # ── 2. Compute dt ─────────────────────────────────────────────
            now = time.perf_counter()
            dt  = max(now - last_t, 1e-4)
            last_t = now

            # ── 3. Update position (dead reckoning) ───────────────────────
            throttle_sign = 1.0 if last_throttle >= 0 else -1.0
            pose = localizer.update(data, dt, throttle_sign)

            # ── 4. Update occupancy grid from LiDAR (at GRID_UPDATE_HZ) ──
            if elapsed - last_grid_t >= 1.0 / cfg.GRID_UPDATE_HZ:
                grid.update(pose,
                            data['lidar_distances'],
                            data['lidar_angles'],
                            data['lidar_valid'])
                last_grid_t = elapsed

            # ── 5. A* replan (at ASTAR_REPLAN_HZ) ────────────────────────
            if has_goal and not goal_reached:
                if elapsed - last_replan_t >= 1.0 / cfg.ASTAR_REPLAN_HZ:
                    if planner.needs_replan(pose):
                        planner.replan(pose)
                        if planner.path_found:
                            logging.info(f"[A*] Path found ({len(planner.get_path_points())} waypoints)")
                        else:
                            logging.info("[A*] No path found -- reactive mode")
                    last_replan_t = elapsed

                # Check goal reached
                if planner.is_goal_reached(pose):
                    goal_reached = True
                    logging.info(f"[GOAL] REACHED! at ({pose[0]:.2f}, {pose[1]:.2f})")
                    print(f"\n*** GOAL REACHED at ({pose[0]:.2f}, {pose[1]:.2f}) ***\n")

            # ── 6. Compute goal heading for DWA ───────────────────────────
            if has_goal and not goal_reached and planner.path_found:
                goal_heading = planner.get_goal_heading(pose)
                if goal_heading is not None:
                    detector.set_goal_heading(goal_heading)
                else:
                    detector.set_goal_heading(0.0)
            else:
                detector.set_goal_heading(0.0)

            # ── 7. Push frames to YOLO queues ─────────────────────────────
            frames = data['csi_frames']
            for i, frame in enumerate(frames):
                q = cam_queues[i]
                if q.full():
                    try: q.get_nowait()
                    except queue.Empty: pass
                try: q.put_nowait(frame)
                except queue.Full: pass

            # ── 8. Battery check ──────────────────────────────────────────
            batt = data['battery_voltage']
            if batt < cfg.BATTERY_CRIT_V and batt > 1.0:
                print(f"\n[SAFETY] Battery critical: {batt:.2f}V -- stopping.")
                break

            # ── 9. Obstacle detection (LiDAR + YOLO + depth) ─────────────
            yolo_dets = yolo.get_detections() if yolo else []
            detection = detector.detect(data, yolo_dets)

            # ── 10. Navigate ──────────────────────────────────────────────
            imu_accel = data.get('accelerometer')
            imu_gyro  = data.get('gyroscope')
            nav_result = nav.update(detection, dt,
                                    imu_accel=imu_accel, imu_gyro=imu_gyro)
            throttle = nav_result['throttle']
            steering = nav_result['steering']
            state    = nav_result['state']
            leds     = nav_result['leds']

            # Stop if goal reached
            if goal_reached:
                throttle = 0.0
                steering = 0.0
                state = "ARRIVED"

            last_throttle = throttle

            if nav_result.get('crash_detected'):
                print(f"\n[IMU] CRASH DETECTED -- emergency reverse!")
            if nav_result.get('oscillation_detected'):
                print(f"\n[IMU] OSCILLATION DETECTED -- recovery!")

            # ── 11. Send to hardware ──────────────────────────────────────
            sensors.write_command(throttle=throttle, steering=steering,
                                 leds=leds)

            # ── 12. Push data to dashboard ────────────────────────────────
            if dash:
                dash.update_lidar(data['lidar_distances'],
                                  data['lidar_angles'],
                                  data['lidar_valid'])
                dash.update_nav(detection, nav_result)
                dash.update_detections(yolo_dets)
                dash.update_realsense(data['rs_rgb'], data['rs_depth_m'])
                # Push map + path (at lower rate to save bandwidth)
                if tick_count % 10 == 0:
                    map_pts = grid.get_map_data()
                    path_pts = planner.get_path_points() if has_goal else []
                    dash.update_map(map_pts, path_pts, pose,
                                   (goal_x, goal_y) if has_goal else None)

            # ── 13. Terminal print every 0.5s ─────────────────────────────
            if elapsed - last_print >= 0.5:
                zone     = detection['zone']
                dist     = detection['distance_m']
                rear     = detection['rear_min_m']
                obs_type = detection['obstacle_type']
                fwd_str  = "FWD" if detection['drive_forward'] else "REV"
                path_deg = np.degrees(detection['best_path_angle'])
                if path_deg > 180:
                    path_deg -= 360

                mode = ""
                if state == "ARRIVED":
                    mode = " [GOAL REACHED]"
                elif state.startswith("MANUAL"):
                    mode = " [MANUAL]"
                elif state == "RECOVERING":
                    mode = " [RECOVERY]"

                pos_str = f"pos=({pose[0]:+.2f},{pose[1]:+.2f})"
                goal_str = ""
                if has_goal:
                    dx = goal_x - pose[0]
                    dy = goal_y - pose[1]
                    goal_dist = math.sqrt(dx * dx + dy * dy)
                    goal_str = f" goal={goal_dist:.2f}m"
                    path_status = "A*" if planner.path_found else "??"
                    goal_str += f" [{path_status}]"

                print(
                    f"[{elapsed:5.1f}s] {state:<11}{mode} "
                    f"{pos_str} {goal_str} "
                    f"dist={dist:.2f}m "
                    f"thr={throttle:+.2f} str={steering:+.3f}"
                )
                last_print = elapsed

            # ── 14. Pacing ────────────────────────────────────────────────
            tick_elapsed = time.perf_counter() - tick_start
            sleep_time = cfg.LOOP_DT - tick_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[main] Stopped by Ctrl+C.")

    finally:
        _shutdown.set()
        print("\n[SAFETY] Zeroing throttle and steering...")
        try:
            zero_leds = np.zeros(8, dtype=np.float64)
            for _ in range(3):
                sensors.write_command(throttle=0.0, steering=0.0, leds=zero_leds)
                time.sleep(0.05)
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
        print(f"  Final pos: ({pose[0]:.2f}, {pose[1]:.2f})")
        if has_goal:
            dx = goal_x - pose[0]
            dy = goal_y - pose[1]
            print(f"  Goal dist: {math.sqrt(dx*dx+dy*dy):.2f}m")
        print("[main] Shutdown complete.")

        os._exit(0)


if __name__ == "__main__":
    main()
