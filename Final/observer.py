"""
observer.py  —  THE ONLY FILE YOU RUN
═══════════════════════════════════════════════════════════════════════════════
Autonomous obstacle-avoidance navigation for QCar 2.
Mission: Drive 2.0m forward, avoid all obstacles, stop at destination.

USAGE:
  cd ~/Qcars/test/stop_sign_nav
  python3 observer.py

WHAT IT DOES EVERY TICK (30Hz):
  1. sensors.read()        → all 5 sensors
  2. estimator.update()    → EKF pose [x, y, θ]
  3. detector.detect()     → LiDAR + YOLO + depth fusion → behaviour
  4. navigator.update()    → heading error + distance to goal
  5. sm.update()           → FSM state → throttle + steering
  6. lights.get_leds()     → LED array for current state
  7. qcar.read_write_std() → send commands to hardware
  8. dashboard.update()    → push all data to browser

SAFETY:
  - MAX_THROTTLE = 0.15 (slow — safe for testing)
  - Ctrl+C always zeroes motors before closing
  - try/finally guarantees sensors.close() runs even on crash

DASHBOARD:
  Open http://<jetson-ip>:5000 in your browser while this runs.

DO NOT RUN ANY OTHER SCRIPT WHILE THIS IS RUNNING.
One process, one hardware access. Always.
═══════════════════════════════════════════════════════════════════════════════
"""

import time
import numpy as np
import signal
import sys

from perception       import SensorManager, Config as PerceptionConfig
from estimation       import PoseEstimator, EstimationConfig
from perceiver        import Perceiver, PerceiverConfig
from obstacle_detector import ObstacleDetector, DetectorConfig
from navigator        import Navigator, NavigatorConfig
from lights           import LightController
from state_machine    import StateMachine, StateMachineConfig
from dashboard        import Dashboard


# ═════════════════════════════════════════════════════════════════════════════
#  MISSION PARAMETERS  —  edit here before each run
# ═════════════════════════════════════════════════════════════════════════════

GOAL_DISTANCE_M  = 2.0     # metres forward to drive
MAX_THROTTLE     = 0.15    # safety ceiling — increase only after testing
DASHBOARD_PORT   = 5000
RUN_TIMEOUT_S    = 60.0    # hard timeout — stops even if goal not reached


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # ── Build all modules ─────────────────────────────────────────────────
    p_cfg  = PerceptionConfig()
    e_cfg  = EstimationConfig()
    pc_cfg = PerceiverConfig()
    d_cfg  = DetectorConfig()
    n_cfg  = NavigatorConfig()
    n_cfg.GOAL_DISTANCE_M = GOAL_DISTANCE_M
    sm_cfg = StateMachineConfig()
    sm_cfg.MAX_THROTTLE   = MAX_THROTTLE

    sensors   = SensorManager(p_cfg)
    estimator = PoseEstimator(e_cfg)
    perceiver = Perceiver(pc_cfg)
    detector  = ObstacleDetector(d_cfg, perceiver)
    navigator = Navigator(n_cfg)
    lights    = LightController()
    sm        = StateMachine(sm_cfg)
    dash      = Dashboard(port=DASHBOARD_PORT)

    print("═" * 62)
    print("  QCar 2 — Autonomous Obstacle Avoidance")
    print(f"  Goal: drive {GOAL_DISTANCE_M:.1f}m forward, avoid obstacles, stop.")
    print(f"  Max throttle: {MAX_THROTTLE}")
    print(f"  Timeout: {RUN_TIMEOUT_S:.0f}s")
    print("  Ctrl+C to stop at any time.")
    print("═" * 62)

    # ── Open hardware (ONE process, all sensors) ───────────────────────────
    sensors.open()          # QCar + RealSense + LiDAR + CSI cameras
    perceiver.open()        # YOLOv8 TensorRT engine
    dash.start()            # Flask dashboard (background thread)
    estimator.reset()       # EKF at origin
    sm.reset()              # FSM → NAVIGATING

    # Get first sensor reading to set navigator goal from initial pose
    first_data = sensors.read()
    first_pose = estimator.update(first_data, dt=0.033)
    navigator.reset(first_pose)

    # ── Convenience: get QCar object for read_write_std ───────────────────
    qcar = sensors._qcar

    # ── Timing state ──────────────────────────────────────────────────────
    start_t      = time.perf_counter()
    tick_start   = start_t
    last_t       = start_t
    last_print   = 0.0
    tick_count   = 0
    lidar_ticks  = 0
    dt_history   = []

    print(f"\n[observer] Running. Dashboard: open browser at port {DASHBOARD_PORT}\n")

    try:
        while True:
            now     = time.perf_counter()
            elapsed = now - start_t

            # Hard timeout
            if elapsed >= RUN_TIMEOUT_S:
                print(f"\n[observer] Timeout after {elapsed:.1f}s.")
                break

            # ── 1. Read all sensors ───────────────────────────────────────
            data = sensors.read()
            tick_count += 1

            # ── 2. Compute dt ─────────────────────────────────────────────
            dt = max(now - last_t, 1e-4)
            last_t = now
            dt_history.append(dt)
            if len(dt_history) > 30:
                dt_history.pop(0)

            if data['lidar_new_scan']:
                lidar_ticks += 1

            # ── 3. Pose estimation ────────────────────────────────────────
            pose = estimator.update(data, dt, steering_rad=0.0)

            # ── 4. Obstacle detection (LiDAR + YOLO + depth) ─────────────
            detection = detector.detect(data)

            # ── 5. Navigation ─────────────────────────────────────────────
            nav_result = navigator.update(pose, dt)

            # ── 6. State machine (decides throttle + steering) ────────────
            sm_result = sm.update(detection, nav_result, data, dt)

            throttle  = sm_result['throttle']
            steering  = sm_result['steering']
            sm_state  = sm_result['state']

            # Update estimator with actual steering command
            pose = estimator.update(data, dt, steering_rad=steering)

            # ── 7. Lights ─────────────────────────────────────────────────
            leds = lights.get_leds(sm_state, detection['avoid_side'])

            # ── 8. Send to hardware ───────────────────────────────────────
            qcar.read_write_std(throttle=throttle, steering=steering, LEDs=leds)

            # ── 9. Dashboard update ───────────────────────────────────────
            loop_ms  = float(np.mean(dt_history)) * 1000.0
            lidar_hz = lidar_ticks / max(elapsed, 1.0)
            timing   = {
                'loop_ms':   loop_ms,
                'lidar_hz':  lidar_hz,
                'ekf_ticks': tick_count,
            }
            dash.update(
                detection   = detection,
                pose        = pose,
                speed_mps   = sm_result['speed_mps'],
                dist_goal   = nav_result['distance_remaining'],
                leds        = leds.tolist(),
                timing      = timing,
                sm_state    = sm_state,
                throttle    = throttle,
                steering    = steering,
                heading_err = nav_result['heading_error'],
            )

            # ── 10. Terminal print every 0.5s ─────────────────────────────
            if elapsed - last_print >= 0.5:
                x, y, theta = pose
                dist_rem    = nav_result['distance_remaining']
                zone        = detection['zone']
                beh         = detection['behaviour']
                otype       = detection['obstacle_type']
                led_desc    = lights.describe(leds)
                print(
                    f"[{elapsed:6.1f}s] {sm_state:<11}  "
                    f"x={x:+.3f}m  y={y:+.3f}m  θ={np.degrees(theta):+.1f}°  "
                    f"goal={dist_rem:.2f}m  "
                    f"zone={zone:<5}  beh={beh:<15}  type={otype:<6}  "
                    f"thr={throttle:+.3f}  str={steering:+.3f}  "
                    f"LEDs={led_desc}  "
                    f"dt={loop_ms:.1f}ms"
                )
                last_print = elapsed

            # ── Check ARRIVED ─────────────────────────────────────────────
            if sm_state == 'ARRIVED':
                print(f"\n[observer] ARRIVED! Goal reached at {elapsed:.1f}s.")
                print(f"  Final pose: x={pose[0]:+.4f}m  y={pose[1]:+.4f}m")
                time.sleep(1.0)
                break

            # ── Soft real-time pacing ─────────────────────────────────────
            tick_elapsed = time.perf_counter() - tick_start
            sleep_time   = p_cfg.LOOP_DT - tick_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            tick_start = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[observer] Stopped by user (Ctrl+C).")

    finally:
        # SAFETY: always zero motors before closing
        print("\n[SAFETY] Zeroing throttle and steering...")
        try:
            qcar.read_write_std(throttle=0.0, steering=0.0,
                                LEDs=np.zeros(8, dtype=np.float64))
            time.sleep(0.1)
            qcar.read_write_std(throttle=0.0, steering=0.0,
                                LEDs=np.zeros(8, dtype=np.float64))
        except Exception as e:
            print(f"  Warning during motor zero: {e}")

        perceiver.close()
        sensors.close()
        print("\n[observer] Shutdown complete.")

        # Final summary
        if len(dt_history) > 0:
            total = time.perf_counter() - start_t
            print(f"\n  Run time:   {total:.1f}s")
            print(f"  Ticks:      {tick_count}")
            print(f"  Loop mean:  {np.mean(dt_history)*1000:.1f}ms")
            print(f"  LiDAR scans:{lidar_ticks} ({lidar_ticks/max(total,1):.1f}Hz)")


if __name__ == "__main__":
    main()
