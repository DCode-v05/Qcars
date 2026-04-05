"""
observer.py  —  THE ONLY FILE YOU RUN
═══════════════════════════════════════════════════════════════════════════════
Autonomous obstacle-avoidance navigation for QCar 2.
Mission: Drive 2.0m forward, avoid all obstacles, stop at destination.

USAGE:
  cd ~/Final
  python3 observer.py

PERFORMANCE FIX:
  - Dashboard rendering runs in background thread (0ms in main loop)
  - YOLO pre-warmed during sensor warmup (no lazy TRT load during driving)
  - Single EKF update per tick (encoder + gyro fusion)
  - Target loop rate: 10-20Hz (was 1.5Hz with matplotlib blocking)

AUTONOMY:
  - Forward throttle is default
  - Front blocked + rear clear → immediate reverse
  - Front blocked + rear blocked → stop and wait
  - Proportional avoidance steering (softer far, harder close)

SAFETY:
  - MAX_THROTTLE = 0.10 (safe for testing)
  - Ctrl+C / SIGINT → graceful shutdown
  - Battery critical → auto-stop
  - try/finally always zeroes motors
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

GOAL_DISTANCE_M  = 15.0    # metres — large enough for 30s of autonomous driving
MAX_THROTTLE     = 0.10    # safety ceiling — increase only after testing
DASHBOARD_PORT   = 5000
RUN_TIMEOUT_S    = 35.0    # hard timeout — car runs for ~30s then stops safely


# ═════════════════════════════════════════════════════════════════════════════
#  SIGNAL HANDLER
# ═════════════════════════════════════════════════════════════════════════════

_shutdown_requested = False

def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    print(f"\n[observer] Received {sig_name} — shutting down...")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    global _shutdown_requested

    signal.signal(signal.SIGINT, _signal_handler)
    try:
        signal.signal(signal.SIGTERM, _signal_handler)
    except (OSError, AttributeError):
        pass

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

    # ── Open hardware ─────────────────────────────────────────────────────
    sensors.open()
    perceiver.open()

    # Pre-warm YOLO: forces TensorRT engine load NOW, not during driving
    first_data = sensors.read()
    perceiver.warmup(
        bgr_frame=first_data['csi_front'],
        depth_frame=first_data['rs_depth_m'],
    )

    dash.start()
    estimator.reset()
    sm.reset()

    # Set navigator goal from initial pose
    first_pose = estimator.update(first_data, dt=0.033)
    navigator.reset(first_pose)

    # ── Timing ────────────────────────────────────────────────────────────
    start_t     = time.perf_counter()
    last_t      = start_t
    last_print  = 0.0
    tick_count  = 0
    lidar_ticks = 0
    dt_sum      = 0.0
    dt_count    = 0

    print(f"\n[observer] Running. Dashboard: http://..:{DASHBOARD_PORT}\n")

    try:
        while not _shutdown_requested:
            tick_start = time.perf_counter()
            elapsed    = tick_start - start_t

            if elapsed >= RUN_TIMEOUT_S:
                print(f"\n[observer] Timeout after {elapsed:.1f}s.")
                break

            # ── 1. Read sensors ───────────────────────────────────────────
            data = sensors.read()
            tick_count += 1

            # ── 2. Compute dt ─────────────────────────────────────────────
            now = time.perf_counter()
            dt  = max(now - last_t, 1e-4)
            last_t = now
            dt_sum   += dt
            dt_count += 1

            if data['lidar_new_scan']:
                lidar_ticks += 1

            # ── 3. Pose estimation (single call: encoder + gyro) ──────────
            pose = estimator.update(data, dt)

            # ── 4. Compute heading to goal (for path planner bias) ───────
            dx = navigator._goal_x - pose[0]
            dy = navigator._goal_y - pose[1]
            goal_heading_world = float(np.arctan2(dy, dx))
            # Convert to body frame: how far the goal is from where we're facing
            goal_heading_body = goal_heading_world - pose[2]
            # Normalize to [0, 2π] for the LiDAR angle convention
            goal_heading_lidar = goal_heading_body % (2 * np.pi)
            detector.set_goal_heading(goal_heading_lidar)

            # ── 5. Obstacle detection (LiDAR + YOLO + depth + path plan) ─
            detection = detector.detect(data)

            # ── 6. Navigation ─────────────────────────────────────────────
            nav_result = navigator.update(pose, dt)

            # ── 7. State machine ──────────────────────────────────────────
            sm_result = sm.update(detection, nav_result, data, dt)
            throttle  = sm_result['throttle']
            steering  = sm_result['steering']
            sm_state  = sm_result['state']

            if not sm_result.get('battery_ok', True):
                print(f"\n[observer] Battery critical — stopping.")
                break

            # ── 7. Lights ─────────────────────────────────────────────────
            leds = lights.get_leds(sm_state, detection['avoid_side'])

            # ── 8. Send to hardware ───────────────────────────────────────
            sensors.write_command(throttle=throttle, steering=steering, leds=leds)

            # ── 9. Dashboard (instant — rendering is in background) ───────
            avg_dt_ms = (dt_sum / max(dt_count, 1)) * 1000.0
            lidar_hz  = lidar_ticks / max(elapsed, 1.0)
            timing = {
                'loop_ms':   avg_dt_ms,
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
                rear_m      = detection.get('rear_min_m', 99.0)
                rear_str    = f"{rear_m:.1f}" if rear_m < 90 else "---"
                gap_w       = detection.get('best_gap_width_deg', 0)
                path_ang    = detection.get('best_path_angle', 0)
                path_ang_n  = path_ang if path_ang <= np.pi else path_ang - 2*np.pi
                fwd_str     = "FWD" if detection.get('drive_forward', True) else "REV"
                print(
                    f"[{elapsed:5.1f}s] {sm_state:<11}  "
                    f"x={x:+.2f} y={y:+.2f} θ={np.degrees(theta):+.0f}°  "
                    f"goal={dist_rem:.2f}m  "
                    f"zone={zone:<5} beh={beh:<12}  "
                    f"path={fwd_str} {np.degrees(path_ang_n):+.0f}° gap={gap_w:.0f}°  "
                    f"rear={rear_str}m  "
                    f"thr={throttle:+.3f} str={steering:+.2f}  "
                    f"dt={avg_dt_ms:.0f}ms"
                )
                last_print = elapsed

            # ── Check ARRIVED ─────────────────────────────────────────────
            if sm_state == 'ARRIVED':
                print(f"\n[observer] ARRIVED at {elapsed:.1f}s.  "
                      f"x={pose[0]:+.3f}m  y={pose[1]:+.3f}m")
                time.sleep(1.0)
                break

            # ── Soft real-time pacing ─────────────────────────────────────
            tick_elapsed = time.perf_counter() - tick_start
            sleep_time   = p_cfg.LOOP_DT - tick_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[observer] Stopped by user (Ctrl+C).")

    finally:
        print("\n[SAFETY] Zeroing throttle and steering...")
        try:
            zero_leds = np.zeros(8, dtype=np.float64)
            sensors.write_command(throttle=0.0, steering=0.0, leds=zero_leds)
            time.sleep(0.1)
            sensors.write_command(throttle=0.0, steering=0.0, leds=zero_leds)
        except Exception as e:
            print(f"  Warning during motor zero: {e}")

        dash.stop()
        perceiver.close()
        sensors.close()
        print("\n[observer] Shutdown complete.")

        if dt_count > 0:
            total = time.perf_counter() - start_t
            print(f"\n  Run time:    {total:.1f}s")
            print(f"  Ticks:       {tick_count}")
            print(f"  Loop mean:   {(dt_sum/dt_count)*1000:.0f}ms  ({dt_count/max(total,1):.1f}Hz)")
            print(f"  LiDAR scans: {lidar_ticks} ({lidar_hz:.1f}Hz)")


if __name__ == "__main__":
    main()
