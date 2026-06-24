# Qcars — Autonomous Obstacle-Avoidance for the Quanser QCar 2

**A full self-driving stack for the 1/10-scale Quanser QCar 2: read every sensor, fuse them into a pose, find a gap through the obstacles, and drive to a goal — with a live web dashboard watching the whole thing.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) ![YOLOv8](https://img.shields.io/badge/YOLOv8-seg-00FFFF?style=flat&logo=ultralytics&logoColor=black) ![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white) ![NVIDIA Jetson](https://img.shields.io/badge/Jetson%20Orin%20Nano-76B900?style=flat&logo=nvidia&logoColor=white) ![Quanser](https://img.shields.io/badge/Quanser%20QCar%202-FF6600?style=flat)

## Overview

The QCar 2 is a 1/10-scale self-driving car from Quanser, running on a Jetson Orin Nano with a LiDAR, IMU, wheel encoder, four CSI cameras, and an Intel RealSense depth camera. This repo is the autonomy stack I wrote for it: the car picks a goal point a fixed distance ahead, then drives toward it on its own while detecting and steering around whatever is in the way. If the front is blocked it backs up; if it sees a person it waits; if it loses every path it stops.

I built it during the iQube applied-robotics work and the two-day *Autonomous Vehicles Using QCar* workshop at the Kumaraguru School of Innovation (26-27 March 2026). The whole thing is a single Python control loop running on the car's Jetson at roughly 10-20 Hz, plus a Flask dashboard you open in a browser to watch the camera feed, LiDAR scan, pose, and state machine in real time.

The stack lives in `src/` and is split into eight modules with one clear job each (sensing, pose, vision, path planning, navigation, decision, lights, dashboard). The `tests/` folder holds the bring-up experiments I used to get each sensor working before wiring them together, and `vendor/` is the Quanser SDK the car ships with.

## Key Features

- **One-command autonomous run** — `observer.py` is the only file you launch. It opens every sensor, pre-warms the YOLO engine, sets a goal, and drives there.
- **Multi-sensor fusion** — RPLiDAR A2 (384 measurements per scan, 0.1-6 m), IMU gyroscope, wheel encoder, front CSI camera, and Intel RealSense D435 RGB-D, all read each tick.
- **EKF pose estimation** — an Extended Kalman Filter fuses encoder odometry with the bias-corrected gyroscope to track `(x, y, heading)` by dead reckoning. A correction step is wired in for future scan-matching or GPS measurements.
- **Vector Field Histogram path planner** — bins the full 360° LiDAR scan into 36 angular bins, finds contiguous open gaps, scores each gap on width, depth, and how well it lines up with the goal, then steers toward the winner.
- **Forward and reverse logic** — if the best gap is behind the car and the front is blocked, the planner switches to reversing with inverted steering instead of getting stuck.
- **YOLOv8s-seg object classification** — instance segmentation on the front camera classifies each detection as PERSON, MOVING (bike/car/dog), STATIC (boxes, chairs), or NONE, and that class changes the car's behaviour.
- **Camera/LiDAR correlation** — YOLO detections are matched back to LiDAR bearings so a "person" label only triggers a wait if there's an actual obstacle at that angle within range.
- **RealSense depth gap check** — a horizontal strip of the depth image is scanned for the widest clear column band, used as a second opinion on which side is open.
- **Finite state machine** — IDLE → NAVIGATING → AVOIDING → WAITING → REVERSING → STOPPED → ARRIVED, with a PID speed controller per state and different cruise speeds for driving, avoiding, and reversing.
- **Safety envelope** — a hard throttle ceiling (0.10 by default), battery low-voltage warning and critical auto-stop, a run timeout, rear-clear checks before reversing, graceful Ctrl+C / SIGTERM handling, and a `try/finally` that always zeroes the motors on exit.
- **Mapped LED behaviour** — the FSM state drives the QCar's 8-channel LED array: headlights while driving, blinking indicators on the avoidance side, hazards while waiting, brake lights while stopped, reverse warning while backing up.
- **Live Flask dashboard** — a web page at port 5000 that loads once and polls a JSON API, showing the annotated camera stream, a LiDAR polar plot, depth view, pose, distance to goal, loop rate, and current state. Rendering runs in a background thread so it adds 0 ms to the control loop.

## How It Works

The control loop in `observer.py` runs ten steps every tick. Each step is one module, and the modules pass plain dictionaries to each other.

### 1. Sensing — `perception.py`

`SensorManager` opens the QCar IMU/encoder, the RealSense D435 (RGB + depth in metres), the RPLiDAR A2, and the front CSI camera, in that order (cameras last to dodge a GStreamer timeout). It runs a 2-second warm-up so every buffer is populated before driving. Each `read()` returns one dictionary with accelerometer, gyroscope, motor speed and encoder counts, battery voltage, the camera frame, the depth frame in metres, and the LiDAR distances/angles with a validity mask. The sensor config targets 30 Hz; LiDAR is 384 measurements per scan with a 0.1-6.0 m valid window.

### 2. Pose estimation — `estimation.py`

`PoseEstimator` wraps Quanser's EKF. The prediction model is a unicycle: wheel speed comes from differencing encoder counts (converted to m/s through the real gear ratio and wheel radius in `constants.py`), and the heading rate comes straight from the gyroscope's z-axis with a measured bias of 0.024 rad/s subtracted. The Jacobian is computed analytically. Process noise on heading is kept tight because the gyro is trusted more than the motion model. There's also a `correct()` step ready for an external `(x, y, θ)` measurement.

### 3. Vision — `perceiver.py`

`Perceiver` runs YOLOv8s-seg (through Quanser's `pit.YOLO` wrapper, on the Jetson's TensorRT engine). The engine is pre-warmed once during startup so the first real frame doesn't pay the ~2-5 s deserialization cost mid-drive. Inference runs on a background thread: the control loop posts the latest frame and reads back the most recent result without blocking, so a 15-30 ms inference never stalls the loop. Detections are filtered to a COCO subset and mapped to PERSON / MOVING / STATIC / NONE, keeping only the nearest object inside the forward column and within 3 m.

### 4. Obstacle detection and path planning — `obstacle_detector.py`

This is the core. The Vector Field Histogram:

1. Splits the 360° scan into 36 bins (10° each) and records the minimum range per bin.
2. Marks a bin "open" if its nearest return is past the 0.6 m gap threshold.
3. Finds contiguous runs of open bins, handling wrap-around by doubling the array, and keeps gaps at least 3 bins (30°) wide.
4. Scores each gap by width, average depth, and angular distance from the goal heading — the goal term is weighted 2× so the car prefers gaps pointing where it wants to go.
5. Returns the best gap's centre angle, a proportional steering command clamped to ±30°, and a `drive_forward` flag (false when the only good gap is behind, which triggers reversing).

On top of the planner it also classifies a forward sector into CLEAR / WARN / STOP zones (warn at 1.5 m, stop at 0.4 m), checks a rear sector before any reverse, reads the RealSense depth strip for a gap side, and correlates YOLO detections with LiDAR bearings to decide the final behaviour: NAVIGATE, AVOID, WAIT, or EMERGENCY_STOP.

### 5. Navigation — `navigator.py`

`Navigator` fixes a goal point a set distance straight ahead of the car's starting pose, then runs a PD heading controller that keeps the nose pointed at that goal. Steering is rate-limited (max π/3 rad/s of change) so transitions out of an avoidance maneuver aren't jerky. It reports distance remaining and flips an `arrived` flag inside a 0.15 m radius.

### 6. Decision — `state_machine.py`

The FSM takes the planner's behaviour plus the navigator's output and picks a state. When the path is clear it uses the navigator's heading steering; when avoiding or reversing it uses the planner's steering directly (it already points at the best gap). A PID speed controller tracks per-state reference speeds (0.2 m/s cruise, 0.12 m/s avoiding, 0.10 m/s reverse) and everything is clamped to the throttle ceiling. It also handles the battery cutoffs, the wait timeout for people, the reverse duration, and a stuck-timer that forces a reverse if the car has been pinned too long.

### 7. Lights and output — `lights.py`

`LightController` maps the current state to the QCar's 8-channel LED array — headlights on while driving, the correct-side indicator blinking while avoiding, both indicators as hazards while waiting, brake lights while stopped, and a reverse warning while backing up. The throttle, steering, and LEDs are then written to the hardware with the steering-calibration offset applied.

### 8. Telemetry — `dashboard.py`

A Flask server holds shared state behind a lock. The HTML page loads once, then JavaScript polls `/api/data` for JSON and swaps the camera JPEG, LiDAR polar plot, and depth PNG with cache-busting — no full page reloads. All the Matplotlib rendering happens on a background thread, so the dashboard costs the main loop nothing.

## Results / Highlights

- Single control loop targeting **10-20 Hz** on the Jetson Orin Nano — the dashboard-in-a-thread and pre-warmed TensorRT fixes brought it up from ~1.5 Hz when Matplotlib was blocking the loop.
- **384-point** RPLiDAR A2 scans turned into a **36-bin** VFH gap search every new scan, with goal-biased scoring.
- YOLOv8s-seg running at **15-30 ms** per inference off the main thread, pre-warmed so there's no lazy load mid-drive.
- EKF pose from encoder + gyro with a measured **0.024 rad/s** gyro bias correction and the exact drivetrain gear ratio baked into the unit conversions.
- Default mission as configured in `observer.py`: drive **15 m** forward, max throttle **0.10**, **35 s** hard timeout, dashboard on port **5000**.

## Tech Stack

- **Language:** Python 3.8+ (Jetson Linux), plus an HTML/JS dashboard front end and a small shell installer.
- **Vision / ML:** YOLOv8s-seg via Quanser PIT on the Jetson TensorRT engine; OpenCV for image handling.
- **Numerics / control:** NumPy, an Extended Kalman Filter, Vector Field Histogram path planning, PID/PD controllers, a finite state machine.
- **Web / telemetry:** Flask JSON API and stream, Matplotlib (Agg) for the LiDAR and depth plots.
- **Hardware / SDK:** Quanser QCar 2, NVIDIA Jetson Orin Nano, RPLiDAR A2, Intel RealSense D435, Quanser QUARC runtime with the `pal` / `hal` / `pit` / `qvl` Python libraries.

## Getting Started

### Prerequisites

- A Quanser QCar 2 with the Jetson Orin Nano image and QUARC runtime (this is what provides the `pal`, `hal`, and `pit` libraries — they aren't installable from PyPI).
- The RPLiDAR A2 and Intel RealSense D435 connected and powered.
- A charged battery (the stack warns below 10.5 V and auto-stops below 10.0 V).

### Installation

```bash
git clone https://github.com/DCode-v05/Qcars.git
cd Qcars
# Quanser libs (pal/hal/pit/qvl) are already on the QCar image.
# Extra Python packages:
pip install numpy opencv-python matplotlib flask
```

### Running

```bash
cd src
python3 observer.py
```

On launch the car initialises every sensor, warms up YOLO, sets a goal 15 m ahead of its current pose, drives autonomously while avoiding obstacles, and stops when it arrives or the 35 s timeout hits. Press Ctrl+C at any point — the `finally` block always zeroes the motors.

Then open the dashboard from any device on the same network:

```
http://<jetson-ip>:5000
```

## Usage

- **Tune the mission** at the top of `src/observer.py`: `GOAL_DISTANCE_M`, `MAX_THROTTLE`, `DASHBOARD_PORT`, and `RUN_TIMEOUT_S`. Start with a low throttle ceiling and raise it only after testing.
- **Tune behaviour** in the per-module config classes — VFH thresholds and bin count in `DetectorConfig`, PID gains in `NavigatorConfig` and `StateMachineConfig`, YOLO confidence in `PerceiverConfig`.
- **Bring up sensors individually** with the scripts under `tests/` (LiDAR, IMU, cameras, wheels, headlights, stereo mic) before running the full stack — most of them are standalone and print live readings.
- **Watch the dashboard** during a run for the annotated camera feed, LiDAR polar plot, pose, distance to goal, FSM state, and loop timing. The terminal also prints a one-line status every 0.5 s.

## Project Structure

```
Qcars/
├── src/                       # The autonomy stack — this is the project
│   ├── observer.py            # Main entry point; the 10-step control loop
│   ├── perception.py          # SensorManager — opens and reads every sensor
│   ├── estimation.py          # EKF pose (encoder + gyro fusion)
│   ├── perceiver.py           # YOLOv8s-seg, threaded + TensorRT pre-warm
│   ├── obstacle_detector.py   # VFH path planner + zone/depth/YOLO fusion
│   ├── navigator.py           # PD heading controller to the goal point
│   ├── state_machine.py       # FSM + per-state speed PID + safety
│   ├── lights.py              # FSM state -> 8-channel LED array
│   ├── dashboard.py           # Flask live telemetry server
│   └── constants.py           # Vehicle geometry, gear ratio, safety limits
│
├── tests/                     # Sensor bring-up and experiments
│   ├── autonomous/            # Earlier full-autonomy build (DWA planner, occupancy grid)
│   ├── imu/                   # IMU connect / read / fusion + drive-with-IMU
│   ├── lidar/                 # LiDAR discovery, viz, streaming server
│   ├── lidar_wheel/           # Combined LiDAR + motor safety, VFH driver, heatmaps
│   ├── light/                 # Headlight test
│   ├── multimedia/            # CSI streaming, RealSense RGBD
│   ├── stereo/                # Stereo mic capture
│   ├── stop_sign_nav/         # Stop-sign detection + navigation prototype
│   └── wheel/                 # Encoder, steering, rotation, calibration
│
├── vendor/                    # Quanser SDK shipped with the car (not my code)
│   ├── libraries/python/      # pal, hal, pit, qvl libraries
│   └── examples/sdcs/qcar2/   # Official QCar 2 example applications
│
├── docs/
│   └── QCar2_Python_Reference.pdf
└── README.md
```

---

## Contact

<table>
  <tr><td><b>Portfolio:</b> <a href="https://www.denistan.me">Denistan</a></td><td><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/denistanb">denistanb</a></td></tr>
  <tr><td><b>GitHub:</b> <a href="https://github.com/DCode-v05">DCode-v05</a></td><td><b>LeetCode:</b> <a href="https://leetcode.com/u/Denistan_B">Denistan_B</a></td></tr>
  <tr><td colspan="2" align="center"><b>Email:</b> <a href="mailto:denistanb05@gmail.com">denistanb05@gmail.com</a></td></tr>
</table>

Made with ❤️ by **Denistan B**
