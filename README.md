# Qcars

Autonomous obstacle avoidance for the Quanser QCar 2, a 1/10-scale car with a Jetson Orin Nano, an RPLiDAR A2, an IMU, a wheel encoder, CSI cameras and a RealSense D435. The car picks a goal a fixed distance ahead of its start pose, drives there, and steers around whatever is in the way. A Flask dashboard shows the run live in a browser.

I wrote it during iQube robotics work and the *Autonomous Vehicles Using QCar* workshop at Kumaraguru School of Innovation (26-27 March 2026).

## How it works

`src/observer.py` is the entry point. It runs one control loop (roughly 10-20 Hz on the Jetson) and each tick passes plain dictionaries between these modules:

| Module | Job |
|---|---|
| `perception.py` | `SensorManager` opens the IMU/encoder, RealSense, LiDAR and front CSI camera, warms them up for 2 s, and returns one reading per tick |
| `estimation.py` | EKF pose `(x, y, heading)` from encoder speed and the gyro z-axis, with a 0.024 rad/s gyro bias subtracted. It wraps Quanser's EKF |
| `perceiver.py` | YOLOv8s-seg through Quanser's `pit.YOLO`, run on a background thread and pre-warmed at startup. Classifies detections as person, moving, static or none |
| `obstacle_detector.py` | Vector Field Histogram planner, forward and rear safety zones, a RealSense depth-strip gap check, and matching of YOLO detections to LiDAR bearings |
| `navigator.py` | Fixes the goal ahead of the start pose and runs a rate-limited PD heading controller. Arrival radius is 0.15 m |
| `state_machine.py` | IDLE, NAVIGATING, AVOIDING, WAITING, REVERSING, STOPPED, ARRIVED, with a speed PID per state and battery cutoffs |
| `lights.py` | Maps the state to the 8-channel LED array (headlights, indicators, hazards, brake, reverse) |
| `dashboard.py` | Flask server with a JSON API; the page polls it for the camera frame, LiDAR polar plot, depth view, pose and state. Plots render on a background thread |

The planner bins the 360-degree scan into 36 bins of 10 degrees, marks a bin open if its nearest return is past 0.6 m, finds runs of at least 3 open bins, and scores each gap on width, depth and alignment with the goal (goal term weighted 2x). The front zone is WARN at 1.5 m and STOP at 0.4 m. If the front is blocked and the rear is clear it reverses; if a person is detected in the path it waits.

## Defaults

Set at the top of `observer.py`: goal 15 m ahead, max throttle 0.10, 35 s run timeout, dashboard on port 5000. The stack warns below 10.5 V and stops below 10.0 V. Cruise speeds in the state machine are 0.2 m/s driving, 0.12 m/s avoiding, 0.10 m/s reversing, all clamped to the throttle ceiling.

## Setup and run

You need a QCar 2 with the Quanser image and QUARC runtime, which provides `pal`, `hal`, `pit` and `qvl` (not on PyPI). Extra packages:

```bash
pip install numpy opencv-python matplotlib flask
cd src
python3 observer.py
```

Then open `http://<jetson-ip>:5000` from another device on the network. Ctrl+C is handled, and a `finally` block always zeroes the motors. Start with a low throttle ceiling and raise it only after testing.

## Layout

```
src/        the autonomy stack (modules above, plus constants.py for geometry, gear ratio and limits)
tests/      sensor bring-up scripts and earlier experiments
  autonomous/     earlier full build with a DWA planner and occupancy grid
  imu/ lidar/ lidar_wheel/ light/ multimedia/ stereo/ stop_sign_nav/ wheel/
vendor/     Quanser SDK and example apps that ship with the car (not my code, unmodified)
docs/       QCar2_Python_Reference.pdf
```

The scripts under `tests/` are mostly standalone and print live readings, so they are the way to check one sensor before running the full stack.

## Limitations

- Dead-reckoning pose only (encoder plus gyro). The EKF has a correction step, but nothing feeds it yet.
- Runs only on the car; the Quanser libraries are not installable elsewhere.
