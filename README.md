# QCar 2 — Autonomous Obstacle-Avoidance Navigation

## Project Description
This project implements a fully autonomous obstacle-avoidance navigation system for the **Quanser QCar 2**, a 1/10-scale self-driving car platform. The system fuses data from multiple sensors — LiDAR, IMU, wheel encoders, CSI cameras, and an Intel RealSense depth camera — to navigate toward a goal while detecting and avoiding obstacles in real time. Object classification is powered by **YOLOv8** running on the Jetson's TensorRT engine, and a live **Flask web dashboard** provides real-time telemetry during operation.

---

## Project Details

### Problem Statement
Autonomous navigation in cluttered environments requires real-time perception, localisation, and decision-making. The QCar 2 must drive a configurable distance (default 15m) from its start position, dynamically avoiding static and moving obstacles, and safely stop at its destination — all within a strict safety envelope.

### System Architecture
The autonomy stack is a modular pipeline executed in a single control loop at 10–30 Hz:

1. **Perception** (`perception.py`) — Reads all sensors: 4× CSI cameras, RealSense RGB-D, RPLiDAR A2 (384 beams), wheel encoder, IMU, and battery voltage.
2. **Estimation** (`estimation.py`) — Extended Kalman Filter (EKF) fusing encoder odometry + gyroscope for real-time pose estimation (x, y, heading).
3. **Perceiver** (`perceiver.py`) — YOLOv8s-seg inference on the front camera. Classifies detections as PERSON, MOVING, STATIC, or NONE to decide behaviour.
4. **Obstacle Detector** (`obstacle_detector.py`) — Vector Field Histogram (VFH) gap-finder on 360° LiDAR data. Finds the best drivable gap, biased toward the goal heading. Fuses RealSense depth for gap confirmation.
5. **Navigator** (`navigator.py`) — PID heading controller steering toward a fixed goal point set at startup.
6. **State Machine** (`state_machine.py`) — Finite state machine (IDLE → NAVIGATING → AVOIDING → WAITING → REVERSING → STOPPED → ARRIVED) with speed PID and battery safety.
7. **Lights** (`lights.py`) — Maps FSM state to QCar 2 LED array (headlights, indicators, brake lights).
8. **Dashboard** (`dashboard.py`) — Flask web server with live JSON API + annotated camera stream, LiDAR polar plot, pose, and timing.

### Key Algorithms
- **Extended Kalman Filter (EKF):** Encoder + gyroscope fusion with bias correction for accurate dead-reckoning.
- **Vector Field Histogram (VFH):** 36-bin angular histogram of LiDAR ranges. Scores contiguous open gaps (width × depth) biased toward goal. Decides forward/reverse and steering angle.
- **YOLOv8s-seg:** Real-time instance segmentation on TensorRT. Pre-warmed during sensor warmup to avoid lazy loading during driving.
- **PID Controllers:** Separate PIDs for speed regulation and heading correction, with steering rate-limiting for smooth transitions.

### Safety Features
- Maximum throttle ceiling (configurable, default 0.10)
- Battery voltage monitoring with low-voltage warning and critical auto-stop
- Configurable run timeout (default 35s hard limit)
- Graceful shutdown on Ctrl+C / SIGINT / SIGTERM
- `try/finally` block always zeroes motors on exit
- Rear-clear checks before reversing

---

## Tech Stack
- Python 3.8+ (Jetson Linux)
- Quanser PAL / HAL / PIT libraries
- YOLOv8 (TensorRT / Jetson inference)
- NumPy
- OpenCV
- Matplotlib
- Flask
- Intel RealSense SDK
- RPLiDAR A2

---

## Getting Started

### Prerequisites
- Quanser QCar 2 hardware with Jetson Orin Nano
- Quanser QUARC runtime installed (provides `pal`, `hal`, `pit` libraries)
- RPLiDAR A2 connected and accessible
- Intel RealSense D435 connected

### 1. Clone the repository
```bash
git clone https://github.com/DCode-v05/QCars.git
cd QCars
```

### 2. Install dependencies
The Quanser libraries (`pal`, `hal`, `pit`) are pre-installed on the QCar 2 Jetson image. Additional Python packages:
```bash
pip install numpy opencv-python matplotlib flask
```

### 3. Run the autonomous system
```bash
cd src
python3 observer.py
```
The car will:
1. Initialise all sensors and pre-warm YOLO
2. Set a goal 15m forward from its current position
3. Drive autonomously, avoiding obstacles
4. Stop when it arrives or the timeout expires

### 4. View the live dashboard
Open a browser on any device on the same network:
```
http://<jetson-ip>:5000
```

---

## Usage
- Edit mission parameters at the top of `observer.py`: goal distance, max throttle, dashboard port, and run timeout.
- Use the `tests/` directory to individually test sensors (LiDAR, IMU, cameras, wheels, stereo mic) before running the full stack.
- Use the `tests/autonomous/` scripts for DWA-based planning experiments and data recording.
- Use the live web dashboard to monitor camera feed, LiDAR scan, pose, obstacle zones, and FSM state.

---

## Project Structure
```
QCars/
│
├── src/                            # Production autonomy stack
│   ├── observer.py                 # Main entry point — run this
│   ├── perception.py               # Sensor manager (LiDAR, cameras, IMU, encoders)
│   ├── estimation.py               # EKF pose estimation (encoder + gyro fusion)
│   ├── perceiver.py                # YOLOv8 object classification
│   ├── obstacle_detector.py        # VFH path planner + sensor fusion
│   ├── navigator.py                # Goal heading PID controller
│   ├── state_machine.py            # FSM decision engine + speed PID
│   ├── lights.py                   # LED state mapping
│   ├── dashboard.py                # Flask live telemetry dashboard
│   └── constants.py                # Shared vehicle parameters
│
├── tests/                          # Sensor tests and experiments
│   ├── autonomous/                 # Full autonomy experiments (DWA planner, YOLO)
│   ├── imu/                        # IMU connection, reading, and fusion tests
│   ├── lidar/                      # LiDAR discovery, visualisation, and streaming
│   ├── lidar_wheel/                # Combined LiDAR + wheel safety and VFH tests
│   ├── light/                      # Headlight and LED tests
│   ├── multimedia/                 # CSI camera streaming and RealSense RGBD
│   ├── stereo/                     # Stereo microphone capture
│   ├── stop_sign_nav/              # Stop sign detection + navigation prototype
│   └── wheel/                      # Encoder, steering, and calibration tests
│
├── vendor/                         # Quanser reference libraries and examples
│   ├── examples/                   # Official QCar 2 example applications
│   ├── libraries/                  # PAL, HAL, PIT, QVL Python libraries
│   └── sdcs/                       # SDCS lab activities (sensor, estimation, control)
│
├── docs/
│   └── QCar2_Python_Reference.pdf  # QCar 2 API reference
│
└── README.md
```

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request describing your changes.

---

## Contact
- **GitHub:** [DCode-v05](https://github.com/DCode-v05)
- **Email:** denistanb05@gmail.com
