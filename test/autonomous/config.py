"""
config.py — All constants for QCar 2 autonomous obstacle avoidance.
"""
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
#  VEHICLE HARDWARE & GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════════
WHEEL_RADIUS_M    = 0.033       # 33mm
WHEEL_BASE_M      = 0.256       # axle-to-axle
CAR_WIDTH_M       = 0.20        # ~200mm body width
CAR_LENGTH_M      = 0.40        # ~400mm body length
GEAR_RATIO        = (13 * 19) / (70 * 37)  # ~0.0948
ENCODER_CPR       = 720 * 4     # 2880 counts/rev (quadrature)
TACH_TO_MPS       = GEAR_RATIO * WHEEL_RADIUS_M
CPS_TO_MPS        = (1.0 / ENCODER_CPR) * GEAR_RATIO * 2 * np.pi * WHEEL_RADIUS_M

# Turning geometry (Ackermann)
MAX_STEERING_RAD   = np.radians(30)   # ±30° max steering lock
MIN_TURN_RADIUS_M  = WHEEL_BASE_M / np.tan(MAX_STEERING_RAD)  # ~0.44m
CAR_HALF_WIDTH_M   = CAR_WIDTH_M / 2.0 + 0.03  # +3cm safety margin
OBSTACLE_INFLATE_M = CAR_HALF_WIDTH_M

# IMU calibration
GYRO_BIAS_Z       = 0.024       # rad/s

# Battery
BATTERY_LOW_V     = 10.5
BATTERY_CRIT_V    = 10.0

# ═══════════════════════════════════════════════════════════════════════════════
#  DRIVING — car NEVER fully stops (always creeping)
# ═══════════════════════════════════════════════════════════════════════════════
THROTTLE           = 0.10       # constant speed — same forward AND reverse
CREEP_THROTTLE     = 0.10       # same as THROTTLE — car never slows down
MAX_STEER_RATE     = np.radians(120)   # 120°/s — fast response
STEERING_GAIN      = 1.2               # aggressive steering

# ═══════════════════════════════════════════════════════════════════════════════
#  SENSORS
# ═══════════════════════════════════════════════════════════════════════════════
QCAR_FREQUENCY     = 500
QCAR_READ_MODE     = 0

# CSI cameras (PAL QCarCameras)
CSI_WIDTH          = 820
CSI_HEIGHT         = 410
CSI_FPS            = 30

# RealSense D435 (RGBD)
RS_WIDTH           = 640
RS_HEIGHT          = 480
RS_FPS             = 30

# LiDAR (RPLIDAR A2)
LIDAR_NUM_MEAS     = 384
LIDAR_RANGE_MODE   = 2
LIDAR_FRONT_DEG    = 180.0      # LiDAR 180° = car's actual front (calibrated)
LIDAR_INTERP       = 0
LIDAR_MIN_M        = 0.10
LIDAR_MAX_M        = 6.0

# ═══════════════════════════════════════════════════════════════════════════════
#  YOLO
# ═══════════════════════════════════════════════════════════════════════════════
YOLO_WIDTH         = 640
CONF_THRESH        = 0.40
H_FOV_DEG          = 160.0

CAMERA_CONFIG = [
    {"id": 0, "name": "RIGHT", "centre_deg":  90.0},
    {"id": 1, "name": "BACK",  "centre_deg": 180.0},
    {"id": 2, "name": "FRONT", "centre_deg":   0.0},
    {"id": 3, "name": "LEFT",  "centre_deg": 270.0},
]

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","yield sign","baseball bat",
    "baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
    "cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave",
    "oven","toaster","sink","refrigerator","book","clock","vase",
    "scissors","teddy bear","hair drier","toothbrush",
]

PERSON_IDS  = {0}
MOVING_IDS  = {1, 2, 3, 16}
STATIC_IDS  = {24, 25, 26, 28, 56, 57, 58, 59, 60, 62, 63, 64, 72, 73}

# ═══════════════════════════════════════════════════════════════════════════════
#  VFH PATH PLANNER — distance-aware, always finds a path
# ═══════════════════════════════════════════════════════════════════════════════
VFH_NUM_BINS       = 72         # 5° per bin
VFH_BIN_WIDTH_DEG  = 360.0 / VFH_NUM_BINS
VFH_BIN_WIDTH_RAD  = np.radians(VFH_BIN_WIDTH_DEG)
VFH_GAP_THRESH_M   = 0.30      # lowered: 30cm = open (was 0.5)
VFH_PLAN_RANGE_M   = 4.0
VFH_MIN_GAP_BINS   = 4         # 20° minimum gap (was 30°)
VFH_GOAL_BIAS      = 2.0
VFH_TURN_PENALTY   = 0.2

# Distance-based obstacle weight: far obstacles matter less
VFH_DIST_DECAY     = 2.0       # obstacles at 2m have half the weight of 0.5m

# ═══════════════════════════════════════════════════════════════════════════════
#  OBSTACLE ZONES
# ═══════════════════════════════════════════════════════════════════════════════
ZONE_CLEAR_M       = 1.5
ZONE_WARN_M        = 0.35      # lowered (was 0.4)
FRONT_SECTOR_DEG   = 50.0
REAR_SECTOR_DEG    = 50.0
SIDE_SECTOR_DEG    = 30.0
REAR_CLEAR_M       = 0.35      # lowered (was 0.5)
SIDE_CLEAR_M       = 0.20      # lowered (was 0.3)

# YOLO-LiDAR fusion
YOLO_CORR_ANGLE    = 15.0
YOLO_INJECT_DIST_M = 1.0

# RealSense depth — catches low obstacles LiDAR misses
DEPTH_ROI_TOP      = 300        # bottom portion of image (floor level)
DEPTH_ROI_BOT      = 480
DEPTH_CLOSE_M      = 0.15      # objects closer than this = immediate obstacle
DEPTH_WARN_M       = 0.50      # objects closer than this = inject into VFH
DEPTH_SECTOR_DEG   = 60.0      # RealSense H-FOV mapped to front ±30°

SMOOTH_WINDOW      = 3

# ═══════════════════════════════════════════════════════════════════════════════
#  TIMING
# ═══════════════════════════════════════════════════════════════════════════════
LOOP_RATE_HZ       = 30
LOOP_DT            = 1.0 / LOOP_RATE_HZ
WARMUP_S           = 2.0

# Navigator (no timers — car never stops on its own)
PERSON_SLOW_S      = 1.0        # slow near person for 1s then resume

# Dashboard
DASHBOARD_PORT     = 5000
JPEG_QUALITY       = 75
