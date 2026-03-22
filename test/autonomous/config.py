"""
config.py — All constants for QCar 2 autonomous obstacle avoidance.
Adapted from Final/constants.py + Final/obstacle_detector.py configs.
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
TACH_TO_MPS       = GEAR_RATIO * WHEEL_RADIUS_M  # rad/s → m/s
CPS_TO_MPS        = (1.0 / ENCODER_CPR) * GEAR_RATIO * 2 * np.pi * WHEEL_RADIUS_M

# Turning geometry (Ackermann)
MAX_STEERING_RAD   = np.radians(30)   # ±30° max steering lock
MIN_TURN_RADIUS_M  = WHEEL_BASE_M / np.tan(MAX_STEERING_RAD)  # ~0.44m
CAR_HALF_WIDTH_M   = CAR_WIDTH_M / 2.0 + 0.05  # +5cm safety margin
OBSTACLE_INFLATE_M = CAR_HALF_WIDTH_M  # inflate obstacles by car half-width

# IMU calibration
GYRO_BIAS_Z       = 0.024       # rad/s (subtract from raw gyro_z)

# Battery
BATTERY_LOW_V     = 10.5
BATTERY_CRIT_V    = 10.0

# ═══════════════════════════════════════════════════════════════════════════════
#  DRIVING
# ═══════════════════════════════════════════════════════════════════════════════
THROTTLE           = 0.10       # always 0.1 (forward and reverse)
MAX_STEER_RATE     = np.radians(60)   # 60°/s max rate of change
STEERING_GAIN      = 0.8              # steer = gain × angle_error

# ═══════════════════════════════════════════════════════════════════════════════
#  SENSORS
# ═══════════════════════════════════════════════════════════════════════════════
# QCar
QCAR_FREQUENCY     = 500
QCAR_READ_MODE     = 0

# CSI cameras (PAL QCarCameras)
CSI_WIDTH          = 820
CSI_HEIGHT         = 410
CSI_FPS            = 30

# LiDAR (RPLIDAR A2)
LIDAR_NUM_MEAS     = 384
LIDAR_RANGE_MODE   = 2
LIDAR_INTERP       = 0
LIDAR_MIN_M        = 0.10
LIDAR_MAX_M        = 6.0

# ═══════════════════════════════════════════════════════════════════════════════
#  YOLO
# ═══════════════════════════════════════════════════════════════════════════════
YOLO_WIDTH         = 640
CONF_THRESH        = 0.40
H_FOV_DEG          = 160.0      # horizontal FOV per camera

# Camera layout (matches QCarCameras for cartype 2)
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

# YOLO class categorization
PERSON_IDS  = {0}
MOVING_IDS  = {1, 2, 3, 16}     # bicycle, car, motorcycle, dog
STATIC_IDS  = {24, 25, 26, 28, 56, 57, 58, 59, 60, 62, 63, 64, 72, 73}

# ═══════════════════════════════════════════════════════════════════════════════
#  VFH (Vector Field Histogram) PATH PLANNER — vehicle-aware
# ═══════════════════════════════════════════════════════════════════════════════
VFH_NUM_BINS       = 72         # 360°/72 = 5° per bin (finer resolution)
VFH_BIN_WIDTH_DEG  = 360.0 / VFH_NUM_BINS
VFH_BIN_WIDTH_RAD  = np.radians(VFH_BIN_WIDTH_DEG)
VFH_GAP_THRESH_M   = 0.5       # bins with min_dist > this are "open"
VFH_PLAN_RANGE_M   = 4.0       # ignore LiDAR beyond this for planning

# Minimum passable gap: car must physically fit through
# At distance d, gap angular width must be > 2*arctan(car_half_width/d)
VFH_MIN_GAP_BINS   = 6         # minimum 30° gap (6 bins × 5°)
VFH_GOAL_BIAS      = 2.0       # weight for goal-aligned gaps
VFH_TURN_PENALTY   = 0.3       # penalty for gaps requiring sharp turns

# ═══════════════════════════════════════════════════════════════════════════════
#  OBSTACLE ZONES — 360° sector analysis
# ═══════════════════════════════════════════════════════════════════════════════
ZONE_CLEAR_M       = 1.5       # > this = CLEAR
ZONE_WARN_M        = 0.4       # WARN < dist <= CLEAR
                                # <= WARN = STOP
FRONT_SECTOR_DEG   = 50.0      # ±50° from heading (wider for turning)
REAR_SECTOR_DEG    = 50.0      # ±50° from rear
SIDE_SECTOR_DEG    = 30.0      # ±30° from 90°/270° for side clearance
REAR_CLEAR_M       = 0.5       # rear is clear if min > this
SIDE_CLEAR_M       = 0.30      # side clearance for safe turning

# YOLO-LiDAR fusion
YOLO_CORR_ANGLE    = 15.0      # ±15° matching window
YOLO_INJECT_DIST_M = 1.0       # inject YOLO objects as obstacles at this dist

# Smoothing
SMOOTH_WINDOW      = 3         # rolling average over N readings

# ═══════════════════════════════════════════════════════════════════════════════
#  TIMING
# ═══════════════════════════════════════════════════════════════════════════════
LOOP_RATE_HZ       = 30
LOOP_DT            = 1.0 / LOOP_RATE_HZ
RUN_TIMEOUT_S      = 60.0      # hard timeout
WARMUP_S           = 2.0       # sensor warmup

# State machine timeouts
WAIT_TIMEOUT_S     = 8.0       # wait for person before resuming
REVERSE_TIMEOUT_S  = 2.0       # max reverse duration
STUCK_RETRY_S      = 3.0       # retry after stuck

# Dashboard
DASHBOARD_PORT     = 5000
JPEG_QUALITY       = 75
