"""
config.py — All constants for QCar 2 autonomous obstacle avoidance.
"""
import numpy as np
 
WHEEL_RADIUS_M    = 0.033
WHEEL_BASE_M      = 0.256
CAR_WIDTH_M       = 0.20
CAR_LENGTH_M      = 0.40
GEAR_RATIO        = (13 * 19) / (70 * 37)
ENCODER_CPR       = 720 * 4
TACH_TO_MPS       = GEAR_RATIO * WHEEL_RADIUS_M
CPS_TO_MPS        = (1.0 / ENCODER_CPR) * GEAR_RATIO * 2 * np.pi * WHEEL_RADIUS_M

MAX_STEERING_RAD   = np.radians(40)
MIN_TURN_RADIUS_M  = WHEEL_BASE_M / np.tan(MAX_STEERING_RAD)
CAR_HALF_WIDTH_M   = CAR_WIDTH_M / 2.0 + 0.03
OBSTACLE_INFLATE_M = 0.06

GYRO_BIAS_Z       = 0.024

BATTERY_LOW_V     = 10.5
BATTERY_CRIT_V    = 10.0

THROTTLE           = 0.10
CREEP_THROTTLE     = 0.10
MAX_STEER_RATE     = np.radians(120)
STEERING_GAIN      = 1.2

QCAR_FREQUENCY     = 500
QCAR_READ_MODE     = 0

CSI_WIDTH          = 820
CSI_HEIGHT         = 410
CSI_FPS            = 30

RS_WIDTH           = 640
RS_HEIGHT          = 480
RS_FPS             = 30

LIDAR_NUM_MEAS     = 384
LIDAR_RANGE_MODE   = 1
LIDAR_FRONT_DEG    = 0.0
LIDAR_INTERP       = 0
LIDAR_MIN_M        = 0.10
LIDAR_MAX_M        = 6.0

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

VFH_NUM_BINS       = 72
VFH_BIN_WIDTH_DEG  = 360.0 / VFH_NUM_BINS
VFH_BIN_WIDTH_RAD  = np.radians(VFH_BIN_WIDTH_DEG)
VFH_GAP_THRESH_M   = 0.30
VFH_PLAN_RANGE_M   = 4.0
VFH_MIN_GAP_BINS   = 4
VFH_GOAL_BIAS      = 2.0
VFH_TURN_PENALTY   = 0.2
VFH_DIST_DECAY     = 2.0

DWA_SPEED_MPS       = 0.4
DWA_SIM_TIME_S      = 1.5
DWA_SIM_STEPS       = 8
DWA_NUM_STEER_FWD   = 21
DWA_NUM_STEER_REV   = 7
DWA_COLLISION_MARGIN = 0.05

DWA_W_PROGRESS      = 0.20
DWA_W_GOAL          = 0.15
DWA_W_CLEARANCE     = 0.20
DWA_W_SMOOTH        = 0.05
DWA_W_FORWARD       = 0.40

STUCK_THRESHOLD      = 45
STUCK_STEER_FRAC     = 0.85
RECOVERY_DURATION_S  = 2.5

DIR_COMMIT_TICKS     = 15

ZONE_CLEAR_M       = 1.5
ZONE_WARN_M        = 0.35
FRONT_SECTOR_DEG   = 50.0
REAR_SECTOR_DEG    = 50.0
SIDE_SECTOR_DEG    = 30.0
REAR_CLEAR_M       = 0.35
SIDE_CLEAR_M       = 0.20

YOLO_CORR_ANGLE    = 15.0
YOLO_INJECT_DIST_M = 1.0

DEPTH_ROI_TOP      = 300
DEPTH_ROI_BOT      = 480
DEPTH_CLOSE_M      = 0.15
DEPTH_WARN_M       = 0.50
DEPTH_SECTOR_DEG   = 60.0

SMOOTH_WINDOW      = 3

LOOP_RATE_HZ       = 30
LOOP_DT            = 1.0 / LOOP_RATE_HZ
WARMUP_S           = 2.0

PERSON_SLOW_S      = 1.0

DASHBOARD_PORT     = 5000
JPEG_QUALITY       = 75

GRID_RESOLUTION_M   = 0.05
GRID_SIZE_M         = 10.0
GRID_UPDATE_HZ      = 10
GRID_LOG_OCC        = 0.7
GRID_LOG_FREE       = -0.3
GRID_LOG_MAX        = 5.0
GRID_LOG_MIN        = -5.0
GRID_OCC_THRESH     = 0.55

ASTAR_REPLAN_HZ     = 2.0
ASTAR_MAX_NODES     = 30000
ASTAR_INFLATE_CELLS = 3
WAYPOINT_LOOKAHEAD_M = 0.40
WAYPOINT_REACHED_M   = 0.15
GOAL_REACHED_M       = 0.20

YOLO_AFFECT_DIST_M  = 0.20
