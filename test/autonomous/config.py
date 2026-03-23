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
MAX_STEERING_RAD   = np.radians(40)   # ±40° max steering lock (pushed to hardware limit)
MIN_TURN_RADIUS_M  = WHEEL_BASE_M / np.tan(MAX_STEERING_RAD)  # ~0.31m
CAR_HALF_WIDTH_M   = CAR_WIDTH_M / 2.0 + 0.03  # +3cm safety margin
OBSTACLE_INFLATE_M = 0.06                       # reduced — DWA handles car width via trajectory sim

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
LIDAR_RANGE_MODE   = 1
LIDAR_FRONT_DEG    = 0.0      # LiDAR 180° = car's actual front (calibrated)
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
#  VFH HISTOGRAM (kept for obstacle mapping — DWA uses this as its obstacle map)
# ═══════════════════════════════════════════════════════════════════════════════
VFH_NUM_BINS       = 72         # 5° per bin
VFH_BIN_WIDTH_DEG  = 360.0 / VFH_NUM_BINS
VFH_BIN_WIDTH_RAD  = np.radians(VFH_BIN_WIDTH_DEG)
VFH_GAP_THRESH_M   = 0.30
VFH_PLAN_RANGE_M   = 4.0
VFH_MIN_GAP_BINS   = 4
VFH_GOAL_BIAS      = 2.0
VFH_TURN_PENALTY   = 0.2
VFH_DIST_DECAY     = 2.0

# ═══════════════════════════════════════════════════════════════════════════════
#  DWA PATH PLANNER — samples Ackermann trajectories, checks against histogram
# ═══════════════════════════════════════════════════════════════════════════════
DWA_SPEED_MPS       = 0.4        # estimated car speed at THROTTLE=0.10
DWA_SIM_TIME_S      = 1.5        # simulate 1.5s — enough to turn 75° at max steer
DWA_SIM_STEPS       = 8          # 8 steps
DWA_NUM_STEER_FWD   = 21         # forward samples: -30° to +30° in 3° steps
DWA_NUM_STEER_REV   = 7          # reverse samples: -30° to +30° in 10° steps
DWA_COLLISION_MARGIN = 0.05              # radial safety buffer (histogram already inflates angularly)

# Scoring weights — forward/side STRONGLY preferred over reverse
DWA_W_PROGRESS      = 0.20       # reward forward distance travelled
DWA_W_GOAL          = 0.15       # reward alignment with goal heading
DWA_W_CLEARANCE     = 0.20       # reward distance from obstacles
DWA_W_SMOOTH        = 0.05       # reward smooth steering (low so side-steer beats straight)
DWA_W_FORWARD       = 0.40       # strong forward bias — side gap always beats reverse

# Stuck detection & recovery
STUCK_THRESHOLD      = 45         # ticks at near-max-lock → stuck (~1.5s at 30Hz)
STUCK_STEER_FRAC     = 0.85      # fraction of max steering considered "locked"
RECOVERY_DURATION_S  = 2.5       # reverse for 2.5s during recovery (longer = more space to turn)

# Direction commitment: once a direction is chosen, hold it for minimum ticks
# Prevents rapid FWD↔REV oscillation
DIR_COMMIT_TICKS     = 15         # hold direction for ~0.5s before allowing change

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

# ═══════════════════════════════════════════════════════════════════════════════
#  OCCUPANCY GRID & A* PATH PLANNING
# ═══════════════════════════════════════════════════════════════════════════════
GRID_RESOLUTION_M   = 0.05    # 5cm per cell
GRID_SIZE_M         = 10.0    # 10m x 10m map
GRID_UPDATE_HZ      = 10      # update grid at 10Hz
GRID_LOG_OCC        = 0.7     # log-odds for occupied cell
GRID_LOG_FREE       = -0.3    # log-odds for free cell
GRID_LOG_MAX        = 5.0     # clamp
GRID_LOG_MIN        = -5.0    # clamp
GRID_OCC_THRESH     = 0.55    # probability threshold for occupied

ASTAR_REPLAN_HZ     = 2.0     # replan frequency
ASTAR_MAX_NODES     = 30000   # search limit
ASTAR_INFLATE_CELLS = 3       # inflate obstacles ~15cm
WAYPOINT_LOOKAHEAD_M = 0.40   # pure-pursuit lookahead
WAYPOINT_REACHED_M   = 0.15   # within this = advance waypoint
GOAL_REACHED_M       = 0.20   # within this = goal reached

# YOLO: only affect navigation when object within this distance
YOLO_AFFECT_DIST_M  = 0.20    # 20cm — ignore far YOLO detections for path decisions
