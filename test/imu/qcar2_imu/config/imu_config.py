import numpy as np

# ── Board Identity ─────────────────────────────────────────────────
BOARD_TYPE = "qcar2"
BOARD_ID   = "0"

# ── IMU Other-Channel Addresses ────────────────────────────────────
# Gyroscope: angular velocity (rad/s)
CH_GYRO_X  = 11000
CH_GYRO_Y  = 11001
CH_GYRO_Z  = 11002

# Accelerometer: linear acceleration (m/s²)
CH_ACCEL_X = 11003
CH_ACCEL_Y = 11004
CH_ACCEL_Z = 11005

# Magnetometer: magnetic field (Tesla)
CH_MAG_X   = 11006
CH_MAG_Y   = 11007
CH_MAG_Z   = 11008

# All 9 channels as a single numpy array
ALL_IMU_CHANNELS = np.array([
    CH_GYRO_X,  CH_GYRO_Y,  CH_GYRO_Z,
    CH_ACCEL_X, CH_ACCEL_Y, CH_ACCEL_Z,
    CH_MAG_X,   CH_MAG_Y,   CH_MAG_Z
], dtype=np.uint32)
NUM_IMU_CHANNELS = 6

# Index slices into the 9-element read buffer
IDX_GYRO  = slice(0, 3)
IDX_ACCEL = slice(3, 6)
IDX_MAG   = slice(6, 9)

# ── Physical Limits ────────────────────────────────────────────────
GYRO_MAX_RADS  = 34.9
ACCEL_MAX_MS2  = 156.9
MAG_MAX_TESLA  = 400e-6

# ── Timing ─────────────────────────────────────────────────────────
SAMPLE_RATE_HZ = 100
DT             = 1.0 / SAMPLE_RATE_HZ

# ── Motor / Steering Output Channels ───────────────────────────────
CH_MOTOR_OUT = np.array([0], dtype=np.uint32)
CH_STEER_OUT = np.array([1], dtype=np.uint32)

THROTTLE_MAX =  0.3
THROTTLE_MIN = -0.3
STEER_MAX    =  0.5
STEER_MIN    = -0.5

# ── CSV column headers ─────────────────────────────────────────────
IMU_LABELS = [
    "gyro_x_rads", "gyro_y_rads", "gyro_z_rads",
    "accel_x_ms2", "accel_y_ms2", "accel_z_ms2",
    "mag_x_T",     "mag_y_T",     "mag_z_T"
]
