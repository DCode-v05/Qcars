"""
constants.py  —  Shared physical constants for QCar 2
Single source of truth for all vehicle parameters.
Import from here — never redefine in individual modules.
"""
import numpy as np

# ── Vehicle geometry ─────────────────────────────────────────────────────────
WHEEL_RADIUS      = 0.066 / 2          # metres (33 mm)
WHEEL_BASE        = 0.256              # metres (axle to axle)

# ── Drivetrain ───────────────────────────────────────────────────────────────
PIN_TO_SPUR_RATIO = (13.0 * 19.0) / (70.0 * 37.0)   # motor → wheel gear ratio
ENCODER_CPR       = 720                # counts per revolution
ENCODER_PPR       = 4                  # pulses per revolution (quadrature)

# ── Derived conversion factors ───────────────────────────────────────────────

# Encoder counts/s  →  wheel linear m/s
CPS_TO_MPS = (1.0 / (ENCODER_CPR * ENCODER_PPR)) * PIN_TO_SPUR_RATIO \
             * 2.0 * np.pi * WHEEL_RADIUS

# Motor tachometer (rad/s at motor shaft)  →  wheel linear m/s
TACH_TO_MPS = PIN_TO_SPUR_RATIO * WHEEL_RADIUS

# ── IMU calibration ──────────────────────────────────────────────────────────
GYRO_BIAS_Z = 0.024      # rad/s  —  measured at rest, subtract from gyro_z

# ── Safety limits ────────────────────────────────────────────────────────────
BATTERY_LOW_VOLTAGE   = 10.5    # volts — warn below this
BATTERY_CRIT_VOLTAGE  = 10.0    # volts — auto-stop below this
