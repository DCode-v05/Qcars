"""
constants.py  —  Shared physical constants for QCar 2
Single source of truth for all vehicle parameters.
Import from here — never redefine in individual modules.
"""
import numpy as np

WHEEL_RADIUS      = 0.066 / 2
WHEEL_BASE        = 0.256

PIN_TO_SPUR_RATIO = (13.0 * 19.0) / (70.0 * 37.0)
ENCODER_CPR       = 720
ENCODER_PPR       = 4

CPS_TO_MPS = (1.0 / (ENCODER_CPR * ENCODER_PPR)) * PIN_TO_SPUR_RATIO \
             * 2.0 * np.pi * WHEEL_RADIUS

TACH_TO_MPS = PIN_TO_SPUR_RATIO * WHEEL_RADIUS

GYRO_BIAS_Z = 0.024

BATTERY_LOW_VOLTAGE   = 10.5
BATTERY_CRIT_VOLTAGE  = 10.0
