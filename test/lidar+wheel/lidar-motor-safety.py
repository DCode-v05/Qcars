#!/usr/bin/env python3
"""
QCar 2 — LiDAR Safety Monitor + Motor Drive (Integrated)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Safety zones:
  UNSAFE    : distance <= 0.5 m  -> motor forced to 0
  MODERATE  : distance >  0.5 m and <= 1.0 m  -> warning, motor allowed
  SAFE      : distance >  1.0 m  -> clear, motor allowed

Modes (set at startup):
  drive   -- forward throttle, LiDAR monitors FRONT arc
  reverse -- reverse throttle, LiDAR monitors REAR  arc

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CALIBRATION — run lidar_diag.py FIRST:
  1. Put an object ~0.4 m in front of QCar nose
  2. Run: python3 lidar_diag.py
  3. Note the "CLOSEST POINT" angle shown
  4. Paste that value into FRONT_CENTER_DEG below
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Lift QCar 2 off ground before running.
Run: python3 lidar_motor_safety.py
"""

import numpy as np
import time
import os

from quanser.hardware import HIL, EncoderQuadratureMode
from quanser.hardware.exceptions import HILError
from pal.products.qcar import QCarLidar

# ══════════════════════════════════════════════════════════════
# Channel Map  (verified from rot.py)
# ══════════════════════════════════════════════════════════════
CARD_TYPE         = "qcar2"
CARD_ID           = "0"
THROTTLE_CH       = 11000
STEERING_CH       = 1000
ENCODER_CH        = 0

MOTOR_CHANNELS    = np.array([THROTTLE_CH, STEERING_CH], dtype=np.uint32)
ENC_CHANNELS      = np.array([ENCODER_CH],               dtype=np.uint32)

# ══════════════════════════════════════════════════════════════
# LiDAR Config
# ══════════════════════════════════════════════════════════════
NUM_MEASUREMENTS  = 384
RANGING_MODE      = 2       # LONG range
INTERP_MODE       = 0

# ══════════════════════════════════════════════════════════════
# CALIBRATION  <-- only value you need to change
# ──────────────────────────────────────────────────────────────
# Step 1: python3 lidar_diag.py  (place object ~0.4m at QCar nose)
# Step 2: note the "CLOSEST POINT" angle printed each scan
# Step 3: paste that angle into FRONT_CENTER_DEG below
# REAR_CENTER_DEG is auto-computed as FRONT + 180 deg
# ══════════════════════════════════════════════════════════════
FRONT_CENTER_DEG  = 0.0     # <-- paste value from lidar_diag.py here
REAR_CENTER_DEG   = (FRONT_CENTER_DEG + 180.0) % 360.0
ARC_HALF_DEG      = 30.0    # half-width of detection cone (deg)

# ══════════════════════════════════════════════════════════════
# Safety Thresholds
# ══════════════════════════════════════════════════════════════
UNSAFE_M          = 0.50
MODERATE_M        = 1.00

# ══════════════════════════════════════════════════════════════
# Motor
# ══════════════════════════════════════════════════════════════
DRIVE_THROTTLE    =  0.12
REVERSE_THROTTLE  = -0.12
STEERING_ANGLE    =  0.0

# ══════════════════════════════════════════════════════════════
# Loop timing
# ══════════════════════════════════════════════════════════════
LOOP_HZ           = 10
LOOP_PERIOD       = 1.0 / LOOP_HZ

# Encoder math (from rot.py)
COUNTS_PER_REV_X4 = 720 * 4
WHEEL_CIRCUM_M    = 2 * np.pi * 0.0328
METRES_PER_COUNT  = WHEEL_CIRCUM_M / COUNTS_PER_REV_X4

# ── ANSI colours ──────────────────────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
DIM    = "\033[2m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
CLEAR  = "\033[2K\r"


# ══════════════════════════════════════════════════════════════
# Arc distance — wrap-safe
# ══════════════════════════════════════════════════════════════
def arc_min_distance(distances, angles_rad, center_deg, half_deg):
    """
    Minimum valid distance within a directional cone.
    Uses atan2 wrap-safe difference so arcs crossing 0/360 work correctly.
    """
    center_rad = np.deg2rad(center_deg)
    half_rad   = np.deg2rad(half_deg)

    angular_diff = np.abs(
        np.arctan2(
            np.sin(angles_rad - center_rad),
            np.cos(angles_rad - center_rad)
        )
    )

    in_arc = angular_diff <= half_rad
    valid  = (distances > 0.05) & in_arc

    if valid.sum() == 0:
        return None

    return float(distances[valid].min())


# ══════════════════════════════════════════════════════════════
# Safety classifier
# ══════════════════════════════════════════════════════════════
def safety_state(min_dist):
    """Returns (label, colour, motor_allowed)."""
    if min_dist is None:
        return "SAFE     (no arc data)         ", GREEN,  True
    if min_dist <= UNSAFE_M:
        return f"UNSAFE   ({min_dist:.3f} m)  MOTOR CUT ", RED,    False
    if min_dist <= MODERATE_M:
        return f"MODERATE ({min_dist:.3f} m)  WARNING   ", YELLOW, True
    return     f"SAFE     ({min_dist:.3f} m)             ", GREEN,  True


# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════
def print_header(mode):
    os.system('clear')
    print(f"{BOLD}{CYAN}{'━'*68}{RESET}")
    print(f"  {BOLD}QCar 2 — LiDAR Safety Monitor + Motor Drive{RESET}")
    print(f"  Mode         : {BOLD}{WHITE}{mode.upper()}{RESET}")
    print(f"  FRONT center : {FRONT_CENTER_DEG:.1f} deg   "
          f"REAR center : {REAR_CENTER_DEG:.1f} deg   "
          f"arc : +-{ARC_HALF_DEG:.0f} deg")
    print(f"  SAFE         : > {MODERATE_M:.1f} m")
    print(f"  MODERATE     : > {UNSAFE_M:.1f} m  and  <= {MODERATE_M:.1f} m")
    print(f"  UNSAFE (cut) : <= {UNSAFE_M:.1f} m")
    print(f"{BOLD}{CYAN}{'━'*68}{RESET}")
    print(f"  {'Time':>6}  {'Mode':<8}  {'Throttle':>9}  "
          f"{'Arc':<5}  {'Min Dist':>9}  Safety State")
    print(f"  {'─'*66}")


def get_mode():
    print(f"\n{BOLD}QCar 2 — LiDAR Safety Integration{RESET}")
    print(f"  Calibrated FRONT : {FRONT_CENTER_DEG:.1f} deg")
    print(f"  Calibrated REAR  : {REAR_CENTER_DEG:.1f} deg")
    print()
    print(f"  {GREEN}drive{RESET}   -- forward,  monitors FRONT arc")
    print(f"  {YELLOW}reverse{RESET} -- backward, monitors REAR  arc")
    print()
    while True:
        choice = input("  Enter mode [drive / reverse]: ").strip().lower()
        if choice in ("drive", "reverse"):
            return choice
        print(f"  {RED}Invalid. Type 'drive' or 'reverse'.{RESET}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():

    mode         = get_mode()
    is_drive     = (mode == "drive")
    throttle_cmd = DRIVE_THROTTLE   if is_drive else REVERSE_THROTTLE
    arc_center   = FRONT_CENTER_DEG if is_drive else REAR_CENTER_DEG
    arc_label    = "FRONT" if is_drive else "REAR "

    print(f"\n{RED}Lift QCar 2 OFF THE GROUND before continuing.{RESET}")
    if input("   Type 'yes' to continue: ").strip().lower() != "yes":
        print("Aborted.")
        return

    # Open HIL
    print("\n[1/3] Opening HIL board ...")
    try:
        card = HIL(CARD_TYPE, CARD_ID)
        print(f"      OK  '{CARD_TYPE}' opened")
    except HILError as e:
        print(f"      FAIL  {e.get_error_message()}")
        return

    # Encoder
    print("[2/3] Configuring encoder ...")
    enc_buf  = np.zeros(1, dtype=np.int32)
    enc_mode = np.array([EncoderQuadratureMode.X4], dtype=np.int32)
    card.set_encoder_quadrature_mode(ENC_CHANNELS, 1, enc_mode)
    card.set_encoder_counts(ENC_CHANNELS, 1, np.zeros(1, dtype=np.int32))
    print("      OK  X4 quadrature, counter zeroed")

    # LiDAR
    print("[3/3] Starting LiDAR ...")
    lidar = QCarLidar(
        numMeasurements=NUM_MEASUREMENTS,
        rangingDistanceMode=RANGING_MODE,
        interpolationMode=INTERP_MODE,
    )
    print("      Waiting 2 s for spin-up ...")
    time.sleep(2.0)
    print("      OK  LiDAR ready")

    # Zero motor
    motor_buf = np.array([0.0, STEERING_ANGLE], dtype=np.float64)
    card.write_other(MOTOR_CHANNELS, 2, motor_buf)

    print_header(mode)

    start_time = time.time()
    loop_count = 0
    motor_cuts = 0

    try:
        while True:
            t0 = time.time()

            # Read LiDAR
            lidar.read()
            dists  = lidar.distances.flatten()
            angles = lidar.angles.flatten()

            # Check only the arc in the direction of travel
            min_dist = arc_min_distance(
                dists, angles, arc_center, ARC_HALF_DEG
            )
            state_str, colour, motor_ok = safety_state(min_dist)

            # Motor decision
            if motor_ok:
                motor_buf[0] = throttle_cmd
            else:
                motor_buf[0] = 0.0
                motor_cuts  += 1

            motor_buf[1] = STEERING_ANGLE
            card.write_other(MOTOR_CHANNELS, 2, motor_buf)

            # Encoder
            card.read_encoder(ENC_CHANNELS, 1, enc_buf)

            # Print one updating line
            elapsed  = time.time() - start_time
            dist_str = f"{min_dist:.3f} m" if min_dist is not None else "  no data"
            motor_str = f"{GREEN}RUN{RESET}" if motor_ok else f"{RED}CUT{RESET}"

            print(
                f"{CLEAR}  {elapsed:>6.1f}s  "
                f"{WHITE}{mode.upper():<8}{RESET}  "
                f"{motor_buf[0]:>+9.2f}  "
                f"{arc_label}  "
                f"{dist_str:>9}  "
                f"{colour}{state_str}{RESET}  {motor_str}",
                end="", flush=True
            )

            loop_count += 1

            sleep_t = LOOP_PERIOD - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Ctrl+C — stopping ...{RESET}")

    except HILError as e:
        print(f"\n{RED}HILError: {e.get_error_message()}{RESET}")

    finally:
        print(f"\n{'─'*68}")
        try:
            card.write_other(MOTOR_CHANNELS, 2,
                             np.array([0.0, 0.0], dtype=np.float64))
            print("  Motor stopped")
        except Exception as e:
            print(f"  Motor stop failed: {e}")

        try:
            lidar.terminate()
            print("  LiDAR terminated")
        except Exception as e:
            print(f"  LiDAR terminate: {e}")

        try:
            card.close()
            print("  HIL board closed")
        except HILError as e:
            print(f"  HIL close: {e.get_error_message()}")

        total = time.time() - start_time
        print(f"\n  mode={mode.upper()}  duration={total:.1f}s  "
              f"loops={loop_count}  motor_cuts={motor_cuts}")
        print(f"{'─'*68}\n")


if __name__ == "__main__":
    main()