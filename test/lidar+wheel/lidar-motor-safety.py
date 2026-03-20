#!/usr/bin/env python3
"""
QCar 2 — LiDAR Safety Monitor + Motor Drive (Integrated)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Safety zones (front arc ±30° from forward):
  🔴 UNSAFE    : distance <= 0.5 m  → motor forced to 0
  🟡 MODERATE  : distance >  0.5 m and <= 1.0 m  → warning, motor allowed
  🟢 SAFE      : distance >  1.0 m  → clear, motor allowed

Modes (set at startup):
  drive   — forward throttle, LiDAR monitors FRONT arc
  reverse — reverse throttle, LiDAR monitors REAR  arc

⚠  LIFT QCAR 2 OFF GROUND BEFORE RUNNING
Run: python3 lidar_motor_safety.py
"""

import numpy as np
import time
import sys
import os

from quanser.hardware import HIL, EncoderQuadratureMode
from quanser.hardware.exceptions import HILError
from pal.products.qcar import QCarLidar

# ══════════════════════════════════════════════════════════════
# Channel Map  (from rot.py — verified on hardware)
# ══════════════════════════════════════════════════════════════
CARD_TYPE        = "qcar2"
CARD_ID          = "0"
THROTTLE_CH      = 11000        # write_other — motor throttle (-1 to +1)
STEERING_CH      = 1000         # write_other — steering (rad)
ENCODER_CH       = 0            # read_encoder

MOTOR_CHANNELS   = np.array([THROTTLE_CH, STEERING_CH], dtype=np.uint32)
ENC_CHANNELS     = np.array([ENCODER_CH],               dtype=np.uint32)

# ══════════════════════════════════════════════════════════════
# LiDAR Config
# ══════════════════════════════════════════════════════════════
NUM_MEASUREMENTS = 384          # match your working lidar_server.py value
RANGING_MODE     = 2            # LONG range
INTERP_MODE      = 0

# ══════════════════════════════════════════════════════════════
# Safety Thresholds
# ══════════════════════════════════════════════════════════════
UNSAFE_M         = 0.50         # <= this → UNSAFE, motor cut
MODERATE_M       = 1.00         # <= this (and > UNSAFE) → MODERATE warning

# Front arc  : angles near 0 rad (forward)   ±30°
# Rear  arc  : angles near π rad (backward)  ±30°
FRONT_ARC_DEG    = 30.0
REAR_ARC_DEG     = 30.0

# ══════════════════════════════════════════════════════════════
# Drive Profile  (throttle per mode)
# ══════════════════════════════════════════════════════════════
DRIVE_THROTTLE   =  0.12        # forward  (+)
REVERSE_THROTTLE = -0.12        # reverse  (-)
STEERING_ANGLE   =  0.0         # straight

# ══════════════════════════════════════════════════════════════
# Loop Rate
# ══════════════════════════════════════════════════════════════
LOOP_HZ          = 10
LOOP_PERIOD      = 1.0 / LOOP_HZ

# Encoder math (from rot.py)
COUNTS_PER_REV_X4 = 720 * 4
WHEEL_CIRCUM_M    = 2 * np.pi * 0.0328
METRES_PER_COUNT  = WHEEL_CIRCUM_M / COUNTS_PER_REV_X4

# ── ANSI colour helpers ───────────────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
DIM    = "\033[2m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
CLEAR  = "\033[2K\r"           # clear current line


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def arc_min_distance(distances, angles, center_deg, half_arc_deg):
    """
    Return the minimum valid distance within a directional arc.

    center_deg : 0   = forward (drive)
                 180 = backward (reverse)
    """
    center_rad   = np.deg2rad(center_deg)
    half_rad     = np.deg2rad(half_arc_deg)

    # Wrap-aware angular difference
    diff = np.abs(np.arctan2(np.sin(angles - center_rad),
                             np.cos(angles - center_rad)))

    in_arc = diff <= half_rad
    valid  = (distances > 0.05) & in_arc

    if valid.sum() == 0:
        return None                 # no valid readings in arc

    return float(distances[valid].min())


def safety_state(min_dist):
    """
    Returns (state_str, colour, motor_allowed).
    min_dist=None means LiDAR returned no data in that arc → treat as SAFE.
    """
    if min_dist is None:
        return "SAFE  (no data)", GREEN, True
    if min_dist <= UNSAFE_M:
        return f"UNSAFE   ({min_dist:.3f} m)", RED, False
    if min_dist <= MODERATE_M:
        return f"MODERATE ({min_dist:.3f} m)", YELLOW, True
    return f"SAFE     ({min_dist:.3f} m)", GREEN, True


def print_header(mode):
    os.system('clear')
    print(f"{BOLD}{CYAN}{'━'*62}{RESET}")
    print(f"  {BOLD}QCar 2 — LiDAR Safety Monitor + Motor Drive{RESET}")
    print(f"  Mode : {BOLD}{WHITE}{mode.upper()}{RESET}")
    print(f"  Safe threshold     : > {MODERATE_M:.1f} m")
    print(f"  Moderate threshold : > {UNSAFE_M:.1f} m  and  <= {MODERATE_M:.1f} m")
    print(f"  UNSAFE threshold   : <= {UNSAFE_M:.1f} m  (motor CUT)")
    print(f"{BOLD}{CYAN}{'━'*62}{RESET}")
    print(f"  {'Time':>6}  {'Mode':<8}  {'Throttle':>9}  "
          f"{'Arc Dist':>10}  {'Safety State':<28}  {'Motor'}")
    print(f"  {'─'*60}")


def get_mode():
    print(f"\n{BOLD}QCar 2 — LiDAR Safety Integration{RESET}")
    print(f"  Select mode:")
    print(f"  {GREEN}drive{RESET}   — forward motion, monitors FRONT arc (±{FRONT_ARC_DEG:.0f}°)")
    print(f"  {YELLOW}reverse{RESET} — reverse motion, monitors REAR  arc (±{REAR_ARC_DEG:.0f}°)")
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

    # ── Mode selection ────────────────────────────────────────
    mode = get_mode()

    is_drive     = (mode == "drive")
    throttle_cmd = DRIVE_THROTTLE if is_drive else REVERSE_THROTTLE
    arc_center   = 0.0   if is_drive else 180.0
    arc_half     = FRONT_ARC_DEG if is_drive else REAR_ARC_DEG
    arc_label    = "FRONT" if is_drive else "REAR "

    # ── Safety confirmation ───────────────────────────────────
    print(f"\n{RED}⚠  CONFIRM: QCar 2 is LIFTED OFF THE GROUND{RESET}")
    if input("   Type 'yes' to continue: ").strip().lower() != "yes":
        print("Aborted.")
        return

    # ── Open HIL ─────────────────────────────────────────────
    print("\n[1/3] Opening HIL board ...")
    try:
        card = HIL(CARD_TYPE, CARD_ID)
        print(f"      ✅ '{CARD_TYPE}' opened")
    except HILError as e:
        print(f"      ❌ {e.get_error_message()}")
        return

    # ── Configure encoder ─────────────────────────────────────
    print("[2/3] Configuring encoder ...")
    enc_buf  = np.zeros(1, dtype=np.int32)
    enc_mode = np.array([EncoderQuadratureMode.X4], dtype=np.int32)
    card.set_encoder_quadrature_mode(ENC_CHANNELS, 1, enc_mode)
    card.set_encoder_counts(ENC_CHANNELS, 1, np.zeros(1, dtype=np.int32))
    print("      ✅ X4 quadrature, counter zeroed")

    # ── Start LiDAR ──────────────────────────────────────────
    print("[3/3] Starting LiDAR ...")
    lidar = QCarLidar(
        numMeasurements   = NUM_MEASUREMENTS,
        rangingDistanceMode = RANGING_MODE,
        interpolationMode = INTERP_MODE,
    )
    print("      Waiting 2 s for motor to spin up ...")
    time.sleep(2.0)
    print("      ✅ LiDAR ready")

    # ── Zero motor before loop ────────────────────────────────
    motor_buf = np.array([0.0, STEERING_ANGLE], dtype=np.float64)
    card.write_other(MOTOR_CHANNELS, 2, motor_buf)

    print_header(mode)

    start_time   = time.time()
    loop_count   = 0
    motor_cuts   = 0

    try:
        while True:
            t0 = time.time()

            # ── Read LiDAR ────────────────────────────────────
            lidar.read()
            dists  = lidar.distances.flatten()
            angles = lidar.angles.flatten()

            min_dist = arc_min_distance(dists, angles, arc_center, arc_half)
            state_str, colour, motor_ok = safety_state(min_dist)

            # ── Motor decision ────────────────────────────────
            if motor_ok:
                motor_buf[0] = throttle_cmd
            else:
                motor_buf[0] = 0.0
                motor_cuts  += 1

            motor_buf[1] = STEERING_ANGLE
            card.write_other(MOTOR_CHANNELS, 2, motor_buf)

            # ── Encoder read ──────────────────────────────────
            card.read_encoder(ENC_CHANNELS, 1, enc_buf)
            dist_m = int(enc_buf[0]) * METRES_PER_COUNT

            # ── Terminal output ───────────────────────────────
            elapsed    = time.time() - start_time
            motor_str  = f"{GREEN}RUN{RESET}" if motor_ok else f"{RED}CUT{RESET}"
            dist_str   = f"{min_dist:.3f} m" if min_dist is not None else "  no data"

            print(
                f"{CLEAR}  {elapsed:>6.1f}s  "
                f"{WHITE}{mode.upper():<8}{RESET}  "
                f"{motor_buf[0]:>+9.2f}  "
                f"{arc_label} {dist_str:>8}  "
                f"{colour}{state_str:<28}{RESET}  "
                f"{motor_str}",
                end="", flush=True
            )

            loop_count += 1

            # ── Pace the loop ─────────────────────────────────
            elapsed_loop = time.time() - t0
            sleep_t = LOOP_PERIOD - elapsed_loop
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}⚠  Ctrl+C — stopping ...{RESET}")

    except HILError as e:
        print(f"\n{RED}❌ HILError: {e.get_error_message()}{RESET}")

    finally:
        # ── Always stop motor ─────────────────────────────────
        print(f"\n{'─'*62}")
        try:
            stop_buf = np.array([0.0, 0.0], dtype=np.float64)
            card.write_other(MOTOR_CHANNELS, 2, stop_buf)
            print(f"  ✅ Motor stopped (throttle=0.0, steering=0.0)")
        except Exception as e:
            print(f"  ⚠  Motor stop failed: {e}")

        try:
            lidar.terminate()
            print(f"  ✅ LiDAR terminated")
        except Exception as e:
            print(f"  ⚠  LiDAR terminate: {e}")

        try:
            card.close()
            print(f"  ✅ HIL board closed")
        except HILError as e:
            print(f"  ⚠  HIL close: {e.get_error_message()}")

        # ── Session summary ───────────────────────────────────
        total_time = time.time() - start_time if 'start_time' in dir() else 0
        print(f"\n{'─'*62}")
        print(f"  Session summary")
        print(f"  Mode        : {mode.upper()}")
        print(f"  Duration    : {total_time:.1f} s")
        print(f"  Loops       : {loop_count}")
        print(f"  Motor cuts  : {motor_cuts}  (UNSAFE triggers)")
        print(f"{'─'*62}\n")


if __name__ == "__main__":
    main()