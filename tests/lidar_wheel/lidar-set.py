#!/usr/bin/env python3
"""
QCar 2 — LiDAR Direction Diagnostic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Run this ONCE to find your QCar's true FRONT angle in LiDAR space.

Steps:
  1. Place an object ~0.4 m directly in FRONT of the QCar nose
  2. Run: python3 lidar_diag.py
  3. Watch CLOSEST POINT angle printed every scan
  4. That angle = your FRONT_CENTER_DEG
  5. Paste it into lidar_motor_safety.py  FRONT_CENTER_DEG = <value>
"""

import numpy as np
import time
from pal.products.qcar import QCarLidar

RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

NUM_MEASUREMENTS = 384
RANGING_MODE     = 2
INTERP_MODE      = 0

# ── ASCII sector rose ─────────────────────────────────────────
def ascii_rose(distances, angles_rad, n_sectors=16):
    sector_size = 2 * np.pi / n_sectors
    sector_min  = [None] * n_sectors
    for d, a in zip(distances, angles_rad):
        if d < 0.05:
            continue
        s = int((a % (2 * np.pi)) / sector_size) % n_sectors
        if sector_min[s] is None or d < sector_min[s]:
            sector_min[s] = d

    print(f"\n  {'─'*52}")
    print(f"  Sector map — 0 deg = LiDAR raw 0 rad, clockwise")
    print(f"  {'─'*52}")
    for i in range(0, n_sectors, 2):
        def fmt(idx):
            v   = sector_min[idx]
            deg = idx * (360.0 / n_sectors)
            if v is None:
                return f"  [{idx:2d}] {deg:5.1f}deg  {'---':<14}"
            bar = "█" * max(1, int((1 - min(v, 3.0) / 3.0) * 12))
            col = RED if v < 0.5 else YELLOW if v < 1.0 else GREEN
            return f"  [{idx:2d}] {deg:5.1f}deg  {col}{bar:<12}{RESET} {v:.2f}m"
        print(fmt(i) + fmt(i + 1))
    print(f"  {'─'*52}\n")


def main():
    print(f"\n{BOLD}{CYAN}QCar 2 — LiDAR Direction Diagnostic{RESET}")
    print(f"  Place object ~0.4 m in FRONT of QCar nose.")
    print(f"  Closest angle shown = FRONT_CENTER_DEG to paste into safety script.\n")

    lidar = QCarLidar(
        numMeasurements=NUM_MEASUREMENTS,
        rangingDistanceMode=RANGING_MODE,
        interpolationMode=INTERP_MODE,
    )
    print("  Waiting 2 s for LiDAR spin-up ...")
    time.sleep(2.0)
    print("  Ready. Press Ctrl+C to stop and get the copy-paste line.\n")
    print(f"  {'Scan':>5}  {'Closest dist':>12}  {'Closest angle (deg)':>20}  Rear angle")
    print(f"  {'─'*65}")

    scan = 0
    closest_deg = None

    try:
        while True:
            lidar.read()
            dists  = lidar.distances.flatten()
            angles = lidar.angles.flatten()

            valid = dists > 0.05
            if valid.sum() == 0:
                print(f"\r  {scan:>5}  no valid data", end="", flush=True)
                time.sleep(0.2)
                scan += 1
                continue

            vd = dists[valid]
            va = angles[valid]
            idx         = np.argmin(vd)
            closest_d   = vd[idx]
            closest_a   = va[idx]
            closest_deg = np.rad2deg(closest_a) % 360.0
            rear_deg    = (closest_deg + 180.0) % 360.0

            col = RED if closest_d < 0.5 else YELLOW if closest_d < 1.0 else GREEN

            print(
                f"\r  {scan:>5}  "
                f"{col}{closest_d:>9.3f} m{RESET}  "
                f"{col}{closest_deg:>17.2f} deg{RESET}  "
                f"rear={rear_deg:.2f} deg"
                f"  <-- FRONT_CENTER_DEG if object is at nose",
                end="", flush=True
            )

            if scan % 15 == 0:
                print()
                ascii_rose(dists, angles)

            scan += 1
            time.sleep(0.15)

    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Stopped after {scan} scans.{RESET}")
        print(f"\n  ┌──────────────────────────────────────────────────┐")
        print(f"  │  Paste this into lidar_motor_safety.py :         │")
        if closest_deg is not None:
            print(f"  │                                                  │")
            print(f"  │    FRONT_CENTER_DEG = {closest_deg:>6.1f}                    │")
            print(f"  │    (REAR auto-computed as FRONT + 180 deg)       │")
        print(f"  └──────────────────────────────────────────────────┘\n")

    finally:
        lidar.terminate()


if __name__ == "__main__":
    main()