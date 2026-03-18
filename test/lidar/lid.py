#!/usr/bin/env python3
"""
QCar 2 — RPLIDAR Connect + 3D Surface Plot
Official API: https://docs.quanser.com/quarc/documentation/python/devices/Devices/rplidar.html

Steps:
  1. Connect to LiDAR via serial URI
  2. Read multiple scans (INTERPOLATED mode — consistent angles)
  3. Convert polar (angle, distance) → Cartesian (X, Y)
  4. Accumulate scans over time → Z axis = scan index
  5. Save 3D surface plot as PNG (headless Jetson)

Run: python3 lidar_3d.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')              # headless — no display needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os

from quanser.devices import RPLIDAR, RangingMeasurements
from quanser.devices.enumerations import RangingDistance, RangingMeasurementMode
from quanser.devices.exceptions import DeviceError

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
# QCar 2 LiDAR serial URI — UART1 connector J26
LIDAR_URI = (
    "serial-cpu://localhost:1"
    "?baud='115200'"
    ",word='8'"
    ",parity='none'"
    ",stop='1'"
    ",flow='none'"
    ",dsr='on'"
)

# 720 measurements = one reading every 0.5 degrees (360/720)
NUM_MEASUREMENTS  = 720

# How many full scans to collect for the 3D surface
NUM_SCANS         = 30             # 30 scans = Z depth of surface

# Ranging mode
RANGING_DISTANCE  = RangingDistance.LONG    # 28m range
RANGING_MODE      = RangingMeasurementMode.INTERPOLATED

# Interpolation limits
MAX_INTERP_DIST   = 0.05           # max 5cm gap to interpolate
MAX_INTERP_ANGLE  = 0.1            # max 0.1 rad gap to interpolate

# Valid distance filter (ignore beyond 20m or below 0.1m)
MIN_DIST_M        = 0.10
MAX_DIST_M        = 20.0

OUTPUT_DIR        = "lidar_output"
# ══════════════════════════════════════════════════════════════

os.makedirs(OUTPUT_DIR, exist_ok=True)


def polar_to_cartesian(headings_rad, distances_m):
    """
    Convert polar LiDAR data → Cartesian X, Y.

    LiDAR gives:
      heading  = angle in radians (0 to 2π)
      distance = metres from sensor centre

    Conversion:
      X = distance × cos(heading)   ← left/right
      Y = distance × sin(heading)   ← forward/back
    """
    x = distances_m * np.cos(headings_rad)
    y = distances_m * np.sin(headings_rad)
    return x, y


def save_2d_scan(headings, distances, quality, scan_num, output_dir):
    """Save a single 2D polar scan as a top-down map."""
    # Filter valid points (quality > 0 and distance in range)
    valid = (quality > 0) & (distances > MIN_DIST_M) & (distances < MAX_DIST_M)
    h_valid = headings[valid]
    d_valid = distances[valid]

    x, y = polar_to_cartesian(h_valid, d_valid)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(x, y, c=d_valid, cmap='plasma',
               s=3, alpha=0.8, vmin=0, vmax=MAX_DIST_M)
    ax.set_xlabel("X (metres)")
    ax.set_ylabel("Y (metres)")
    ax.set_title(f"QCar 2 LiDAR — 2D Scan #{scan_num}\n"
                 f"Valid points: {valid.sum()}/{NUM_MEASUREMENTS}")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Mark LiDAR centre
    ax.plot(0, 0, 'r*', markersize=15, label='LiDAR')
    ax.legend()

    path = os.path.join(output_dir, f"scan_2d_{scan_num:03d}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def save_3d_surface(all_distances, all_headings, all_quality, output_dir):
    """
    Build and save a 3D surface from accumulated LiDAR scans.

    X, Y = Cartesian from polar (spatial)
    Z    = scan index (time axis — shows how environment changes)

    This creates a "scan tunnel" — each ring is one LiDAR rotation.
    """
    print("\n── Building 3D Surface ───────────────────────────────")

    # Stack all scans into arrays
    # Shape: (NUM_SCANS, NUM_MEASUREMENTS)
    dist_arr    = np.array(all_distances)    # shape (S, N)
    head_arr    = np.array(all_headings)     # shape (S, N)
    qual_arr    = np.array(all_quality)      # shape (S, N)

    # Filter invalid readings → set to NaN
    invalid = (qual_arr == 0) | (dist_arr < MIN_DIST_M) | (dist_arr > MAX_DIST_M)
    dist_arr[invalid] = np.nan

    # Convert to Cartesian
    x_arr = dist_arr * np.cos(head_arr)   # shape (S, N)
    y_arr = dist_arr * np.sin(head_arr)   # shape (S, N)

    # Z axis = scan index (time)
    z_arr = np.zeros_like(x_arr)
    for i in range(len(all_distances)):
        z_arr[i, :] = i

    print(f"  Scans       : {dist_arr.shape[0]}")
    print(f"  Points/scan : {dist_arr.shape[1]}")
    print(f"  Valid pts   : {(~invalid).sum()}")
    print(f"  Dist range  : {np.nanmin(dist_arr):.2f}m – {np.nanmax(dist_arr):.2f}m")

    # ── Plot 1: 3D Scatter (all valid points) ─────────────────
    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(121, projection='3d')
    mask = ~invalid
    sc = ax1.scatter(
        x_arr[mask], y_arr[mask], z_arr[mask],
        c=dist_arr[mask],
        cmap='plasma',
        s=1,
        alpha=0.6,
        vmin=0, vmax=MAX_DIST_M
    )
    plt.colorbar(sc, ax=ax1, label='Distance (m)', shrink=0.5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Scan Index (time)')
    ax1.set_title(f'QCar 2 LiDAR — 3D Scatter\n'
                  f'{(~invalid).sum()} valid points from {NUM_SCANS} scans')

    # ── Plot 2: Top-down 2D density map ───────────────────────
    ax2 = fig.add_subplot(122)
    all_x = x_arr[mask]
    all_y = y_arr[mask]
    all_d = dist_arr[mask]

    sc2 = ax2.scatter(all_x, all_y,
                      c=all_d, cmap='plasma',
                      s=2, alpha=0.4,
                      vmin=0, vmax=MAX_DIST_M)
    plt.colorbar(sc2, ax=ax2, label='Distance (m)')
    ax2.plot(0, 0, 'r*', markersize=15, label='LiDAR')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'Top-Down View ({NUM_SCANS} scans overlaid)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle('QCar 2 RPLIDAR — 3D Surface Analysis', fontsize=14)
    plt.tight_layout()

    path_3d = os.path.join(output_dir, "lidar_3d_surface.png")
    plt.savefig(path_3d, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {path_3d}")

    # ── Plot 3: Individual scan rings as 3D rings ─────────────
    fig2 = plt.figure(figsize=(12, 8))
    ax3  = fig2.add_subplot(111, projection='3d')

    cmap = plt.cm.viridis
    for i in range(min(NUM_SCANS, 30)):
        valid_i = ~invalid[i]
        if valid_i.sum() < 10:
            continue
        color = cmap(i / NUM_SCANS)
        ax3.plot(
            x_arr[i, valid_i],
            y_arr[i, valid_i],
            z_arr[i, valid_i],
            '.', markersize=1,
            color=color, alpha=0.7
        )

    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Scan Index')
    ax3.set_title('QCar 2 LiDAR — Scan Rings Over Time\n'
                  '(each colour = one full 360° rotation)')

    path_rings = os.path.join(output_dir, "lidar_rings.png")
    plt.savefig(path_rings, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {path_rings}")

    return path_3d, path_rings


def save_raw_data(all_distances, all_headings, all_quality, output_dir):
    """Save raw numpy arrays for later analysis."""
    np.save(os.path.join(output_dir, "lidar_distances.npy"),
            np.array(all_distances))
    np.save(os.path.join(output_dir, "lidar_headings.npy"),
            np.array(all_headings))
    np.save(os.path.join(output_dir, "lidar_quality.npy"),
            np.array(all_quality))
    print(f"  ✅ Raw data saved to {output_dir}/lidar_*.npy")


def main():
    print("═" * 55)
    print("  QCar 2 — RPLIDAR 3D Surface Test")
    print(f"  URI            : {LIDAR_URI[:40]}...")
    print(f"  Measurements   : {NUM_MEASUREMENTS} per scan (0.5° steps)")
    print(f"  Scans to collect: {NUM_SCANS}")
    print(f"  Ranging mode   : INTERPOLATED, LONG range (28m)")
    print("═" * 55)

    # ── Step 1: Create measurement buffer ─────────────────────
    print("\n[1/4] Allocating measurement buffer ...")
    measurements = RangingMeasurements(NUM_MEASUREMENTS)
    print(f"      ✅ Buffer: {NUM_MEASUREMENTS} measurements")

    # ── Step 2: Open LiDAR ────────────────────────────────────
    print("\n[2/4] Opening RPLIDAR ...")
    lidar = RPLIDAR()
    try:
        lidar.open(LIDAR_URI, RANGING_DISTANCE)
        print("      ✅ LiDAR opened")
        print("      ℹ️  LiDAR motor spinning up (2s) ...")
        time.sleep(2)
    except DeviceError as e:
        print(f"      ❌ DeviceError: {e}")
        print("\n  Troubleshooting:")
        print("  - Check LiDAR is connected to J26 (UART1)")
        print("  - Try: ls /dev/ttyTHS*")
        print("  - Try URI with localhost:1 instead of localhost:2")
        return

    # Storage for all scans
    all_distances = []
    all_headings  = []
    all_quality   = []

    try:
        # ── Step 3: Read scans ────────────────────────────────
        print(f"\n[3/4] Reading {NUM_SCANS} scans ...")
        print(f"      (Rotating LiDAR — keep environment still)\n")
        print(f"  {'Scan':>5}  {'Valid':>6}  {'Min(m)':>7}  "
              f"{'Max(m)':>7}  {'Mean(m)':>8}  {'Time(s)':>8}")
        print(f"  {'-'*55}")

        start_time  = time.time()
        scan_count  = 0
        empty_count = 0

        while scan_count < NUM_SCANS:
            t0 = time.time()

            # read() returns number of valid measurements
            n_valid = lidar.read(
                RANGING_MODE,
                MAX_INTERP_DIST,
                MAX_INTERP_ANGLE,
                measurements
            )

            if n_valid == 0:
                empty_count += 1
                if empty_count > 50:
                    print("\n  ⚠️  Too many empty reads — LiDAR may not be spinning")
                    break
                time.sleep(0.02)
                continue

            empty_count = 0

            # Extract data from measurements buffer
            distances = np.array([measurements.distance[i]
                                   for i in range(NUM_MEASUREMENTS)])
            headings  = np.array([measurements.heading[i]
                                   for i in range(NUM_MEASUREMENTS)])
            quality   = np.array([measurements.quality[i]
                                   for i in range(NUM_MEASUREMENTS)])

            # Store scan
            all_distances.append(distances.copy())
            all_headings.append(headings.copy())
            all_quality.append(quality.copy())

            scan_count += 1

            # Stats for this scan
            valid_mask = (quality > 0) & (distances > MIN_DIST_M) & \
                         (distances < MAX_DIST_M)
            n_good  = valid_mask.sum()
            elapsed = time.time() - start_time

            if n_good > 0:
                d_valid = distances[valid_mask]
                print(f"  {scan_count:>5}  {n_good:>6}  "
                      f"{d_valid.min():>7.2f}  "
                      f"{d_valid.max():>7.2f}  "
                      f"{d_valid.mean():>8.2f}  "
                      f"{elapsed:>8.2f}")
            else:
                print(f"  {scan_count:>5}  {n_good:>6}  "
                      f"{'---':>7}  {'---':>7}  {'---':>8}  "
                      f"{elapsed:>8.2f}")

            # Save first scan as 2D image immediately
            if scan_count == 1:
                path = save_2d_scan(headings, distances, quality,
                                    scan_count, OUTPUT_DIR)
                print(f"         → First scan saved: {path}")

    except KeyboardInterrupt:
        print(f"\n  ⚠️  Stopped at scan {scan_count}")

    except DeviceError as e:
        print(f"\n  ❌ Read error: {e}")

    finally:
        lidar.close()
        print("\n✅ LiDAR closed")

    # ── Step 4: Generate 3D surface plots ─────────────────────
    if len(all_distances) < 2:
        print("\n❌ Not enough scans for 3D surface")
        return

    print(f"\n[4/4] Generating 3D surface from {len(all_distances)} scans ...")
    save_3d_surface(all_distances, all_headings, all_quality, OUTPUT_DIR)
    save_raw_data(all_distances, all_headings, all_quality, OUTPUT_DIR)

    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'═'*55}")
    print(f"  Scans collected : {len(all_distances)}")
    print(f"  Total time      : {total_time:.1f}s")
    print(f"  Output dir      : {os.path.abspath(OUTPUT_DIR)}/")
    print(f"\n  Files saved:")
    for f in os.listdir(OUTPUT_DIR):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"    {f:<35} ({size:.0f} KB)")
    print("═" * 55)


if __name__ == "__main__":
    main()
