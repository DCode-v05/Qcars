"""
lidar_hello.py — QCar 2 LIDAR Hello World
Confirmed: QCar 2, serial-cpu://localhost:1, baud 256000
"""

import sys
import time
import numpy as np

# Add Quanser libraries to path
sys.path.insert(0, '/home/nvidia/Documents/Quanser/libraries/python')

from pal.utilities.lidar import Lidar

# ── QCar 2 CONFIG (confirmed) ─────────────────────────────────────────────────
NUM_MEASUREMENTS = 384
NUM_SCANS        = 10

print("=" * 55)
print("  QCar 2 — LIDAR Hello World")
print("  Port: localhost:1 | Baud: 256000 | RPLidar")
print("=" * 55)

try:
    # Patch the URL for QCar 2 BEFORE Lidar.__init__ opens the device
    # We subclass just enough to override the url attribute
    class QCar2Lidar(Lidar):
        url = ("serial-cpu://localhost:1?baud='256000',"
               "word='8',parity='none',stop='1',flow='none',dsr='on'")

    with QCar2Lidar(
        type='RPLidar',
        numMeasurements=NUM_MEASUREMENTS,
        rangingDistanceMode=2,   # LONG
        interpolationMode=0      # NORMAL
    ) as lidar:

        print("\n[OK]  LIDAR opened on localhost:1 @ 256000 baud")
        print("[..] Waiting for motor to spin up...\n")
        time.sleep(1.0)

        valid_scans = 0

        for i in range(NUM_SCANS):
            got_data = lidar.read()

            if got_data:
                distances = lidar.distances.flatten()
                angles    = np.degrees(lidar.angles.flatten())

                valid_mask = distances > 0
                valid_dist = distances[valid_mask]
                valid_ang  = angles[valid_mask]

                valid_scans += 1

                if valid_mask.sum() > 0:
                    closest_idx  = valid_dist.argmin()
                    farthest_idx = valid_dist.argmax()
                    print(
                        f"  Scan {i+1:02d}/{NUM_SCANS} | "
                        f"Valid: {valid_mask.sum():3d}/{NUM_MEASUREMENTS} | "
                        f"Closest:  {valid_dist[closest_idx]:.3f}m "
                        f"@ {valid_ang[closest_idx]:6.1f}° | "
                        f"Farthest: {valid_dist[farthest_idx]:.3f}m "
                        f"@ {valid_ang[farthest_idx]:6.1f}°"
                    )
                else:
                    print(f"  Scan {i+1:02d}/{NUM_SCANS} | All readings invalid (quality=0)")
            else:
                print(f"  Scan {i+1:02d}/{NUM_SCANS} | No data returned yet...")

            time.sleep(0.15)

        print(f"\n{'='*55}")
        print(f"  RESULT: {valid_scans}/{NUM_SCANS} scans had data")
        print(f"  LIDAR closed cleanly.")
        print(f"{'='*55}")

except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    print("\n-- Troubleshooting --")
    print("  Permission issue?  → sudo chmod a+rw /dev/ttyTHS*")
    print("  Wrong port?        → check /home/nvidia/Documents/Quanser/libraries/python/pal/products/qcar_config.json")
    sys.exit(1)
