import sys
sys.path.insert(0, '/home/nvidia/Documents/Quanser/libraries/python')

import time
import numpy as np
from pal.products.qcar import QCarLidar

print("=== Quanser QCar2 Official LiDAR Test ===\n")

numMeasurements      = 1000
lidarMeasurementMode = 2
lidarInterpolationMode = 0

print("Initializing QCarLidar...")
myLidar = QCarLidar(
    numMeasurements=numMeasurements,
    rangingDistanceMode=lidarMeasurementMode,
    interpolationMode=lidarInterpolationMode
)
print("✅ LiDAR initialized!\n")

runTime = 10.0
t0 = time.time()
scan_count = 0

print(f"Reading LiDAR for {runTime} seconds...\n")

try:
    while time.time() - t0 < runTime:
        myLidar.read()
        scan_count += 1

        distances = np.array(myLidar.distances)
        angles    = np.array(myLidar.angles)

        valid = distances[distances > 0]

        print(f"Scan #{scan_count:03d} | "
              f"Points: {len(distances)} | "
              f"Valid hits: {len(valid)} | "
              f"Min: {valid.min():.3f}m | "
              f"Max: {valid.max():.3f}m"
              if len(valid) > 0 else
              f"Scan #{scan_count:03d} | ⚠️  No valid hits")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    myLidar.terminate()
    print(f"\n✅ Done. Total scans: {scan_count}")
    print("LiDAR terminated cleanly.")

