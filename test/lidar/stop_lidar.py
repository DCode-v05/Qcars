import sys
sys.path.insert(0, '/home/nvidia/Documents/Quanser/libraries/python')

from pal.products.qcar import QCarLidar

print("Stopping LiDAR motor...")
myLidar = QCarLidar(
    numMeasurements=1000,
    rangingDistanceMode=2,
    interpolationMode=0
)
myLidar.terminate()
print("✅ LiDAR motor stopped.")
