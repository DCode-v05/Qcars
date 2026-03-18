import numpy as np
from pal.products.qcar import QCar, QCarLidar
from pal.utilities.vision import Camera2D, Camera3D

print("1. Testing QCar...")
qcar = QCar(frequency=30, readMode=0)
qcar.read()
print(f"   accel={qcar.accelerometer}")
print("   QCar DONE")

print("2. Testing CSI camera...")
cam = Camera2D(cameraId="0", frameWidth=820, frameHeight=410, frameRate=30.0)
flag = cam.read()
print(f"   flag={flag}  shape={cam.imageData.shape}")
print("   CSI DONE")

print("3. Testing RealSense...")
rs = Camera3D(mode='RGB&Depth',
              frameWidthRGB=640, frameHeightRGB=480, frameRateRGB=30.0,
              frameWidthDepth=640, frameHeightDepth=480, frameRateDepth=30.0)
rs.read_RGB()
print(f"   RGB={rs.imageBufferRGB.shape}")
print("   RealSense DONE")

print("4. Testing LiDAR...")
lidar = QCarLidar(numMeasurements=384, rangingDistanceMode=2, interpolationMode=0)
flag = lidar.read()
print(f"   flag={flag}  shape={lidar.distances.shape}")
print("   LiDAR DONE")

print("ALL SENSORS OK")
