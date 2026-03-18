from quanser.devices import RPLIDAR
import time

ports = ["/dev/ttyTHS0", "/dev/ttyTHS1", "/dev/ttyTHS3"]

for port in ports:
    print(f"\nTrying {port}...")
    try:
        lidar = RPLIDAR(connection_mode=0, port=port)
        lidar.open()
        time.sleep(1)
        distances, angles = lidar.read()
        print(f"SUCCESS on {port}! Got {len(distances)} points")
        lidar.close()
        break
    except Exception as e:
        print(f"Failed: {e}")
