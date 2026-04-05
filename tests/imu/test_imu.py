import time
import numpy as np
from quanser.hardware import HIL, HILError

card = HIL("qcar2", "0")

try:
    imu_channels = np.array(
        [3000, 3001, 3002,
         4000, 4001, 4002,
         10000],
        dtype=np.uint32
    )
    num_channels = len(imu_channels)
    imu_buffer = np.zeros(num_channels, dtype=np.float64)

    print("Reading IMU. Press Ctrl+C to stop.\n")
    while True:
        card.read_other(imu_channels, num_channels, imu_buffer)

        print(f"Gyro  (rad/s): x={imu_buffer[0]:+.4f}  y={imu_buffer[1]:+.4f}  z={imu_buffer[2]:+.4f}")
        print(f"Accel (m/s²):  x={imu_buffer[3]:+.4f}  y={imu_buffer[4]:+.4f}  z={imu_buffer[5]:+.4f}")
        print(f"Temp  (°C):    {imu_buffer[6]:.2f}\n")

        time.sleep(0.1)   # 10 Hz — adjust as needed

except HILError as e:
    print("HIL Error:", e.get_error_message())
except KeyboardInterrupt:
    print("Stopped.")
finally:
    card.close()
