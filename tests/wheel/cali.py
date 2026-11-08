import numpy as np
import time
from hal.utilities.image_processing import ImageProcessing
from pal.products.qcar import QCar

STEER_BIAS = -0.04

myCar = QCar()

try:
    print(f"Applying steer_bias = {STEER_BIAS} rad")
    print("Driving straight for 3 seconds... observe wheel alignment.")

    start = time.time()
    while time.time() - start < 3.0:
        steering = 0.0 + STEER_BIAS
        throttle = 0.05

        myCar.read_write_std(throttle=throttle, steering=steering)
        time.sleep(0.01)

    print("Done. Check if car drove straight.")

finally:
    myCar.read_write_std(throttle=0, steering=steering)
