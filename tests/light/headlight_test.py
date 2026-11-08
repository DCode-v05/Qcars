import sys
sys.path.insert(0, '/home/nvidia/Documents/Quanser/libraries/python')

import time
import numpy as np
from pal.products.qcar import QCar

def main():
    car = QCar(readMode=0, frequency=500)

    LEDs = np.zeros(8, dtype=np.int8)

    try:
        print("=== QCar 2 Headlight Test ===")

        LEDs[0] = 1
        LEDs[1] = 1
        LEDs[2] = 1
        LEDs[3] = 1
        LEDs[4] = 1
        LEDs[5] = 1
        LEDs[6] = 1
        LEDs[7] = 1

        car.write(throttle=0, steering=0, LEDs=LEDs)
        print("Headlights ON — look at the front of the car!")

        time.sleep(5)

        LEDs[0] = 0
        LEDs[1] = 0
        LEDs[2] = 0
        LEDs[3] = 0
        LEDs[4] = 0
        LEDs[5] = 0
        LEDs[6] = 0
        LEDs[7] = 0

        car.write(throttle=0, steering=0, LEDs=LEDs)
        print("Headlights OFF.")

    finally:
        car.terminate()
        print("Done. QCar terminated safely.")

if __name__ == '__main__':
    main()
