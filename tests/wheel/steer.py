#!/usr/bin/env python3
"""
QCar 2 — Steering Test
Official channel from docs.quanser.com/quarc/documentation/qcar2.html

Other Output:
  channel 1000  = Steering angle (radians)
   0.0  rad = straight ahead
  +0.5  rad = full RIGHT (~28 degrees)
  -0.5  rad = full LEFT  (~28 degrees)

Motor stays at 0.0 (stationary) — only steering moves.
Run: python3 test_steer.py
"""

import numpy as np
import time
from quanser.hardware import HIL
from quanser.hardware.exceptions import HILError

CARD_TYPE        = "qcar2"
CARD_ID          = "0"

THROTTLE_CHANNEL = 11000
STEERING_CHANNEL = 1000

MAX_STEER_RAD    =  0.5
MIN_STEER_RAD    = -0.5
STEER_STEP_RAD   =  0.05

TEST_PHASES = [
    ( 0.00, 2.0, "STRAIGHT      "),
    ( 0.10, 2.0, "RIGHT SLIGHT  "),
    ( 0.25, 2.0, "RIGHT MEDIUM  "),
    ( 0.50, 2.0, "RIGHT FULL    "),
    ( 0.00, 2.0, "STRAIGHT      "),
    (-0.10, 2.0, "LEFT  SLIGHT  "),
    (-0.25, 2.0, "LEFT  MEDIUM  "),
    (-0.50, 2.0, "LEFT  FULL    "),
    ( 0.00, 2.0, "STRAIGHT      "),
]


def rad_to_deg(rad):
    return rad * (180.0 / np.pi)


def main():
    print("═" * 55)
    print("  QCar 2 — Steering Test")
    print(f"  Steering channel : {STEERING_CHANNEL}  (write_other)")
    print(f"  Throttle channel : {THROTTLE_CHANNEL}  (write_other)")
    print(f"  Range            : {MIN_STEER_RAD} to {MAX_STEER_RAD} rad")
    print(f"  Range (degrees)  : {rad_to_deg(MIN_STEER_RAD):.1f}° to "
          f"{rad_to_deg(MAX_STEER_RAD):.1f}°")
    print(f"  Motor throttle   : 0.0 (stationary — only servo moves)")
    print(f"  Total duration   : {sum(p[1] for p in TEST_PHASES):.0f}s")
    print("═" * 55)

    print("\n⚠️  Car can stay on ground — motor is OFF (throttle=0.0)")
    print("   Watch the FRONT WHEELS physically turn left/right.\n")
    confirm = input("   Type 'yes' to start: ").strip().lower()
    if confirm != "yes":
        print("Aborted.")
        return

    print("\n[1/2] Opening HIL board ...")
    try:
        card = HIL(CARD_TYPE, CARD_ID)
        print(f"      ✅ '{CARD_TYPE}' opened")
    except HILError as e:
        print(f"      ❌ {e.get_error_message()}")
        return

    channels = np.array([THROTTLE_CHANNEL, STEERING_CHANNEL],
                         dtype=np.uint32)
    buffer   = np.array([0.0, 0.0], dtype=np.float64)

    try:
        card.write_other(channels, 2, buffer)
        print("[2/2] Outputs zeroed. Starting steering test ...\n")

        print(f"  {'Time':>6}  {'Phase':<18}  "
              f"{'Steer(rad)':>10}  {'Steer(deg)':>10}  "
              f"{'Direction':>12}")
        print(f"  {'-'*65}")

        start_time = time.time()

        for steer_rad, hold_sec, label in TEST_PHASES:

            buffer[0] = 0.0
            buffer[1] = steer_rad
            card.write_other(channels, 2, buffer)

            elapsed   = time.time() - start_time
            steer_deg = rad_to_deg(steer_rad)

            if steer_rad > 0.01:
                direction = "→ RIGHT"
            elif steer_rad < -0.01:
                direction = "← LEFT"
            else:
                direction = "↑ STRAIGHT"

            print(f"  {elapsed:>6.2f}  {label:<18}  "
                  f"{steer_rad:>+10.3f}  "
                  f"{steer_deg:>+9.1f}°  "
                  f"{direction:>12}")

            time.sleep(hold_sec)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted — centering steering ...")

    except HILError as e:
        print(f"\n❌ HILError: {e.get_error_message()}")

    finally:
        print("\n── Cleanup ───────────────────────────────────────────")
        try:
            buffer[0] = 0.0
            buffer[1] = 0.0
            card.write_other(channels, 2, buffer)
            print("  ✅ Steering centred  (0.0 rad)")
            print("  ✅ Throttle stopped  (0.0)")
        except Exception as e:
            print(f"  ⚠️  Reset warning: {e}")

        try:
            card.close()
            print("  ✅ HIL board closed")
        except HILError as e:
            print(f"  ⚠️  close: {e.get_error_message()}")

        print("\n── Test Complete ─────────────────────────────────────")


if __name__ == "__main__":
    main()
