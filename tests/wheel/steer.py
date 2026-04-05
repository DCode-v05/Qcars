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

# ══════════════════════════════════════════════════════════════
# QCar 2 Channel Map
# ══════════════════════════════════════════════════════════════
CARD_TYPE        = "qcar2"
CARD_ID          = "0"

THROTTLE_CHANNEL = 11000      # motor throttle (%)
STEERING_CHANNEL = 1000       # steering angle (rad)

# Steering limits (radians)
MAX_STEER_RAD    =  0.5       # full right
MIN_STEER_RAD    = -0.5       # full left
STEER_STEP_RAD   =  0.05      # increment per step (~3 degrees)

# ── Test Profile ──────────────────────────────────────────────
# (steering_rad, hold_seconds, label)
TEST_PHASES = [
    ( 0.00, 2.0, "STRAIGHT      "),   # centre
    ( 0.10, 2.0, "RIGHT SLIGHT  "),   # slight right
    ( 0.25, 2.0, "RIGHT MEDIUM  "),   # medium right
    ( 0.50, 2.0, "RIGHT FULL    "),   # full right
    ( 0.00, 2.0, "STRAIGHT      "),   # back to centre
    (-0.10, 2.0, "LEFT  SLIGHT  "),   # slight left
    (-0.25, 2.0, "LEFT  MEDIUM  "),   # medium left
    (-0.50, 2.0, "LEFT  FULL    "),   # full left
    ( 0.00, 2.0, "STRAIGHT      "),   # return to centre
]
# ══════════════════════════════════════════════════════════════


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

    # ── Open HIL board ────────────────────────────────────────
    print("\n[1/2] Opening HIL board ...")
    try:
        card = HIL(CARD_TYPE, CARD_ID)
        print(f"      ✅ '{CARD_TYPE}' opened")
    except HILError as e:
        print(f"      ❌ {e.get_error_message()}")
        return

    # Both throttle + steering written together
    channels = np.array([THROTTLE_CHANNEL, STEERING_CHANNEL],
                         dtype=np.uint32)
    buffer   = np.array([0.0, 0.0], dtype=np.float64)
    #                        ↑   ↑
    #                   throttle  steering

    try:
        # ── Zero both outputs first ───────────────────────────
        card.write_other(channels, 2, buffer)
        print("[2/2] Outputs zeroed. Starting steering test ...\n")

        print(f"  {'Time':>6}  {'Phase':<18}  "
              f"{'Steer(rad)':>10}  {'Steer(deg)':>10}  "
              f"{'Direction':>12}")
        print(f"  {'-'*65}")

        start_time = time.time()

        for steer_rad, hold_sec, label in TEST_PHASES:

            # Write steering — motor stays 0.0
            buffer[0] = 0.0          # throttle = OFF
            buffer[1] = steer_rad    # steering = target angle
            card.write_other(channels, 2, buffer)

            elapsed   = time.time() - start_time
            steer_deg = rad_to_deg(steer_rad)

            # Direction indicator
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
        # ── Always return to straight + stop ──────────────────
        print("\n── Cleanup ───────────────────────────────────────────")
        try:
            buffer[0] = 0.0   # throttle = 0
            buffer[1] = 0.0   # steering = straight
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
