#!/usr/bin/env python3
"""
QCar 2 — Motor Drive + Encoder Test (FIXED)
Official channel map from docs.quanser.com/quarc/documentation/qcar2.html

Other Output channels (write_other):
  11000 = Motor throttle  (units: %)   range: -1.0 to +1.0
  1000  = Steering angle  (units: rad) range: ~-0.5 to +0.5

⚠️  LIFT QCAR 2 OFF GROUND BEFORE RUNNING
Run: python3 enc_motor.py
"""

import numpy as np
import time
from quanser.hardware import HIL, EncoderQuadratureMode
from quanser.hardware.exceptions import HILError

# ══════════════════════════════════════════════════════════════
# QCar 2 — CORRECT Channel Map (from official docs)
# ══════════════════════════════════════════════════════════════
CARD_TYPE         = "qcar2"
CARD_ID           = "0"

THROTTLE_CHANNEL  = 11000     # Other Output — motor throttle (%)
STEERING_CHANNEL  = 1000      # Other Output — steering angle (rad)

ENCODER_CHANNEL   = 0         # Encoder Input — motor encoder (J24)

# Encoder math
COUNTS_PER_REV    = 720
QUADRATURE        = 4
COUNTS_PER_REV_X4 = COUNTS_PER_REV * QUADRATURE   # 2880
WHEEL_RADIUS_M    = 0.0328
WHEEL_CIRCUM_M    = 2 * np.pi * WHEEL_RADIUS_M
METRES_PER_COUNT  = WHEEL_CIRCUM_M / COUNTS_PER_REV_X4

# ── Test Profile ──────────────────────────────────────────────
# (throttle_%, duration_sec, label)
# Throttle: -1.0 = full reverse, 0.0 = stop, +1.0 = full forward
TEST_PHASES = [
    ( 0.00, 1.0, "IDLE    "),   # confirm encoder stable at rest
    ( 0.10, 3.0, "FWD 10% "),   # gentle forward
    ( 0.15, 3.0, "FWD 15% "),   # slightly faster
    ( 0.00, 2.0, "COAST   "),   # cut motor, watch coast
    (-0.10, 3.0, "REV 10% "),   # gentle reverse
    ( 0.00, 2.0, "STOP    "),   # final stop
]

SAMPLE_RATE_HZ  = 50
SAMPLE_PERIOD   = 1.0 / SAMPLE_RATE_HZ
# ══════════════════════════════════════════════════════════════


def counts_to_metres(counts):
    return counts * METRES_PER_COUNT


def main():
    print("═" * 58)
    print("  QCar 2 — Motor Drive + Encoder Test")
    print(f"  Throttle channel : {THROTTLE_CHANNEL}  (write_other)")
    print(f"  Steering channel : {STEERING_CHANNEL}  (write_other)")
    print(f"  Encoder channel  : {ENCODER_CHANNEL}")
    print(f"  Total duration   : {sum(p[1] for p in TEST_PHASES):.0f}s")
    print("═" * 58)

    # ── Safety confirmation ───────────────────────────────────
    print("\n⚠️  CONFIRM: QCar 2 is LIFTED OFF THE GROUND")
    confirm = input("   Type 'yes' to continue: ").strip().lower()
    if confirm != "yes":
        print("Aborted.")
        return

    # ── Open HIL board ────────────────────────────────────────
    print("\n[1/4] Opening HIL board ...")
    try:
        card = HIL(CARD_TYPE, CARD_ID)
        print(f"      ✅ '{CARD_TYPE}' opened")
    except HILError as e:
        print(f"      ❌ {e.get_error_message()}")
        return

    # ── Channel arrays ────────────────────────────────────────
    # Encoder
    enc_channels = np.array([ENCODER_CHANNEL], dtype=np.uint32)
    enc_buffer   = np.zeros(1, dtype=np.int32)

    # Motor + Steering — write_other uses channel numbers 11000, 1000
    motor_channels = np.array([THROTTLE_CHANNEL, STEERING_CHANNEL],
                               dtype=np.uint32)
    motor_buffer   = np.array([0.0, 0.0], dtype=np.float64)
    #                                ↑           ↑
    #                            throttle     steering=0 (straight)

    try:
        # ── Configure encoder ─────────────────────────────────
        print("[2/4] Configuring encoder ...")
        modes = np.array([EncoderQuadratureMode.X4], dtype=np.int32)
        card.set_encoder_quadrature_mode(enc_channels, 1, modes)
        card.set_encoder_counts(enc_channels, 1,
                                np.zeros(1, dtype=np.int32))
        print("      ✅ X4 quadrature, counter zeroed")

        # ── Zero motor at start ───────────────────────────────
        print("[3/4] Zeroing motor + steering ...")
        card.write_other(motor_channels, 2, motor_buffer)
        print(f"      ✅ throttle=0.0  steering=0.0 rad")

        # ── Test loop ─────────────────────────────────────────
        print("[4/4] Starting test profile ...\n")
        print(f"  {'Time':>6}  {'Phase':<12}  {'Throttle':>9}  "
              f"{'Counts':>8}  {'Dist(m)':>9}  {'Speed(m/s)':>11}")
        print(f"  {'-'*65}")

        prev_counts = 0
        prev_time   = time.time()
        start_time  = prev_time

        for throttle, duration, label in TEST_PHASES:

            # Write throttle + keep steering straight (0.0 rad)
            motor_buffer[0] = throttle   # throttle channel 11000
            motor_buffer[1] = 0.0        # steering channel 1000
            card.write_other(motor_channels, 2, motor_buffer)

            phase_end = time.time() + duration

            while time.time() < phase_end:
                t_loop = time.time()

                # Read encoder
                card.read_encoder(enc_channels, 1, enc_buffer)
                now          = time.time()
                curr_counts  = int(enc_buffer[0])
                elapsed      = now - start_time
                delta_counts = curr_counts - prev_counts
                delta_time   = now - prev_time
                speed_ms     = counts_to_metres(delta_counts) / delta_time \
                               if delta_time > 0 else 0.0
                distance_m   = counts_to_metres(curr_counts)

                direction = "→FWD" if delta_counts > 0 \
                       else "←REV" if delta_counts < 0 \
                       else " STP"

                print(f"  {elapsed:>6.2f}  {label:<12}  "
                      f"{throttle:>+9.2f}  "
                      f"{curr_counts:>+8d}  "
                      f"{distance_m:>+9.5f}  "
                      f"{speed_ms:>+10.4f}  {direction}",
                      end="\r")

                prev_counts = curr_counts
                prev_time   = now

                sleep = SAMPLE_PERIOD - (time.time() - t_loop)
                if sleep > 0:
                    time.sleep(sleep)

            print()   # newline between phases

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted — stopping motor ...")

    except HILError as e:
        print(f"\n❌ HILError: {e.get_error_message()}")

    finally:
        # ── ALWAYS stop motor ─────────────────────────────────
        print("\n── Cleanup ───────────────────────────────────────────")
        try:
            stop_buf = np.array([0.0, 0.0], dtype=np.float64)
            card.write_other(motor_channels, 2, stop_buf)
            print("  ✅ Motor stopped  (throttle=0.0, steering=0.0)")
        except Exception as e:
            print(f"  ⚠️  Motor stop: {e}")

        # Final encoder reading
        try:
            card.read_encoder(enc_channels, 1, enc_buffer)
            fc = int(enc_buffer[0])
            print(f"\n── Final Summary ─────────────────────────────────────")
            print(f"  Final count      : {fc:+d}")
            print(f"  Total distance   : {counts_to_metres(fc):+.5f} m")
            print(f"  Total revolutions: {fc/COUNTS_PER_REV_X4:+.4f}")
        except Exception:
            pass

        try:
            card.close()
            print("  ✅ HIL board closed")
        except HILError as e:
            print(f"  ⚠️  close: {e.get_error_message()}")


if __name__ == "__main__":
    main()
