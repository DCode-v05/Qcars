#!/usr/bin/env python3
"""
QCar 2 — Wheel Encoder Test
Uses: quanser.hardware HIL API
Docs: https://docs.quanser.com/quarc/documentation/python/hardware/

Reads rear wheel encoder counts in a live loop.
Converts counts → distance (metres) and velocity (m/s).

Run: python3 test_encoder.py
Manually spin the rear wheel to see counts change.
Press Ctrl+C to stop.
"""

import numpy as np
import time
from quanser.hardware import HIL, EncoderQuadratureMode
from quanser.hardware.exceptions import HILError

# ══════════════════════════════════════════════════════════════
# QCar 2 Hardware Config
# ══════════════════════════════════════════════════════════════
CARD_TYPE         = "qcar2"          # QCar 2 HIL card type
CARD_ID           = "0"              # First (and only) board

ENCODER_CHANNEL   = 0                # Rear wheel encoder channel
COUNTS_PER_REV    = 720              # Base encoder resolution
QUADRATURE        = 4                # X4 mode → 720 × 4 = 2880
COUNTS_PER_REV_X4 = COUNTS_PER_REV * QUADRATURE   # = 2880

WHEEL_RADIUS_M    = 0.0328           # metres
WHEEL_CIRCUM_M    = 2 * np.pi * WHEEL_RADIUS_M     # = 0.2061 m
METRES_PER_COUNT  = WHEEL_CIRCUM_M / COUNTS_PER_REV_X4  # m per count

SAMPLE_RATE_HZ    = 100              # Read rate (100 Hz = 10ms loop)
SAMPLE_PERIOD     = 1.0 / SAMPLE_RATE_HZ
# ══════════════════════════════════════════════════════════════


def counts_to_metres(counts):
    """Convert raw encoder counts → distance in metres."""
    return counts * METRES_PER_COUNT


def main():
    print("═" * 55)
    print("  QCar 2 — Wheel Encoder Test")
    print(f"  Card           : {CARD_TYPE}  ID={CARD_ID}")
    print(f"  Encoder channel: {ENCODER_CHANNEL}")
    print(f"  Counts/rev     : {COUNTS_PER_REV_X4} (X4 quadrature)")
    print(f"  Metres/count   : {METRES_PER_COUNT:.7f} m")
    print(f"  Sample rate    : {SAMPLE_RATE_HZ} Hz")
    print("═" * 55)
    print("\nSpin the rear wheel manually to see counts change.")
    print("Press Ctrl+C to stop.\n")

    # ── Step 1: Open HIL board ────────────────────────────────
    print("[1/4] Opening HIL board ...")
    try:
        card = HIL(CARD_TYPE, CARD_ID)
        print(f"      ✅ '{CARD_TYPE}' opened")
    except HILError as e:
        print(f"      ❌ HILError: {e.get_error_message()}")
        print("\n  Troubleshooting:")
        print("  - Verify card type: try 'qcar2' or 'qcar'")
        print("  - Check: ls /dev/ | grep -i qcar")
        return

    # ── numpy arrays (must be correct dtype per docs) ─────────
    enc_channels = np.array([ENCODER_CHANNEL], dtype=np.uint32)  # channel list
    enc_buffer   = np.zeros(1, dtype=np.int32)                    # int32 for counts
    num_channels = len(enc_channels)

    try:
        # ── Step 2: Set X4 quadrature mode ───────────────────
        print("[2/4] Setting quadrature mode to X4 ...")
        modes = np.array([EncoderQuadratureMode.X4], dtype=np.int32)
        card.set_encoder_quadrature_mode(enc_channels, num_channels, modes)
        print("      ✅ X4 quadrature set")

        # ── Step 3: Zero the encoder counter ─────────────────
        print("[3/4] Zeroing encoder counter ...")
        zero_counts = np.zeros(1, dtype=np.int32)
        card.set_encoder_counts(enc_channels, num_channels, zero_counts)
        print("      ✅ Counter reset to 0")

        # ── Step 4: Live read loop ────────────────────────────
        print("[4/4] Starting read loop ...\n")
        print(f"  {'Time(s)':>8}  {'Counts':>10}  "
              f"{'Distance(m)':>13}  {'Velocity(m/s)':>14}  "
              f"{'Revolutions':>12}")
        print(f"  {'-'*65}")

        prev_counts   = 0
        prev_time     = time.time()
        start_time    = prev_time
        sample_count  = 0

        while True:
            t_loop_start = time.time()

            # ── Read encoder ──────────────────────────────────
            card.read_encoder(enc_channels, num_channels, enc_buffer)

            current_counts = int(enc_buffer[0])
            current_time   = time.time()

            # ── Calculate values ──────────────────────────────
            elapsed        = current_time - start_time
            delta_counts   = current_counts - prev_counts
            delta_time     = current_time - prev_time

            distance_m     = counts_to_metres(current_counts)
            velocity_ms    = counts_to_metres(delta_counts) / delta_time \
                             if delta_time > 0 else 0.0
            revolutions    = current_counts / COUNTS_PER_REV_X4

            # ── Print every sample ────────────────────────────
            direction = "→FWD" if delta_counts > 0 \
                   else "←REV" if delta_counts < 0 \
                   else "STOP"

            print(f"  {elapsed:>8.2f}  "
                  f"{current_counts:>+10d}  "
                  f"{distance_m:>+13.5f}  "
                  f"{velocity_ms:>+14.4f}  "
                  f"{revolutions:>+12.4f}  "
                  f"{direction}",
                  end="\r")

            prev_counts = current_counts
            prev_time   = current_time
            sample_count += 1

            # ── Maintain sample rate ──────────────────────────
            elapsed_loop = time.time() - t_loop_start
            sleep_time   = SAMPLE_PERIOD - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n\n── Final Results ─────────────────────────────────────")
        print(f"  Total samples    : {sample_count}")
        print(f"  Final count      : {current_counts:+d}")
        print(f"  Total distance   : {counts_to_metres(current_counts):+.5f} m")
        print(f"  Total revolutions: {current_counts / COUNTS_PER_REV_X4:+.4f}")
        print(f"  Run time         : {time.time() - start_time:.2f} s")
        print("─────────────────────────────────────────────────────")

    except HILError as e:
        print(f"\n❌ HILError during read: {e.get_error_message()}")

    finally:
        # ── Always close ──────────────────────────────────────
        try:
            card.close()
            print("\n✅ HIL board closed.")
        except HILError as e:
            print(f"⚠️  close warning: {e.get_error_message()}")


if __name__ == "__main__":
    main()
