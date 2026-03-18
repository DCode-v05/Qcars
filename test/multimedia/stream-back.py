#!/usr/bin/env python3
"""
QCar 2 — CSI Camera One-by-One Test
Uses: quanser.multimedia.VideoCapture (official API)
Docs: https://docs.quanser.com/quarc/documentation/python/multimedia/

Tests sensor-id 0,1,2,3 sequentially.
For each camera:
  - Opens connection
  - Starts stream
  - Reads 30 frames
  - Saves one snapshot as .jpg
  - Stops and closes cleanly

Run: python3 test_csi_one_by_one.py
"""

import numpy as np
import cv2
from quanser.multimedia import VideoCapture, ImageFormat, ImageDataType
from quanser.multimedia.exceptions import MediaError

# ══════════════════════════════════════════════════════════════
# CONFIG — IMX219 native Mode 3 (confirmed from your GStreamer)
# ══════════════════════════════════════════════════════════════
FRAME_WIDTH  = 820
FRAME_HEIGHT = 410
FRAME_RATE   = 120.0
FRAMES_TO_READ = 30        # frames per camera test (~0.25s at 120fps)

# All 4 confirmed CSI sensors on QCar 2
CSI_SENSORS = [0, 1, 2, 3]
# ══════════════════════════════════════════════════════════════


def test_camera(sensor_id: int) -> bool:
    """
    Test a single CSI camera by sensor_id.

    Follows the exact API lifecycle from official docs:
      VideoCapture() → open() → start() → read() → stop() → close()

    Returns True if test passed, False if failed.
    """

    # URL format from docs: "video://localhost:id"
    url = f"video://localhost:{sensor_id}"

    print(f"\n{'─'*50}")
    print(f"  Testing sensor-id = {sensor_id}")
    print(f"  URL              : {url}")
    print(f"  Resolution       : {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FRAME_RATE}fps")
    print(f"  Frames to read   : {FRAMES_TO_READ}")
    print(f"{'─'*50}")

    # Step 1 — Allocate image buffer
    # Shape must be (H, W, 3) for ROW_MAJOR_INTERLEAVED_BGR + UINT8
    image_data = np.zeros(
        (FRAME_HEIGHT, FRAME_WIDTH, 3),
        dtype=np.uint8
    )
    print(f"  [1/5] Buffer allocated : shape={image_data.shape}, dtype={image_data.dtype}")

    # Step 2 — Construct VideoCapture object (no connection yet)
    capture = VideoCapture()
    print(f"  [2/5] VideoCapture()   : object created")

    try:
        # Step 3 — open(): connect to the camera
        capture.open(
            url,                                   # video://localhost:N
            FRAME_RATE,                            # float
            FRAME_WIDTH,                           # int
            FRAME_HEIGHT,                          # int
            ImageFormat.ROW_MAJOR_INTERLEAVED_BGR, # BGR, OpenCV-compatible
            ImageDataType.UINT8,                   # 8-bit per channel
            None,                                  # no extra attributes
            0                                      # num_attributes = 0
        )
        print(f"  [3/5] open()           : ✅ connected to {url}")

        # Step 4 — start(): begin streaming frames
        capture.start()
        print(f"  [4/5] start()          : ✅ streaming started")

        # Step 5 — read() loop
        print(f"  [5/5] read() loop      : reading {FRAMES_TO_READ} frames...")

        frames_got  = 0
        frames_miss = 0

        while frames_got < FRAMES_TO_READ:
            # read() → True  : new frame written into image_data
            # read() → False : no new frame yet, poll again
            # read() → raises MediaError on hardware failure
            got = capture.read(image_data)

            if got:
                frames_got += 1
            else:
                frames_miss += 1

        # Report frame stats
        print(f"         frames received : {frames_got}")
        print(f"         missed polls    : {frames_miss}")
        print(f"         last frame mean : {image_data.mean():.2f}  "
              f"(0=black, 127=mid, 255=white)")

        # Save snapshot to verify actual image content
        snapshot = f"snapshot_sensor{sensor_id}.jpg"
        cv2.imwrite(snapshot, image_data)
        print(f"         snapshot saved  : {snapshot}")

        return True   # Test passed

    except MediaError as e:
        # get_error_message() gives the detailed Quanser error string
        print(f"\n  ❌ MediaError: {e.get_error_message()}")
        print(f"     Possible causes:")
        print(f"       - Wrong resolution (try 1640x820)")
        print(f"       - sensor-id={sensor_id} not physically connected")
        print(f"       - Another process is using this camera")
        return False  # Test failed

    finally:
        # Always stop() then close() — even if read() failed
        # stop() before close() is mandatory per the docs lifecycle

        try:
            capture.stop()
            print(f"  stop()  : ✅")
        except MediaError as e:
            print(f"  stop()  : warning → {e.get_error_message()}")

        try:
            capture.close()
            print(f"  close() : ✅")
        except MediaError as e:
            print(f"  close() : warning → {e.get_error_message()}")


def main():
    print("═" * 50)
    print("  QCar 2 — CSI Camera Test (Quanser API)")
    print(f"  Sensors to test : {CSI_SENSORS}")
    print(f"  Resolution      : {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FRAME_RATE}fps")
    print("═" * 50)

    results = {}

    # Test each sensor one by one — sequentially
    for sensor_id in CSI_SENSORS:
        passed = test_camera(sensor_id)
        results[sensor_id] = passed

    # Final summary
    print(f"\n{'═'*50}")
    print("  FINAL RESULTS")
    print(f"{'═'*50}")
    for sensor_id, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        snap   = f"→ snapshot_sensor{sensor_id}.jpg" if passed else ""
        print(f"  sensor-id={sensor_id} : {status}  {snap}")

    total   = len(results)
    passing = sum(results.values())
    print(f"\n  {passing}/{total} cameras passed")
    print("═" * 50)


if __name__ == "__main__":
    main()
