#!/usr/bin/env python3
import numpy as np
import cv2
from quanser.multimedia import VideoCapture, ImageFormat, ImageDataType
from quanser.multimedia.exceptions import MediaError

# ── Config ────────────────────────────────────────────────────
FRAME_WIDTH  = 820
FRAME_HEIGHT = 410
FRAME_RATE   = 120.0
# ─────────────────────────────────────────────────────────────

# Ask user
sensor_id = input("Enter sensor ID (0 / 1 / 2 / 3): ").strip()
url = f"video://localhost:{sensor_id}"
print(f"Opening {url} ... Press 'q' to quit.")

# Buffer — shape (H, W, 3) for ROW_MAJOR_INTERLEAVED_BGR
image_data = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

capture = VideoCapture()
try:
    capture.open(url, FRAME_RATE, FRAME_WIDTH, FRAME_HEIGHT,
                 ImageFormat.ROW_MAJOR_INTERLEAVED_BGR, ImageDataType.UINT8, None, 0)
    capture.start()

    while True:
        if capture.read(image_data):
            cv2.imshow(f"CSI sensor-id={sensor_id}", image_data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except MediaError as e:
    print(f"MediaError: {e.get_error_message()}")

finally:
    capture.stop()
    capture.close()
    cv2.destroyAllWindows()
    print("Done.")
