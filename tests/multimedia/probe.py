#!/usr/bin/env python3

import numpy as np
import time
from quanser.multimedia import VideoCapture, ImageFormat, ImageDataType
from quanser.multimedia.exceptions import MediaError

MODES = [
    (3264, 2464, 21.0),
    (3264, 1848, 28.0),
    (1920, 1080, 30.0),
    (1640, 1232, 30.0),
    (1640,  820, 30.0),
    ( 820,  616, 60.0),
    ( 820,  410, 120.0),
    (1280,  720, 60.0),
    ( 640,  480, 30.0),
]

for sid in [0, 1, 2, 3]:
    print(f"\n── Sensor {sid} ──")
    for w, h, fps in MODES:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cap = VideoCapture()
        try:
            cap.open(f"video://localhost:{sid}", fps, w, h,
                     ImageFormat.ROW_MAJOR_INTERLEAVED_BGR,
                     ImageDataType.UINT8, [], 0)
            cap.start()
            # try reading a frame
            for _ in range(100):
                got = cap.read(img)
                if got:
                    break
                time.sleep(0.01)
            cap.stop()
            cap.close()
            print(f"  ✅ {w}x{h} @ {fps}fps")
        except MediaError as e:
            print(f"  ❌ {w}x{h} @ {fps}fps")
            try: cap.stop()
            except: pass
            try: cap.close()
            except: pass
