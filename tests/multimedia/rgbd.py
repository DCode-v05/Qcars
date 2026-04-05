#!/usr/bin/env python3
"""
QCar 2 — Intel RealSense RGB-D Test
Fixed: removed get_depth_scale() — uses frame.get_meters() directly
"""

import numpy as np
import cv2
from quanser.multimedia import (
    Video3D,
    ImageFormat, ImageDataType, Video3DStreamType
)
from quanser.multimedia.exceptions import MediaError

# ══════════════════════════════════════════════════════════════
DEVICE_ID     = "0"
STREAM_INDEX  = 0
FRAME_RATE    = 30.0
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
WARMUP_FRAMES = 60
# ══════════════════════════════════════════════════════════════

def depth_colormap(depth_uint16):
    norm = cv2.normalize(depth_uint16, None, 0, 255,
                         cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

def main():
    print("═" * 52)
    print("  QCar 2 — Intel RealSense RGB-D Test")
    print(f"  Device     : {DEVICE_ID}")
    print(f"  Resolution : {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FRAME_RATE}fps")
    print("═" * 52)

    # ── Step 1: Open device ───────────────────────────────────
    print("\n[1/4] Opening Video3D device ...")
    try:
        video3d = Video3D(DEVICE_ID)
        print("      ✅ Device opened")
    except MediaError as e:
        print(f"      ❌ {e.get_error_message()}")
        return

    color_stream = None
    depth_stream = None

    try:
        # ── Step 2: Open streams ──────────────────────────────
        print("\n[2/4] Opening streams ...")

        color_stream = video3d.stream_open(
            Video3DStreamType.COLOR, STREAM_INDEX,
            FRAME_RATE, FRAME_WIDTH, FRAME_HEIGHT,
            ImageFormat.ROW_MAJOR_INTERLEAVED_BGR,
            ImageDataType.UINT8
        )
        print("      ✅ COLOR opened  (BGR UINT8)")

        depth_stream = video3d.stream_open(
            Video3DStreamType.DEPTH, STREAM_INDEX,
            FRAME_RATE, FRAME_WIDTH, FRAME_HEIGHT,
            ImageFormat.ROW_MAJOR_GREYSCALE,
            ImageDataType.UINT16
        )
        print("      ✅ DEPTH opened  (GREYSCALE UINT16)")

        # ── Step 3: Start streaming ───────────────────────────
        print("\n[3/4] Starting streams ...")
        video3d.start_streaming()
        print("      ✅ start_streaming() done")

        # ── Allocate buffers ──────────────────────────────────
        color_buf = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        depth_buf = np.zeros((FRAME_HEIGHT, FRAME_WIDTH),    dtype=np.uint16)
        # get_meters() needs float32 (H, W) buffer
        depth_m   = np.zeros((FRAME_HEIGHT, FRAME_WIDTH),    dtype=np.float32)

        # ── Step 4: Capture loop ──────────────────────────────
        print(f"\n[4/4] Capturing ... warmup={WARMUP_FRAMES} frames\n")

        color_count = 0
        depth_count = 0

        while color_count < WARMUP_FRAMES or depth_count < WARMUP_FRAMES:

            # ── COLOR ─────────────────────────────────────────
            try:
                c_frame = color_stream.get_frame()
                if c_frame is not None:
                    c_frame.get_data(color_buf)
                    c_num = c_frame.get_number()
                    c_ts  = c_frame.get_timestamp()
                    c_frame.release()           # mandatory before next get_frame
                    color_count += 1
                    print(f"  COLOR  #{c_num:>4}  ts={c_ts:.2f}  "
                          f"mean={color_buf.mean():.1f}  "
                          f"[{color_count}/{WARMUP_FRAMES}]",
                          end="\r")
            except MediaError:
                pass  # QERR_WOULD_BLOCK — no frame yet

            # ── DEPTH ─────────────────────────────────────────
            try:
                d_frame = depth_stream.get_frame()
                if d_frame is not None:
                    d_frame.get_data(depth_buf)     # raw UINT16
                    d_frame.get_meters(depth_m)     # direct metres (no scale needed)
                    d_num = d_frame.get_number()
                    d_ts  = d_frame.get_timestamp()
                    d_frame.release()               # mandatory before next get_frame
                    depth_count += 1
            except MediaError:
                pass

        # ── Save snapshots ────────────────────────────────────
        print("\n")

        cv2.imwrite("rgbd_color.jpg",         color_buf)
        cv2.imwrite("rgbd_depth_visual.jpg",  depth_colormap(depth_buf))
        np.save("rgbd_depth_raw.npy",         depth_buf)
        np.save("rgbd_depth_metres.npy",      depth_m)

        # Stats
        cx, cy      = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
        centre_m    = depth_m[cy, cx]
        valid_depth = depth_m[depth_m > 0]
        min_m       = valid_depth.min() if valid_depth.size > 0 else 0.0
        max_m       = valid_depth.max() if valid_depth.size > 0 else 0.0

        print("── Results ───────────────────────────────────────────")
        print(f"  COLOR  frames read   : {color_count}")
        print(f"  DEPTH  frames read   : {depth_count}")
        print(f"  COLOR  mean bright   : {color_buf.mean():.2f}")
        print(f"  DEPTH  centre ({cx},{cy}) : {centre_m:.3f} m")
        print(f"  DEPTH  min / max     : {min_m:.3f} m / {max_m:.3f} m")
        print(f"\n  Saved:")
        print(f"    rgbd_color.jpg         ← colour image")
        print(f"    rgbd_depth_visual.jpg  ← depth colourmap")
        print(f"    rgbd_depth_raw.npy     ← raw UINT16")
        print(f"    rgbd_depth_metres.npy  ← metres float32")
        print("─────────────────────────────────────────────────────")

    except MediaError as e:
        print(f"\n❌ MediaError: {e.get_error_message()}")

    finally:
        # ── Cleanup — strict order ────────────────────────────
        print("\n── Cleanup ───────────────────────────────────────────")
        try:
            video3d.stop_streaming()
            print("  ✅ stop_streaming()")
        except Exception as e:
            print(f"  ⚠️  stop_streaming: {e}")
        if color_stream:
            try:
                color_stream.close()
                print("  ✅ color_stream.close()")
            except Exception as e:
                print(f"  ⚠️  color_stream.close: {e}")
        if depth_stream:
            try:
                depth_stream.close()
                print("  ✅ depth_stream.close()")
            except Exception as e:
                print(f"  ⚠️  depth_stream.close: {e}")
        try:
            video3d.close()
            print("  ✅ video3d.close()")
        except Exception as e:
            print(f"  ⚠️  video3d.close: {e}")

if __name__ == "__main__":
    main()
