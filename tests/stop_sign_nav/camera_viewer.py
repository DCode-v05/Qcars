"""
camera_viewer.py  —  Live camera viewer with YOLO detections
═══════════════════════════════════════════════════════════════════════════════
Shows the front CSI camera feed in real-time in your browser with:
  - Raw camera frame via MJPEG stream (no page reloads!)
  - YOLO bounding boxes + segmentation masks overlaid
  - Detection list: name, class_id, confidence (refreshed via JSON poll)
  - FPS counter

HOW TO USE
──────────
  1. Run on QCar:   python3 camera_viewer.py
  2. Open browser:  http://192.168.2.2:5000   (replace with your Jetson IP)
  3. You'll see the live camera feed with YOLO detections overlaid

CAMERA INDEX
────────────
  Front camera on this QCar = index "2"
  If image looks wrong, try indices "0", "1", "3"

CONTROLS
────────
  The video is a persistent MJPEG stream — no page reloads.
  The detection panel refreshes every 500ms via a lightweight JSON poll.
  Terminal also prints what YOLO detects every second.

TROUBLESHOOTING
────────────────
  Black frame → wrong camera index, change CAM_ID below
  No detections → lower CONFIDENCE or move closer to camera
  Blurry/distorted → camera resolution mismatch (check CSI_WIDTH/HEIGHT)
═══════════════════════════════════════════════════════════════════════════════
"""

import time
import threading
import numpy as np
import cv2
from flask import Flask, Response, render_template_string, jsonify


CAM_ID      = "2"
CAM_WIDTH   = 820
CAM_HEIGHT  = 410
CAM_FPS     = 30.0

YOLO_WIDTH  = 640
YOLO_HEIGHT = 480
CONFIDENCE  = 0.25
FLASK_PORT  = 5000


_lock         = threading.Lock()
_jpeg_frame   = b''
_detections   = []
_yolo_fps     = 0.0
_cam_ok       = False

def _set_frame(jpeg: bytes, dets: list, fps: float):
    global _jpeg_frame, _detections, _yolo_fps
    with _lock:
        _jpeg_frame = jpeg
        _detections = dets
        _yolo_fps   = fps

def _get_frame():
    with _lock:
        return _jpeg_frame, list(_detections), _yolo_fps


app = Flask(__name__)

_PAGE = """<!DOCTYPE html>
<html>
<head>
  <title>QCar 2 — Camera Viewer</title>
  <!-- NO meta-refresh: browser keeps one persistent MJPEG connection -->
  <style>
    body { background:#0d1117; color:#c9d1d9; font-family:monospace;
           margin:0; padding:12px; }
    h2   { color:#3fb950; margin:0 0 8px; font-size:14px; }
    .grid { display:grid; grid-template-columns:2fr 1fr; gap:12px; }
    .card { background:#161b22; border:1px solid #21262d;
            border-radius:6px; padding:10px; }
    img  { width:100%; border-radius:4px; }
    .det { padding:4px 8px; margin:3px 0; border-radius:4px;
           font-size:12px; background:#21262d; }
    .PERSON { border-left:3px solid #f85149; }
    .MOVING { border-left:3px solid #e3b341; }
    .STATIC { border-left:3px solid #388bfd; }
    .conf   { color:#484f58; float:right; }
    .fps    { color:#3fb950; font-size:11px; margin-top:6px; }
    .none   { color:#484f58; font-size:12px; margin-top:8px; }
  </style>
</head>
<body>
  <h2>QCar 2 — Live Camera (cam{{ cam_id }}) + YOLO &nbsp;
    <span style="font-size:11px;color:#484f58;">MJPEG stream</span>
  </h2>
  <div class="grid">
    <div class="card">
      <!-- Single persistent HTTP connection; browser decodes each pushed JPEG in-place -->
      <img src="/stream" alt="camera">
      <div class="fps" id="fps-label">{{ fps }} fps &bull; {{ width }}×{{ height }}</div>
    </div>
    <div class="card" id="det-panel">
      <div style="font-size:11px;color:#8b949e;margin-bottom:6px;">
        DETECTIONS  (conf &gt; {{ conf }})
      </div>
      {% if dets %}
        {% for d in dets %}
        <div class="det {{ d.type }}">
          <b>{{ d.name }}</b>
          <span style="color:#8b949e;font-size:11px;"> id={{ d.cid }}</span>
          <span class="conf">{{ d.conf }}%</span>
          <div style="font-size:11px;color:#484f58;">
            {{ d.type }} &bull; {{ d.dist }}
          </div>
        </div>
        {% endfor %}
      {% else %}
        <div class="none">Nothing detected<br>
          <span style="font-size:11px;">Point camera at a person or object</span>
        </div>
      {% endif %}
    </div>
  </div>

  <script>
    // Poll /detections (lightweight JSON) every 500ms to refresh the
    // detection panel without touching the video stream at all.
    const CONF = "{{ conf }}";
    const W = "{{ width }}";
    const H = "{{ height }}";

    async function refreshDets() {
      try {
        const r = await fetch('/detections');
        const data = await r.json();

        document.getElementById('fps-label').textContent =
          data.fps + ' fps \u2022 ' + W + '\xd7' + H;

        const panel = document.getElementById('det-panel');
        let html = '<div style="font-size:11px;color:#8b949e;margin-bottom:6px;">DETECTIONS  (conf &gt; ' + CONF + ')</div>';
        if (data.dets.length === 0) {
          html += '<div class="none">Nothing detected<br><span style="font-size:11px;">Point camera at a person or object</span></div>';
        } else {
          for (const d of data.dets) {
            html += `<div class="det ${d.type}">
              <b>${d.name}</b>
              <span style="color:#8b949e;font-size:11px;"> id=${d.cid}</span>
              <span class="conf">${d.conf}%</span>
              <div style="font-size:11px;color:#484f58;">${d.type} &bull; ${d.dist}</div>
            </div>`;
          }
        }
        panel.innerHTML = html;
      } catch (e) { /* ignore transient network errors */ }
    }

    setInterval(refreshDets, 500);
  </script>
</body>
</html>
"""


@app.route('/')
def index():
    _, dets, fps = _get_frame()
    return render_template_string(
        _PAGE,
        cam_id = CAM_ID,
        fps    = f"{fps:.0f}",
        width  = YOLO_WIDTH,
        height = YOLO_HEIGHT,
        conf   = CONFIDENCE,
        dets   = dets,
    )


def _mjpeg_generator():
    """Yield a continuous MJPEG stream — one boundary chunk per new JPEG frame."""
    boundary = b'--frame'
    last_jpeg = b''
    while True:
        jpeg, _, _ = _get_frame()
        if not jpeg:
            blank = np.zeros((YOLO_HEIGHT, YOLO_WIDTH, 3), dtype=np.uint8)
            cv2.putText(blank, 'Waiting for camera...', (140, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
            _, buf = cv2.imencode('.jpg', blank)
            jpeg = buf.tobytes()

        if jpeg != last_jpeg:
            last_jpeg = jpeg
            yield (boundary + b'\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   jpeg + b'\r\n')
        else:
            time.sleep(0.005)


@app.route('/stream')
def stream():
    """MJPEG endpoint — one persistent HTTP response, frames pushed server-side."""
    return Response(
        _mjpeg_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={'Cache-Control': 'no-cache, no-store, must-revalidate'},
    )


@app.route('/detections')
def detections():
    """Lightweight JSON endpoint polled by JS every 500ms for the detection panel."""
    _, dets, fps = _get_frame()
    return jsonify(fps=f"{fps:.0f}", dets=dets)


def camera_loop():
    """Opens camera and YOLO, runs detection, updates shared state."""
    global _cam_ok

    from pal.utilities.vision import Camera2D
    from pit.YOLO.nets import YOLOv8

    PERSON_IDS = {0}
    MOVING_IDS = {1, 2, 3, 16}

    def get_type(cid):
        if cid in PERSON_IDS: return 'PERSON'
        if cid in MOVING_IDS: return 'MOVING'
        return 'STATIC'

    print(f"  Opening camera id='{CAM_ID}'...")
    cam = Camera2D(
        cameraId    = CAM_ID,
        frameWidth  = CAM_WIDTH,
        frameHeight = CAM_HEIGHT,
        frameRate   = CAM_FPS,
    )
    print(f"  Camera OK  buffer={cam.imageData.shape}")

    print("  Loading YOLOv8s-seg (uses cached engine)...")
    net = YOLOv8(imageWidth=YOLO_WIDTH, imageHeight=YOLO_HEIGHT)
    print("  YOLO OK")
    _cam_ok = True

    for _ in range(10):
        cam.read()
        time.sleep(0.05)

    print("\n  Camera loop running.")
    print(f"  Dashboard: check your browser at http://192.168.55.1:{FLASK_PORT}\n")

    last_print = time.perf_counter()

    while True:
        t0 = time.perf_counter()

        cam.read()
        frame_raw = cam.imageData

        img  = net.pre_process(frame_raw)
        pred = net.predict(img, classes=None,
                           confidence=CONFIDENCE, verbose=False)

        fps  = float(net.FPS)
        n    = len(pred.boxes.cls)

        dets = []
        if n > 0:
            net.post_processing(alignedDepth=None)
            for j in range(n):
                cid   = int(pred.boxes.cls[j].item())
                conf  = float(pred.boxes.conf[j].item())
                name  = pred.names[cid]
                dtype = get_type(cid)
                dets.append({
                    'cid':  cid,
                    'name': name,
                    'type': dtype,
                    'conf': f"{conf*100:.0f}",
                    'dist': '—',
                })
            annotated = net.post_process_render(showFPS=True)
        else:
            annotated = img.copy()
            cv2.putText(annotated, f'FPS:{fps:.0f}  No detections',
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (100, 100, 100), 1)

        _, buf = cv2.imencode('.jpg', annotated,
                              [cv2.IMWRITE_JPEG_QUALITY, 80])
        jpeg = buf.tobytes()

        _set_frame(jpeg, dets, fps)

        now = time.perf_counter()
        if now - last_print >= 1.0:
            if dets:
                names = ', '.join(f"{d['name']}({d['conf']}%)" for d in dets)
                print(f"  [{now-last_print:.0f}s] {n} detected: {names}  fps={fps:.0f}")
            else:
                print(f"  fps={fps:.0f}  nothing detected — point at an object")
            last_print = now

        elapsed = time.perf_counter() - t0
        sleep   = max(0, 1/30 - elapsed)
        if sleep > 0:
            time.sleep(sleep)


def main():
    import socket, logging

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = '10.1.77.73'

    print("═" * 55)
    print("  QCar 2 — Live Camera Viewer")
    print(f"  Front camera: id='{CAM_ID}'")
    print(f"\n  Open in browser: http://{ip}:{FLASK_PORT}")
    print("  Ctrl+C to stop.")
    print("═" * 55 + "\n")

    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    print("  Waiting for camera and YOLO to initialise...")
    while not _cam_ok:
        time.sleep(0.1)
    print("  Ready.\n")

    app.run(host='0.0.0.0', port=FLASK_PORT, threaded=True)


if __name__ == '__main__':
    main()
