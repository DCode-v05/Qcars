"""
dashboard.py  —  Live web dashboard for QCar 2
Flask server at http://<jetson-ip>:5000

CRITICAL FIX: All image rendering runs in a background thread at max 5Hz.
Dashboard.update() is now instant (~0ms) — never blocks the main control loop.
Depth chart uses OpenCV instead of matplotlib (~1ms vs ~200ms).
"""
import io
import threading
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, Response, render_template_string

from obstacle_detector import (
    ZONE_CLEAR, ZONE_WARN, ZONE_STOP,
    BEHAVIOUR_NAVIGATE, BEHAVIOUR_WAIT,
    BEHAVIOUR_AVOID, BEHAVIOUR_EMERGENCY_STOP,
)


# ── Shared state ──────────────────────────────────────────────────────────────

class _State:
    def __init__(self):
        self._lock       = threading.Lock()
        self.detection   = {}
        self.pose        = np.zeros(3)
        self.speed_mps   = 0.0
        self.dist_goal   = 99.0
        self.leds        = [0]*8
        self.timing      = {}
        self.sm_state    = 'IDLE'
        self._raw_frame  = None     # raw BGR for render thread
        self._cam_jpeg   = b''
        self._lidar_png  = b''
        self._depth_png  = b''
        self._dirty      = False    # new data available for rendering

    def push(self, detection, pose, speed_mps, dist_goal, leds, timing, sm_state):
        """Called from main loop — instant, no rendering."""
        with self._lock:
            self.detection  = detection
            self.pose       = pose
            self.speed_mps  = speed_mps
            self.dist_goal  = dist_goal
            self.leds       = leds
            self.timing     = timing
            self.sm_state   = sm_state
            self._raw_frame = detection.get('annotated_frame', None)
            self._dirty     = True

    def pop_for_render(self):
        """Called from render thread — grabs snapshot and clears dirty flag."""
        with self._lock:
            if not self._dirty:
                return None
            self._dirty = False
            return (dict(self.detection), self._raw_frame,
                    self.pose.copy(), self.sm_state)

    def set_images(self, cam_jpeg, lidar_png, depth_png):
        with self._lock:
            if cam_jpeg:   self._cam_jpeg  = cam_jpeg
            if lidar_png:  self._lidar_png = lidar_png
            if depth_png:  self._depth_png = depth_png

    def get(self):
        with self._lock:
            return (dict(self.detection), self.pose.copy(),
                    self.speed_mps, self.dist_goal,
                    list(self.leds), dict(self.timing), self.sm_state)

    def get_cam(self):
        with self._lock: return self._cam_jpeg
    def get_lidar(self):
        with self._lock: return self._lidar_png
    def get_depth(self):
        with self._lock: return self._depth_png


_s = _State()


# ── Fast image generators ────────────────────────────────────────────────────

def _cam_jpeg(frame):
    """Encode BGR frame to JPEG. ~5ms."""
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, 'No frame', (220, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (60,60,60), 2)
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()


def _depth_cv2(det):
    """
    Depth clearance bar chart using OpenCV — replaces matplotlib.
    ~1ms instead of ~200ms.
    """
    W, H = 280, 180
    img = np.full((H, W, 3), (22, 27, 13), dtype=np.uint8)  # #0d1117 BGR

    lm = det.get('left_clear_m',  99.0)
    rm = det.get('right_clear_m', 99.0)
    gs = det.get('gap_side',      'left')
    gw = det.get('gap_width_px',  0)

    max_d = 3.0
    bar_w = 55
    max_h = H - 50
    lh = int((min(lm, max_d) / max_d) * max_h)
    rh = int((min(rm, max_d) / max_d) * max_h)

    # Colours (BGR): green=(80,185,63), blue=(253,139,56), yellow=(65,179,227)
    lcolor = (80, 185, 63) if gs == 'left' else (253, 139, 56)
    rcolor = (80, 185, 63) if gs == 'right' else (65, 179, 227)

    # Left bar
    x1l = 50
    cv2.rectangle(img, (x1l, H-25-lh), (x1l+bar_w, H-25), lcolor, -1)
    # Right bar
    x1r = W - 50 - bar_w
    cv2.rectangle(img, (x1r, H-25-rh), (x1r+bar_w, H-25), rcolor, -1)

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    grey = (142, 149, 158)
    white = (201, 209, 217)
    cv2.putText(img, 'Left', (x1l+5, H-8), font, 0.38, grey, 1)
    cv2.putText(img, 'Right', (x1r+2, H-8), font, 0.38, grey, 1)

    lt = f'{lm:.1f}m' if lm < 90 else '---'
    rt = f'{rm:.1f}m' if rm < 90 else '---'
    cv2.putText(img, lt, (x1l+5, H-30-lh), font, 0.38, white, 1)
    cv2.putText(img, rt, (x1r+5, H-30-rh), font, 0.38, white, 1)

    # Title
    title = f'Gap: {gs.upper()}  {gw}px'
    cv2.putText(img, title, (10, 16), font, 0.42, white, 1)

    # Scale marks
    for i, label in enumerate(['0m', '1m', '2m', '3m']):
        y = H - 25 - int((i / 3.0) * max_h)
        cv2.line(img, (40, y), (W-40, y), (33, 38, 48), 1)
        cv2.putText(img, label, (5, y+4), font, 0.3, grey, 1)

    _, buf = cv2.imencode('.png', img)
    return buf.tobytes()


class _LidarRenderer:
    """Reusable polar plot for LiDAR scans. Figure created once."""

    def __init__(self, max_r=4.0):
        self.max_r = max_r
        self._fig = plt.figure(figsize=(3.5, 3.5), facecolor='#0d1117')
        self._ax  = self._fig.add_subplot(111, projection='polar')
        self._setup_axes()

    def _setup_axes(self):
        ax = self._ax
        ax.set_facecolor('#0d1117')
        ax.set_ylim(0, self.max_r)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.tick_params(colors='#484f58', labelsize=6)
        ax.grid(color='#21262d', linewidth=0.5)

    def render(self, det) -> bytes:
        ax = self._ax
        ax.cla()
        self._setup_axes()

        max_r = self.max_r
        fha = np.radians(40)
        distances = det.get('all_distances', np.array([]))
        angles    = det.get('all_angles',    np.array([]))
        valid     = det.get('all_valid',     np.array([], dtype=bool))

        if len(distances) > 0 and valid.sum() > 0:
            d_v = distances[valid]; a_v = angles[valid]; clip = d_v < max_r
            il  = (a_v >= 0) & (a_v <= fha)
            ir  = a_v >= (2*np.pi - fha)
            ins = il | ir
            sm  = ins & (d_v <= 0.4)
            wm  = ins & (d_v > 0.4) & (d_v <= 1.5)
            sf  = ~sm & ~wm
            if (sf & clip).sum(): ax.scatter(a_v[sf&clip], d_v[sf&clip], s=3, c='#388bfd', alpha=0.7, linewidths=0)
            if (wm & clip).sum(): ax.scatter(a_v[wm&clip], d_v[wm&clip], s=8, c='#e3b341', alpha=0.9, linewidths=0)
            if (sm & clip).sum(): ax.scatter(a_v[sm&clip], d_v[sm&clip], s=12,c='#f85149', alpha=1.0, linewidths=0)

            # Also show rear points (dimmer)
            rear = ~ins & clip
            if rear.sum(): ax.scatter(a_v[rear], d_v[rear], s=2, c='#484f58', alpha=0.4, linewidths=0)

        ax.fill_between(np.linspace(-fha, fha, 60), 0, max_r, color='#e3b341', alpha=0.05)
        ring = np.linspace(0, 2*np.pi, 200)
        ax.plot(ring, [1.5]*200, '--', c='#e3b341', lw=0.7, alpha=0.5)
        ax.plot(ring, [0.4]*200, '-',  c='#f85149', lw=0.7, alpha=0.7)
        ax.scatter([0],[0], s=60, c='#3fb950', zorder=5, linewidths=0)
        for r in [1,2,3]:
            if r < max_r:
                ax.text(np.pi/6, r, f'{r}m', color='#484f58', fontsize=6, ha='center')
        plt.tight_layout(pad=0.1)

        buf = io.BytesIO()
        self._fig.savefig(buf, format='png', dpi=80, facecolor='#0d1117', bbox_inches='tight')
        buf.seek(0)
        return buf.read()


# ── Flask ─────────────────────────────────────────────────────────────────────

_app = Flask(__name__)

_HTML = """<!DOCTYPE html>
<html>
<head>
<title>QCar 2 — Live Observer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#0d1117;color:#c9d1d9;font-family:monospace;font-size:13px;padding:8px;}
.topbar{background:#161b22;border:1px solid #21262d;border-radius:6px;
        padding:7px 12px;margin-bottom:8px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:8px;}
.grid3b{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;}
.card{background:#161b22;border:1px solid #21262d;border-radius:6px;padding:8px;}
.title{color:#8b949e;font-size:10px;text-transform:uppercase;letter-spacing:.05em;margin-bottom:5px;}
.big{font-size:19px;font-weight:bold;margin:3px 0;}
.row{display:flex;justify-content:space-between;padding:2px 0;
     border-bottom:1px solid #21262d;font-size:12px;}
.row:last-child{border:none;}
.lbl{color:#8b949e;}.val{color:#c9d1d9;}
img{width:100%;border-radius:3px;display:block;}
.CLEAR,.NAVIGATE{color:#3fb950;}.WARN,.AVOID{color:#388bfd;}
.STOP,.EMERGENCY_STOP{color:#f85149;}.WAITING,.WAIT{color:#e3b341;}
.PERSON{color:#f85149;}.MOVING{color:#e3b341;}
.STATIC{color:#388bfd;}.NONE,.ARRIVED,.IDLE{color:#484f58;}
.REVERSING{color:#d2a8ff;}
.led-on{display:inline-block;width:11px;height:11px;border-radius:50%;
        background:#e3b341;margin:0 2px;vertical-align:middle;}
.led-off{display:inline-block;width:11px;height:11px;border-radius:50%;
         background:#21262d;border:1px solid #30363d;margin:0 2px;vertical-align:middle;}
</style>
</head>
<body>
<div class="topbar">
  <span style="color:#3fb950;font-size:15px;font-weight:bold;">QCar 2</span>
  <span>FSM: <b class="{{ sm_state }}">{{ sm_state }}</b></span>
  <span>ZONE: <b class="{{ zone }}">{{ zone }}</b></span>
  <span>TYPE: <b class="{{ obj_type }}">{{ obj_type }}</b></span>
  <span>BEH: <b class="{{ behaviour }}">{{ behaviour }}</b></span>
  <span>obs: <b>{{ dist }}</b></span>
  <span>goal: <b>{{ goal_dist }}</b></span>
  <span>spd: <b>{{ speed }}</b></span>
  <span>batt: <b>{{ battery }}V</b></span>
  <span>loop: <b>{{ loop_ms }}ms</b></span>
</div>
<div class="grid3">
  <div class="card">
    <div class="title">CSI front + YOLO</div>
    <img src="/camera" alt="camera">
    <div style="color:#484f58;font-size:11px;margin-top:3px;">
      {{ n_dets }} dets | {{ yolo_fps }}fps
    </div>
  </div>
  <div class="card">
    <div class="title">LiDAR (polar, up=fwd)</div>
    <img src="/lidar" alt="lidar">
    <div style="color:#484f58;font-size:11px;margin-top:3px;">
      fwd min={{ sector_min }}m | rear={{ rear_min }}m
    </div>
  </div>
  <div class="card">
    <div class="title">Depth clearance</div>
    <img src="/depth" alt="depth">
    <div style="color:#484f58;font-size:11px;margin-top:3px;">
      steer: <b>{{ avoid_side }}</b>
    </div>
  </div>
</div>
<div class="grid3b">
  <div class="card">
    <div class="title">Obstacle</div>
    <div class="big {{ obj_type }}">{{ obj_type }}</div>
    <div class="row"><span class="lbl">LiDAR fwd</span><span class="val {{ zone }}">{{ dist }}</span></div>
    <div class="row"><span class="lbl">LiDAR rear</span><span class="val">{{ rear_min }}m</span></div>
    <div class="row"><span class="lbl">Behaviour</span><span class="val {{ behaviour }}">{{ behaviour }}</span></div>
    <div class="row"><span class="lbl">Avoid</span><span class="val">{{ avoid_side }}</span></div>
    <div class="row"><span class="lbl">L/R clear</span><span class="val">{{ left_m }} / {{ right_m }}</span></div>
  </div>
  <div class="card">
    <div class="title">Pose + nav</div>
    <div class="row"><span class="lbl">x</span><span class="val">{{ px }}m</span></div>
    <div class="row"><span class="lbl">y</span><span class="val">{{ py }}m</span></div>
    <div class="row"><span class="lbl">th</span><span class="val">{{ pth }}rad ({{ pth_deg }}deg)</span></div>
    <div class="row"><span class="lbl">goal</span><span class="val">{{ goal_dist }}</span></div>
    <div class="row"><span class="lbl">h_err</span><span class="val">{{ h_err }}rad</span></div>
    <div class="row"><span class="lbl">FSM</span><span class="val {{ sm_state }}">{{ sm_state }}</span></div>
    <div class="row"><span class="lbl">progress</span>
      <span class="val">
        <div style="width:80px;height:7px;background:#21262d;border-radius:3px;display:inline-block;vertical-align:middle;">
          <div style="width:{{ progress }}%;height:7px;background:#3fb950;border-radius:3px;"></div>
        </div> {{ progress }}%
      </span>
    </div>
  </div>
  <div class="card">
    <div class="title">LEDs + perf</div>
    <div style="margin-bottom:8px;line-height:2;">
      <span class="lbl">Head</span>
      <span class="{{ 'led-on' if leds6 else 'led-off' }}"></span>
      <span class="{{ 'led-on' if leds7 else 'led-off' }}"></span>
      <span class="lbl">Brk</span>
      <span class="{{ 'led-on' if leds4 else 'led-off' }}"></span>
      <span class="lbl">L</span>
      <span class="{{ 'led-on' if leds0 else 'led-off' }}"></span>
      <span class="lbl">R</span>
      <span class="{{ 'led-on' if leds2 else 'led-off' }}"></span>
    </div>
    <div class="row"><span class="lbl">Loop</span><span class="val">{{ loop_ms }}ms</span></div>
    <div class="row"><span class="lbl">LiDAR</span><span class="val">{{ lidar_hz }}Hz</span></div>
    <div class="row"><span class="lbl">YOLO</span><span class="val">{{ yolo_fps }}fps</span></div>
    <div class="row"><span class="lbl">Thr</span><span class="val">{{ throttle }}</span></div>
    <div class="row"><span class="lbl">Str</span><span class="val">{{ steering }}rad</span></div>
    <div class="row"><span class="lbl">Batt</span><span class="val">{{ battery }}V</span></div>
  </div>
</div>
<script>setTimeout(()=>location.reload(), 500);</script>
</body></html>"""


@_app.route('/')
def index():
    det, pose, speed, dist_goal, leds, timing, sm_state = _s.get()
    GOAL_M = 2.0
    zone     = det.get('zone',           ZONE_CLEAR)
    behav    = det.get('behaviour',      BEHAVIOUR_NAVIGATE)
    obj_type = det.get('obstacle_type',  'NONE')
    dist_m   = det.get('distance_m',     99.0)
    avoid    = det.get('avoid_side',     'left')
    lm       = det.get('left_clear_m',   99.0)
    rm       = det.get('right_clear_m',  99.0)
    smin     = det.get('sector_min_m',   99.0)
    rmin     = det.get('rear_min_m',     99.0)
    gw       = det.get('gap_width_px',   0)
    yfps     = det.get('yolo_fps',       0.0)
    ndets    = det.get('n_yolo_dets',    0)
    batt     = det.get('battery_v',      0.0)
    throttle = timing.get('throttle',    0.0)
    steering = timing.get('steering',    0.0)
    h_err    = timing.get('heading_err', 0.0)
    prog     = max(0, min(100, int((1 - min(dist_goal, GOAL_M)/GOAL_M)*100)))

    return render_template_string(
        _HTML,
        sm_state  = sm_state,
        zone      = zone,
        behaviour = behav,
        obj_type  = obj_type,
        dist      = f"{dist_m:.2f}m" if dist_m<90 else "---",
        goal_dist = f"{dist_goal:.2f}m" if dist_goal<90 else "---",
        speed     = f"{speed:.2f}",
        battery   = f"{batt:.1f}",
        n_dets    = ndets,
        yolo_fps  = f"{yfps:.0f}",
        sector_min= f"{smin:.2f}",
        rear_min  = f"{rmin:.1f}" if rmin < 90 else "---",
        avoid_side= avoid.upper(),
        px=f"{pose[0]:+.3f}", py=f"{pose[1]:+.3f}",
        pth=f"{pose[2]:+.3f}", pth_deg=f"{np.degrees(pose[2]):+.1f}",
        h_err     = f"{h_err:+.3f}",
        progress  = prog,
        left_m    = f"{lm:.1f}m" if lm<90 else "---",
        right_m   = f"{rm:.1f}m" if rm<90 else "---",
        leds0=leds[0], leds2=leds[2], leds4=leds[4],
        leds6=leds[6], leds7=leds[7],
        loop_ms   = f"{timing.get('loop_ms',0):.0f}",
        lidar_hz  = f"{timing.get('lidar_hz',0):.1f}",
        throttle  = f"{throttle:+.3f}",
        steering  = f"{steering:+.3f}",
    )


@_app.route('/camera')
def camera():
    return Response(_s.get_cam(), mimetype='image/jpeg',
                    headers={'Cache-Control':'no-cache'})

@_app.route('/lidar')
def lidar():
    return Response(_s.get_lidar(), mimetype='image/png',
                    headers={'Cache-Control':'no-cache'})

@_app.route('/depth')
def depth():
    return Response(_s.get_depth(), mimetype='image/png',
                    headers={'Cache-Control':'no-cache'})


# ── Public API ────────────────────────────────────────────────────────────────

class Dashboard:
    """
    Dashboard.update() is now INSTANT (~0ms).
    All rendering runs in a background thread at max 5Hz.
    """

    def __init__(self, port: int = 5000):
        self.port          = port
        self._lidar_render = _LidarRenderer()
        self._running      = False
        self._render_thread = None

    def start(self):
        import socket, logging
        logging.getLogger('werkzeug').setLevel(logging.ERROR)

        # Start Flask
        t = threading.Thread(
            target=lambda: _app.run(host='0.0.0.0', port=self.port, threaded=True),
            daemon=True,
        )
        t.start()

        # Start render thread
        self._running = True
        self._render_thread = threading.Thread(
            target=self._render_loop, daemon=True, name='dash-render'
        )
        self._render_thread.start()

        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            ip = '<jetson-ip>'
        print(f"\n  Dashboard: http://{ip}:{self.port}")
        print("  Render thread: started (5Hz)")
        print("  Open in your PC browser.\n")

    def update(self, detection: dict, pose: np.ndarray,
               speed_mps: float, dist_goal: float,
               leds: list, timing: dict, sm_state: str,
               throttle: float = 0.0, steering: float = 0.0,
               heading_err: float = 0.0):
        """Push data to shared state — NO rendering, returns instantly."""
        timing = dict(timing)
        timing['throttle']    = throttle
        timing['steering']    = steering
        timing['heading_err'] = heading_err
        _s.push(detection, pose, speed_mps, dist_goal, leds, timing, sm_state)

    def _render_loop(self):
        """Background: renders images at max 5Hz. Never blocks main loop."""
        lidar_counter = 0
        while self._running:
            time.sleep(0.2)  # 5Hz max

            snap = _s.pop_for_render()
            if snap is None:
                continue

            det, frame, pose, sm_state = snap

            # Camera JPEG (~5ms)
            cam_j = _cam_jpeg(frame)

            # Depth chart via OpenCV (~1ms)
            depth_p = _depth_cv2(det)

            # LiDAR polar plot — only on new scans, max 2Hz (~100ms)
            lidar_p = None
            if det.get('new_lidar_scan', False):
                lidar_counter += 1
                if lidar_counter % 2 == 0:  # every other scan = ~2Hz
                    lidar_p = self._lidar_render.render(det)

            _s.set_images(cam_j, lidar_p, depth_p)

    def stop(self):
        self._running = False
