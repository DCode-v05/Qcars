"""
dashboard.py  —  Live web dashboard for QCar 2
Flask server at http://<jetson-ip>:5000

FIX: NO full page reloads. HTML loads once, JS fetches /api/data (JSON)
and swaps image src with cache-busting. ~1 small JSON + 1 image per cycle.
"""
import io
import threading
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, Response, jsonify

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
        self._raw_frame  = None
        self._cam_jpeg   = b''
        self._lidar_png  = b''
        self._depth_png  = b''
        self._dirty      = False

    def push(self, detection, pose, speed_mps, dist_goal, leds, timing, sm_state):
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

    def get_json(self):
        with self._lock:
            det   = dict(self.detection)
            pose  = self.pose.copy()
            speed = self.speed_mps
            dg    = self.dist_goal
            leds  = list(self.leds)
            tim   = dict(self.timing)
            sm    = self.sm_state
        GOAL_M = 15.0
        dist_m = det.get('distance_m', 99.0)
        rmin   = det.get('rear_min_m', 99.0)
        smin   = det.get('sector_min_m', 99.0)
        prog   = max(0, min(100, int((1 - min(dg, GOAL_M)/GOAL_M)*100)))
        return {
            'sm': sm,
            'zone': det.get('zone', ZONE_CLEAR),
            'beh': det.get('behaviour', BEHAVIOUR_NAVIGATE),
            'otype': det.get('obstacle_type', 'NONE'),
            'dist': f"{dist_m:.2f}m" if dist_m < 90 else "---",
            'gdist': f"{dg:.2f}m" if dg < 90 else "---",
            'spd': f"{speed:.2f}",
            'bat': f"{det.get('battery_v',0):.1f}",
            'nd': det.get('n_yolo_dets', 0),
            'yfps': f"{det.get('yolo_fps',0):.0f}",
            'smin': f"{smin:.2f}",
            'rmin': f"{rmin:.1f}" if rmin < 90 else "---",
            'aside': det.get('avoid_side','left').upper(),
            'lm': f"{det.get('left_clear_m',99):.1f}m" if det.get('left_clear_m',99)<90 else "---",
            'rm': f"{det.get('right_clear_m',99):.1f}m" if det.get('right_clear_m',99)<90 else "---",
            'px': f"{pose[0]:+.3f}", 'py': f"{pose[1]:+.3f}",
            'pth': f"{pose[2]:+.3f}", 'pthd': f"{np.degrees(pose[2]):+.1f}",
            'herr': f"{tim.get('heading_err',0):+.3f}",
            'prog': prog,
            'leds': leds,
            'lms': f"{tim.get('loop_ms',0):.0f}",
            'lhz': f"{tim.get('lidar_hz',0):.1f}",
            'thr': f"{tim.get('throttle',0):+.3f}",
            'str': f"{tim.get('steering',0):+.3f}",
        }

    def get_cam(self):
        with self._lock: return self._cam_jpeg
    def get_lidar(self):
        with self._lock: return self._lidar_png
    def get_depth(self):
        with self._lock: return self._depth_png


_s = _State()


# ── Fast image generators ────────────────────────────────────────────────────

def _cam_jpeg(frame):
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, 'No frame', (220, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (60,60,60), 2)
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()


def _depth_cv2(det):
    W, H = 280, 180
    img = np.full((H, W, 3), (22, 27, 13), dtype=np.uint8)
    lm = det.get('left_clear_m', 99.0)
    rm = det.get('right_clear_m', 99.0)
    gs = det.get('gap_side', 'left')
    gw = det.get('gap_width_px', 0)
    max_d, bar_w, max_h = 3.0, 55, H - 50
    lh = int((min(lm, max_d) / max_d) * max_h)
    rh = int((min(rm, max_d) / max_d) * max_h)
    lcolor = (80, 185, 63) if gs == 'left' else (253, 139, 56)
    rcolor = (80, 185, 63) if gs == 'right' else (65, 179, 227)
    x1l, x1r = 50, W - 50 - bar_w
    cv2.rectangle(img, (x1l, H-25-lh), (x1l+bar_w, H-25), lcolor, -1)
    cv2.rectangle(img, (x1r, H-25-rh), (x1r+bar_w, H-25), rcolor, -1)
    font, grey, white = cv2.FONT_HERSHEY_SIMPLEX, (142,149,158), (201,209,217)
    cv2.putText(img, 'Left', (x1l+5, H-8), font, 0.38, grey, 1)
    cv2.putText(img, 'Right', (x1r+2, H-8), font, 0.38, grey, 1)
    cv2.putText(img, f'{lm:.1f}m' if lm<90 else '---', (x1l+5, H-30-lh), font, 0.38, white, 1)
    cv2.putText(img, f'{rm:.1f}m' if rm<90 else '---', (x1r+5, H-30-rh), font, 0.38, white, 1)
    cv2.putText(img, f'Gap: {gs.upper()}  {gw}px', (10, 16), font, 0.42, white, 1)
    for i, label in enumerate(['0m','1m','2m','3m']):
        y = H - 25 - int((i / 3.0) * max_h)
        cv2.line(img, (40, y), (W-40, y), (33, 38, 48), 1)
        cv2.putText(img, label, (5, y+4), font, 0.3, grey, 1)
    _, buf = cv2.imencode('.png', img)
    return buf.tobytes()


class _LidarRenderer:
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
            il = (a_v >= 0) & (a_v <= fha)
            ir = a_v >= (2*np.pi - fha)
            ins = il | ir
            sm = ins & (d_v <= 0.4)
            wm = ins & (d_v > 0.4) & (d_v <= 1.5)
            sf = ~sm & ~wm
            if (sf & clip).sum(): ax.scatter(a_v[sf&clip], d_v[sf&clip], s=3, c='#388bfd', alpha=0.7, linewidths=0)
            if (wm & clip).sum(): ax.scatter(a_v[wm&clip], d_v[wm&clip], s=8, c='#e3b341', alpha=0.9, linewidths=0)
            if (sm & clip).sum(): ax.scatter(a_v[sm&clip], d_v[sm&clip], s=12,c='#f85149', alpha=1.0, linewidths=0)
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

# HTML loaded ONCE by the browser — never reloaded.
# All data updates via fetch('/api/data') + DOM manipulation.
_HTML = """<!DOCTYPE html><html><head>
<title>QCar 2</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#c9d1d9;font-family:monospace;font-size:13px;padding:8px}
.tb{background:#161b22;border:1px solid #21262d;border-radius:6px;padding:7px 12px;margin-bottom:8px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:8px}
.g3b{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}
.c{background:#161b22;border:1px solid #21262d;border-radius:6px;padding:8px}
.t{color:#8b949e;font-size:10px;text-transform:uppercase;letter-spacing:.05em;margin-bottom:5px}
.big{font-size:19px;font-weight:bold;margin:3px 0}
.r{display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid #21262d;font-size:12px}
.r:last-child{border:none}
.l{color:#8b949e}.v{color:#c9d1d9}
img{width:100%;border-radius:3px;display:block}
.lon{display:inline-block;width:11px;height:11px;border-radius:50%;background:#e3b341;margin:0 2px;vertical-align:middle}
.lof{display:inline-block;width:11px;height:11px;border-radius:50%;background:#21262d;border:1px solid #30363d;margin:0 2px;vertical-align:middle}
.bb{width:80px;height:7px;background:#21262d;border-radius:3px;display:inline-block;vertical-align:middle}
.bf{height:7px;background:#3fb950;border-radius:3px}
</style></head><body>

<div class="tb">
  <span style="color:#3fb950;font-size:15px;font-weight:bold">QCar 2</span>
  <span>FSM: <b id="d_sm">---</b></span>
  <span>ZONE: <b id="d_zone">---</b></span>
  <span>TYPE: <b id="d_otype">---</b></span>
  <span>BEH: <b id="d_beh">---</b></span>
  <span>obs: <b id="d_dist">---</b></span>
  <span>goal: <b id="d_gdist">---</b></span>
  <span>spd: <b id="d_spd">---</b></span>
  <span>batt: <b id="d_bat">---</b>V</span>
  <span>loop: <b id="d_lms">---</b>ms</span>
</div>

<div class="g3">
  <div class="c"><div class="t">CSI front + YOLO</div>
    <img id="ic" src="/camera" alt="cam">
    <div style="color:#484f58;font-size:11px;margin-top:3px"><span id="d_nd">0</span> dets | <span id="d_yfps">0</span>fps</div>
  </div>
  <div class="c"><div class="t">LiDAR (polar)</div>
    <img id="il" src="/lidar" alt="lidar">
    <div style="color:#484f58;font-size:11px;margin-top:3px">fwd=<span id="d_smin">---</span>m rear=<span id="d_rmin">---</span>m</div>
  </div>
  <div class="c"><div class="t">Depth clearance</div>
    <img id="id" src="/depth" alt="depth">
    <div style="color:#484f58;font-size:11px;margin-top:3px">steer: <b id="d_aside">---</b></div>
  </div>
</div>

<div class="g3b">
  <div class="c"><div class="t">Obstacle</div>
    <div class="big" id="d_ot2">NONE</div>
    <div class="r"><span class="l">LiDAR fwd</span><span class="v" id="d_d2">---</span></div>
    <div class="r"><span class="l">LiDAR rear</span><span class="v" id="d_r2">---</span></div>
    <div class="r"><span class="l">Behaviour</span><span class="v" id="d_b2">---</span></div>
    <div class="r"><span class="l">Avoid</span><span class="v" id="d_a2">---</span></div>
    <div class="r"><span class="l">L/R clear</span><span class="v" id="d_lr">---</span></div>
  </div>
  <div class="c"><div class="t">Pose + nav</div>
    <div class="r"><span class="l">x</span><span class="v" id="d_px">---</span></div>
    <div class="r"><span class="l">y</span><span class="v" id="d_py">---</span></div>
    <div class="r"><span class="l">th</span><span class="v" id="d_pth">---</span></div>
    <div class="r"><span class="l">goal</span><span class="v" id="d_g2">---</span></div>
    <div class="r"><span class="l">h_err</span><span class="v" id="d_he">---</span></div>
    <div class="r"><span class="l">FSM</span><span class="v" id="d_f2">---</span></div>
    <div class="r"><span class="l">progress</span><span class="v"><div class="bb"><div class="bf" id="d_bar" style="width:0%"></div></div> <span id="d_pr">0</span>%</span></div>
  </div>
  <div class="c"><div class="t">LEDs + perf</div>
    <div style="margin-bottom:8px;line-height:2">
      <span class="l">Head</span><span id="l6" class="lof"></span><span id="l7" class="lof"></span>
      <span class="l">Brk</span><span id="l4" class="lof"></span>
      <span class="l">L</span><span id="l0" class="lof"></span>
      <span class="l">R</span><span id="l2" class="lof"></span>
    </div>
    <div class="r"><span class="l">Loop</span><span class="v" id="d_lm2">---</span></div>
    <div class="r"><span class="l">LiDAR</span><span class="v" id="d_lhz">---</span></div>
    <div class="r"><span class="l">YOLO</span><span class="v" id="d_yf2">---</span></div>
    <div class="r"><span class="l">Thr</span><span class="v" id="d_thr">---</span></div>
    <div class="r"><span class="l">Str</span><span class="v" id="d_str">---</span></div>
    <div class="r"><span class="l">Batt</span><span class="v" id="d_bt2">---</span></div>
  </div>
</div>

<script>
var CM={CLEAR:'#3fb950',NAVIGATE:'#3fb950',NAVIGATING:'#3fb950',
WARN:'#388bfd',AVOID:'#388bfd',AVOIDING:'#388bfd',
STOP:'#f85149',EMERGENCY_STOP:'#f85149',STOPPED:'#f85149',
WAITING:'#e3b341',WAIT:'#e3b341',
PERSON:'#f85149',MOVING:'#e3b341',STATIC:'#388bfd',
NONE:'#484f58',ARRIVED:'#484f58',IDLE:'#484f58',REVERSING:'#d2a8ff'};

function S(id,v,col){var e=document.getElementById(id);if(!e)return;e.textContent=v;if(col)e.style.color=CM[v]||'#c9d1d9'}
function L(id,on){var e=document.getElementById(id);if(e)e.className=on?'lon':'lof'}

var tk=0;
function poll(){
  fetch('/api/data').then(function(r){return r.json()}).then(function(d){
    S('d_sm',d.sm,1);S('d_zone',d.zone,1);S('d_otype',d.otype,1);S('d_beh',d.beh,1);
    S('d_dist',d.dist);S('d_gdist',d.gdist);S('d_spd',d.spd);S('d_bat',d.bat);S('d_lms',d.lms);
    S('d_nd',d.nd);S('d_yfps',d.yfps);S('d_smin',d.smin);S('d_rmin',d.rmin);S('d_aside',d.aside);
    var o2=document.getElementById('d_ot2');o2.textContent=d.otype;o2.style.color=CM[d.otype]||'#c9d1d9';
    S('d_d2',d.dist);document.getElementById('d_d2').style.color=CM[d.zone]||'#c9d1d9';
    S('d_r2',d.rmin+'m');S('d_b2',d.beh);document.getElementById('d_b2').style.color=CM[d.beh]||'#c9d1d9';
    S('d_a2',d.aside);S('d_lr',d.lm+' / '+d.rm);
    S('d_px',d.px+'m');S('d_py',d.py+'m');S('d_pth',d.pth+'rad ('+d.pthd+'\u00b0)');
    S('d_g2',d.gdist);S('d_he',d.herr+'rad');
    S('d_f2',d.sm);document.getElementById('d_f2').style.color=CM[d.sm]||'#c9d1d9';
    document.getElementById('d_bar').style.width=d.prog+'%';S('d_pr',d.prog);
    L('l0',d.leds[0]);L('l2',d.leds[2]);L('l4',d.leds[4]);L('l6',d.leds[6]);L('l7',d.leds[7]);
    S('d_lm2',d.lms+'ms');S('d_lhz',d.lhz+'Hz');S('d_yf2',d.yfps+'fps');
    S('d_thr',d.thr);S('d_str',d.str+'rad');S('d_bt2',d.bat+'V');
    // Stagger images: only refresh 1 per cycle
    var t=Date.now();tk++;
    if(tk%3===0)document.getElementById('ic').src='/camera?'+t;
    if(tk%3===1)document.getElementById('il').src='/lidar?'+t;
    if(tk%3===2)document.getElementById('id').src='/depth?'+t;
  }).catch(function(){});
}
setInterval(poll,333);
</script>
</body></html>"""


@_app.route('/')
def index():
    return _HTML

@_app.route('/api/data')
def api_data():
    return jsonify(_s.get_json())

@_app.route('/camera')
def camera():
    return Response(_s.get_cam(), mimetype='image/jpeg',
                    headers={'Cache-Control': 'no-store'})

@_app.route('/lidar')
def lidar():
    return Response(_s.get_lidar(), mimetype='image/png',
                    headers={'Cache-Control': 'no-store'})

@_app.route('/depth')
def depth():
    return Response(_s.get_depth(), mimetype='image/png',
                    headers={'Cache-Control': 'no-store'})


# ── Public API ────────────────────────────────────────────────────────────────

class Dashboard:

    def __init__(self, port: int = 5000):
        self.port          = port
        self._lidar_render = _LidarRenderer()
        self._running      = False
        self._render_thread = None

    def start(self):
        import socket, logging
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        t = threading.Thread(
            target=lambda: _app.run(host='0.0.0.0', port=self.port, threaded=True),
            daemon=True,
        )
        t.start()
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
        print("  Open in your PC browser.\n")

    def update(self, detection: dict, pose: np.ndarray,
               speed_mps: float, dist_goal: float,
               leds: list, timing: dict, sm_state: str,
               throttle: float = 0.0, steering: float = 0.0,
               heading_err: float = 0.0):
        timing = dict(timing)
        timing['throttle']    = throttle
        timing['steering']    = steering
        timing['heading_err'] = heading_err
        _s.push(detection, pose, speed_mps, dist_goal, leds, timing, sm_state)

    def _render_loop(self):
        lidar_counter = 0
        while self._running:
            time.sleep(0.2)
            snap = _s.pop_for_render()
            if snap is None:
                continue
            det, frame, pose, sm_state = snap
            cam_j   = _cam_jpeg(frame)
            depth_p = _depth_cv2(det)
            lidar_p = None
            if det.get('new_lidar_scan', False):
                lidar_counter += 1
                if lidar_counter % 2 == 0:
                    lidar_p = self._lidar_render.render(det)
            _s.set_images(cam_j, lidar_p, depth_p)

    def stop(self):
        self._running = False
