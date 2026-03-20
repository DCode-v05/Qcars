#!/usr/bin/env python3
"""
QCar 2 — VFH+ Navigation + Terminal LiDAR Heatmap
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Algorithm : VFH+ (Vector Field Histogram Plus)
            Ulrich & Borenstein, IEEE ICRA 1998
            Best algorithm for dense/cluttered small spaces.

LiDAR mode: rangingDistanceMode=0  (SHORT ~2m)
            Best for 10×10 ft room — faster scan, less noise at close range.

How VFH+ works here (3-stage data reduction):
  Stage 1 — Raw 384-pt scan → obstacle density per angular sector (polar histogram)
  Stage 2 — Threshold polar histogram → find candidate FREE valleys (gaps)
  Stage 3 — Cost function picks best valley considering:
               • alignment with current heading (continuity)
               • valley width (prefer wide gaps)
               • proximity to obstacles (safety margin)
  Output  — steer angle + adaptive throttle 0.0→MAX

Behaviours:
  DRIVE   — move forward through best gap
  REVERSE — triggered when forward hemisphere fully blocked
  STOP    — all directions blocked (only then)
  AUTO    — switching is automatic, no user input needed for motion

Keys (instant, no Enter):
  q  quit     d  force DRIVE     r  force REVERSE
  +  increase max throttle       -  decrease max throttle

Calibrate:
  python3 lidar_diag.py  →  paste FRONT_CENTER_DEG below

Run:
  python3 lidar_vfh_drive.py
"""

import sys, os, math, time, select, tty, termios
import numpy as np

# ── optional hardware ──────────────────────────────────────────
try:
    from pal.products.qcar import QCarLidar
    from quanser.hardware import HIL, EncoderQuadratureMode
    from quanser.hardware.exceptions import HILError
    HARDWARE = True
except ImportError:
    HARDWARE = False

# ══════════════════════════════════════════════════════════════
#  CALIBRATION
# ══════════════════════════════════════════════════════════════
FRONT_CENTER_DEG = 0.0
REAR_CENTER_DEG  = (FRONT_CENTER_DEG + 180.0) % 360.0

# ══════════════════════════════════════════════════════════════
#  LIDAR
#  MODE 0 = SHORT range (~2m) — best for 10x10 ft room
#  faster spin, lower noise, exactly what you need indoors
# ══════════════════════════════════════════════════════════════
RANGING_MODE     = 0            # SHORT — best for closed room
NUM_MEASUREMENTS = 384
INTERP_MODE      = 0
LOOP_HZ          = 10
LOOP_PERIOD      = 1.0 / LOOP_HZ

# ══════════════════════════════════════════════════════════════
#  VFH+ PARAMETERS
# ══════════════════════════════════════════════════════════════
# Polar histogram sectors (α = sector resolution in degrees)
SECTOR_ALPHA     = 5            # degrees per sector → 360/5 = 72 sectors
N_SECTORS        = 360 // SECTOR_ALPHA   # 72

# Obstacle magnitude weights
DIST_INFLUENCE   = 2.0          # max distance (m) where obstacles contribute
CERTAINTY_WEIGHT = 1.0          # weight for certainty grid contribution

# Polar histogram smoothing window
SMOOTH_WINDOW    = 3            # sectors each side to smooth over

# Threshold for binary free/blocked (h_threshold)
# Lower = more conservative (detects more things as obstacles)
H_THRESHOLD      = 0.6          # tune: 0.3 (cautious) to 1.0 (aggressive)
H_HYSTERESIS     = 0.1          # high=H_THRESHOLD+hyst, low=H_THRESHOLD-hyst

# Minimum valley (gap) width in sectors to be considered passable
MIN_VALLEY_WIDTH = 3            # sectors = 3×5° = 15° minimum gap

# Robot body radius (used to inflate obstacles)
ROBOT_RADIUS_M   = 0.20         # QCar ~0.4m wide, half = 0.2m
SAFETY_DIST_M    = 0.10         # extra safety margin

# VFH+ cost function weights  c1=heading c2=continuity c3=previous
COST_W1          = 5.0          # w1: steer angle vs current heading (continuity)
COST_W2          = 2.0          # w2: steer angle vs previous steer (smoothness)
COST_W3          = 2.0          # w3: steer angle vs forward direction

# Blocked detection (for DRIVE→REVERSE switch)
FRONT_BLOCKED_DEG   = 60        # degrees each side of front to check
FRONT_BLOCKED_FRAC  = 0.75      # fraction of those sectors that must be above threshold

# ══════════════════════════════════════════════════════════════
#  THROTTLE  (adaptive 0.0 → MAX)
# ══════════════════════════════════════════════════════════════
MAX_THROTTLE     = 0.20         # upper limit (adjustable with +/-)
MIN_THROTTLE     = 0.08         # minimum when moving (below this = stop)
THROTTLE_STEP    = 0.02         # step for +/- keys

# Throttle scales DOWN as:
#   a) nearest obstacle gets closer
#   b) required steer angle increases
NEAR_SLOW_DIST   = 0.8          # start slowing below this (m)
NEAR_STOP_DIST   = 0.25         # hard stop below this (m)
STEER_SLOW_DEG   = 20.0         # start slowing when steer > this (deg)

# ══════════════════════════════════════════════════════════════
#  STEERING
# ══════════════════════════════════════════════════════════════
MAX_STEER_RAD    = 0.40
STEER_SMOOTH     = 0.45         # IIR smoothing (0=instant 1=frozen)

# ══════════════════════════════════════════════════════════════
#  HARDWARE
# ══════════════════════════════════════════════════════════════
CARD_TYPE        = "qcar2"
CARD_ID          = "0"
THROTTLE_CH      = 11000
STEERING_CH      = 1000
ENCODER_CH       = 0
MOTOR_CH         = np.array([THROTTLE_CH, STEERING_CH], dtype=np.uint32)
ENC_CH           = np.array([ENCODER_CH],               dtype=np.uint32)
METRES_PER_COUNT = (2 * math.pi * 0.0328) / (720 * 4)

# ══════════════════════════════════════════════════════════════
#  TERMINAL CANVAS
# ══════════════════════════════════════════════════════════════
CANVAS_COLS = 74
CANVAS_ROWS = 38
PIX_COLS    = CANVAS_COLS
PIX_ROWS    = CANVAS_ROWS * 2
RADAR_CX    = PIX_COLS  // 2
RADAR_CY    = PIX_ROWS  // 2
RADAR_R     = min(RADAR_CX, RADAR_CY) - 2
PANEL_COL   = CANVAS_COLS + 2
PANEL_WIDTH = 46

# ── Colours ───────────────────────────────────────────────────
C_BG      = (  4,   8,  15)
C_PANEL   = (  8,  15,  26)
C_BORDER  = ( 15,  42,  74)
C_ACCENT  = (  0, 212, 255)
C_SAFE    = (  0, 220, 100)
C_MOD     = (255, 180,   0)
C_UNSAFE  = (220,  40,  40)
C_DIM     = ( 20,  45,  70)
C_WHITE   = (200, 230, 255)
C_GREY    = ( 80, 110, 140)
C_GRID    = ( 16,  34,  58)
C_GAP     = (  0, 255, 160)
C_HIST    = ( 80,  40, 140)
C_HIST_HI = (180,  80, 255)

ESC = "\033"
def fg(r,g,b):     return f"{ESC}[38;2;{r};{g};{b}m"
def bg(r,g,b):     return f"{ESC}[48;2;{r};{g};{b}m"
def rst():         return f"{ESC}[0m"
def bld():         return f"{ESC}[1m"
def dim_a():       return f"{ESC}[2m"
def mv(r,c):       return f"{ESC}[{r};{c}H"
def clr():         return f"{ESC}[2J"
def hcur():        return f"{ESC}[?25l"
def scur():        return f"{ESC}[?25h"


def dist_color(d, max_r=2.0):
    t = max(0.0, min(1.0, d / max_r))
    if   t < 0.25: r,g,b = 220, int(t/0.25*130), 20
    elif t < 0.50: r,g,b = 220, 130+int((t-0.25)/0.25*100), 0
    elif t < 0.75: r,g,b = int(220*(1-(t-0.50)/0.25)), 220, 0
    else:          r,g,b = 0, int(220-(t-0.75)/0.25*80), int((t-0.75)/0.25*200)
    return (r, g, b)


# ══════════════════════════════════════════════════════════════
#  VFH+ ALGORITHM
# ══════════════════════════════════════════════════════════════
class VFHPlus:
    """
    VFH+ implementation for 2D LiDAR navigation.
    Reference: Ulrich & Borenstein, IEEE ICRA 1998
    """

    def __init__(self):
        self.hist       = np.zeros(N_SECTORS)   # raw polar histogram
        self.hist_sm    = np.zeros(N_SECTORS)   # smoothed polar histogram
        self.binary     = np.zeros(N_SECTORS, dtype=bool)   # free sectors
        self.prev_steer = 0.0                   # previous steering angle (rad)
        self.h_high     = H_THRESHOLD + H_HYSTERESIS
        self.h_low      = H_THRESHOLD - H_HYSTERESIS

    def sector_of(self, angle_deg):
        """Map angle (0-360 deg) → sector index (0..N_SECTORS-1)."""
        return int(angle_deg / SECTOR_ALPHA) % N_SECTORS

    def build_histogram(self, distances, angles_rad):
        """
        Stage 1: Build polar obstacle density histogram.
        Each obstacle point contributes magnitude m = (a - b*d)^2
        where d = distance, a/b tuned so m→0 at DIST_INFLUENCE.
        """
        self.hist[:] = 0.0
        a_coeff = CERTAINTY_WEIGHT
        b_coeff = CERTAINTY_WEIGHT / DIST_INFLUENCE

        for i in range(len(distances)):
            d = float(distances[i])
            if d < 0.03 or d > DIST_INFLUENCE:
                continue
            # Obstacle magnitude (increases as d decreases)
            mag = (a_coeff - b_coeff * d) ** 2

            # Angular sector
            angle_deg = math.degrees(float(angles_rad[i])) % 360.0
            # Account for front calibration offset
            rel_deg   = (angle_deg - FRONT_CENTER_DEG) % 360.0
            sec       = int(rel_deg / SECTOR_ALPHA) % N_SECTORS

            # Spread contribution over neighbouring sectors (robot size inflation)
            spread_deg = math.degrees(math.asin(
                min(1.0, (ROBOT_RADIUS_M + SAFETY_DIST_M) / max(d, 0.01))
            ))
            spread_sec = max(1, int(spread_deg / SECTOR_ALPHA))

            for ds in range(-spread_sec, spread_sec + 1):
                self.hist[(sec + ds) % N_SECTORS] += mag

        # Stage 2: Smooth histogram
        self.hist_sm[:] = 0.0
        l = SMOOTH_WINDOW
        for k in range(N_SECTORS):
            total = 0.0
            for s in range(-l, l + 1):
                total += self.hist[(k + s) % N_SECTORS]
            self.hist_sm[k] = total / (2 * l + 1)

    def find_valleys(self):
        """
        Stage 2: Apply threshold with hysteresis to get binary free/blocked.
        Find all contiguous free valley sequences.
        Returns list of (start_sec, end_sec, center_sec, width_sec).
        """
        # Hysteresis thresholding
        for k in range(N_SECTORS):
            if self.hist_sm[k] >= self.h_high:
                self.binary[k] = False   # blocked
            elif self.hist_sm[k] <= self.h_low:
                self.binary[k] = True    # free
            # else: keep previous state (hysteresis)

        # Find contiguous free runs (wrap-around aware)
        valleys = []
        n = N_SECTORS

        # Double the array to handle wrap-around
        b2 = np.concatenate([self.binary, self.binary])
        i  = 0
        while i < n:
            if b2[i]:
                j = i
                while j < i + n and b2[j]:
                    j += 1
                width = j - i
                if width >= MIN_VALLEY_WIDTH:
                    start  = i % n
                    end    = (j - 1) % n
                    center = ((i + j - 1) // 2) % n
                    valleys.append((start, end, center, width))
                i = j
            else:
                i += 1

        return valleys

    def sector_to_angle_rad(self, sec):
        """Convert sector index → heading angle in radians (robot frame)."""
        deg = (sec * SECTOR_ALPHA + SECTOR_ALPHA / 2.0) % 360.0
        # Convert to signed heading: 0=forward, +left, -right
        if deg > 180:
            deg -= 360.0
        return math.radians(deg)

    def cost(self, candidate_sec, current_heading_sec, forward_sec=0):
        """
        VFH+ cost function:
        c = w1*Δ(candidate, forward)
          + w2*Δ(candidate, current_heading)
          + w3*Δ(candidate, prev_steer)
        Lower cost = better.
        """
        def sec_diff(a, b):
            d = abs(a - b)
            return min(d, N_SECTORS - d) * SECTOR_ALPHA   # in degrees

        c1 = COST_W3 * sec_diff(candidate_sec, forward_sec)
        c2 = COST_W1 * sec_diff(candidate_sec, current_heading_sec)
        c3 = COST_W2 * math.degrees(abs(
            math.atan2(math.sin(self.sector_to_angle_rad(candidate_sec)
                                - self.prev_steer),
                       math.cos(self.sector_to_angle_rad(candidate_sec)
                                - self.prev_steer))
        ))
        return c1 + c2 + c3

    def navigate(self, distances, angles_rad, mode):
        """
        Full VFH+ pipeline. Returns:
          steer_rad   — steering command (-MAX..+MAX)
          throttle    — 0.0 to MAX_THROTTLE (adaptive)
          best_valley — (start,end,center,width) or None
          all_valleys — list of all found valleys
          status      — string label
        """
        self.build_histogram(distances, angles_rad)
        valleys = self.find_valleys()

        # Global minimum distance (safety)
        valid_d = distances[distances > 0.03]
        min_d   = float(valid_d.min()) if len(valid_d) else 99.0

        # Hard stop if too close all around
        if min_d < NEAR_STOP_DIST:
            self.prev_steer = 0.0
            return 0.0, 0.0, None, valleys, f"ESTOP {min_d:.2f}m"

        if not valleys:
            # No free valley found — all blocked
            return 0.0, 0.0, None, valleys, "ALL BLOCKED"

        # Current heading sector (forward=0 for DRIVE, rear for REVERSE)
        if mode == "DRIVE":
            forward_sec     = 0
            current_heading = self.sector_of(0)
        else:
            forward_sec     = N_SECTORS // 2
            current_heading = forward_sec

        # Previous steer as sector
        prev_sec = self.sector_of(
            (math.degrees(self.prev_steer) % 360.0)
        )

        # Pick best valley by cost function
        best_cost   = float('inf')
        best_valley = None
        best_sec    = forward_sec

        for v in valleys:
            start, end, center, width = v

            # For narrow valleys: aim at center
            # For wide valleys: pick edge closest to forward direction
            if width <= 2 * MIN_VALLEY_WIDTH:
                candidate = center
            else:
                # Pick side of valley closest to forward
                d_left  = abs(start - forward_sec)
                d_right = abs(end   - forward_sec)
                if d_left < d_right:
                    candidate = (start + MIN_VALLEY_WIDTH) % N_SECTORS
                else:
                    candidate = (end   - MIN_VALLEY_WIDTH) % N_SECTORS

            c = self.cost(candidate, current_heading, forward_sec)
            if c < best_cost:
                best_cost   = c
                best_valley = v
                best_sec    = candidate

        # Convert best sector → steer angle
        steer_rad = self.sector_to_angle_rad(best_sec)
        if mode == "REVERSE":
            steer_rad = -steer_rad   # mirror for reverse

        # Clamp steering
        steer_rad = float(np.clip(steer_rad, -MAX_STEER_RAD, MAX_STEER_RAD))

        # Smooth steering (IIR filter)
        steer_rad = (STEER_SMOOTH * self.prev_steer
                     + (1.0 - STEER_SMOOTH) * steer_rad)
        self.prev_steer = steer_rad

        # Adaptive throttle
        # Scale 1: slow near obstacles
        near_scale = 1.0
        if min_d < NEAR_SLOW_DIST:
            near_scale = max(0.0, (min_d - NEAR_STOP_DIST)
                             / (NEAR_SLOW_DIST - NEAR_STOP_DIST))

        # Scale 2: slow on sharp turns
        steer_scale = 1.0
        steer_deg   = abs(math.degrees(steer_rad))
        if steer_deg > STEER_SLOW_DEG:
            steer_scale = max(0.2,
                1.0 - (steer_deg - STEER_SLOW_DEG) / (90.0 - STEER_SLOW_DEG))

        throttle_raw = MAX_THROTTLE * near_scale * steer_scale
        throttle     = float(np.clip(throttle_raw, 0.0, MAX_THROTTLE))
        if throttle < MIN_THROTTLE:
            throttle = 0.0

        if mode == "REVERSE":
            throttle = -throttle

        # Status string
        steer_col_str = f"steer={math.degrees(steer_rad):+.0f}°"
        status_str    = f"VALLEY w={best_valley[3]*SECTOR_ALPHA}° {steer_col_str}"

        return steer_rad, throttle, best_valley, valleys, status_str

    def is_hemisphere_blocked(self, center_deg):
        """
        Returns True if most sectors in the hemisphere around center_deg
        are blocked (above threshold).
        """
        center_sec = self.sector_of(center_deg)
        half       = FRONT_BLOCKED_DEG // SECTOR_ALPHA
        blocked    = 0
        total      = 0
        for ds in range(-half, half + 1):
            k = (center_sec + ds) % N_SECTORS
            total += 1
            if not self.binary[k]:
                blocked += 1
        if total == 0:
            return False
        return (blocked / total) >= FRONT_BLOCKED_FRAC


# ══════════════════════════════════════════════════════════════
#  PIXEL CANVAS + RENDERER
# ══════════════════════════════════════════════════════════════
def make_canvas():
    c = np.zeros((PIX_ROWS, PIX_COLS, 3), dtype=np.uint8)
    c[:,:] = C_BG
    return c

def sp(canvas, px, py, color):
    if 0 <= py < PIX_ROWS and 0 <= px < PIX_COLS:
        canvas[py, px] = color

def bline(canvas, x0, y0, x1, y1, col):
    dx,dy = abs(x1-x0), abs(y1-y0)
    sx = 1 if x0<x1 else -1
    sy = 1 if y0<y1 else -1
    err = dx-dy
    while True:
        sp(canvas,x0,y0,col)
        if x0==x1 and y0==y1: break
        e2=2*err
        if e2>-dy: err-=dy; x0+=sx
        if e2< dx: err+=dx; y0+=sy

def dot(canvas, px, py, col, r=2):
    for dy in range(-r,r+1):
        for dx in range(-r,r+1):
            if dx*dx+dy*dy<=r*r:
                sp(canvas, px+dx, py+dy, col)

def circ(canvas, cx, cy, r, col):
    x,y,d = r,0,1-r
    while x>=y:
        for sx,sy in [(cx+x,cy+y),(cx-x,cy+y),(cx+x,cy-y),(cx-x,cy-y),
                      (cx+y,cy+x),(cx-y,cy+x),(cx+y,cy-x),(cx-y,cy-x)]:
            sp(canvas,sx,sy,col)
        y+=1
        if d<0: d+=2*y+1
        else: x-=1; d+=2*(y-x)+1

def polar_to_pixel(dist_m, angle_rad, max_r=2.0):
    r_px  = min(dist_m / max_r, 1.0) * RADAR_R
    front = math.radians(FRONT_CENTER_DEG)
    sa    = (angle_rad - front) - math.pi/2
    return RADAR_CX+int(r_px*math.cos(sa)), RADAR_CY+int(r_px*math.sin(sa))

def sector_to_pixel(sec, frac=1.0):
    """Sector index → pixel on radar perimeter (or inner ring if frac<1)."""
    deg   = (sec * SECTOR_ALPHA + SECTOR_ALPHA/2) % 360.0
    front = math.radians(FRONT_CENTER_DEG)
    rad   = math.radians(deg) + front - math.pi/2
    r     = int(RADAR_R * frac)
    return RADAR_CX+int(r*math.cos(rad)), RADAR_CY+int(r*math.sin(rad))

def draw_grid(canvas):
    for rm in [0.5, 1.0, 1.5, 2.0]:
        rp = int(rm/2.0*RADAR_R)
        circ(canvas, RADAR_CX, RADAR_CY, rp, C_GRID)
    circ(canvas, RADAR_CX, RADAR_CY, RADAR_R, C_BORDER)
    front = math.radians(FRONT_CENTER_DEG)
    for deg in range(0,360,45):
        a  = math.radians(deg)+front-math.pi/2
        ex = RADAR_CX+int(RADAR_R*math.cos(a))
        ey = RADAR_CY+int(RADAR_R*math.sin(a))
        bline(canvas, RADAR_CX, RADAR_CY, ex, ey, C_GRID)

def draw_scan(canvas, distances, angles_rad):
    for i in range(len(distances)):
        d = float(distances[i])
        if d < 0.03: continue
        col    = dist_color(d, max_r=2.0)
        px,py  = polar_to_pixel(d, float(angles_rad[i]), max_r=2.0)
        r      = 3 if d < 0.4 else 2 if d < 0.8 else 1
        dot(canvas, px, py, col, r)

def draw_histogram_ring(canvas, hist_sm):
    """Draw normalised polar histogram as a ring of bars just inside radar edge."""
    max_h = max(hist_sm.max(), 1e-6)
    for k in range(N_SECTORS):
        h_frac = hist_sm[k] / max_h
        inner_r = int(RADAR_R * 0.88)
        outer_r = int(RADAR_R * (0.88 + 0.10 * h_frac))
        col = C_HIST_HI if hist_sm[k] > H_THRESHOLD else C_HIST
        deg   = (k * SECTOR_ALPHA + SECTOR_ALPHA/2) % 360.0
        front = math.radians(FRONT_CENTER_DEG)
        a     = math.radians(deg) + front - math.pi/2
        for r in range(inner_r, outer_r+1):
            px = RADAR_CX + int(r * math.cos(a))
            py = RADAR_CY + int(r * math.sin(a))
            sp(canvas, px, py, col)

def draw_free_valleys(canvas, valleys, binary):
    """Shade free sectors with faint green."""
    for k in range(N_SECTORS):
        if binary[k]:
            px,py = sector_to_pixel(k, frac=0.78)
            dot(canvas, px, py, (0,60,30), 1)
    for v in valleys:
        start, end, center, width = v
        # Draw arc for each valley
        k = start
        while True:
            px,py = sector_to_pixel(k, frac=0.72)
            dot(canvas, px, py, C_GAP, 1)
            if k == end: break
            k = (k+1) % N_SECTORS

def draw_best_valley(canvas, valley):
    if valley is None: return
    start, end, center, width = valley
    # bright arc
    k = start
    while True:
        px,py = sector_to_pixel(k, frac=0.68)
        dot(canvas, px, py, C_ACCENT, 2)
        if k == end: break
        k = (k+1) % N_SECTORS
    # center line
    px,py = sector_to_pixel(center, frac=0.65)
    bline(canvas, RADAR_CX, RADAR_CY, px, py, C_ACCENT)

def draw_steer_arrow(canvas, steer_rad, throttle, mode):
    col   = C_SAFE if mode == "DRIVE" else C_MOD
    front = math.radians(FRONT_CENTER_DEG)
    sign  = 1 if mode == "DRIVE" else -1
    length = max(8, int(abs(throttle)/MAX_THROTTLE * RADAR_R * 0.4))
    angle  = front - math.pi/2 + steer_rad*2.0
    ex = RADAR_CX + int(length * sign * math.cos(angle))
    ey = RADAR_CY + int(length * sign * math.sin(angle))
    bline(canvas, RADAR_CX, RADAR_CY, ex, ey, col)
    for da in (-0.5, 0.5):
        hx = ex+int(8*math.cos(angle+math.pi+da))
        hy = ey+int(8*math.sin(angle+math.pi+da))
        bline(canvas, ex, ey, hx, hy, col)

def draw_car(canvas, mode):
    col = C_SAFE if mode == "DRIVE" else C_MOD
    dot(canvas, RADAR_CX, RADAR_CY, col, 5)
    dot(canvas, RADAR_CX, RADAR_CY, C_BG, 3)
    dot(canvas, RADAR_CX, RADAR_CY, col, 1)

def canvas_to_term(canvas):
    out = []
    for row in range(0, PIX_ROWS-1, 2):
        out.append(mv(row//2+2, 1))
        pt = pb = None
        for col in range(PIX_COLS):
            top = tuple(canvas[row,   col])
            bot = tuple(canvas[row+1, col])
            s = ""
            if top != pt: s += fg(*top); pt = top
            if bot != pb: s += bg(*bot); pb = bot
            s += "▀"
            out.append(s)
        out.append(rst())
    return "".join(out)


# ══════════════════════════════════════════════════════════════
#  RIGHT PANEL
# ══════════════════════════════════════════════════════════════
def bar(frac, w=22, c_on=C_SAFE, c_off=C_BORDER):
    frac = max(0.0, min(1.0, frac))
    f    = int(frac*w)
    return fg(*c_on)+"█"*f + fg(*c_off)+"░"*(w-f)+rst()

def panel(mode, status, steer_rad, throttle,
          min_d, n_valleys, best_v, scan_n, fps,
          auto_sw, enc_m, max_thr, hist_sm, binary):

    sc  = C_SAFE if mode=="DRIVE" else C_MOD
    sym = "▲ DRIVE  " if mode=="DRIVE" else "▼ REVERSE"
    sd  = math.degrees(steer_rad)
    sc2 = C_MOD if abs(sd)>20 else C_WHITE
    dc  = (C_UNSAFE if min_d<=0.25 else C_MOD if min_d<=0.8 else C_SAFE)
    tc  = C_SAFE if throttle>0 else C_MOD if throttle<0 else C_GREY

    # Mini histogram bar (72 sectors compressed to 22 chars)
    block_w = N_SECTORS // 22
    hist_bar = ""
    max_h    = max(hist_sm.max(), 1e-6)
    for i in range(22):
        chunk  = hist_sm[i*block_w:(i+1)*block_w]
        val    = chunk.mean() / max_h
        free   = binary[i*block_w:(i+1)*block_w].all()
        col    = C_GAP if free else C_HIST_HI if val>0.5 else C_HIST
        shade  = " ░▒▓█"[min(4,int(val*5))]
        hist_bar += fg(*col) + shade
    hist_bar += rst()

    bv_str = f"w={best_v[3]*SECTOR_ALPHA}° c={best_v[2]*SECTOR_ALPHA}°" if best_v else "none"
    th_sign = "▲" if throttle>0 else "▼" if throttle<0 else "■"

    def row(label, val, col=C_WHITE):
        pad = 13-len(label)
        return fg(*C_GREY)+label+" "*pad+fg(*col)+str(val)+rst()

    lines = [
        bld()+fg(*C_ACCENT)+"QCar 2 · VFH+ NAVIGATOR"+rst(),
        fg(*C_GREY)+"LiDAR mode=SHORT (rangingDistanceMode=0)"+rst(),
        fg(*C_BORDER)+"─"*PANEL_WIDTH+rst(),

        fg(*C_GREY)+" MODE"+rst(),
        "  "+bld()+fg(*sc)+sym+rst(),
        "",
        fg(*C_GREY)+" STATUS"+rst(),
        "  "+fg(*C_SAFE if "VALLEY" in status else C_MOD)+bld()+status[:PANEL_WIDTH-2]+rst(),
        "",
        fg(*C_GREY)+" VFH+ HISTOGRAM  (free="+fg(*C_GAP)+"░"+rst()+fg(*C_GREY)+" blocked="+fg(*C_HIST_HI)+"█"+fg(*C_GREY)+")"+rst(),
        "  "+hist_bar,
        row(" valleys found", str(n_valleys), C_ACCENT),
        row(" best valley",   bv_str,         C_ACCENT),
        "",
        fg(*C_BORDER)+"─"*PANEL_WIDTH+rst(),
        fg(*C_GREY)+" MOTION"+rst(),
        row(" steer",   f"{sd:+.1f}° ({steer_rad:+.3f}rad)", sc2),
        "  "+bar((steer_rad+MAX_STEER_RAD)/(2*MAX_STEER_RAD), c_on=sc2),
        row(" throttle", f"{th_sign} {abs(throttle):.3f} / {max_thr:.2f}", tc),
        "  "+bar(abs(throttle)/max(max_thr,0.01), c_on=tc),
        "",
        fg(*C_BORDER)+"─"*PANEL_WIDTH+rst(),
        fg(*C_GREY)+" ENVIRONMENT"+rst(),
        row(" nearest",  f"{min_d:.3f} m", dc),
        "  "+bar(1.0-min(min_d/2.0,1.0), w=22, c_on=dc),
        row(" odometry", f"{enc_m:+.3f} m"),
        "",
        fg(*C_BORDER)+"─"*PANEL_WIDTH+rst(),
        fg(*C_GREY)+" SESSION"+rst(),
        row(" scan #",   str(scan_n)),
        row(" fps",      f"{fps:.1f}"),
        row(" switches", str(auto_sw), C_MOD),
        row(" max thr",  f"{max_thr:.2f}  (+/- to adjust)"),
        "",
        fg(*C_BORDER)+"─"*PANEL_WIDTH+rst(),
        fg(*C_GREY)+" HEATMAP  near→far"+rst(),
        "  "+"".join(fg(*dist_color(i*2.0/26,max_r=2.0))+"█" for i in range(26))+rst(),
        "  "+fg(*C_UNSAFE)+"▲near "+rst()+fg(*C_MOD)+"▲med "+rst()+fg(*C_SAFE)+"▲far"+rst(),
        "",
        fg(*C_BORDER)+"─"*PANEL_WIDTH+rst(),
        fg(*C_GREY)+" d=DRIVE  r=REVERSE  +/-=throttle  q=quit"+rst(),
        fg(*C_GREY)+" "+("HARDWARE" if HARDWARE else "DEMO MODE")+rst(),
    ]
    return lines


# ══════════════════════════════════════════════════════════════
#  DEMO DATA
# ══════════════════════════════════════════════════════════════
_dt = 0.0
def demo_scan():
    global _dt
    angles = np.linspace(0, 2*math.pi, NUM_MEASUREMENTS, endpoint=False)
    # Simulated 10×10 ft room walls at ~1.5m + random obstacles
    base = 1.5 + 0.3*np.sin(4*angles) + 0.2*np.cos(7*angles+1)
    # Moving obstacle in front
    front = math.radians(FRONT_CENTER_DEG)
    wall  = 0.35 + 0.4*abs(math.sin(_dt*0.3))
    blob  = wall * np.exp(-15*(angles-front)**2)
    # Side obstacle
    side  = math.radians(FRONT_CENTER_DEG+90)
    blob2 = 0.4 * np.exp(-20*(angles-side)**2)
    dists = np.clip(base-blob-blob2, 0.10, 2.0)
    dists += np.random.normal(0, 0.01, NUM_MEASUREMENTS)
    _dt  += LOOP_PERIOD
    return dists, angles


# ══════════════════════════════════════════════════════════════
#  KEYBOARD  (raw, non-blocking)
# ══════════════════════════════════════════════════════════════
def setup_term():
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setraw(fd)
    return old

def restore_term(old):
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)

def readkey():
    if select.select([sys.stdin],[],[],0)[0]:
        return sys.stdin.read(1)
    return None


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    card=None; lidar=None
    motor_buf = np.array([0.0,0.0], dtype=np.float64)
    enc_buf   = np.zeros(1, dtype=np.int32)
    enc_m     = 0.0

    if HARDWARE:
        print("[init] Opening HIL board ...")
        try:
            card = HIL(CARD_TYPE, CARD_ID)
            em   = np.array([EncoderQuadratureMode.X4], dtype=np.int32)
            card.set_encoder_quadrature_mode(ENC_CH, 1, em)
            card.set_encoder_counts(ENC_CH, 1, np.zeros(1, dtype=np.int32))
            card.write_other(MOTOR_CH, 2, motor_buf)
            print("[init] HIL OK")
        except HILError as e:
            print(f"[init] HIL FAIL: {e.get_error_message()}")
            card = None

        print(f"[init] LiDAR SHORT mode (rangingDistanceMode={RANGING_MODE}) ...")
        lidar = QCarLidar(
            numMeasurements=NUM_MEASUREMENTS,
            rangingDistanceMode=RANGING_MODE,
            interpolationMode=INTERP_MODE,
        )
        print("[init] Waiting 2s spin-up ...")
        time.sleep(2.0)
        print("[init] Ready.")
        time.sleep(0.5)
    else:
        print("DEMO MODE — no hardware. Starting ...")
        time.sleep(1.0)

    old_term = setup_term()
    sys.stdout.write(clr())
    sys.stdout.flush()

    canvas    = make_canvas()
    vfh       = VFHPlus()

    mode         = "DRIVE"
    steer_rad    = 0.0
    throttle     = 0.0
    scan_n       = 0
    fps_s        = 0.0
    t_last       = time.time()
    auto_sw      = 0
    max_thr      = MAX_THROTTLE
    distances    = np.ones(NUM_MEASUREMENTS) * 1.5
    angles_rad   = np.linspace(0, 2*math.pi, NUM_MEASUREMENTS, endpoint=False)
    status       = "INIT"
    best_valley  = None
    all_valleys  = []

    try:
        while True:
            t0 = time.time()

            # ── key ───────────────────────────────────────────
            k = readkey()
            if k:
                if k in ('q','Q','\x03'): break
                elif k in ('d','D'): mode="DRIVE"
                elif k in ('r','R'): mode="REVERSE"
                elif k in ('+','='): max_thr=min(1.0, round(max_thr+THROTTLE_STEP,3))
                elif k in ('-','_'): max_thr=max(0.0, round(max_thr-THROTTLE_STEP,3))

            # ── scan ──────────────────────────────────────────
            if HARDWARE and lidar:
                lidar.read()
                distances  = lidar.distances.flatten().copy()
                angles_rad = lidar.angles.flatten().copy()
            else:
                distances, angles_rad = demo_scan()

            # ── VFH+ ──────────────────────────────────────────
            steer_rad, throttle, best_valley, all_valleys, status = \
                vfh.navigate(distances, angles_rad, mode)

            # ── Auto mode switch ──────────────────────────────
            if throttle == 0.0 and "ALL BLOCKED" in status:
                if mode == "DRIVE":
                    if not vfh.is_hemisphere_blocked(REAR_CENTER_DEG):
                        mode = "REVERSE"
                        vfh.prev_steer = 0.0
                        auto_sw += 1
                        status = "AUTO→REVERSE"
                    else:
                        status = "BOTH BLOCKED — STOPPED"
                else:
                    if not vfh.is_hemisphere_blocked(FRONT_CENTER_DEG):
                        mode = "DRIVE"
                        vfh.prev_steer = 0.0
                        auto_sw += 1
                        status = "AUTO→DRIVE"
                    else:
                        status = "BOTH BLOCKED — STOPPED"

            # Update MAX_THROTTLE from user adjustment
            vfh_max = min(abs(throttle), max_thr) if throttle != 0 else 0.0
            throttle = math.copysign(vfh_max, throttle) if throttle != 0 else 0.0

            # ── Motor write ───────────────────────────────────
            if HARDWARE and card:
                motor_buf[0] = throttle
                motor_buf[1] = steer_rad
                try:
                    card.write_other(MOTOR_CH, 2, motor_buf)
                    card.read_encoder(ENC_CH, 1, enc_buf)
                    enc_m = int(enc_buf[0]) * METRES_PER_COUNT
                except HILError:
                    pass

            # ── FPS ───────────────────────────────────────────
            now   = time.time()
            fps_s = fps_s*0.85 + (1.0/max(now-t_last,1e-4))*0.15
            t_last= now
            scan_n+= 1

            # ── Render ────────────────────────────────────────
            canvas[:] = C_BG
            draw_grid(canvas)
            draw_histogram_ring(canvas, vfh.hist_sm)
            draw_free_valleys(canvas, all_valleys, vfh.binary)
            draw_scan(canvas, distances, angles_rad)
            draw_best_valley(canvas, best_valley)
            draw_steer_arrow(canvas, steer_rad, throttle, mode)
            draw_car(canvas, mode)

            valid_d = distances[distances>0.03]
            min_d   = float(valid_d.min()) if len(valid_d) else 99.0

            plines = panel(
                mode, status, steer_rad, throttle,
                min_d, len(all_valleys), best_valley,
                scan_n, fps_s, auto_sw, enc_m, max_thr,
                vfh.hist_sm, vfh.binary
            )

            buf = [hcur(), canvas_to_term(canvas)]

            # Header
            mc = C_SAFE if mode=="DRIVE" else C_MOD
            buf.append(mv(1,1))
            buf.append(
                bg(*C_PANEL)+fg(*C_ACCENT)+bld()+
                "  QCar2 VFH+ Navigator "+rst()+
                bg(*C_PANEL)+fg(*C_GREY)+
                f" SHORT-mode LiDAR | sectors={N_SECTORS} α={SECTOR_ALPHA}° | "+
                f"threshold={H_THRESHOLD} "+
                fg(*mc)+bld()+("▲DRIVE" if mode=="DRIVE" else "▼REVERSE")+
                rst()+" "*8
            )

            # Right panel
            for i, line in enumerate(plines):
                buf.append(mv(i+2, PANEL_COL))
                buf.append(line)
                buf.append(fg(*C_BG)+"  "+rst())

            # Bottom bar
            bot = CANVAS_ROWS+3
            buf.append(mv(bot,1))
            dc = C_UNSAFE if min_d<0.25 else C_MOD if min_d<0.8 else C_SAFE
            buf.append(
                bg(*C_PANEL)+fg(*C_GREY)+
                f"  nearest={fg(*dc)}{min_d:.2f}m{rst()}{bg(*C_PANEL)}{fg(*C_GREY)}"
                f"  steer={fg(*C_WHITE)}{math.degrees(steer_rad):+.0f}°{rst()}{bg(*C_PANEL)}{fg(*C_GREY)}"
                f"  thr={fg(*C_SAFE if throttle>0 else C_MOD if throttle<0 else C_GREY)}{throttle:+.3f}{rst()}{bg(*C_PANEL)}{fg(*C_GREY)}"
                f"  valleys={len(all_valleys)}"
                f"  scan={scan_n}  fps={fps_s:.1f}"+
                " "*10+rst()
            )

            sys.stdout.write("".join(buf))
            sys.stdout.flush()

            # ── pace ──────────────────────────────────────────
            sleep_t = LOOP_PERIOD - (time.time()-t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except Exception as e:
        restore_term(old_term)
        sys.stdout.write(scur()+"\n")
        print(f"Error: {e}")
        import traceback; traceback.print_exc()

    finally:
        restore_term(old_term)
        sys.stdout.write(scur())
        sys.stdout.write(mv(CANVAS_ROWS+6,1))
        sys.stdout.flush()
        print("\n[shutdown] ...")
        if HARDWARE:
            if card:
                try:
                    card.write_other(MOTOR_CH,2,np.array([0.0,0.0],dtype=np.float64))
                    print("[shutdown] Motor stopped")
                    card.close()
                except HILError as e:
                    print(f"[shutdown] {e.get_error_message()}")
            if lidar:
                try:
                    lidar.terminate()
                    print("[shutdown] LiDAR terminated")
                except: pass
        print("[shutdown] Done.")

if __name__ == "__main__":
    main()