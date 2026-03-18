#!/usr/bin/env python3
"""
QCar 2 — LiDAR Heatmap + Autonomous Steering + Auto-Reverse
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • Reads 384-point LiDAR scan every loop
  • Renders full 360° polar heatmap in a pygame window
    (red = near / yellow = moderate / green = far)
  • Finds the widest free gap in the forward hemisphere
  • Steers toward that gap centre
  • If forward hemisphere is fully blocked → auto-switches to REVERSE
  • If rear hemisphere is blocked in reverse → stops and waits

Install deps (once on Jetson):
    pip3 install pygame numpy

Calibrate first:
    python3 lidar_diag.py  →  note FRONT_CENTER_DEG  →  paste below

Run:
    python3 lidar_heatmap_drive.py
    Press Q or Esc to quit cleanly.
"""

import sys
import math
import time
import numpy as np

# ── optional hardware imports ──────────────────────────────────
try:
    from pal.products.qcar import QCarLidar
    from quanser.hardware import HIL, EncoderQuadratureMode
    from quanser.hardware.exceptions import HILError
    HARDWARE = True
except ImportError:
    HARDWARE = False

import pygame
from pygame import gfxdraw

# ══════════════════════════════════════════════════════════════
#  CALIBRATION  ← paste your value from lidar_diag.py here
# ══════════════════════════════════════════════════════════════
FRONT_CENTER_DEG = 180.0          # angle (deg) that points to QCar nose
REAR_CENTER_DEG  = (FRONT_CENTER_DEG + 180.0) % 360.0

# ══════════════════════════════════════════════════════════════
#  SAFETY & MOTION CONFIG
# ══════════════════════════════════════════════════════════════
UNSAFE_M         = 0.50         # <= this  →  UNSAFE / motor cut
MODERATE_M       = 1.00         # <= this  →  MODERATE warning
MAX_RANGE_M      = 6.0          # display + logic clamp

DRIVE_THROTTLE   =  0.12
REVERSE_THROTTLE = -0.12
MAX_STEER_RAD    =  0.40        # physical steering limit (rad)
STEER_GAIN       =  0.6         # how aggressively to chase gap centre

# Forward / rear scan arcs for blocked detection
FORWARD_ARC_DEG  = 60.0         # ±60° cone in front
BLOCKED_FRAC     = 0.80         # fraction of arc readings unsafe → "blocked"

# Gap search arc (wider — where to look for best gap)
GAP_SEARCH_DEG   = 150.0        # search ±150° in direction of travel

# ══════════════════════════════════════════════════════════════
#  HARDWARE CONFIG  (from rot.py)
# ══════════════════════════════════════════════════════════════
CARD_TYPE        = "qcar2"
CARD_ID          = "0"
THROTTLE_CH      = 11000
STEERING_CH      = 1000
ENCODER_CH       = 0
MOTOR_CH         = np.array([THROTTLE_CH, STEERING_CH], dtype=np.uint32)
ENC_CH           = np.array([ENCODER_CH],               dtype=np.uint32)

NUM_MEASUREMENTS = 384
RANGING_MODE     = 2
INTERP_MODE      = 0

LOOP_HZ          = 10
LOOP_PERIOD      = 1.0 / LOOP_HZ

# ══════════════════════════════════════════════════════════════
#  PYGAME WINDOW
# ══════════════════════════════════════════════════════════════
WIN_W, WIN_H     = 1100, 700
RADAR_CX         = 370           # polar plot centre X
RADAR_CY         = 340           # polar plot centre Y
RADAR_R          = 300           # polar plot radius (pixels)

# Colour palette
C_BG             = ( 4,  8, 15)
C_PANEL          = ( 8, 15, 26)
C_BORDER         = (15, 42, 74)
C_ACCENT         = ( 0,212,255)
C_SAFE           = ( 0,220,100)
C_MOD            = (255,180,  0)
C_UNSAFE         = (220, 40, 40)
C_DIM            = (20, 50, 80)
C_WHITE          = (200,230,255)
C_GREY           = ( 80,110,140)

# ── distance → RGB heatmap colour ─────────────────────────────
def dist_color(d: float) -> tuple:
    """Blue(near) → Cyan → Green → Yellow → Red(far) — inverted for danger."""
    t = max(0.0, min(1.0, d / MAX_RANGE_M))
    if t < 0.25:
        # Red → Orange  (very close = danger)
        r = 220
        g = int(t / 0.25 * 140)
        b = 0
    elif t < 0.50:
        # Orange → Yellow
        r = 220
        g = 140 + int((t - 0.25) / 0.25 * 115)
        b = 0
    elif t < 0.75:
        # Yellow → Green
        r = int(220 - (t - 0.50) / 0.25 * 220)
        g = 220
        b = 0
    else:
        # Green → Cyan/Blue  (far = safe)
        r = 0
        g = int(220 - (t - 0.75) / 0.25 * 80)
        b = int((t - 0.75) / 0.25 * 180)
    return (r, g, b)


# ══════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════
def angle_diff(a, b):
    """Signed angular difference a-b wrapped to [-π, π]."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def polar_to_screen(dist_m, angle_rad, cx, cy, radius, max_range):
    """Convert (distance, angle) → pixel (x, y) on radar plot."""
    r_px = min(dist_m / max_range, 1.0) * radius
    # angle_rad 0 = front = up on screen
    front_rad = math.radians(FRONT_CENTER_DEG)
    screen_angle = (angle_rad - front_rad) - math.pi / 2
    x = cx + r_px * math.cos(screen_angle)
    y = cy + r_px * math.sin(screen_angle)
    return int(x), int(y)


# ══════════════════════════════════════════════════════════════
#  STEERING LOGIC — find widest free gap
# ══════════════════════════════════════════════════════════════
def find_best_gap(distances, angles_rad, travel_center_deg, search_half_deg):
    """
    Within ±search_half_deg of travel_center:
      - find continuous angular gap where dist > MODERATE_M
      - return (gap_center_rad, gap_width_deg, steer_error_rad)
    If no gap found → return None.
    """
    center_rad  = math.radians(travel_center_deg)
    search_half = math.radians(search_half_deg)

    # Filter to search arc, sorted by angle
    mask = np.array([
        abs(angle_diff(float(a), center_rad)) <= search_half
        for a in angles_rad
    ])
    if mask.sum() == 0:
        return None

    arc_angles = angles_rad[mask]
    arc_dists  = distances[mask]

    # Sort by angle
    order      = np.argsort(arc_angles)
    arc_angles = arc_angles[order]
    arc_dists  = arc_dists[order]

    # Label each reading as free (> MODERATE_M) or blocked
    free = arc_dists > MODERATE_M

    # Find all contiguous free runs
    best_width  = 0.0
    best_center = None
    run_start   = None

    for i in range(len(free)):
        if free[i] and run_start is None:
            run_start = i
        if (not free[i] or i == len(free) - 1) and run_start is not None:
            run_end   = i if not free[i] else i + 1
            width_rad = float(arc_angles[run_end - 1] - arc_angles[run_start])
            if width_rad > best_width:
                best_width  = width_rad
                best_center = float(
                    (arc_angles[run_start] + arc_angles[run_end - 1]) / 2.0
                )
            run_start = None

    if best_center is None:
        return None

    steer_err = angle_diff(best_center, center_rad)   # signed rad
    return best_center, math.degrees(best_width), steer_err


def arc_blocked(distances, angles_rad, center_deg, half_deg, threshold_m, frac):
    """
    Returns True if >= frac of arc readings are <= threshold_m.
    """
    center_rad = math.radians(center_deg)
    half_rad   = math.radians(half_deg)
    in_arc     = np.array([
        abs(angle_diff(float(a), center_rad)) <= half_rad
        for a in angles_rad
    ])
    valid = (distances > 0.05) & in_arc
    if valid.sum() == 0:
        return False
    blocked = distances[valid] <= threshold_m
    return (blocked.sum() / valid.sum()) >= frac


# ══════════════════════════════════════════════════════════════
#  DEMO DATA  (used when no hardware)
# ══════════════════════════════════════════════════════════════
_demo_t = 0.0

def demo_scan():
    global _demo_t
    angles = np.linspace(0, 2 * np.pi, NUM_MEASUREMENTS, endpoint=False)
    base   = 3.5 + 1.5 * np.sin(2 * angles) + 0.8 * np.cos(3 * angles + 1)
    # moving wall in front
    front  = math.radians(FRONT_CENTER_DEG)
    wall   = 0.45 + 0.3 * abs(math.sin(_demo_t * 0.4))
    blob   = wall * np.exp(-10 * (angles - front) ** 2)
    dists  = np.clip(base - blob, 0.12, MAX_RANGE_M)
    dists += np.random.normal(0, 0.02, NUM_MEASUREMENTS)
    _demo_t += LOOP_PERIOD
    return dists, angles


# ══════════════════════════════════════════════════════════════
#  PYGAME DRAW HELPERS
# ══════════════════════════════════════════════════════════════
def draw_grid(surf, cx, cy, r):
    rings = [1.0, 2.0, 3.0, MAX_RANGE_M]
    for rm in rings:
        rp = int(rm / MAX_RANGE_M * r)
        pygame.draw.circle(surf, C_BORDER, (cx, cy), rp, 1)
        # label
        lbl = font_sm.render(f"{rm:.0f}m", True, C_DIM)
        surf.blit(lbl, (cx + rp + 3, cy - 8))

    # spokes every 30°
    front_rad = math.radians(FRONT_CENTER_DEG)
    for deg in range(0, 360, 30):
        a = math.radians(deg) + front_rad - math.pi / 2   # screen coords
        ex = cx + r * math.cos(a)
        ey = cy + r * math.sin(a)
        pygame.draw.line(surf, C_BORDER, (cx, cy), (int(ex), int(ey)), 1)

    # Outer ring
    pygame.draw.circle(surf, C_ACCENT, (cx, cy), r, 2)

    # Compass labels  N=front S=rear E=right W=left
    compass = [("FWD", 0), ("R", 90), ("REV", 180), ("L", 270)]
    for label, deg in compass:
        a = math.radians(deg) - math.pi / 2
        lx = cx + int((r + 18) * math.cos(a))
        ly = cy + int((r + 18) * math.sin(a))
        t  = font_sm.render(label, True, C_ACCENT if deg == 0 else C_GREY)
        surf.blit(t, (lx - t.get_width() // 2, ly - t.get_height() // 2))


def draw_radar(surf, distances, angles_rad):
    """Draw all 384 LiDAR points as coloured dots on polar plot."""
    for i in range(len(distances)):
        d = distances[i]
        if d < 0.05:
            continue
        col  = dist_color(d)
        sx, sy = polar_to_screen(d, float(angles_rad[i]),
                                 RADAR_CX, RADAR_CY, RADAR_R, MAX_RANGE_M)
        # glow
        gfxdraw.filled_circle(surf, sx, sy, 4,  (*col, 40))
        gfxdraw.filled_circle(surf, sx, sy, 2,  (*col, 180))
        gfxdraw.aacircle(surf,      sx, sy, 2,  col)


def draw_gap_arc(surf, gap_center_rad, gap_width_deg, travel_center_deg):
    """Draw a cyan arc showing the detected free gap."""
    half  = math.radians(gap_width_deg / 2)
    front = math.radians(FRONT_CENTER_DEG)
    # Draw arc as series of line segments
    steps = 30
    pts   = []
    for i in range(steps + 1):
        a   = (gap_center_rad - half) + i * (2 * half / steps)
        sca = a - front - math.pi / 2
        r   = RADAR_R * 0.85
        pts.append((
            RADAR_CX + int(r * math.cos(sca)),
            RADAR_CY + int(r * math.sin(sca))
        ))
    if len(pts) > 1:
        pygame.draw.lines(surf, C_ACCENT, False, pts, 3)


def draw_steering_arrow(surf, steer_rad, mode):
    """Arrow from car centre showing steering direction."""
    col   = C_SAFE if mode == "DRIVE" else C_MOD
    front = math.radians(FRONT_CENTER_DEG)
    sign  = 1 if mode == "DRIVE" else -1
    angle = front - math.pi / 2 + steer_rad * 2.0   # exaggerate for visibility
    length = 60
    ex = RADAR_CX + int(length * sign * math.cos(angle))
    ey = RADAR_CY + int(length * sign * math.sin(angle))
    pygame.draw.line(surf, col, (RADAR_CX, RADAR_CY), (ex, ey), 3)
    # arrowhead
    for da in (-0.4, 0.4):
        hx = ex + int(12 * math.cos(angle + math.pi + da))
        hy = ey + int(12 * math.sin(angle + math.pi + da))
        pygame.draw.line(surf, col, (ex, ey), (hx, hy), 2)


def draw_car(surf, mode):
    """Small car icon at radar centre."""
    col = C_SAFE if mode == "DRIVE" else C_MOD
    pygame.draw.circle(surf, col, (RADAR_CX, RADAR_CY), 8, 0)
    pygame.draw.circle(surf, C_BG,  (RADAR_CX, RADAR_CY), 4, 0)


def stat_block(surf, x, y, label, value, col=None):
    col = col or C_WHITE
    lbl = font_sm.render(label, True, C_GREY)
    val = font_lg.render(str(value), True, col)
    surf.blit(lbl, (x, y))
    surf.blit(val, (x, y + 16))
    return y + 50


def hbar(surf, x, y, w, h, frac, col):
    pygame.draw.rect(surf, C_BORDER, (x, y, w, h), 0, 3)
    fill = max(2, int(frac * w))
    pygame.draw.rect(surf, col,      (x, y, fill, h), 0, 3)


def color_scale_bar(surf, x, y, w, h):
    for i in range(w):
        t   = i / w
        col = dist_color(t * MAX_RANGE_M)
        pygame.draw.line(surf, col, (x + i, y), (x + i, y + h))
    pygame.draw.rect(surf, C_BORDER, (x, y, w, h), 1)
    lbl_near = font_sm.render("near", True, C_GREY)
    lbl_far  = font_sm.render("far",  True, C_GREY)
    surf.blit(lbl_near, (x,         y + h + 3))
    surf.blit(lbl_far,  (x + w - 24, y + h + 3))


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    global font_sm, font_lg, font_xl, font_hud

    # ── pygame init ───────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("QCar 2 · LiDAR Heatmap + Steering")
    clock  = pygame.time.Clock()

    font_sm  = pygame.font.SysFont("monospace", 11)
    font_lg  = pygame.font.SysFont("monospace", 18, bold=True)
    font_xl  = pygame.font.SysFont("monospace", 28, bold=True)
    font_hud = pygame.font.SysFont("monospace", 13)

    # ── hardware init ─────────────────────────────────────────
    card  = None
    lidar = None
    enc_buf   = np.zeros(1, dtype=np.int32)
    motor_buf = np.array([0.0, 0.0], dtype=np.float64)

    if HARDWARE:
        print("[init] Opening HIL board ...")
        try:
            card = HIL(CARD_TYPE, CARD_ID)
            enc_mode = np.array([EncoderQuadratureMode.X4], dtype=np.int32)
            card.set_encoder_quadrature_mode(ENC_CH, 1, enc_mode)
            card.set_encoder_counts(ENC_CH, 1, np.zeros(1, dtype=np.int32))
            card.write_other(MOTOR_CH, 2, motor_buf)
            print("[init] HIL OK")
        except HILError as e:
            print(f"[init] HIL FAILED: {e.get_error_message()}")
            card = None

        print("[init] Starting LiDAR (2s spin-up) ...")
        lidar = QCarLidar(
            numMeasurements=NUM_MEASUREMENTS,
            rangingDistanceMode=RANGING_MODE,
            interpolationMode=INTERP_MODE,
        )
        time.sleep(2.0)
        print("[init] LiDAR ready")
    else:
        print("[init] DEMO mode — no hardware detected")

    # ── state ─────────────────────────────────────────────────
    mode        = "DRIVE"       # "DRIVE" or "REVERSE"
    steer_rad   = 0.0
    throttle    = 0.0
    scan_count  = 0
    fps_smooth  = 0.0
    t_last      = time.time()
    distances   = np.zeros(NUM_MEASUREMENTS)
    angles_rad  = np.linspace(0, 2 * np.pi, NUM_MEASUREMENTS, endpoint=False)
    min_front   = None
    min_rear    = None
    gap_result  = None
    safety_str  = "INIT"
    safety_col  = C_GREY
    motor_cut   = False
    auto_switch_count = 0

    running = True
    while running:

        # ── events ────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                if ev.key == pygame.K_d:
                    mode = "DRIVE"
                if ev.key == pygame.K_r:
                    mode = "REVERSE"

        # ── read LiDAR ────────────────────────────────────────
        t0 = time.time()
        if HARDWARE and lidar:
            lidar.read()
            distances = lidar.distances.flatten().copy()
            angles_rad = lidar.angles.flatten().copy()
        else:
            distances, angles_rad = demo_scan()

        # ── safety checks ─────────────────────────────────────
        front_blocked = arc_blocked(
            distances, angles_rad,
            FRONT_CENTER_DEG, FORWARD_ARC_DEG,
            UNSAFE_M, BLOCKED_FRAC
        )
        rear_blocked = arc_blocked(
            distances, angles_rad,
            REAR_CENTER_DEG, FORWARD_ARC_DEG,
            UNSAFE_M, BLOCKED_FRAC
        )

        # min distance in each arc
        def arc_min(center_deg, half_deg):
            center_rad = math.radians(center_deg)
            half_rad   = math.radians(half_deg)
            in_arc = np.array([
                abs(angle_diff(float(a), center_rad)) <= half_rad
                for a in angles_rad
            ])
            valid = (distances > 0.05) & in_arc
            return float(distances[valid].min()) if valid.sum() > 0 else None

        min_front = arc_min(FRONT_CENTER_DEG, FORWARD_ARC_DEG)
        min_rear  = arc_min(REAR_CENTER_DEG,  FORWARD_ARC_DEG)

        # ── auto mode switch ──────────────────────────────────
        if mode == "DRIVE" and front_blocked:
            if not rear_blocked:
                mode = "REVERSE"
                auto_switch_count += 1
            else:
                # both blocked — stop
                throttle  = 0.0
                steer_rad = 0.0
                motor_cut = True
                safety_str = "BOTH BLOCKED — STOPPED"
                safety_col  = C_UNSAFE
                gap_result  = None

        elif mode == "REVERSE" and rear_blocked:
            if not front_blocked:
                mode = "DRIVE"
                auto_switch_count += 1
            else:
                throttle  = 0.0
                steer_rad = 0.0
                motor_cut = True
                safety_str = "BOTH BLOCKED — STOPPED"
                safety_col  = C_UNSAFE
                gap_result  = None

        # ── gap finding & steering ────────────────────────────
        travel_center = FRONT_CENTER_DEG if mode == "DRIVE" else REAR_CENTER_DEG
        gap_result = find_best_gap(
            distances, angles_rad,
            travel_center, GAP_SEARCH_DEG
        )

        if not motor_cut:
            if gap_result:
                gap_center, gap_width, steer_err = gap_result
                steer_rad = float(np.clip(steer_err * STEER_GAIN,
                                          -MAX_STEER_RAD, MAX_STEER_RAD))
                throttle  = DRIVE_THROTTLE if mode == "DRIVE" else REVERSE_THROTTLE

                # current arc min distance
                cur_min = min_front if mode == "DRIVE" else min_rear
                if cur_min is None or cur_min > MODERATE_M:
                    safety_str = f"SAFE  {cur_min:.2f}m" if cur_min else "SAFE"
                    safety_col  = C_SAFE
                elif cur_min > UNSAFE_M:
                    safety_str = f"MODERATE  {cur_min:.2f}m"
                    safety_col  = C_MOD
                else:
                    safety_str = f"UNSAFE  {cur_min:.2f}m"
                    safety_col  = C_UNSAFE
                    throttle    = 0.0
                    motor_cut   = True
            else:
                # no gap at all → slow stop
                throttle  = 0.0
                steer_rad = 0.0
                safety_str = "NO GAP FOUND"
                safety_col  = C_MOD

        # ── write motor & steering ────────────────────────────
        if HARDWARE and card:
            motor_buf[0] = throttle
            motor_buf[1] = steer_rad
            try:
                card.write_other(MOTOR_CH, 2, motor_buf)
                card.read_encoder(ENC_CH, 1, enc_buf)
            except HILError:
                pass

        # reset motor_cut for next loop
        motor_cut = False

        # ── FPS ───────────────────────────────────────────────
        now = time.time()
        dt  = now - t_last
        t_last = now
        fps_smooth = fps_smooth * 0.85 + (1.0 / max(dt, 1e-4)) * 0.15
        scan_count += 1

        # ══════════════════════════════════════════════════════
        #  DRAW
        # ══════════════════════════════════════════════════════
        screen.fill(C_BG)

        # ── left panel background ─────────────────────────────
        pygame.draw.rect(screen, C_PANEL, (0, 0, RADAR_CX * 2, WIN_H))
        pygame.draw.rect(screen, C_BORDER,(RADAR_CX * 2, 0, 1, WIN_H))

        # ── radar grid ────────────────────────────────────────
        draw_grid(screen, RADAR_CX, RADAR_CY, RADAR_R)

        # ── scan points ───────────────────────────────────────
        draw_radar(screen, distances, angles_rad)

        # ── gap arc ───────────────────────────────────────────
        if gap_result:
            draw_gap_arc(screen, gap_result[0], gap_result[1], travel_center)

        # ── steering arrow ────────────────────────────────────
        draw_steering_arrow(screen, steer_rad, mode)

        # ── car icon ──────────────────────────────────────────
        draw_car(screen, mode)

        # ── right panel ───────────────────────────────────────
        PX = RADAR_CX * 2 + 20
        PY = 20

        # Title
        title = font_xl.render("QCar 2  LIDAR", True, C_ACCENT)
        screen.blit(title, (PX, PY)); PY += 36
        sub = font_hud.render("Heatmap + Steering + Auto-Reverse", True, C_GREY)
        screen.blit(sub, (PX, PY)); PY += 26
        pygame.draw.line(screen, C_BORDER, (PX, PY), (WIN_W - 20, PY)); PY += 12

        # Mode
        mode_col = C_SAFE if mode == "DRIVE" else C_MOD
        PY = stat_block(screen, PX, PY, "MODE",
                        f"{'▲ DRIVE' if mode == 'DRIVE' else '▼ REVERSE'}",
                        mode_col); PY -= 6

        # Safety
        PY = stat_block(screen, PX, PY, "SAFETY STATE", safety_str, safety_col)

        pygame.draw.line(screen, C_BORDER, (PX, PY), (WIN_W - 20, PY)); PY += 10

        # Steering
        steer_deg = math.degrees(steer_rad)
        steer_col = C_MOD if abs(steer_deg) > 15 else C_WHITE
        PY = stat_block(screen, PX, PY, "STEER CMD",
                        f"{steer_deg:+.1f} deg  ({steer_rad:+.3f} rad)",
                        steer_col)

        # Steering bar
        hbar(screen, PX, PY, 340, 10,
             (steer_rad + MAX_STEER_RAD) / (2 * MAX_STEER_RAD), steer_col)
        PY += 18

        # Throttle
        thr_col = C_SAFE if throttle > 0 else C_MOD if throttle < 0 else C_GREY
        PY = stat_block(screen, PX, PY, "THROTTLE",
                        f"{throttle:+.3f}", thr_col)
        hbar(screen, PX, PY, 340, 10,
             abs(throttle) / 0.20, thr_col)
        PY += 18

        pygame.draw.line(screen, C_BORDER, (PX, PY), (WIN_W - 20, PY)); PY += 10

        # Front / rear min distance
        fd_col = (C_UNSAFE if min_front and min_front <= UNSAFE_M
                  else C_MOD if min_front and min_front <= MODERATE_M
                  else C_SAFE)
        rd_col = (C_UNSAFE if min_rear and min_rear <= UNSAFE_M
                  else C_MOD if min_rear and min_rear <= MODERATE_M
                  else C_SAFE)

        PY = stat_block(screen, PX, PY, "FRONT MIN DIST",
                        f"{min_front:.3f} m" if min_front else "no data", fd_col)
        PY = stat_block(screen, PX, PY, "REAR MIN DIST",
                        f"{min_rear:.3f} m"  if min_rear  else "no data", rd_col)

        # Gap info
        if gap_result:
            PY = stat_block(screen, PX, PY, "BEST GAP WIDTH",
                            f"{gap_result[1]:.1f} deg", C_ACCENT)
        else:
            PY = stat_block(screen, PX, PY, "BEST GAP WIDTH", "none", C_GREY)

        pygame.draw.line(screen, C_BORDER, (PX, PY), (WIN_W - 20, PY)); PY += 10

        # Scan stats
        PY = stat_block(screen, PX, PY, "SCAN #",
                        f"{scan_count}", C_WHITE)
        PY = stat_block(screen, PX, PY, "FPS",
                        f"{fps_smooth:.1f}", C_WHITE)
        PY = stat_block(screen, PX, PY, "AUTO-SWITCHES",
                        f"{auto_switch_count}", C_MOD)

        pygame.draw.line(screen, C_BORDER, (PX, PY), (WIN_W - 20, PY)); PY += 10

        # Color scale legend
        lbl = font_sm.render("DISTANCE COLOR SCALE", True, C_GREY)
        screen.blit(lbl, (PX, PY)); PY += 14
        color_scale_bar(screen, PX, PY, 340, 14); PY += 34

        # Key hints
        pygame.draw.line(screen, C_BORDER, (PX, PY), (WIN_W - 20, PY)); PY += 8
        hints = [
            "D = force DRIVE mode",
            "R = force REVERSE mode",
            "Q / Esc = quit",
            f"{'HARDWARE' if HARDWARE else 'DEMO MODE'}",
        ]
        for h in hints:
            t = font_sm.render(h, True, C_GREY)
            screen.blit(t, (PX, PY)); PY += 15

        # ── header strip ──────────────────────────────────────
        pygame.draw.rect(screen, C_PANEL,  (0, 0, WIN_W, 30))
        pygame.draw.line(screen, C_BORDER, (0, 30), (WIN_W, 30))
        hdr = font_hud.render(
            f"QCar2 · LiDAR 384pt · "
            f"FRONT={FRONT_CENTER_DEG:.0f}°  REAR={REAR_CENTER_DEG:.0f}°  "
            f"arc=±{FORWARD_ARC_DEG:.0f}°  max={MAX_RANGE_M:.0f}m",
            True, C_GREY
        )
        screen.blit(hdr, (10, 8))
        # live dot
        dot_col = C_SAFE if HARDWARE else C_MOD
        pygame.draw.circle(screen, dot_col, (WIN_W - 20, 15), 6)

        pygame.display.flip()

        # ── pace loop ─────────────────────────────────────────
        elapsed = time.time() - t0
        sleep_t = LOOP_PERIOD - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    # ── shutdown ──────────────────────────────────────────────
    print("\n[shutdown] Stopping ...")
    if HARDWARE:
        if card:
            try:
                card.write_other(MOTOR_CH, 2, np.array([0.0, 0.0], dtype=np.float64))
                print("[shutdown] Motor stopped")
                card.close()
                print("[shutdown] HIL closed")
            except HILError as e:
                print(f"[shutdown] HIL error: {e.get_error_message()}")
        if lidar:
            try:
                lidar.terminate()
                print("[shutdown] LiDAR terminated")
            except Exception as e:
                print(f"[shutdown] LiDAR error: {e}")

    pygame.quit()
    print("[shutdown] Done.")


if __name__ == "__main__":
    main()
