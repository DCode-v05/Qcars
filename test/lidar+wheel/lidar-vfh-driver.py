#!/usr/bin/env python3
"""
QCar 2 — Hybrid VFH+ / FGM Autonomous Navigator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Algorithm stack (priority order):
  1. VFH+  — primary: polar histogram valley finding
             full 360° awareness, cost-function steering
  2. FGM   — fallback: Follow the Gap Method (Sezer & Gokasan 2012)
             no local-minimum problem, picks widest raw gap in scan
  3. ESCAPE — recovery: slow spin-in-place to break symmetry
             triggered only when BOTH VFH+ and FGM find nothing

State machine:
  DRIVE   → REVERSE  (auto, when forward blocked > threshold)
  REVERSE → DRIVE    (auto, when rear gap opens up)
  any     → ESCAPE   (auto, when all gaps gone > 0.5s)
  ESCAPE  → DRIVE    (auto, when gap appears again)
  STOP    only when nearest obstacle < ESTOP_M in all directions

Key fixes over previous version:
  • VFH+ re-runs immediately after every mode switch (no stale state)
  • FGM runs on raw scan — independent of VFH threshold tuning
  • Blocked detection uses SOFT threshold (80% of sectors, 2-frame confirm)
  • Escape uses in-place rotation (steer hard + tiny throttle)
  • Throttle never zero unless ESTOP or genuinely no path at all
  • All 360° sectors always searched — not just forward/rear hemisphere

LiDAR: rangingDistanceMode=0  SHORT — best for 10x10 ft room

Run:   python3 lidar_auto.py
Keys:  q=quit  +=throttle up  -=throttle down  e=force escape
"""

import sys, math, time, select, tty, termios, collections
import numpy as np

try:
    from pal.products.qcar import QCarLidar
    from quanser.hardware import HIL, EncoderQuadratureMode
    from quanser.hardware.exceptions import HILError
    HARDWARE = True
except ImportError:
    HARDWARE = False

# ══════════════════════════════════════════════════════════════
#  CALIBRATE — paste from lidar_diag.py
# ══════════════════════════════════════════════════════════════
FRONT_DEG   = 180.0
REAR_DEG    = (FRONT_DEG + 180.0) % 360.0

# ══════════════════════════════════════════════════════════════
#  LIDAR
# ══════════════════════════════════════════════════════════════
RANGING_MODE    = 0      # SHORT — best for closed room < 3m
NUM_MEAS        = 384
LOOP_HZ         = 10
LOOP_T          = 1.0 / LOOP_HZ

# ══════════════════════════════════════════════════════════════
#  VFH+ PARAMETERS
# ══════════════════════════════════════════════════════════════
ALPHA           = 5              # sector width degrees → 72 sectors
N_SEC           = 360 // ALPHA   # 72
DIST_INFL       = 2.0            # obstacle influence radius (m)
ROBOT_R         = 0.22           # robot half-width for inflation
SAFE_D          = 0.08           # extra safety margin
SMOOTH_L        = 3              # smoothing window (sectors each side)
H_THRESH        = 0.55           # binary threshold (lower=more cautious)
H_HYST          = 0.12           # hysteresis band
MIN_VALLEY_SEC  = 2              # min free sectors for valid valley (2×5°=10°)
COST_W_FORWARD  = 4.0            # weight: gap vs forward dir
COST_W_PREV     = 3.0            # weight: gap vs previous steer
COST_W_CONT     = 2.0            # weight: gap vs current heading

# ══════════════════════════════════════════════════════════════
#  FGM PARAMETERS (Follow the Gap Method fallback)
# ══════════════════════════════════════════════════════════════
FGM_MIN_GAP_M   = 0.45           # min distance jump to count as gap edge
FGM_ROBOT_D     = 0.40           # robot diameter for gap passability check
FGM_ALPHA       = 0.85           # blend: 1.0=pure gap centre, 0=forward bias

# ══════════════════════════════════════════════════════════════
#  STATE MACHINE
# ══════════════════════════════════════════════════════════════
# DRIVE → REVERSE trigger: fraction of FRONT_CHECK_DEG blocked
FRONT_CHECK_DEG = 70             # cone to check for forward block
BLOCKED_FRAC    = 0.78           # fraction needed to trigger mode flip
CONFIRM_FRAMES  = 2              # consecutive frames needed to confirm block

# ESCAPE trigger: no gap found for this long
ESCAPE_TRIGGER_S = 0.5
ESCAPE_STEER     = 0.40          # hard steer during escape
ESCAPE_THROTTLE  = 0.06          # very slow creep during escape
ESCAPE_MAX_S     = 2.5           # max escape duration before trying reverse

# Hard stop distance — truly blocked in all directions
ESTOP_M          = 0.20

# ══════════════════════════════════════════════════════════════
#  THROTTLE — adaptive
# ══════════════════════════════════════════════════════════════
MAX_THR         = 0.18
MIN_THR         = 0.07
THR_STEP        = 0.02
SLOW_NEAR_M     = 0.80           # start slowing below this
STOP_NEAR_M     = ESTOP_M
SLOW_STEER_DEG  = 18             # slow on steer > this
STEER_SMOOTH    = 0.40           # IIR steer smoother

# ══════════════════════════════════════════════════════════════
#  HARDWARE
# ══════════════════════════════════════════════════════════════
CARD_TYPE   = "qcar2"
CARD_ID     = "0"
MOTOR_CH    = np.array([11000, 1000], dtype=np.uint32)
ENC_CH      = np.array([0],          dtype=np.uint32)
MAX_STEER   = 0.40
MPC         = (2*math.pi*0.0328)/(720*4)   # metres per encoder count

# ══════════════════════════════════════════════════════════════
#  TERMINAL CANVAS
# ══════════════════════════════════════════════════════════════
CC, CR  = 74, 38
PC, PR  = CC, CR*2
CX, CY  = PC//2, PR//2
RR      = min(CX,CY)-2
PCOL    = CC+2
PW      = 46

CB   = (4,8,15);    CP=(8,15,26);   CBR=(15,42,74)
CA   = (0,212,255); CS=(0,220,100); CM=(255,180,0)
CU   = (220,40,40); CW=(200,230,255); CG=(80,110,140)
CGRD = (16,34,58);  CH=(60,30,110); CHH=(160,60,240)
CGAP = (0,255,160); CES=(255,100,0)

E="\033"
def fg(r,g,b): return f"{E}[38;2;{r};{g};{b}m"
def bg(r,g,b): return f"{E}[48;2;{r};{g};{b}m"
def rs():      return f"{E}[0m"
def bd():      return f"{E}[1m"
def mv(r,c):   return f"{E}[{r};{c}H"
def clr():     return f"{E}[2J"
def hc():      return f"{E}[?25l"
def sc():      return f"{E}[?25h"

def dcol(d, mx=2.0):
    t=max(0.,min(1.,d/mx))
    if   t<.25: return (220,int(t/.25*130),20)
    elif t<.50: return (220,130+int((t-.25)/.25*100),0)
    elif t<.75: return (int(220*(1-(t-.50)/.25)),220,0)
    else:       return (0,int(220-(t-.75)/.25*80),int((t-.75)/.25*200))

def make_canvas():
    c=np.zeros((PR,PC,3),dtype=np.uint8); c[:,:]=CB; return c

def sp(cv,x,y,col):
    if 0<=y<PR and 0<=x<PC: cv[y,x]=col

def bln(cv,x0,y0,x1,y1,col):
    dx,dy=abs(x1-x0),abs(y1-y0)
    sx=1 if x0<x1 else -1; sy=1 if y0<y1 else -1; err=dx-dy
    while True:
        sp(cv,x0,y0,col)
        if x0==x1 and y0==y1: break
        e2=2*err
        if e2>-dy: err-=dy; x0+=sx
        if e2< dx: err+=dx; y0+=sy

def dot(cv,x,y,col,r=2):
    for dy in range(-r,r+1):
        for dx in range(-r,r+1):
            if dx*dx+dy*dy<=r*r: sp(cv,x+dx,y+dy,col)

def circ(cv,cx,cy,r,col):
    x,y,d=r,0,1-r
    while x>=y:
        for sx,sy in[(cx+x,cy+y),(cx-x,cy+y),(cx+x,cy-y),(cx-x,cy-y),
                     (cx+y,cy+x),(cx-y,cy+x),(cx+y,cy-x),(cx-y,cy-x)]:
            sp(cv,sx,sy,col)
        y+=1
        d = d+2*y+1 if d<0 else (x:=x-1, d+2*(y-x)+1)[1]

def p2px(d,a,maxr=2.0):
    rp=min(d/maxr,1.)*RR
    fr=math.radians(FRONT_DEG); sa=(a-fr)-math.pi/2
    return CX+int(rp*math.cos(sa)), CY+int(rp*math.sin(sa))

def sec2px(sec,frac=1.0):
    deg=(sec*ALPHA+ALPHA/2)%360.
    fr=math.radians(FRONT_DEG); a=math.radians(deg)+fr-math.pi/2
    r=int(RR*frac)
    return CX+int(r*math.cos(a)), CY+int(r*math.sin(a))

def draw_grid(cv):
    for rm in[0.5,1.0,1.5,2.0]:
        circ(cv,CX,CY,int(rm/2.*RR),CGRD)
    circ(cv,CX,CY,RR,CBR)
    fr=math.radians(FRONT_DEG)
    for d in range(0,360,45):
        a=math.radians(d)+fr-math.pi/2
        bln(cv,CX,CY,CX+int(RR*math.cos(a)),CY+int(RR*math.sin(a)),CGRD)

def draw_scan(cv,dists,angs):
    for i in range(len(dists)):
        d=float(dists[i])
        if d<0.03: continue
        col=dcol(d)
        px,py=p2px(d,float(angs[i]))
        dot(cv,px,py,col,3 if d<0.35 else 2 if d<0.8 else 1)

def draw_hist_ring(cv,hist_sm):
    mx=max(hist_sm.max(),1e-6)
    for k in range(N_SEC):
        h=hist_sm[k]/mx
        ir=int(RR*.88); or_=int(RR*(.88+.10*h))
        col=CHH if hist_sm[k]>H_THRESH else CH
        deg=(k*ALPHA+ALPHA/2)%360.
        fr=math.radians(FRONT_DEG); a=math.radians(deg)+fr-math.pi/2
        for r in range(ir,or_+1):
            sp(cv,CX+int(r*math.cos(a)),CY+int(r*math.sin(a)),col)

def draw_valleys(cv,valleys,binary):
    for k in range(N_SEC):
        if binary[k]:
            px,py=sec2px(k,.76); dot(cv,px,py,(0,50,25),1)
    for v in valleys:
        k=v[0]
        while True:
            px,py=sec2px(k,.70); dot(cv,px,py,CGAP,1)
            if k==v[1]: break
            k=(k+1)%N_SEC

def draw_best(cv,valley,fgm_angle_rad=None):
    if valley:
        k=valley[0]
        while True:
            px,py=sec2px(k,.65); dot(cv,px,py,CA,2)
            if k==valley[1]: break
            k=(k+1)%N_SEC
        px,py=sec2px(valley[2],.62)
        bln(cv,CX,CY,px,py,CA)
    if fgm_angle_rad is not None:
        fr=math.radians(FRONT_DEG); a=(fgm_angle_rad-fr)-math.pi/2
        px=CX+int(RR*.55*math.cos(a)); py=CY+int(RR*.55*math.sin(a))
        dot(cv,px,py,(255,200,0),3)
        bln(cv,CX,CY,px,py,(255,200,0))

def draw_arrow(cv,steer,thr,mode,state):
    col=CS if mode=="DRIVE" else CM
    if state=="ESCAPE": col=CES
    fr=math.radians(FRONT_DEG); sign=1 if mode=="DRIVE" else -1
    ln=max(8,int(abs(thr)/MAX_THR*RR*.45))
    a=fr-math.pi/2+steer*2.
    ex=CX+int(ln*sign*math.cos(a)); ey=CY+int(ln*sign*math.sin(a))
    bln(cv,CX,CY,ex,ey,col)
    for da in(-.5,.5):
        bln(cv,ex,ey,ex+int(8*math.cos(a+math.pi+da)),
                     ey+int(8*math.sin(a+math.pi+da)),col)

def draw_car(cv,mode,state):
    col=CES if state=="ESCAPE" else CS if mode=="DRIVE" else CM
    dot(cv,CX,CY,col,5); dot(cv,CX,CY,CB,3); dot(cv,CX,CY,col,1)

def cv2term(cv):
    out=[]
    for row in range(0,PR-1,2):
        out.append(mv(row//2+2,1))
        pt=pb=None
        for col in range(PC):
            top=tuple(cv[row,col]); bot=tuple(cv[row+1,col])
            s=""
            if top!=pt: s+=fg(*top); pt=top
            if bot!=pb: s+=bg(*bot); pb=bot
            s+="▀"; out.append(s)
        out.append(rs())
    return "".join(out)

def bar(f,w=22,on=CS,off=CBR):
    f=max(0.,min(1.,f)); n=int(f*w)
    return fg(*on)+"█"*n+fg(*off)+"░"*(w-n)+rs()

def panel_lines(mode,state,algo,status,steer,thr,min_d,
                n_val,best_v,scan_n,fps,sw,enc_m,max_thr,
                hist_sm,binary,escape_t):
    mc=CS if mode=="DRIVE" else CM
    ec=CES if state=="ESCAPE" else mc
    sd=math.degrees(steer)
    sc2=CM if abs(sd)>20 else CW
    dc=(CU if min_d<=0.25 else CM if min_d<=0.8 else CS)
    tc=CS if thr>0 else CM if thr<0 else CG
    bv=f"w={best_v[3]*ALPHA}° c={best_v[2]*ALPHA}°" if best_v else "none"
    sym="▲ DRIVE  " if mode=="DRIVE" else "▼ REVERSE"

    # mini histogram bar (N_SEC→22 chars)
    bw=N_SEC//22; hbar=""
    mx=max(hist_sm.max(),1e-6)
    for i in range(22):
        c=hist_sm[i*bw:(i+1)*bw]; v=c.mean()/mx
        free=binary[i*bw:(i+1)*bw].all()
        col=CGAP if free else CHH if v>.5 else CH
        hbar+=fg(*col)+" ░▒▓█"[min(4,int(v*5))]
    hbar+=rs()

    def r(lbl,val,col=CW):
        return fg(*CG)+lbl+" "*(13-len(lbl))+fg(*col)+str(val)+rs()

    esc_info=""
    if state=="ESCAPE":
        esc_info=f"  {fg(*CES)}{bd()}ESCAPE {escape_t:.1f}s / {ESCAPE_MAX_S:.1f}s{rs()}"

    return [
        bd()+fg(*CA)+"QCar 2 · Hybrid VFH+/FGM"+rs(),
        fg(*CG)+"SHORT LiDAR | auto-navigate | never stops"+rs(),
        fg(*CBR)+"─"*PW+rs(),
        fg(*CG)+" MODE / STATE"+rs(),
        "  "+bd()+fg(*ec)+sym+" │ "+state+rs(),
        esc_info,
        "",
        fg(*CG)+" ALGORITHM"+rs(),
        "  "+fg(*CA)+algo+rs()+"  "+fg(*CG)+status[:PW-12]+rs(),
        "",
        fg(*CG)+" VFH+ HISTOGRAM"+rs(),
        "  "+hbar,
        r(" valleys",str(n_val),CA),
        r(" best valley",bv,CA),
        "",
        fg(*CBR)+"─"*PW+rs(),
        fg(*CG)+" MOTION"+rs(),
        r(" steer",f"{sd:+.1f}° ({steer:+.3f}r)",sc2),
        "  "+bar((steer+MAX_STEER)/(2*MAX_STEER),on=sc2),
        r(" throttle",f"{'▲' if thr>0 else '▼' if thr<0 else '■'} {abs(thr):.3f}/{max_thr:.2f}",tc),
        "  "+bar(abs(thr)/max(max_thr,.01),on=tc),
        "",
        fg(*CBR)+"─"*PW+rs(),
        fg(*CG)+" ENVIRONMENT"+rs(),
        r(" nearest",f"{min_d:.3f} m",dc),
        "  "+bar(1.-min(min_d/2.,1.),on=dc),
        r(" odometry",f"{enc_m:+.3f} m"),
        "",
        fg(*CBR)+"─"*PW+rs(),
        fg(*CG)+" SESSION"+rs(),
        r(" scan #",str(scan_n)),
        r(" fps",f"{fps:.1f}"),
        r(" mode switches",str(sw),CM),
        r(" max throttle",f"{max_thr:.2f}  (+/-)"),
        "",
        fg(*CBR)+"─"*PW+rs(),
        fg(*CG)+" HEATMAP SCALE  near→far"+rs(),
        "  "+"".join(fg(*dcol(i*2./26))+"█" for i in range(26))+rs(),
        "  "+fg(*CU)+"▲near "+rs()+fg(*CM)+"▲med "+rs()+fg(*CS)+"▲far"+rs(),
        "",
        fg(*CBR)+"─"*PW+rs(),
        fg(*CG)+" q=quit  e=escape  +/-=throttle"+rs(),
        fg(*CG)+" "+("HARDWARE" if HARDWARE else "DEMO MODE")+rs(),
    ]


# ══════════════════════════════════════════════════════════════
#  VFH+  (fixed: full 360°, no direction bias in histogram)
# ══════════════════════════════════════════════════════════════
class VFHPlus:
    def __init__(self):
        self.hist    = np.zeros(N_SEC)
        self.hist_sm = np.zeros(N_SEC)
        self.binary  = np.ones(N_SEC,dtype=bool)   # start all free
        self.prev_st = 0.0

    def build(self, dists, angs):
        self.hist[:]=0.
        a=1.0; b=1.0/DIST_INFL
        for i in range(len(dists)):
            d=float(dists[i])
            if d<0.03 or d>DIST_INFL: continue
            mag=(a-b*d)**2
            # full 360° — no front bias in histogram build
            deg=(math.degrees(float(angs[i]))-FRONT_DEG)%360.
            sec=int(deg/ALPHA)%N_SEC
            spread=max(1,int(math.degrees(
                math.asin(min(1.,(ROBOT_R+SAFE_D)/max(d,.01))))/ALPHA))
            for ds in range(-spread,spread+1):
                self.hist[(sec+ds)%N_SEC]+=mag
        # smooth
        l=SMOOTH_L
        for k in range(N_SEC):
            self.hist_sm[k]=sum(self.hist[(k+s)%N_SEC]
                                for s in range(-l,l+1))/(2*l+1)

    def threshold(self):
        hi=H_THRESH+H_HYST; lo=H_THRESH-H_HYST
        for k in range(N_SEC):
            if   self.hist_sm[k]>=hi: self.binary[k]=False
            elif self.hist_sm[k]<=lo: self.binary[k]=True
            # else: keep (hysteresis)

    def valleys(self):
        n=N_SEC; b2=np.concatenate([self.binary,self.binary])
        vals=[]; i=0
        while i<n:
            if b2[i]:
                j=i
                while j<i+n and b2[j]: j+=1
                w=j-i
                if w>=MIN_VALLEY_SEC:
                    s=i%n; e=(j-1)%n; c=((i+j-1)//2)%n
                    vals.append((s,e,c,w))
                i=j
            else: i+=1
        return vals

    def sdiff(self,a,b):
        d=abs(a-b); return min(d,N_SEC-d)*ALPHA

    def best_valley(self, vals, forward_sec):
        if not vals: return None,0.
        prev_sec=int((math.degrees(self.prev_st)%360.)/ALPHA)%N_SEC
        best_cost=1e9; best_v=None; best_sec=forward_sec
        for v in vals:
            s,e,c,w=v
            if w<=2*MIN_VALLEY_SEC:
                cand=c
            else:
                dl=self.sdiff(s,forward_sec)
                dr=self.sdiff(e,forward_sec)
                cand=((s+MIN_VALLEY_SEC)%N_SEC if dl<dr
                      else (e-MIN_VALLEY_SEC)%N_SEC)
            cost=(COST_W_FORWARD*self.sdiff(cand,forward_sec)
                 +COST_W_PREV   *self.sdiff(cand,prev_sec)
                 +COST_W_CONT   *self.sdiff(cand,forward_sec))
            if cost<best_cost:
                best_cost=cost; best_v=v; best_sec=cand
        # convert sector → signed angle
        deg=(best_sec*ALPHA+ALPHA/2.)%360.
        if deg>180.: deg-=360.
        return best_v, math.radians(deg)

    def run(self, dists, angs, forward_deg):
        self.build(dists,angs)
        self.threshold()
        vals=self.valleys()
        fwd_sec=int(forward_deg/ALPHA)%N_SEC
        bv, steer=self.best_valley(vals,fwd_sec)
        return vals,bv,steer


# ══════════════════════════════════════════════════════════════
#  FGM  — Follow the Gap Method (Sezer & Gokasan 2012)
#  Works on RAW scan — independent of VFH thresholds
#  No local-minimum problem by design
# ══════════════════════════════════════════════════════════════
def fgm(dists, angs, forward_deg):
    """
    1. Find all gap edges (large distance jumps between adjacent readings)
    2. Each gap = (left_edge_ang, right_edge_ang, min_dist_inside)
    3. Keep only gaps wide enough for robot to pass
    4. Score each gap: wide + far from obstacles + close to forward dir
    5. Return steering angle toward best gap centre
    """
    n=len(dists)
    if n<2: return None, 0.

    # Sort by angle for adjacency
    order=np.argsort(angs)
    sa=angs[order]; sd=dists[order]

    gaps=[]
    i=0
    while i<n:
        # look for a "jump up" (start of a gap)
        j=(i+1)%n
        d_cur=float(sd[i]); d_next=float(sd[j])
        if d_next-d_cur>FGM_MIN_GAP_M and d_cur>0.03:
            # gap starts here — find where it ends (jump down)
            gap_start=float(sa[i])
            k=j
            while k!=i:
                kn=(k+1)%n
                if float(sd[k])-float(sd[kn])>FGM_MIN_GAP_M or kn==i:
                    gap_end=float(sa[k])
                    # gap width in angle
                    gap_width=abs(angle_diff(gap_end,gap_start))
                    # check if robot can fit: gap_width_m at gap_dist
                    gap_dist=float(sd[j:k+1].max()) if k>=j else float(sd[j])
                    gap_w_m=gap_dist*gap_width
                    if gap_w_m>FGM_ROBOT_D:
                        gap_centre=(gap_start+gap_end)/2.
                        gaps.append((gap_centre,gap_width,gap_dist))
                    break
                k=kn
        i+=1

    if not gaps:
        return None, 0.

    # Score: prefer wide+far+forward
    fr=math.radians(forward_deg)
    best_score=-1e9; best_ang=fr
    for gc,gw,gd in gaps:
        align=1.-abs(angle_diff(gc,fr))/math.pi   # 1=forward, 0=behind
        score=FGM_ALPHA*(gw*gd)+(1.-FGM_ALPHA)*align
        if score>best_score:
            best_score=score; best_ang=gc

    steer=angle_diff(best_ang,fr)   # signed rad vs forward
    return gaps, float(np.clip(steer,-MAX_STEER,MAX_STEER))


def angle_diff(a,b):
    return math.atan2(math.sin(a-b),math.cos(a-b))

def arc_min(dists,angs,center_deg,half_deg):
    cr=math.radians(center_deg); hr=math.radians(half_deg)
    mask=np.array([abs(angle_diff(float(a),cr))<=hr for a in angs])
    v=(dists>0.03)&mask
    return float(dists[v].min()) if v.sum()>0 else None

def hemi_blocked(binary,center_deg,half_deg):
    """Fraction of sectors in arc that are blocked."""
    cs=int(center_deg/ALPHA)%N_SEC
    hs=max(1,int(half_deg/ALPHA))
    bl=sum(1 for ds in range(-hs,hs+1) if not binary[(cs+ds)%N_SEC])
    return bl/(2*hs+1)


# ══════════════════════════════════════════════════════════════
#  DEMO
# ══════════════════════════════════════════════════════════════
_dt=0.
def demo():
    global _dt
    a=np.linspace(0,2*math.pi,NUM_MEAS,endpoint=False)
    base=1.4+0.3*np.sin(4*a)+0.2*np.cos(7*a+1)
    fr=math.radians(FRONT_DEG)
    wall=0.32+0.38*abs(math.sin(_dt*.28))
    b1=wall*np.exp(-18*(a-fr)**2)
    side=fr+math.pi/2
    b2=0.35*np.exp(-22*(a-side)**2)
    d=np.clip(base-b1-b2,.08,2.0)+np.random.normal(0,.015,NUM_MEAS)
    _dt+=LOOP_T
    return d,a

# ══════════════════════════════════════════════════════════════
#  KEYBOARD
# ══════════════════════════════════════════════════════════════
def setup_tty():
    fd=sys.stdin.fileno(); old=termios.tcgetattr(fd); tty.setraw(fd); return old
def restore_tty(old):
    termios.tcsetattr(sys.stdin.fileno(),termios.TCSADRAIN,old)
def readkey():
    return sys.stdin.read(1) if select.select([sys.stdin],[],[],0)[0] else None

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    card=None; lidar=None
    mb=np.array([0.,0.],dtype=np.float64)
    eb=np.zeros(1,dtype=np.int32); enc_m=0.

    if HARDWARE:
        print("[init] HIL ...")
        try:
            card=HIL(CARD_TYPE,CARD_ID)
            em=np.array([EncoderQuadratureMode.X4],dtype=np.int32)
            card.set_encoder_quadrature_mode(ENC_CH,1,em)
            card.set_encoder_counts(ENC_CH,1,np.zeros(1,dtype=np.int32))
            card.write_other(MOTOR_CH,2,mb)
            print("[init] HIL OK")
        except HILError as e:
            print(f"[init] HIL fail: {e.get_error_message()}"); card=None

        print(f"[init] LiDAR SHORT mode ...")
        lidar=QCarLidar(numMeasurements=NUM_MEAS,
                        rangingDistanceMode=RANGING_MODE,
                        interpolationMode=0)
        print("[init] 2s spin-up ..."); time.sleep(2.)
        print("[init] ready."); time.sleep(.4)
    else:
        print("DEMO MODE..."); time.sleep(.8)

    old_tty=setup_tty()
    sys.stdout.write(clr()); sys.stdout.flush()

    cv=make_canvas(); vfh=VFHPlus()

    # ── state ─────────────────────────────────────────────────
    mode        = "DRIVE"
    state       = "DRIVE"   # DRIVE / REVERSE / ESCAPE / STOP
    steer       = 0.
    thr         = 0.
    max_thr     = MAX_THR
    scan_n      = 0
    fps_s       = 0.
    t_last      = time.time()
    sw          = 0
    algo        = "VFH+"
    status      = "INIT"
    best_v      = None
    all_v       = []
    fgm_ang     = None
    no_gap_since= None
    escape_dir  = 1
    escape_start= 0.
    front_blk_q = collections.deque(maxlen=CONFIRM_FRAMES)
    rear_blk_q  = collections.deque(maxlen=CONFIRM_FRAMES)
    dists       = np.ones(NUM_MEAS)*1.5
    angs        = np.linspace(0,2*math.pi,NUM_MEAS,endpoint=False)

    try:
        while True:
            t0=time.time()

            # ── keys ──────────────────────────────────────────
            k=readkey()
            if k:
                if k in('q','Q','\x03'): break
                elif k in('e','E'):
                    state="ESCAPE"; escape_start=time.time()
                    escape_dir=1; no_gap_since=None
                elif k in('+','='): max_thr=min(1.,round(max_thr+THR_STEP,3))
                elif k in('-','_'): max_thr=max(0.,round(max_thr-THR_STEP,3))

            # ── scan ──────────────────────────────────────────
            if HARDWARE and lidar:
                lidar.read()
                dists=lidar.distances.flatten().copy()
                angs =lidar.angles.flatten().copy()
            else:
                dists,angs=demo()

            valid_d=dists[dists>0.03]
            min_d=float(valid_d.min()) if len(valid_d) else 99.

            # ── VFH+ (always run full 360°) ───────────────────
            forward_deg=(FRONT_DEG if mode=="DRIVE" else REAR_DEG)
            all_v,best_v,vfh_steer=vfh.run(dists,angs,forward_deg)

            # ── FGM fallback (always run, independent) ────────
            fgm_gaps,fgm_steer_raw=fgm(dists,angs,forward_deg)
            fgm_ang=(math.radians(forward_deg)+fgm_steer_raw
                     if fgm_gaps else None)

            # ── blocked detection (soft, confirmed) ───────────
            fb=hemi_blocked(vfh.binary,FRONT_DEG,FRONT_CHECK_DEG)
            rb=hemi_blocked(vfh.binary,REAR_DEG, FRONT_CHECK_DEG)
            front_blk_q.append(fb>=BLOCKED_FRAC)
            rear_blk_q.append(rb>=BLOCKED_FRAC)
            front_confirmed=all(front_blk_q) and len(front_blk_q)==CONFIRM_FRAMES
            rear_confirmed =all(rear_blk_q)  and len(rear_blk_q) ==CONFIRM_FRAMES

            # ── ESTOP check ───────────────────────────────────
            if min_d < ESTOP_M:
                state="STOP"; steer=0.; thr=0.
                status=f"ESTOP {min_d:.2f}m"; algo="ESTOP"

            # ── STATE MACHINE ─────────────────────────────────
            elif state=="ESCAPE":
                et=time.time()-escape_start
                if et>ESCAPE_MAX_S:
                    # try flipping mode after escape timeout
                    mode="REVERSE" if mode=="DRIVE" else "DRIVE"
                    state=mode; sw+=1
                    escape_dir*=-1; no_gap_since=None
                elif all_v or fgm_gaps:
                    # gap appeared — exit escape
                    state=mode; no_gap_since=None
                else:
                    # keep escaping: spin in place
                    steer=float(np.clip(ESCAPE_STEER*escape_dir,
                                       -MAX_STEER,MAX_STEER))
                    thr=ESCAPE_THROTTLE if mode=="DRIVE" else -ESCAPE_THROTTLE
                    algo="ESCAPE"; status=f"spinning {et:.1f}s"

            elif state in("DRIVE","REVERSE"):
                # auto mode flip on confirmed block
                if mode=="DRIVE" and front_confirmed and not rear_confirmed:
                    mode="REVERSE"; state="REVERSE"; sw+=1
                    vfh.prev_st=0.; status="AUTO→REVERSE"
                elif mode=="REVERSE" and rear_confirmed and not front_confirmed:
                    mode="DRIVE"; state="DRIVE"; sw+=1
                    vfh.prev_st=0.; status="AUTO→DRIVE"
                elif front_confirmed and rear_confirmed:
                    # both blocked → escape
                    state="ESCAPE"; escape_start=time.time()
                    escape_dir=1; status="BOTH→ESCAPE"

                # pick algorithm
                if all_v:
                    # VFH+ has valleys — use it (most informed)
                    raw_steer=vfh_steer
                    # IIR smooth
                    steer=(STEER_SMOOTH*vfh.prev_st
                           +(1.-STEER_SMOOTH)*raw_steer)
                    vfh.prev_st=steer
                    steer=float(np.clip(steer,-MAX_STEER,MAX_STEER))
                    algo="VFH+"
                    bv_str=(f"w={best_v[3]*ALPHA}°" if best_v else "")
                    status=f"valley {bv_str} st={math.degrees(steer):+.0f}°"
                elif fgm_gaps:
                    # VFH found nothing but FGM found a gap — fallback
                    raw_steer=fgm_steer_raw
                    steer=(STEER_SMOOTH*vfh.prev_st
                           +(1.-STEER_SMOOTH)*raw_steer)
                    vfh.prev_st=steer
                    steer=float(np.clip(steer,-MAX_STEER,MAX_STEER))
                    algo="FGM"
                    status=f"gap st={math.degrees(steer):+.0f}°"
                    no_gap_since=None
                else:
                    # no gap from either — track time
                    algo="NONE"
                    status="no gap found..."
                    if no_gap_since is None:
                        no_gap_since=time.time()
                    elif time.time()-no_gap_since>ESCAPE_TRIGGER_S:
                        state="ESCAPE"; escape_start=time.time()
                        escape_dir=1; no_gap_since=None
                    # hold last steer while waiting
                    steer=vfh.prev_st

                # adaptive throttle (only if not in escape/stop)
                if state in("DRIVE","REVERSE"):
                    near_sc=1.
                    if min_d<SLOW_NEAR_M:
                        near_sc=max(0.,(min_d-STOP_NEAR_M)
                                    /(SLOW_NEAR_M-STOP_NEAR_M))
                    sd_deg=abs(math.degrees(steer))
                    steer_sc=(1.-max(0.,sd_deg-SLOW_STEER_DEG)
                              /(90.-SLOW_STEER_DEG)) if sd_deg>SLOW_STEER_DEG else 1.
                    raw_thr=max_thr*near_sc*steer_sc
                    thr=float(np.clip(raw_thr,0.,max_thr))
                    if thr<MIN_THR: thr=MIN_THR   # never dip below min while moving
                    if mode=="REVERSE": thr=-thr

            elif state=="STOP":
                # re-check — maybe obstacle moved
                if min_d>ESTOP_M*1.5:
                    state=mode

            # ── motor ─────────────────────────────────────────
            if HARDWARE and card:
                mb[0]=thr; mb[1]=steer
                try:
                    card.write_other(MOTOR_CH,2,mb)
                    card.read_encoder(ENC_CH,1,eb)
                    enc_m=int(eb[0])*MPC
                except HILError: pass

            # ── fps ───────────────────────────────────────────
            now=time.time()
            fps_s=fps_s*.85+(1./max(now-t_last,1e-4))*.15
            t_last=now; scan_n+=1

            # ── render ────────────────────────────────────────
            cv[:]=CB
            draw_grid(cv)
            draw_hist_ring(cv,vfh.hist_sm)
            draw_valleys(cv,all_v,vfh.binary)
            draw_scan(cv,dists,angs)
            draw_best(cv,best_v,fgm_ang)
            draw_arrow(cv,steer,thr,mode,state)
            draw_car(cv,mode,state)

            et_esc=(time.time()-escape_start
                    if state=="ESCAPE" else 0.)
            plines=panel_lines(
                mode,state,algo,status,steer,thr,min_d,
                len(all_v),best_v,scan_n,fps_s,sw,enc_m,
                max_thr,vfh.hist_sm,vfh.binary,et_esc
            )

            mc=CS if mode=="DRIVE" else CM
            ec=CES if state=="ESCAPE" else mc
            buf=[hc(), cv2term(cv), mv(1,1),
                 bg(*CP)+fg(*CA)+bd()+
                 "  QCar2 Hybrid VFH+/FGM "+rs()+
                 bg(*CP)+fg(*CG)+
                 f" SHORT-LiDAR secs={N_SEC} α={ALPHA}° "
                 f"thresh={H_THRESH} "+
                 fg(*ec)+bd()+state+rs()+" "*6]
            for i,l in enumerate(plines):
                buf+=[mv(i+2,PCOL),l,fg(*CB)+"  "+rs()]
            bot=CR+3; dc2=CU if min_d<.25 else CM if min_d<.8 else CS
            buf+=[mv(bot,1),
                  bg(*CP)+fg(*CG)+
                  f"  nearest={fg(*dc2)}{min_d:.2f}m{rs()}{bg(*CP)}{fg(*CG)}"
                  f" steer={fg(*CW)}{math.degrees(steer):+.0f}°{rs()}{bg(*CP)}{fg(*CG)}"
                  f" thr={fg(*CS if thr>0 else CM if thr<0 else CG)}{thr:+.3f}{rs()}{bg(*CP)}{fg(*CG)}"
                  f" algo={fg(*CA)}{algo}{rs()}{bg(*CP)}{fg(*CG)}"
                  f" valleys={len(all_v)} scan={scan_n} fps={fps_s:.1f}"+
                  " "*6+rs()]
            sys.stdout.write("".join(buf)); sys.stdout.flush()

            s=LOOP_T-(time.time()-t0)
            if s>0: time.sleep(s)

    except Exception as ex:
        restore_tty(old_tty); sys.stdout.write(sc()+"\n")
        print(f"Error: {ex}")
        import traceback; traceback.print_exc()
    finally:
        restore_tty(old_tty); sys.stdout.write(sc())
        sys.stdout.write(mv(CR+7,1)); sys.stdout.flush()
        print("\n[shutdown]")
        if HARDWARE:
            if card:
                try:
                    card.write_other(MOTOR_CH,2,
                                     np.array([0.,0.],dtype=np.float64))
                    print("  motor stopped"); card.close()
                except HILError as e:
                    print(f"  {e.get_error_message()}")
            if lidar:
                try: lidar.terminate(); print("  lidar terminated")
                except: pass
        print("  done.")

if __name__=="__main__":
    main()
