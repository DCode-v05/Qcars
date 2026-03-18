# Quanser QCar2 - LiDAR Web Visualization
# Run on QCar, view in browser on your PC

import sys
sys.path.insert(0, '/home/nvidia/Documents/Quanser/libraries/python')

import threading
import time
import numpy as np
from flask import Flask, render_template_string
from pal.products.qcar import QCarLidar

# --- LiDAR setup (official Quanser settings) ---
myLidar = QCarLidar(
    numMeasurements=1000,
    rangingDistanceMode=2,
    interpolationMode=0
)

# Shared data
lidar_data = {"angles": [], "distances": []}
lock = threading.Lock()

# --- Background thread: keep reading LiDAR ---
def lidar_loop():
    while True:
        myLidar.read()
        angles    = np.array(myLidar.angles).tolist()
        distances = np.array(myLidar.distances).tolist()
        with lock:
            lidar_data["angles"]    = angles
            lidar_data["distances"] = distances
        time.sleep(0.05)

threading.Thread(target=lidar_loop, daemon=True).start()

# --- Flask web server ---
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>QCar2 LiDAR Viewer</title>
  <style>
    body { background:#111; color:#0f0; font-family:monospace; text-align:center; margin:0; }
    h2   { margin:10px 0; color:#0ff; }
    canvas { border: 1px solid #0ff; border-radius:50%; background:#000; }
    #stats { margin:8px; font-size:14px; }
  </style>
</head>
<body>
  <h2>🚗 QCar2 LiDAR — Live View</h2>
  <canvas id="c" width="600" height="600"></canvas>
  <div id="stats">Loading...</div>

<script>
const canvas  = document.getElementById('c');
const ctx     = canvas.getContext('2d');
const cx      = canvas.width / 2;
const cy      = canvas.height / 2;
const scale   = 60; // pixels per meter

function drawGrid() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#1a1a1a';
  ctx.lineWidth = 1;
  for (let r = 1; r <= 5; r++) {
    ctx.beginPath();
    ctx.arc(cx, cy, r * scale, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.fillStyle = '#333';
    ctx.font = '10px monospace';
    ctx.fillText(r + 'm', cx + r * scale + 2, cy);
  }
  // crosshair
  ctx.strokeStyle = '#222';
  ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, canvas.height); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(canvas.width, cy); ctx.stroke();
  // car icon
  ctx.fillStyle = '#0ff';
  ctx.fillRect(cx - 6, cy - 10, 12, 20);
}

async function fetchAndDraw() {
  try {
    const res  = await fetch('/data');
    const json = await res.json();
    const ang  = json.angles;
    const dist = json.distances;

    drawGrid();

    ctx.fillStyle = '#00ff88';
    let valid = 0;
    let minD = Infinity, maxD = 0;

    for (let i = 0; i < ang.length; i++) {
      const d = dist[i];
      if (d <= 0 || d > 6) continue;
      valid++;
      if (d < minD) minD = d;
      if (d > maxD) maxD = d;

      // LiDAR angle: convert to canvas coords
      const a = ang[i];  // radians
      const x = cx + d * scale * Math.sin(a);
      const y = cy - d * scale * Math.cos(a);

      ctx.beginPath();
      ctx.arc(x, y, 2, 0, 2 * Math.PI);
      ctx.fill();
    }

    document.getElementById('stats').innerHTML =
      `Points: <b>${ang.length}</b> &nbsp;|&nbsp; `+
      `Valid: <b>${valid}</b> &nbsp;|&nbsp; `+
      `Min: <b>${minD < Infinity ? minD.toFixed(2)+'m' : '--'}</b> &nbsp;|&nbsp; `+
      `Max: <b>${maxD > 0 ? maxD.toFixed(2)+'m' : '--'}</b>`;

  } catch(e) {
    document.getElementById('stats').innerHTML = '⚠️ Error: ' + e;
  }
  setTimeout(fetchAndDraw, 80); // ~12 fps
}

drawGrid();
fetchAndDraw();
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/data')
def data():
    with lock:
        return {
            "angles":    lidar_data["angles"],
            "distances": lidar_data["distances"]
        }

if __name__ == '__main__':
    print("✅ LiDAR Web Viewer running!")
    print("👉 Open in your PC browser: http://<QCAR_IP>:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
