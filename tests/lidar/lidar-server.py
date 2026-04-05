#!/usr/bin/env python3
"""
QCar 2 LiDAR WebSocket Server
Streams live LiDAR scan data to the browser GUI over WebSocket.

Usage:
    python3 lidar_server.py

Then open lidar_heatmap.html in a browser on the same machine.
"""

import asyncio
import json
import time
import numpy as np
import websockets

# ── QCar LiDAR import (falls back to synthetic demo data if not on hardware) ──
try:
    from pal.products.qcar import QCarLidar
    HARDWARE = True
    print("[lidar_server] QCar hardware detected — using real LiDAR")
except ImportError:
    HARDWARE = False
    print("[lidar_server] QCar libraries not found — running DEMO mode (synthetic data)")

HOST = "0.0.0.0"
PORT = 8765
NUM_MEASUREMENTS = 384

# ── Demo data generator (used when not on real QCar hardware) ──
def synthetic_scan(t: float):
    """Generate a fake but realistic-looking LiDAR scan for testing."""
    angles = np.linspace(0, 2 * np.pi, NUM_MEASUREMENTS, endpoint=False)
    # Base room shape + some moving obstacle
    base = 3.0 + 1.2 * np.sin(2 * angles) + 0.8 * np.cos(3 * angles)
    # Moving obstacle blob
    obstacle_angle = (t * 0.5) % (2 * np.pi)
    blob = 1.5 * np.exp(-8 * (angles - obstacle_angle) ** 2)
    distances = np.clip(base - blob, 0.15, 8.0)
    # Add small noise
    distances += np.random.normal(0, 0.02, NUM_MEASUREMENTS)
    return distances, angles

# ── LiDAR reader coroutine ──
async def lidar_producer(queue: asyncio.Queue):
    if HARDWARE:
        lidar = QCarLidar(numMeasurements=NUM_MEASUREMENTS, rangingDistanceMode=2, interpolationMode=0)
        print("[lidar_server] Waiting 2 s for LiDAR motor to spin up...")
        await asyncio.sleep(2.0)
        try:
            while True:
                flag = lidar.read()
                distances = lidar.distances.flatten().tolist()
                angles = lidar.angles.flatten().tolist()
                payload = {
                    "distances": distances,
                    "angles": angles,
                    "flag": int(flag),
                    "timestamp": time.time(),
                }
                await queue.put(payload)
                await asyncio.sleep(0.1)  # ~10 Hz
        finally:
            lidar.terminate()
    else:
        t = 0.0
        while True:
            distances, angles = synthetic_scan(t)
            payload = {
                "distances": distances.tolist(),
                "angles": angles.tolist(),
                "flag": 1,
                "timestamp": time.time(),
            }
            await queue.put(payload)
            t += 0.1
            await asyncio.sleep(0.1)

# ── WebSocket handler ──
async def handler(websocket, path=None):
    client = websocket.remote_address
    print(f"[lidar_server] Client connected: {client}")
    queue: asyncio.Queue = asyncio.Queue(maxsize=2)

    producer_task = asyncio.create_task(lidar_producer(queue))
    try:
        while True:
            payload = await queue.get()
            await websocket.send(json.dumps(payload))
    except websockets.exceptions.ConnectionClosed:
        print(f"[lidar_server] Client disconnected: {client}")
    finally:
        producer_task.cancel()

async def main():
    print(f"[lidar_server] Starting WebSocket server on ws://{HOST}:{PORT}")
    async with websockets.serve(handler, HOST, PORT):
        print(f"[lidar_server] Ready — open lidar_heatmap.html in your browser")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())