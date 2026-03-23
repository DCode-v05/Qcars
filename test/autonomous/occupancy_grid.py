"""
occupancy_grid.py — Incremental 2D occupancy grid built from LiDAR.

The car starts at (0,0) which is the center of the grid. As it drives,
LiDAR rays mark cells as FREE (traversed) or OCCUPIED (endpoint).
Uses log-odds for efficient Bayesian updates.

The grid is used by A* path planner to find global paths.
"""
import math
import numpy as np

import config as cfg


class OccupancyGrid:

    def __init__(self):
        res = cfg.GRID_RESOLUTION_M
        size = cfg.GRID_SIZE_M
        self._res = res
        self._n = int(size / res)  # cells per side (200 for 10m at 5cm)
        self._half = self._n // 2  # center cell index
        # Log-odds grid: 0 = unknown
        self._log_odds = np.zeros((self._n, self._n), dtype=np.float32)

    def reset(self):
        self._log_odds[:] = 0.0

    def update(self, pose, distances, angles, valid):
        """Update grid from one LiDAR scan.

        Args:
            pose: (x, y, theta) in world frame
            distances: np.array of LiDAR distances
            angles: np.array of LiDAR angles (car-relative, 0=front)
            valid: np.array of bools
        """
        if len(distances) == 0:
            return

        px, py, ptheta = pose
        cx, cy = self._world_to_grid(px, py)

        for i in range(len(distances)):
            if not valid[i]:
                continue
            d = distances[i]
            if d > cfg.LIDAR_MAX_M or d < cfg.LIDAR_MIN_M:
                continue

            # World angle of this LiDAR ray
            world_angle = ptheta + angles[i]

            # Endpoint in world frame
            wx = px + d * math.cos(world_angle)
            wy = py + d * math.sin(world_angle)
            ex, ey = self._world_to_grid(wx, wy)

            # Bresenham ray: mark cells along ray as FREE
            self._trace_ray(cx, cy, ex, ey, d)

    def _trace_ray(self, x0, y0, x1, y1, dist):
        """Bresenham line from (x0,y0) to (x1,y1).
        Marks traversed cells as free, endpoint as occupied."""
        n = self._n
        lo = self._log_odds

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            if 0 <= x < n and 0 <= y < n:
                if x == x1 and y == y1:
                    # Endpoint = occupied
                    lo[y, x] = min(lo[y, x] + cfg.GRID_LOG_OCC,
                                   cfg.GRID_LOG_MAX)
                else:
                    # Traversed = free
                    lo[y, x] = max(lo[y, x] + cfg.GRID_LOG_FREE,
                                   cfg.GRID_LOG_MIN)

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def is_occupied(self, wx, wy):
        """Check if a world coordinate is occupied."""
        gx, gy = self._world_to_grid(wx, wy)
        if 0 <= gx < self._n and 0 <= gy < self._n:
            prob = 1.0 / (1.0 + math.exp(-self._log_odds[gy, gx]))
            return prob > cfg.GRID_OCC_THRESH
        return True  # out of bounds = occupied

    def get_binary_grid(self):
        """Return binary occupancy grid (True = occupied/unknown).
        Inflated by ASTAR_INFLATE_CELLS for path planning."""
        prob = 1.0 / (1.0 + np.exp(-self._log_odds))
        occupied = prob > cfg.GRID_OCC_THRESH

        # Inflate obstacles by car radius
        inflate = cfg.ASTAR_INFLATE_CELLS
        if inflate > 0:
            inflated = occupied.copy()
            for dy in range(-inflate, inflate + 1):
                for dx in range(-inflate, inflate + 1):
                    if dx * dx + dy * dy <= inflate * inflate:
                        shifted = np.roll(np.roll(occupied, dy, axis=0),
                                          dx, axis=1)
                        inflated |= shifted
            return inflated
        return occupied

    def _world_to_grid(self, wx, wy):
        gx = int(wx / self._res) + self._half
        gy = int(wy / self._res) + self._half
        return gx, gy

    def _grid_to_world(self, gx, gy):
        wx = (gx - self._half) * self._res
        wy = (gy - self._half) * self._res
        return wx, wy

    def get_map_data(self):
        """Return map data for dashboard visualization.
        Returns list of occupied cell world coordinates."""
        prob = 1.0 / (1.0 + np.exp(-self._log_odds))
        occ_y, occ_x = np.where(prob > cfg.GRID_OCC_THRESH)
        points = []
        for i in range(0, len(occ_x), 2):  # skip every other for bandwidth
            wx, wy = self._grid_to_world(int(occ_x[i]), int(occ_y[i]))
            points.append({'x': round(wx, 2), 'y': round(wy, 2)})
        return points

    @property
    def resolution(self):
        return self._res

    @property
    def size(self):
        return self._n
