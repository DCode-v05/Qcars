"""
path_planner.py — A* global path planner on occupancy grid.

Finds shortest path from car's current position to goal (Point B).
Replans when obstacles block the path. Provides waypoints for DWA to follow.
"""
import math
import heapq
import numpy as np

import config as cfg


class PathPlanner:

    def __init__(self, occupancy_grid):
        self._grid = occupancy_grid
        self._path = []              # list of (wx, wy) world waypoints
        self._waypoint_idx = 0
        self._goal = None            # (gx, gy) world coordinates
        self._has_path = False
        self._last_replan = 0.0

    def set_goal(self, goal_x, goal_y):
        """Set the destination Point B in world coordinates."""
        self._goal = (goal_x, goal_y)
        self._path = []
        self._waypoint_idx = 0
        self._has_path = False

    def has_goal(self):
        return self._goal is not None

    def replan(self, pose):
        """Run A* from current position to goal. Call at ASTAR_REPLAN_HZ."""
        if self._goal is None:
            return

        px, py, _ = pose
        grid = self._grid

        # Convert to grid coordinates
        sx, sy = grid._world_to_grid(px, py)
        gx, gy = grid._world_to_grid(self._goal[0], self._goal[1])

        n = grid.size
        if not (0 <= sx < n and 0 <= sy < n and 0 <= gx < n and 0 <= gy < n):
            self._has_path = False
            return

        # Get inflated binary grid
        obstacles = grid.get_binary_grid()

        # A* search
        path_cells = self._astar(sx, sy, gx, gy, obstacles, n)

        if path_cells:
            # Convert grid cells to world waypoints, simplify
            raw = [(grid._grid_to_world(cx, cy)) for cx, cy in path_cells]
            self._path = self._simplify_path(raw)
            self._waypoint_idx = 0
            self._has_path = True
        else:
            self._path = []
            self._waypoint_idx = 0
            self._has_path = False

    def get_goal_heading(self, pose):
        """Get heading from car to next waypoint (car-relative).
        Returns None if no goal or no path."""
        if not self._has_path or not self._path:
            return None

        px, py, ptheta = pose
        wp = self._get_lookahead_point(px, py)
        if wp is None:
            return None

        wx, wy = wp
        # World heading to waypoint
        world_heading = math.atan2(wy - py, wx - px)
        # Convert to car-relative
        rel_heading = world_heading - ptheta
        # Wrap to [-pi, pi]
        while rel_heading > math.pi:
            rel_heading -= 2 * math.pi
        while rel_heading < -math.pi:
            rel_heading += 2 * math.pi
        return rel_heading

    def is_goal_reached(self, pose):
        """Check if car is within GOAL_REACHED_M of the goal."""
        if self._goal is None:
            return False
        px, py, _ = pose
        dx = self._goal[0] - px
        dy = self._goal[1] - py
        return math.sqrt(dx * dx + dy * dy) < cfg.GOAL_REACHED_M

    def needs_replan(self, pose):
        """Check if we need to replan (path blocked or no path)."""
        if self._goal is None:
            return False
        if not self._has_path:
            return True
        if not self._path:
            return True
        # Check if any remaining path cells are now occupied
        for i in range(self._waypoint_idx, min(self._waypoint_idx + 10,
                                                len(self._path))):
            wx, wy = self._path[i]
            if self._grid.is_occupied(wx, wy):
                return True
        return False

    def get_path_points(self):
        """Return path waypoints for dashboard visualization."""
        return [{'x': round(wx, 2), 'y': round(wy, 2)}
                for wx, wy in self._path]

    @property
    def path_found(self):
        return self._has_path

    @property
    def goal(self):
        return self._goal

    # ══════════════════════════════════════════════════════════════════════
    #  INTERNAL
    # ══════════════════════════════════════════════════════════════════════

    def _get_lookahead_point(self, px, py):
        """Pure-pursuit: find point on path at lookahead distance ahead."""
        if not self._path:
            return None

        lookahead = cfg.WAYPOINT_LOOKAHEAD_M
        reached = cfg.WAYPOINT_REACHED_M

        # Advance past reached waypoints
        while self._waypoint_idx < len(self._path) - 1:
            wx, wy = self._path[self._waypoint_idx]
            d = math.sqrt((wx - px) ** 2 + (wy - py) ** 2)
            if d < reached:
                self._waypoint_idx += 1
            else:
                break

        # Find the farthest waypoint within lookahead
        best = None
        for i in range(self._waypoint_idx, len(self._path)):
            wx, wy = self._path[i]
            d = math.sqrt((wx - px) ** 2 + (wy - py) ** 2)
            if d <= lookahead:
                best = (wx, wy)
            else:
                # Take this one if nothing closer was found
                if best is None:
                    best = (wx, wy)
                break

        if best is None and self._path:
            best = self._path[-1]  # go to final waypoint

        return best

    def _astar(self, sx, sy, gx, gy, obstacles, n):
        """A* on 8-connected grid. Returns list of (gx, gy) cells or None."""
        if obstacles[sy, sx] or obstacles[gy, gx]:
            # Start or goal is inside an obstacle — try to find nearest free cell
            if obstacles[gy, gx]:
                gx, gy = self._nearest_free(gx, gy, obstacles, n)
                if gx is None:
                    return None
            if obstacles[sy, sx]:
                sx, sy = self._nearest_free(sx, sy, obstacles, n)
                if sx is None:
                    return None

        # 8 directions: dx, dy, cost
        dirs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

        # Priority queue: (f_score, x, y)
        open_set = [(0.0, sx, sy)]
        g_score = {(sx, sy): 0.0}
        came_from = {}
        visited = set()
        nodes_explored = 0

        while open_set and nodes_explored < cfg.ASTAR_MAX_NODES:
            _, cx, cy = heapq.heappop(open_set)

            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            nodes_explored += 1

            if cx == gx and cy == gy:
                # Reconstruct path
                path = [(cx, cy)]
                while (cx, cy) in came_from:
                    cx, cy = came_from[(cx, cy)]
                    path.append((cx, cy))
                path.reverse()
                return path

            for dx, dy, cost in dirs:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < n and 0 <= ny < n):
                    continue
                if obstacles[ny, nx]:
                    continue
                if (nx, ny) in visited:
                    continue

                new_g = g_score[(cx, cy)] + cost
                if new_g < g_score.get((nx, ny), float('inf')):
                    g_score[(nx, ny)] = new_g
                    # Euclidean heuristic
                    h = math.sqrt((nx - gx) ** 2 + (ny - gy) ** 2)
                    f = new_g + h
                    came_from[(nx, ny)] = (cx, cy)
                    heapq.heappush(open_set, (f, nx, ny))

        return None  # no path found

    def _nearest_free(self, gx, gy, obstacles, n):
        """Find nearest free cell to (gx, gy)."""
        for r in range(1, 20):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < n and 0 <= ny < n and not obstacles[ny, nx]:
                        return nx, ny
        return None, None

    def _simplify_path(self, path):
        """Remove redundant collinear waypoints."""
        if len(path) <= 2:
            return path

        simplified = [path[0]]
        for i in range(1, len(path) - 1):
            x0, y0 = simplified[-1]
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            # Check if collinear (cross product ~= 0)
            cross = abs((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0))
            if cross > 0.001:  # not collinear — keep
                simplified.append(path[i])
        simplified.append(path[-1])
        return simplified
