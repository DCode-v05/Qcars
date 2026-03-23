"""
dwa_planner.py — Dynamic Window Approach for QCar 2.

Samples candidate Ackermann trajectories (varying steering angle + direction),
simulates each forward in time, checks against the VFH polar histogram for
collisions, and scores surviving trajectories to pick the best one.

Replaces the VFH gap-finding + steering computation. The histogram itself
(built from LiDAR + RealSense + YOLO) is still used as the obstacle map.
"""
import math
import numpy as np

import config as cfg


class DWAPlanner:

    def __init__(self):
        self._last_steering = 0.0

    def plan(self, histogram, current_steering, goal_heading=0.0):
        """
        Pick the best (steering, direction) by simulating Ackermann trajectories
        and scoring them against the histogram obstacle map.

        Args:
            histogram:        np.array (VFH_NUM_BINS,) — min obstacle distance per bin
            current_steering: float — current steering angle (rad)
            goal_heading:     float — desired heading (rad, 0 = forward)

        Returns:
            dict with: steering, drive_forward, best_angle, gap_width_deg, gap_depth
        """
        candidates = self._generate_candidates()
        best = None
        best_score = -float('inf')
        # Track best-of-colliding as fallback
        fallback = None
        fallback_clearance = -float('inf')

        for steer_rad, forward in candidates:
            traj = self._simulate(steer_rad, forward)
            clear, min_clearance = self._check_collision(traj, histogram)

            if clear:
                score = self._score(traj, steer_rad, forward,
                                    min_clearance, current_steering,
                                    goal_heading, histogram)
                if score > best_score:
                    best_score = score
                    best = (steer_rad, forward, traj, min_clearance)
            else:
                # Track the least-bad colliding trajectory
                if min_clearance > fallback_clearance:
                    fallback_clearance = min_clearance
                    fallback = (steer_rad, forward, traj, min_clearance)

        # If nothing is collision-free, use the least-bad trajectory
        if best is None:
            if fallback is not None:
                best = fallback
            else:
                # Absolute fallback: drive straight forward slowly
                best = (0.0, True, [(0, 0, 0)], 0.0)

        steer_rad, forward, traj, clearance = best

        # Compute path angle for dashboard display
        if len(traj) > 1:
            ex, ey, _ = traj[-1]
            path_angle = math.atan2(ey, ex) % (2 * math.pi)
        else:
            path_angle = 0.0

        self._last_steering = steer_rad

        return {
            'steering':      float(steer_rad),
            'drive_forward': forward,
            'best_angle':    path_angle,
            'gap_width_deg': clearance * 20.0,   # approximate for display
            'gap_depth':     clearance,
        }

    # ══════════════════════════════════════════════════════════════════════════
    #  CANDIDATE GENERATION
    # ══════════════════════════════════════════════════════════════════════════

    def _generate_candidates(self):
        """Generate (steering_rad, forward) pairs to evaluate."""
        candidates = []
        max_s = cfg.MAX_STEERING_RAD

        # Forward candidates: fine-grained sampling across full steering range
        n_fwd = cfg.DWA_NUM_STEER_FWD
        for i in range(n_fwd):
            frac = i / max(n_fwd - 1, 1)           # 0.0 → 1.0
            steer = -max_s + frac * 2.0 * max_s     # -30° → +30°
            candidates.append((steer, True))

        # Reverse candidates: coarser sampling
        n_rev = cfg.DWA_NUM_STEER_REV
        for i in range(n_rev):
            frac = i / max(n_rev - 1, 1)
            steer = -max_s + frac * 2.0 * max_s
            candidates.append((steer, False))

        return candidates

    # ══════════════════════════════════════════════════════════════════════════
    #  ACKERMANN TRAJECTORY SIMULATION
    # ══════════════════════════════════════════════════════════════════════════

    def _simulate(self, steering_rad, forward):
        """Simulate an Ackermann trajectory. Returns list of (x, y, theta).
        x = forward, y = left in car frame."""
        dt = cfg.DWA_SIM_TIME_S / cfg.DWA_SIM_STEPS
        v = cfg.DWA_SPEED_MPS * (1.0 if forward else -1.0)

        x, y, theta = 0.0, 0.0, 0.0
        traj = [(x, y, theta)]

        for _ in range(cfg.DWA_SIM_STEPS):
            if abs(steering_rad) < 1e-4:
                # Straight line
                x += v * math.cos(theta) * dt
                y += v * math.sin(theta) * dt
            else:
                # Ackermann bicycle model: R = wheelbase / tan(steering)
                R = cfg.WHEEL_BASE_M / math.tan(steering_rad)
                omega = v / R
                theta += omega * dt
                x += v * math.cos(theta) * dt
                y += v * math.sin(theta) * dt
            traj.append((x, y, theta))

        return traj

    # ══════════════════════════════════════════════════════════════════════════
    #  COLLISION CHECKING AGAINST HISTOGRAM
    # ══════════════════════════════════════════════════════════════════════════

    def _check_collision(self, trajectory, histogram):
        """Check if a trajectory is collision-free against the polar histogram.
        Returns (collision_free, min_clearance)."""
        n_bins = len(histogram)
        min_clearance = float('inf')
        margin = cfg.DWA_COLLISION_MARGIN

        for (x, y, _) in trajectory:
            dist_from_car = math.sqrt(x * x + y * y)
            if dist_from_car < 0.02:
                continue  # skip origin

            # Convert trajectory point to polar (bearing from car)
            angle = math.atan2(y, x) % (2 * math.pi)
            bin_idx = int(angle / (2 * math.pi) * n_bins) % n_bins

            # Check this bin and immediate neighbours for safety
            for offset in range(-1, 2):
                b = (bin_idx + offset) % n_bins
                obstacle_dist = histogram[b]
                clearance = obstacle_dist - dist_from_car
                if clearance < min_clearance:
                    min_clearance = clearance
                if clearance < margin:
                    return False, min_clearance

        return True, min_clearance

    # ══════════════════════════════════════════════════════════════════════════
    #  TRAJECTORY SCORING
    # ══════════════════════════════════════════════════════════════════════════

    def _score(self, trajectory, steering, forward,
               min_clearance, current_steering, goal_heading,
               histogram=None):
        """Score a collision-free trajectory. Higher = better."""
        ex, ey, etheta = trajectory[-1]

        # 1. Forward progress: how far forward (x-axis) the car moves
        max_dist = cfg.DWA_SPEED_MPS * cfg.DWA_SIM_TIME_S
        progress_score = ex / max(max_dist, 0.01)

        # 2. Goal alignment: how close end heading is to goal
        heading_err = abs(etheta - goal_heading)
        if heading_err > math.pi:
            heading_err = 2 * math.pi - heading_err
        goal_score = 1.0 - heading_err / math.pi

        # 3. Clearance from obstacles
        clearance_score = min(min_clearance / 1.0, 1.0)

        # 4. Smoothness: prefer steering close to current
        steer_change = abs(steering - current_steering)
        smooth_score = 1.0 - steer_change / (2.0 * cfg.MAX_STEERING_RAD)

        # 5. Direction bonus: strongly prefer forward
        direction_bonus = 1.0 if forward else 0.0

        score = (cfg.DWA_W_PROGRESS  * progress_score +
                 cfg.DWA_W_GOAL      * goal_score +
                 cfg.DWA_W_CLEARANCE  * clearance_score +
                 cfg.DWA_W_SMOOTH    * smooth_score +
                 cfg.DWA_W_FORWARD   * direction_bonus)

        # 6. Reverse turn bonus: when reversing, prefer turning toward
        #    the side with more space (prepares for next forward move)
        if not forward and histogram is not None:
            # Check which side has more space in the front hemisphere
            n = len(histogram)
            left_space = float(np.mean(histogram[n * 3 // 4:]))    # 270-360° = left
            right_space = float(np.mean(histogram[1:n // 4]))       # 0-90° = right
            if abs(steering) > 0.05:
                # Reward reverse trajectories that turn toward the open side
                turns_left = steering > 0
                if turns_left and left_space > right_space:
                    score += 0.15
                elif not turns_left and right_space > left_space:
                    score += 0.15
                # Always give some bonus for turning while reversing
                # (any turn creates lateral offset for the next forward attempt)
                score += 0.05 * abs(steering) / cfg.MAX_STEERING_RAD

        return score
