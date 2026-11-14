"""
mtah2027_possession/src/physics/ballistics.py
============================================
Ballistic Impulse Detection & Kalman Innovation Latching (Method 1 in technical plan).

Key mathematical principles:
1. Possession is a LATCHED state that only changes at discrete impulse/touch events.
2. 2D Aerodynamic Drag Kalman Filter:
     x_{t+1} = F x_t + w_t,  F = [[I, dt*I], [0, (1 - kappa*dt) I]]
   Computes innovation nu_t = z_t - H x_{t|t-1} and covariance S_t.
3. Generalized Likelihood Ratio Test (GLRT) for touch detection:
     Lambda_t = nu_t^T S_t^{-1} nu_t > chi^2_2(0.99) ≈ 9.21
4. Touch Attribution to candidate players within r < 2.5m (EPTS torso-to-foot extension).
5. Trajectory Continuation Likelihood Ratio (Pass-in-Flight proof):
   When ball is in flight (v > 2.5 m/s), an opponent standing 1.5m away DOES NOT touch
   the ball unless Lambda_t > 9.21 or dv > 1.8 m/s.
"""

import numpy as np
import pandas as pd


class BallisticsEngine:
    def __init__(self, dt=0.04, kappa=0.08, sigma_z=0.15, q_acc=1.5):
        self.dt = dt
        self.kappa = kappa
        self.sigma_z = sigma_z
        self.q_acc = q_acc

        # State transition matrix F (4x4)
        self.F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0 - kappa * dt, 0.0],
            [0.0, 0.0, 0.0, 1.0 - kappa * dt]
        ], dtype=np.float64)

        # Measurement matrix H (2x4)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)

        # Measurement noise covariance R (2x2)
        self.R = np.eye(2, dtype=np.float64) * (sigma_z ** 2)

        # Process noise covariance Q (4x4)
        dt2 = dt * dt
        dt3 = dt2 * dt
        q_pos = q_acc * dt3 / 3.0
        q_pos_vel = q_acc * dt2 / 2.0
        q_vel = q_acc * dt
        self.Q = np.array([
            [q_pos, 0.0, q_pos_vel, 0.0],
            [0.0, q_pos, 0.0, q_pos_vel],
            [q_pos_vel, 0.0, q_vel, 0.0],
            [0.0, q_pos_vel, 0.0, q_vel]
        ], dtype=np.float64)

    def filter_trajectory(self, ball_xy):
        """
        Run forward Kalman filter on 2D ball position trajectory.
        """
        N = len(ball_xy)
        x_est = np.zeros((N, 4), dtype=np.float64)
        innovations = np.zeros((N, 2), dtype=np.float64)
        innov_stats = np.zeros(N, dtype=np.float64)

        p0 = ball_xy[0]
        if np.isnan(p0[0]) or np.isnan(p0[1]):
            p0 = np.array([52.65, 34.0])
        x = np.array([p0[0], p0[1], 0.0, 0.0], dtype=np.float64)
        P = np.eye(4, dtype=np.float64) * 1.0

        for t in range(N):
            if t > 0:
                x = self.F @ x
                P = self.F @ P @ self.F.T + self.Q

            z = ball_xy[t]
            if np.isnan(z[0]) or np.isnan(z[1]):
                x_est[t] = x
                innovations[t] = 0.0
                innov_stats[t] = 0.0
                continue

            nu = z - self.H @ x
            S = self.H @ P @ self.H.T + self.R
            try:
                S_inv = np.linalg.inv(S)
                lambda_t = float(nu.T @ S_inv @ nu)
            except np.linalg.LinAlgError:
                S_inv = np.eye(2) / (self.sigma_z ** 2)
                lambda_t = float(np.sum(nu ** 2) / (self.sigma_z ** 2))

            K = P @ self.H.T @ S_inv
            x = x + K @ nu
            P = (np.eye(4) - K @ self.H) @ P

            x_est[t] = x
            innovations[t] = nu
            innov_stats[t] = lambda_t

        return x_est, innovations, innov_stats

    def detect_touches(self, ball_xy, player_xy, team_of_track, threshold_lambda=9.21, max_touch_dist=1.4):
        """
        Detect and attribute discrete ball touches to candidate players.
        Default max_touch_dist=1.4m reflects the physiological foot-reach limit from the player's torso GPS pod.
        """
        N = len(ball_xy)
        x_est, innovations, innov_stats = self.filter_trajectory(ball_xy)

        vx, vy = x_est[:, 2], x_est[:, 3]
        v_ball = np.sqrt(vx**2 + vy**2)
        dv_x = np.diff(vx, prepend=vx[0])
        dv_y = np.diff(vy, prepend=vy[0])
        dv_mag = np.sqrt(dv_x**2 + dv_y**2)

        bx = ball_xy[:, 0, None]
        by = ball_xy[:, 1, None]
        px = player_xy[:, :, 0]
        py = player_xy[:, :, 1]
        dist_players = np.sqrt((px - bx)**2 + (py - by)**2)
        dist_players = np.nan_to_num(dist_players, nan=99.0)

        min_player_dist = np.min(dist_players, axis=1)

        touch_mask = np.zeros(N, dtype=bool)
        touch_player = np.full(N, -1, dtype=int)
        touch_team = np.full(N, -1, dtype=int)
        touch_strength = np.zeros(N, dtype=np.float32)

        for t in range(N):
            is_candidate = False
            strength = 0.0

            # Condition 1: Ballistic impulse detected by GLRT (Lambda > 9.21)
            # and at least one player within reach r <= 2.6m
            if min_player_dist[t] <= max_touch_dist and innov_stats[t] > threshold_lambda:
                is_candidate = True
                strength = innov_stats[t]
            # Condition 2: Sudden velocity jump dv > 1.8 m/s within reach
            elif min_player_dist[t] <= max_touch_dist and dv_mag[t] > 1.8:
                is_candidate = True
                strength = dv_mag[t] * 10.0
            # Condition 3: Ball is slow (controlled dribble / trap, v < 2.0 m/s) and player is very close (d < 0.85m)
            elif min_player_dist[t] <= 0.85 and v_ball[t] <= 2.5:
                is_candidate = True
                strength = 10.0

            if not is_candidate:
                continue

            # Candidate players within reach
            nearby_mask = dist_players[t] <= max_touch_dist
            nearby_indices = np.where(nearby_mask)[0]
            if len(nearby_indices) == 0:
                continue

            best_player = -1
            best_score = -1.0
            ball_pos = ball_xy[t]
            delta_v = np.array([dv_x[t], dv_y[t]])
            delta_mag = np.linalg.norm(delta_v) + 1e-4

            for j in nearby_indices:
                d_j = dist_players[t, j]
                p_j = player_xy[t, j]
                vec_j_to_ball = ball_pos - p_j
                dist_j = np.linalg.norm(vec_j_to_ball) + 1e-4
                u_dir = vec_j_to_ball / dist_j

                cos_push = np.dot(delta_v / delta_mag, u_dir)

                # Prior score based on distance and push direction
                score = np.exp(- (d_j ** 2) / (2.0 * (1.0 ** 2)))
                if cos_push > -0.2:
                    score *= (1.0 + 0.5 * cos_push)
                else:
                    score *= 0.5

                if score > best_score:
                    best_score = score
                    best_player = j

            if best_player >= 0 and best_score > 0.05:
                touch_mask[t] = True
                touch_player[t] = best_player
                touch_team[t] = int(team_of_track[best_player])
                touch_strength[t] = float(strength)

        return {
            "touch_mask": touch_mask,
            "touch_player": touch_player,
            "touch_team": touch_team,
            "touch_strength": touch_strength,
            "innov_stats": innov_stats,
            "v_ball": v_ball,
            "dist_players": dist_players,
            "min_dist": min_player_dist
        }

    def compute_latched_features(self, ball_xy, player_xy, team_of_track):
        """
        Compute ballistic touch-latched invariant features for the full sequence.
        """
        N = len(ball_xy)
        touch_info = self.detect_touches(ball_xy, player_xy, team_of_track)
        touch_mask = touch_info["touch_mask"]
        touch_team = touch_info["touch_team"]
        innov_stats = touch_info["innov_stats"]
        v_ball = touch_info["v_ball"]
        min_dist = touch_info["min_dist"]

        # 1. Forward latching (causal last touch)
        frames_since_touch = np.zeros(N, dtype=np.float32)
        team_last_touch = np.full(N, -1, dtype=np.float32)

        cur_team = -1
        cur_frames = 9999.0
        for t in range(N):
            if touch_mask[t]:
                cur_team = touch_team[t]
                cur_frames = 0.0
            else:
                cur_frames += 1.0
            frames_since_touch[t] = cur_frames
            team_last_touch[t] = float(cur_team)

        # 2. Backward lookahead (non-causal next touch)
        time_to_next_touch = np.zeros(N, dtype=np.float32)
        next_touch_team = np.full(N, -1, dtype=np.float32)

        next_team = -1
        next_frames = 9999.0
        for t in range(N - 1, -1, -1):
            if touch_mask[t]:
                next_team = touch_team[t]
                next_frames = 0.0
            else:
                next_frames += 1.0
            time_to_next_touch[t] = next_frames
            next_touch_team[t] = float(next_team)

        # 3. Unperturbed Flight Identification (Pass-in-Flight)
        # In flight: v_ball > 2.5 m/s, min_dist > 1.0m, and no current touch changepoint
        is_flight = (v_ball > 2.5) & (min_dist > 1.0) & (~touch_mask)

        # 3B. Phase 2: Pseudo-3D Parabolic Arc Estimation
        z_ball = self.estimate_pseudo_3d_arc(ball_xy, v_ball, touch_info=touch_info)
        is_flight = (is_flight | (z_ball > 1.8)) & (~touch_mask)

        # 4. Latched Ownership (Resolves flight attribution)
        latched_H = np.full(N, 0.5, dtype=np.float32)
        for t in range(N):
            lt = team_last_touch[t]
            nt = next_touch_team[t]

            if is_flight[t]:
                # In unperturbed flight: pass belongs to last toucher!
                if lt == 0:
                    latched_H[t] = 0.95
                elif lt == 1:
                    latched_H[t] = 0.05
                elif nt == 0:
                    latched_H[t] = 0.90
                elif nt == 1:
                    latched_H[t] = 0.10
            else:
                # Close play
                if lt == 0 and frames_since_touch[t] < 25:
                    latched_H[t] = 0.85
                elif lt == 1 and frames_since_touch[t] < 25:
                    latched_H[t] = 0.15

        # 5. Speed continuity
        v_diff = np.abs(np.diff(v_ball, prepend=v_ball[0]))
        ball_speed_continuity = v_diff / (v_ball + 1.0)

        # 6. Max innovation in rolling 1s window (25 frames)
        s_innov = pd.Series(innov_stats)
        innov_max_1s = s_innov.rolling(25, min_periods=1).max().values.astype(np.float32)

        features = np.column_stack([
            frames_since_touch,
            team_last_touch,
            is_flight.astype(np.float32),
            latched_H,
            ball_speed_continuity,
            innov_max_1s,
            time_to_next_touch,
            next_touch_team,
            z_ball
        ])

        return features, touch_info

    def estimate_pseudo_3d_arc(self, ball_xy: np.ndarray, v_ball: np.ndarray, touch_info: dict = None) -> np.ndarray:
        """
        Phase 2: Estimate pseudo-3D ballistic arc height z(t) for lofted passes and clearances
        by kinematic parabolic arc reconstruction.

        Parameters
        ----------
        ball_xy : np.ndarray of shape (N, 2)
            2D ball coordinates in meters.
        v_ball : np.ndarray of shape (N,)
            Ball speed in m/s.
        touch_info : dict, optional
            Touch info dictionary from detect_touches containing touch_mask.

        Returns
        -------
        z_ball : np.ndarray of shape (N,)
            Estimated vertical height above turf in meters (0.1m ground baseline).
        """
        N = len(ball_xy)
        g = 9.81
        dt = self.dt

        # Ensure ball coordinates are linearly interpolated across tracking gaps
        bx = pd.Series(ball_xy[:, 0]).interpolate(method="linear").bfill().ffill().values
        by = pd.Series(ball_xy[:, 1]).interpolate(method="linear").bfill().ffill().values

        # Clean velocity
        vx = np.diff(bx, prepend=bx[0]) / dt
        vy = np.diff(by, prepend=by[0]) / dt
        vb = np.sqrt(vx**2 + vy**2)

        # Identify ballistic flight candidate intervals (sustained motion >= 3.0 m/s)
        is_moving = (vb > 3.0)
        breaks = np.flatnonzero(is_moving[1:] != is_moving[:-1]) + 1
        starts = [0] + list(breaks)
        ends = list(breaks) + [N]

        z_ball = np.full(N, 0.1, dtype=np.float32)

        for s, e in zip(starts, ends):
            if is_moving[s]:
                n_f = e - s
                if n_f >= 12:  # flight duration >= 0.48s
                    dx = bx[e - 1] - bx[s]
                    dy = by[e - 1] - by[s]
                    dist_traveled = np.sqrt(dx**2 + dy**2)
                    max_v = np.max(vb[s:e])

                    # Ballistic arc condition: either traveled >= 8.0m with kick speed >= 7.0 m/s,
                    # or sustained long flight >= 20 frames (0.8s) with net distance >= 6.0m
                    if (dist_traveled >= 8.0 and max_v >= 7.0) or (n_f >= 20 and dist_traveled >= 6.0):
                        T = n_f * dt
                        # Max height from projectile kinematics: z_max = (1/8) * g * T^2
                        z_max = min(6.5, (g * T**2) / 8.0)
                        if z_max >= 0.8:
                            tau = np.arange(n_f)
                            xi = tau / max(1, n_f - 1)
                            # Parabolic profile: z(t) = 4 * z_max * xi * (1 - xi) + z_ground
                            z_ball[s:e] = 4.0 * z_max * xi * (1.0 - xi) + 0.1

        return z_ball

    def estimate_pseudo_3d_height(self, ball_xy: np.ndarray, v_ball: np.ndarray) -> np.ndarray:
        """Alias for estimate_pseudo_3d_arc for backwards compatibility."""
        return self.estimate_pseudo_3d_arc(ball_xy, v_ball)

    def compute_effective_3d_distance(self, d_2d: np.ndarray, z_ball: np.ndarray) -> np.ndarray:
        """
        Compute effective 3D distance between players on turf and airborne ball.
        """
        dz_excess = np.maximum(0.0, z_ball - 1.8)
        return np.sqrt(d_2d**2 + dz_excess**2)

