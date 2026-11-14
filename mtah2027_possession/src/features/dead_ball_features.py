"""
mtah2027_possession/src/features/dead_ball_features.py
======================================================
Method 4: Dead-Ball Geometry and Collective Stillness Signature.

Features engineered:
1. Symmetry-safe boundary proximity:
   - d_line = min(x, 105.3 - x, y, 68.0 - y)
   - d_corners = min dist to 4 corner flags
   - d_penalty_spots = min dist to penalty spots (11m from goal)
   - d_centre_spot = dist to (52.65, 34.0)
2. Out-of-bounds flag & trajectory memory:
   - is_out_of_bounds = (x < 0 | x > 105.3 | y < 0 | y > 68)
   - frames_since_out_of_bounds
3. Collective kinetic energy:
   - E_t = (1/22) * sum ||v_j||^2
   - dE_t/dt (derivative over 5-frame window)
   - ratio of team_H kinetic energy to team_A kinetic energy
4. Wall / cluster structure:
   - n_players_within_9_15m (FIFA wall distance)
   - dispersion of players within 9.15m
5. Dead-ball hysteresis indicator:
   - Sustained stillness + boundary/out proximity
"""

import numpy as np
import pandas as pd


class DeadBallFeatureExtractor:
    def __init__(self, pitch_length=105.3, pitch_width=68.0):
        self.L = pitch_length
        self.W = pitch_width
        self.corners = np.array([
            [0.0, 0.0],
            [0.0, self.W],
            [self.L, 0.0],
            [self.L, self.W]
        ], dtype=np.float32)
        self.penalties = np.array([
            [11.0, self.W / 2.0],
            [self.L - 11.0, self.W / 2.0]
        ], dtype=np.float32)
        self.centre = np.array([self.L / 2.0, self.W / 2.0], dtype=np.float32)

    def extract(self, ball_xy, player_xy, player_v, team_of_track):
        """
        Parameters:
        -----------
        ball_xy : np.ndarray (N, 2) in meters
        player_xy : np.ndarray (N, 46, 2) in meters
        player_v : np.ndarray (N, 46) player speeds (m/s)
        team_of_track : np.ndarray (46,) 0=H, 1=A

        Returns:
        --------
        X_dead : np.ndarray (N, 16) float32
        """
        N = len(ball_xy)
        bx = np.nan_to_num(ball_xy[:, 0], nan=self.L / 2.0)
        by = np.nan_to_num(ball_xy[:, 1], nan=self.W / 2.0)

        # 1. Boundary distances
        d_x = np.minimum(bx, self.L - bx)
        d_y = np.minimum(by, self.W - by)
        d_line = np.minimum(d_x, d_y)

        # Out-of-bounds flag
        is_out = ((bx < 0.0) | (bx > self.L) | (by < 0.0) | (by > self.W)).astype(np.float32)

        # Frames since out-of-bounds
        frames_since_out = np.zeros(N, dtype=np.float32)
        cur_out_cnt = 9999.0
        for t in range(N):
            if is_out[t] > 0.5:
                cur_out_cnt = 0.0
            else:
                cur_out_cnt += 1.0
            frames_since_out[t] = cur_out_cnt

        # Distance to pitch landmarks
        d_corners = np.min(np.sqrt((bx[:, None] - self.corners[:, 0])**2 + (by[:, None] - self.corners[:, 1])**2), axis=1)
        d_penalties = np.min(np.sqrt((bx[:, None] - self.penalties[:, 0])**2 + (by[:, None] - self.penalties[:, 1])**2), axis=1)
        d_centre = np.sqrt((bx - self.centre[0])**2 + (by - self.centre[1])**2)

        # 2. Ball Kinematics
        vx = np.diff(bx, prepend=bx[0]) / 0.04
        vy = np.diff(by, prepend=by[0]) / 0.04
        v_ball = np.sqrt(vx**2 + vy**2)
        a_ball = np.diff(v_ball, prepend=v_ball[0]) / 0.04

        # Ball stillness rolling windows (0.5s, 1.0s, 2.0s)
        s_v = pd.Series(v_ball)
        still_12 = s_v.rolling(12, min_periods=1).mean().values.astype(np.float32)
        still_25 = s_v.rolling(25, min_periods=1).mean().values.astype(np.float32)
        still_50 = s_v.rolling(50, min_periods=1).mean().values.astype(np.float32)

        # 3. Collective Kinetic Energy
        # Select active players on pitch (players with valid coordinates)
        pv = np.nan_to_num(player_v, nan=0.0)
        ke_all = 0.5 * np.mean(pv**2, axis=1)  # average kinetic energy per player
        ke_dot = np.diff(ke_all, prepend=ke_all[0]) / 0.04
        s_ke = pd.Series(ke_all)
        ke_rolling_1s = s_ke.rolling(25, min_periods=1).mean().values.astype(np.float32)

        # 4. Wall / FIFA 9.15m cluster structure
        px = np.nan_to_num(player_xy[:, :, 0], nan=self.L / 2.0)
        py = np.nan_to_num(player_xy[:, :, 1], nan=self.W / 2.0)
        dist_players = np.sqrt((px - bx[:, None])**2 + (py - by[:, None])**2)

        # Count players within 9.15m
        n_within_9_15 = np.sum(dist_players <= 9.15, axis=1).astype(np.float32)
        # Dispersion of players within 9.15m
        dispersion_9_15 = np.zeros(N, dtype=np.float32)
        for t in range(N):
            mask_9 = dist_players[t] <= 9.15
            if np.sum(mask_9) >= 2:
                dispersion_9_15[t] = float(np.std(dist_players[t, mask_9]))

        # 5. Hysteresis rule indicator
        # High likelihood of dead ball: ball still, near boundary or out, or very low player energy
        is_still_near_line = ((still_25 < 0.5) & (d_line < 1.0)).astype(np.float32)
        is_still_corner = ((still_25 < 0.5) & (d_corners < 2.0)).astype(np.float32)

        features = np.column_stack([
            d_line,
            is_out,
            frames_since_out,
            d_corners,
            d_penalties,
            d_centre,
            v_ball,
            a_ball,
            still_12,
            still_25,
            still_50,
            ke_all,
            ke_dot,
            ke_rolling_1s,
            n_within_9_15,
            is_still_near_line
        ]).astype(np.float32)

        return features

