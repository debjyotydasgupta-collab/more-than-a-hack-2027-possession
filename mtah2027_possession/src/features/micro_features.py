"""
src/features/micro_features.py
==============================
Per-frame micro-spatial features (32-dimensional) derived from player and ball positions.
Vectorized implementation for ultra-fast processing across 199,319 frames.

Feature groups:
1. Top-5 player distances to ball — per team (10 features)
2. Proximity ratio & differential (2 features)
3. Closing velocity of nearest 3 players per team (6 features)
4. Ball speed, ball acceleration (2 features)
5. Nearest-player raw distance — per team (2 features)
6. Nearest-player direction cosines — per team (4 features)
7. Distance ratio nearest-to-5th — per team (2 features)
8. Ball-to-pitch-centre distance, ball half (2 features)
9. Team average distance to ball (2 features)

Total: 10 + 2 + 6 + 2 + 2 + 4 + 2 + 2 + 2 = 32 features
"""

import numpy as np

N_TOP = 5
N_CLOSE = 3
N_FEATS = 32
LARGE = 999.0

HOME = 0
AWAY = 1


class MicroFeatureExtractor:
    """Compute 32-dimensional micro features for every frame using vectorized NumPy."""

    def __init__(self, team_of_track, pitch_length=105.3, pitch_width=68.0, dt=0.04):
        self.team_of_track = np.asarray(team_of_track, dtype=np.int8)
        self.pitch_length = float(pitch_length)
        self.pitch_width = float(pitch_width)
        self.dt = float(dt)

        self.home_mask = (self.team_of_track == HOME)
        self.away_mask = (self.team_of_track == AWAY)

    def extract(self, frames_m, ball_xy_m_smooth, ball_speed=None, ball_accel=None):
        """Extract micro features across all N frames using vectorized NumPy."""
        N = frames_m.shape[0]
        X = np.zeros((N, N_FEATS), dtype=np.float32)

        if ball_speed is None:
            ball_speed = np.zeros(N, dtype=np.float32)
        if ball_accel is None:
            ball_accel = np.zeros(N, dtype=np.float32)

        bx = ball_xy_m_smooth[:, 0]
        by = ball_xy_m_smooth[:, 1]
        ball_valid = ~np.isnan(bx)

        # Separate Home and Away player tracks
        H_pts = frames_m[:, self.home_mask, :]  # (N, N_H, 2)
        A_pts = frames_m[:, self.away_mask, :]  # (N, N_A, 2)

        # Distances from ball to players
        d_H = np.sqrt((H_pts[:, :, 0] - bx[:, None]) ** 2 + (H_pts[:, :, 1] - by[:, None]) ** 2)
        d_A = np.sqrt((A_pts[:, :, 0] - bx[:, None]) ** 2 + (A_pts[:, :, 1] - by[:, None]) ** 2)

        # Fill NaNs with LARGE for missing tracks or missing ball
        d_H_clean = np.where(np.isnan(d_H) | (~ball_valid[:, None]), LARGE, d_H)
        d_A_clean = np.where(np.isnan(d_A) | (~ball_valid[:, None]), LARGE, d_A)

        # 1. Top-5 player distances per team
        sort_H = np.sort(d_H_clean, axis=1)
        sort_A = np.sort(d_A_clean, axis=1)

        X[:, 0:5] = sort_H[:, :N_TOP]
        X[:, 5:10] = sort_A[:, :N_TOP]

        # 2. Proximity ratio & differential
        h_near = X[:, 0]
        a_near = X[:, 5]
        denom = h_near + a_near + 1e-6
        X[:, 10] = np.where((h_near < LARGE) & (a_near < LARGE), h_near / denom, 0.5)
        X[:, 11] = np.where((h_near < LARGE) & (a_near < LARGE), a_near - h_near, 0.0)

        # 3. Closing velocities
        # Distance rate of change: (d_prev - d_curr) / dt
        d_H_diff = -np.diff(d_H_clean, axis=0, prepend=d_H_clean[[0]]) / self.dt
        d_A_diff = -np.diff(d_A_clean, axis=0, prepend=d_A_clean[[0]]) / self.dt
        d_H_diff[~ball_valid, :] = 0.0
        d_A_diff[~ball_valid, :] = 0.0

        sort_cv_H = np.sort(np.nan_to_num(d_H_diff, nan=0.0), axis=1)[:, ::-1]
        sort_cv_A = np.sort(np.nan_to_num(d_A_diff, nan=0.0), axis=1)[:, ::-1]

        X[:, 12:15] = sort_cv_H[:, :N_CLOSE]
        X[:, 15:18] = sort_cv_A[:, :N_CLOSE]

        # 4. Ball speed, acceleration
        X[:, 18] = np.nan_to_num(ball_speed, nan=0.0)
        X[:, 19] = np.nan_to_num(ball_accel, nan=0.0)

        # 5. Nearest player raw distance
        X[:, 20] = h_near
        X[:, 21] = a_near

        # 6. Direction cosines of nearest player to ball
        arg_H = np.argmin(d_H_clean, axis=1)
        arg_A = np.argmin(d_A_clean, axis=1)

        # Indexing along axis 1 for each row
        row_idx = np.arange(N)
        near_H_x = H_pts[row_idx, arg_H, 0]
        near_H_y = H_pts[row_idx, arg_H, 1]
        near_A_x = A_pts[row_idx, arg_A, 0]
        near_A_y = A_pts[row_idx, arg_A, 1]

        dx_H = bx - near_H_x
        dy_H = by - near_H_y
        dist_H = np.sqrt(dx_H ** 2 + dy_H ** 2) + 1e-6
        X[:, 22] = np.where(ball_valid, dx_H / dist_H, 0.0)
        X[:, 23] = np.where(ball_valid, dy_H / dist_H, 0.0)

        dx_A = bx - near_A_x
        dy_A = by - near_A_y
        dist_A = np.sqrt(dx_A ** 2 + dy_A ** 2) + 1e-6
        X[:, 24] = np.where(ball_valid, dx_A / dist_A, 0.0)
        X[:, 25] = np.where(ball_valid, dy_A / dist_A, 0.0)

        # 7. Distance ratio: nearest-to-5th per team
        X[:, 26] = np.where(sort_H[:, 4] < LARGE, sort_H[:, 0] / (sort_H[:, 4] + 1e-6), 1.0)
        X[:, 27] = np.where(sort_A[:, 4] < LARGE, sort_A[:, 0] / (sort_A[:, 4] + 1e-6), 1.0)

        # 8. Ball-to-pitch-centre distance & side
        cx = self.pitch_length * 0.5
        cy = self.pitch_width * 0.5
        X[:, 28] = np.where(ball_valid, np.sqrt((bx - cx) ** 2 + (by - cy) ** 2), 0.0)
        X[:, 29] = np.where(ball_valid, np.where(bx >= cx, 1.0, -1.0), 0.0)

        # 9. Team average distance to ball
        X[:, 30] = np.where(ball_valid, np.nanmean(np.where(d_H_clean < LARGE, d_H_clean, np.nan), axis=1), LARGE)
        X[:, 31] = np.where(ball_valid, np.nanmean(np.where(d_A_clean < LARGE, d_A_clean, np.nan), axis=1), LARGE)

        return np.nan_to_num(X, nan=0.0)
