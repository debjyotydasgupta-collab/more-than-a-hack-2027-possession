"""
src/features/invariant_features.py
==================================
Spatially invariant relative kinematic feature extraction for ball possession tracking.
Completely decoupled from absolute pitch coordinates (X, Y, centroid positions, spreads).
"""

import numpy as np


class InvariantFeatureExtractor:
    """
    Extracts pure relative ball-player kinematic features.
    Guaranteed invariant under pitch translations, rotations, and wing shifts.
    """

    def __init__(self, team_of_track, pitch_length=105.3, pitch_width=68.0):
        self.team_of_track = np.asarray(team_of_track)
        self.pitch_length = float(pitch_length)
        self.pitch_width = float(pitch_width)
        self.H_idx = np.where(self.team_of_track == 0)[0]
        self.A_idx = np.where(self.team_of_track == 1)[0]

    def extract(self, player_positions_m, player_speed_ms, ball_xy_m, ball_speed_ms=None, ball_accel_ms2=None):
        """
        Extract invariant relative features.

        Parameters
        ----------
        player_positions_m : (N, 46, 2) float array
            Player (x, y) coordinates in metres.
        player_speed_ms : (N, 46) float array
            Player instantaneous speeds in m/s.
        ball_xy_m : (N, 2) float array
            Ball (x, y) coordinates in metres.
        ball_speed_ms : (N,) float array, optional
        ball_accel_ms2 : (N,) float array, optional

        Returns
        -------
        X_inv : (N, 32) float32 array
            Matrix of coordinate-agnostic invariant relative features.
        """
        N = len(player_positions_m)
        bx = np.nan_to_num(ball_xy_m[:, 0], nan=self.pitch_length / 2.0)
        by = np.nan_to_num(ball_xy_m[:, 1], nan=self.pitch_width / 2.0)

        # Ball kinematics
        if ball_speed_ms is None:
            v_bx = np.diff(bx, prepend=bx[0]) / 0.04
            v_by = np.diff(by, prepend=by[0]) / 0.04
            ball_speed_ms = np.sqrt(v_bx**2 + v_by**2)
        else:
            v_bx = np.diff(bx, prepend=bx[0]) / 0.04
            v_by = np.diff(by, prepend=by[0]) / 0.04

        if ball_accel_ms2 is None:
            ball_accel_ms2 = np.diff(ball_speed_ms, prepend=ball_speed_ms[0]) / 0.04

        px = np.nan_to_num(player_positions_m[:, :, 0], nan=self.pitch_length / 2.0)
        py = np.nan_to_num(player_positions_m[:, :, 1], nan=self.pitch_width / 2.0)
        pv = np.nan_to_num(player_speed_ms, nan=0.0)

        # Relative displacements to ball: (N, 46)
        dx_ball = px - bx[:, None]
        dy_ball = py - by[:, None]
        d_ball = np.sqrt(dx_ball**2 + dy_ball**2)

        # 1. Sorted team distances
        dH_sorted = np.sort(d_ball[:, self.H_idx], axis=1)
        dA_sorted = np.sort(d_ball[:, self.A_idx], axis=1)

        dH1, dH2, dH3 = dH_sorted[:, 0], dH_sorted[:, 1], dH_sorted[:, 2]
        dA1, dA2, dA3 = dA_sorted[:, 0], dA_sorted[:, 1], dA_sorted[:, 2]

        # 2. Differential proximity
        delta_d1 = dA1 - dH1
        delta_d2 = dA2 - dH2
        delta_d3 = dA3 - dH3
        ratio_d1 = np.clip(dA1 / (dH1 + 0.1), 0.05, 20.0)

        # 3. Nearest player speeds
        arg_H1 = self.H_idx[np.argmin(d_ball[:, self.H_idx], axis=1)]
        arg_A1 = self.A_idx[np.argmin(d_ball[:, self.A_idx], axis=1)]
        arg_H2 = self.H_idx[np.argsort(d_ball[:, self.H_idx], axis=1)[:, 1]]
        arg_A2 = self.A_idx[np.argsort(d_ball[:, self.A_idx], axis=1)[:, 1]]

        row_idx = np.arange(N)
        v_H1 = pv[row_idx, arg_H1]
        v_A1 = pv[row_idx, arg_A1]
        v_H2 = pv[row_idx, arg_H2]
        v_A2 = pv[row_idx, arg_A2]
        delta_v1 = v_H1 - v_A1

        # 4. Heading & Velocity Alignment with ball
        cos_H1 = (dx_ball[row_idx, arg_H1] * v_bx + dy_ball[row_idx, arg_H1] * v_by) / ((dH1 + 0.1) * (ball_speed_ms + 0.1))
        cos_A1 = (dx_ball[row_idx, arg_A1] * v_bx + dy_ball[row_idx, arg_A1] * v_by) / ((dA1 + 0.1) * (ball_speed_ms + 0.1))
        delta_cos = cos_H1 - cos_A1

        # 5. Time-to-Intercept (TTI)
        tti_H1 = dH1 / (v_H1 + 1.0)
        tti_A1 = dA1 / (v_A1 + 1.0)
        tti_H2 = dH2 / (v_H2 + 1.0)
        tti_A2 = dA2 / (v_A2 + 1.0)
        delta_tti = tti_A1 - tti_H1

        # 6. Local Pitch Control at Ball Locus
        pc_H = np.sum(np.exp(-d_ball[:, self.H_idx]**2 / 18.0), axis=1)
        pc_A = np.sum(np.exp(-d_ball[:, self.A_idx]**2 / 18.0), axis=1)
        pc_diff = pc_H - pc_A
        pc_ratio = np.clip(pc_H / (pc_A + 1e-4), 0.05, 20.0)

        # 7. Local Pressure between nearest player and opponent team
        pos_H1 = np.stack([px[row_idx, arg_H1], py[row_idx, arg_H1]], axis=1)
        pos_A1 = np.stack([px[row_idx, arg_A1], py[row_idx, arg_A1]], axis=1)

        d_H1_to_A = np.sqrt((px[:, self.A_idx] - pos_H1[:, [0]])**2 + (py[:, self.A_idx] - pos_H1[:, [1]])**2)
        d_A1_to_H = np.sqrt((px[:, self.H_idx] - pos_A1[:, [0]])**2 + (py[:, self.H_idx] - pos_A1[:, [1]])**2)

        press_H1 = np.sum(np.exp(-d_H1_to_A**2 / 8.0), axis=1)
        press_A1 = np.sum(np.exp(-d_A1_to_H**2 / 8.0), axis=1)
        delta_press = press_A1 - press_H1

        # 8. Distance to pitch boundaries (State feature)
        dist_x_bnd = np.minimum(bx, self.pitch_length - bx)
        dist_y_bnd = np.minimum(by, self.pitch_width - by)
        dist_to_boundary = np.minimum(dist_x_bnd, dist_y_bnd)

        # Assemble full 32-column invariant matrix
        features = [
            dH1, dH2, dH3, dA1, dA2, dA3,
            delta_d1, delta_d2, delta_d3, ratio_d1,
            v_H1, v_A1, v_H2, v_A2, delta_v1,
            cos_H1, cos_A1, delta_cos,
            tti_H1, tti_A1, tti_H2, tti_A2, delta_tti,
            pc_H, pc_A, pc_diff, pc_ratio,
            press_H1, press_A1, delta_press,
            ball_speed_ms, dist_to_boundary,
        ]

        X_inv = np.column_stack(features).astype(np.float32)
        return X_inv
