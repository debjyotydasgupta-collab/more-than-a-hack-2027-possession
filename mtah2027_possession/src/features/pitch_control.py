"""
src/features/pitch_control.py
=============================
20-dimensional pitch control model features based on the Spearman/Fernandez-Bornn
time-to-intercept (TTI) physics framework.
Vectorized implementation for ultra-fast processing across 199,319 frames.

Features:
1. Team Pitch Control at ball position: PC_H, PC_A, PC_diff, PC_ratio (4 features)
2. Time-to-Intercept (TTI) of nearest player to ball: Home, Away, diff (3 features)
3. Average TTI of nearest 3 players: Home, Away (2 features)
4. Sigmoidal control probability sum: P_sum_H, P_sum_A (2 features)
5. Spatial control entropy around the ball (1 feature)
6. Pitch Control at 5m ahead of ball trajectory: PC_H_ahead, PC_A_ahead (2 features)
7. Dynamic control dominance flag: 1 (H), 0 (Dead/Contested), -1 (A) (1 feature)
8. Pitch Control gradient: difference between current and ahead control (2 features)
9. Pitch control transition velocity / acceleration (3 features)

Total: 4 + 3 + 2 + 2 + 1 + 2 + 1 + 2 + 3 = 20 features
"""

import numpy as np

N_CONTROL_FEATS = 20
HOME = 0
AWAY = 1

V_MAX = 8.5            # Maximum sprinting speed (m/s)
T_REACTION = 0.5       # Reaction latency (s)
LAMBDA_CONTROL = 4.3   # Scaling parameter for sigmoidal intercept probability


class PitchControlFeatureExtractor:
    """Pitch control probability field feature extractor."""

    def __init__(self, team_of_track, pitch_length=105.3, pitch_width=68.0):
        self.team_of_track = np.asarray(team_of_track, dtype=np.int8)
        self.pitch_length = float(pitch_length)
        self.pitch_width = float(pitch_width)

        self.home_mask = (self.team_of_track == HOME)
        self.away_mask = (self.team_of_track == AWAY)

    def extract(self, frames_m, frames_speed_ms, ball_xy_m_smooth):
        """Extract pitch control features across all frames using vectorized NumPy.

        Parameters
        ----------
        frames_m : (N, 46, 2) player positions in metres
        frames_speed_ms : (N, 46) player speed in m/s
        ball_xy_m_smooth : (N, 2) smoothed ball positions in metres
        """
        N = frames_m.shape[0]
        X = np.zeros((N, N_CONTROL_FEATS), dtype=np.float32)

        bx = ball_xy_m_smooth[:, 0]
        by = ball_xy_m_smooth[:, 1]
        ball_valid = ~np.isnan(bx)

        # Distances to ball for all players: (N, 46)
        dx = frames_m[:, :, 0] - bx[:, None]
        dy = frames_m[:, :, 1] - by[:, None]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        dist[np.isnan(frames_m[:, :, 0])] = np.nan

        # TTI: T_REACTION + max(0, dist - T_REACTION * speed) / V_MAX
        speed_safe = np.nan_to_num(frames_speed_ms, nan=0.0)
        eff_dist = np.maximum(0.0, dist - T_REACTION * speed_safe)
        tti = T_REACTION + eff_dist / V_MAX
        tti[np.isnan(dist)] = np.nan

        tti_H = tti[:, self.home_mask]  # (N, N_H)
        tti_A = tti[:, self.away_mask]  # (N, N_A)

        # Sort TTIs
        sorted_H = np.sort(np.nan_to_num(tti_H, nan=99.0), axis=1)
        sorted_A = np.sort(np.nan_to_num(tti_A, nan=99.0), axis=1)

        min_tti_h = sorted_H[:, 0]
        min_tti_a = sorted_A[:, 0]
        tti_diff = min_tti_a - min_tti_h

        avg_top3_h = np.mean(sorted_H[:, :3], axis=1)
        avg_top3_a = np.mean(sorted_A[:, :3], axis=1)

        # Sigmoid control probabilities
        prob_H = 1.0 / (1.0 + np.exp(LAMBDA_CONTROL * (tti_H - 0.0)))
        prob_A = 1.0 / (1.0 + np.exp(LAMBDA_CONTROL * (tti_A - 0.0)))

        sum_prob_h = np.nansum(prob_H, axis=1)
        sum_prob_a = np.nansum(prob_A, axis=1)
        tot_prob = sum_prob_h + sum_prob_a + 1e-6

        pc_h = sum_prob_h / tot_prob
        pc_a = sum_prob_a / tot_prob
        pc_diff = pc_h - pc_a
        pc_ratio = pc_h / (pc_a + 1e-4)

        # Default to neutral on ball missing
        pc_h[~ball_valid] = 0.5
        pc_a[~ball_valid] = 0.5
        pc_diff[~ball_valid] = 0.0
        pc_ratio[~ball_valid] = 1.0

        # Features 0..3: PC at ball
        X[:, 0] = pc_h
        X[:, 1] = pc_a
        X[:, 2] = pc_diff
        X[:, 3] = pc_ratio

        # Features 4..6: Nearest TTIs
        X[:, 4] = np.where(ball_valid, min_tti_h, 99.0)
        X[:, 5] = np.where(ball_valid, min_tti_a, 99.0)
        X[:, 6] = np.where(ball_valid, tti_diff, 0.0)

        # Features 7..8: Top-3 average TTIs
        X[:, 7] = np.where(ball_valid, avg_top3_h, 99.0)
        X[:, 8] = np.where(ball_valid, avg_top3_a, 99.0)

        # Features 9..10: Probability sums
        X[:, 9] = sum_prob_h
        X[:, 10] = sum_prob_a

        # Feature 11: Control entropy
        p1 = np.clip(pc_h, 1e-4, 1.0 - 1e-4)
        p2 = 1.0 - p1
        entropy = - (p1 * np.log2(p1) + p2 * np.log2(p2))
        X[:, 11] = entropy

        # Ball trajectory ahead (5m)
        ball_diff = np.diff(ball_xy_m_smooth, axis=0, prepend=ball_xy_m_smooth[[0]])
        ball_speed = np.sqrt(ball_diff[:, 0] ** 2 + ball_diff[:, 1] ** 2)
        ball_dir = np.nan_to_num(ball_diff / (ball_speed[:, None] + 1e-6), nan=0.0)
        ahead_xy = ball_xy_m_smooth + 5.0 * ball_dir

        dx_ahead = frames_m[:, :, 0] - ahead_xy[:, 0:1]
        dy_ahead = frames_m[:, :, 1] - ahead_xy[:, 1:2]
        dist_ahead = np.sqrt(dx_ahead ** 2 + dy_ahead ** 2)
        eff_dist_ahead = np.maximum(0.0, dist_ahead - T_REACTION * speed_safe)
        tti_ahead = T_REACTION + eff_dist_ahead / V_MAX

        prob_H_ahead = 1.0 / (1.0 + np.exp(LAMBDA_CONTROL * (tti_ahead[:, self.home_mask] - 0.0)))
        prob_A_ahead = 1.0 / (1.0 + np.exp(LAMBDA_CONTROL * (tti_ahead[:, self.away_mask] - 0.0)))
        sum_h_ahead = np.nansum(prob_H_ahead, axis=1)
        sum_a_ahead = np.nansum(prob_A_ahead, axis=1)
        tot_ahead = sum_h_ahead + sum_a_ahead + 1e-6

        pc_h_ahead = np.where(ball_valid, sum_h_ahead / tot_ahead, 0.5)
        pc_a_ahead = np.where(ball_valid, sum_a_ahead / tot_ahead, 0.5)

        X[:, 12] = pc_h_ahead
        X[:, 13] = pc_a_ahead

        # Dominance flag
        dom = np.zeros(N, dtype=np.float32)
        dom[pc_h > 0.65] = 1.0
        dom[pc_a > 0.65] = -1.0
        X[:, 14] = dom

        # Control gradient ahead
        X[:, 15] = pc_h_ahead - pc_h
        X[:, 16] = pc_a_ahead - pc_a

        # Control velocity & dynamics
        d_pc = np.diff(pc_h, prepend=pc_h[0]) / 0.04
        acc_pc = np.diff(d_pc, prepend=d_pc[0]) / 0.04
        X[:, 17] = d_pc
        X[:, 18] = acc_pc
        X[:, 19] = np.abs(d_pc)

        return np.nan_to_num(X, nan=0.0)
