"""
src/features/spatial_features.py
================================
40-dimensional spatial-geometric and team formation features.
Highly optimized and vectorized for sub-second processing across 199,319 frames.

Features:
1. Bounding box & spatial coverage areas: Home, Away, ratio, difference (4 features)
2. Team centroids (X, Y) and distance to ball: Home, Away (6 features)
3. Team spatial variance / compactness (spread from centroid) (2 features)
4. Inter-team centroid distance and relative vector (3 features)
5. Voronoi dominance (nearest player team) and local cell area estimate (2 features)
6. Outfield player densities in 5m / 10m / 20m radii around the ball (4 features)
7. Ball pitch location: X_norm, Y_norm, distance to goals, penalty box flags (7 features)
8. Length & width of team bounding boxes: Home, Away (4 features)
9. Number of active players on pitch: Home, Away (2 features)
10. Centroid movement velocity / dynamics (6 features)

Total: 4 + 6 + 2 + 3 + 2 + 4 + 7 + 4 + 2 + 6 = 40 features
"""

import numpy as np

N_SPATIAL_FEATS = 40
HOME = 0
AWAY = 1


class SpatialFeatureExtractor:
    """Extract 40-dimensional spatial and geometric features per frame."""

    def __init__(self, team_of_track, pitch_length=105.3, pitch_width=68.0):
        self.team_of_track = np.asarray(team_of_track, dtype=np.int8)
        self.pitch_length = float(pitch_length)
        self.pitch_width = float(pitch_width)

        self.home_mask = (self.team_of_track == HOME)
        self.away_mask = (self.team_of_track == AWAY)

    def extract(self, frames_m, ball_xy_m_smooth):
        """Extract spatial features across all frames with vectorized NumPy operations.

        Parameters
        ----------
        frames_m : (N, 46, 2) player positions in metres
        ball_xy_m_smooth : (N, 2) smoothed ball positions in metres
        """
        N = frames_m.shape[0]
        X = np.zeros((N, N_SPATIAL_FEATS), dtype=np.float32)

        # Slice teams
        H_pts = frames_m[:, self.home_mask, :]  # (N, N_H, 2)
        A_pts = frames_m[:, self.away_mask, :]  # (N, N_A, 2)

        # 1 & 8. Bounding box & Hull area proxy
        h_xmin = np.nanmin(H_pts[:, :, 0], axis=1)
        h_xmax = np.nanmax(H_pts[:, :, 0], axis=1)
        h_ymin = np.nanmin(H_pts[:, :, 1], axis=1)
        h_ymax = np.nanmax(H_pts[:, :, 1], axis=1)
        h_dx = np.maximum(0.0, h_xmax - h_xmin)
        h_dy = np.maximum(0.0, h_ymax - h_ymin)
        h_area = h_dx * h_dy * 0.65  # Convex hull of football team is typically ~65% of bounding box

        a_xmin = np.nanmin(A_pts[:, :, 0], axis=1)
        a_xmax = np.nanmax(A_pts[:, :, 0], axis=1)
        a_ymin = np.nanmin(A_pts[:, :, 1], axis=1)
        a_ymax = np.nanmax(A_pts[:, :, 1], axis=1)
        a_dx = np.maximum(0.0, a_xmax - a_xmin)
        a_dy = np.maximum(0.0, a_ymax - a_ymin)
        a_area = a_dx * a_dy * 0.65

        hull_ratio = h_area / (a_area + 1e-4)
        hull_diff = h_area - a_area

        X[:, 0] = h_area
        X[:, 1] = a_area
        X[:, 2] = hull_ratio
        X[:, 3] = hull_diff

        # 2 & 3. Centroids and spreads
        h_cx = np.nanmean(H_pts[:, :, 0], axis=1)
        h_cy = np.nanmean(H_pts[:, :, 1], axis=1)
        a_cx = np.nanmean(A_pts[:, :, 0], axis=1)
        a_cy = np.nanmean(A_pts[:, :, 1], axis=1)

        # Ball positions
        bx = ball_xy_m_smooth[:, 0]
        by = ball_xy_m_smooth[:, 1]
        ball_valid = ~np.isnan(bx)

        h_dist_b = np.where(ball_valid, np.sqrt((h_cx - bx) ** 2 + (h_cy - by) ** 2), 999.0)
        a_dist_b = np.where(ball_valid, np.sqrt((a_cx - bx) ** 2 + (a_cy - by) ** 2), 999.0)

        X[:, 4] = h_cx
        X[:, 5] = h_cy
        X[:, 6] = h_dist_b
        X[:, 7] = a_cx
        X[:, 8] = a_cy
        X[:, 9] = a_dist_b

        # Compactness / spatial spread
        h_spread = np.nanmean(np.sqrt((H_pts[:, :, 0] - h_cx[:, None]) ** 2 + (H_pts[:, :, 1] - h_cy[:, None]) ** 2), axis=1)
        a_spread = np.nanmean(np.sqrt((A_pts[:, :, 0] - a_cx[:, None]) ** 2 + (A_pts[:, :, 1] - a_cy[:, None]) ** 2), axis=1)
        X[:, 10] = h_spread
        X[:, 11] = a_spread

        # 4. Inter-team centroid distance and relative vector
        cent_dx = a_cx - h_cx
        cent_dy = a_cy - h_cy
        cent_dist = np.sqrt(cent_dx ** 2 + cent_dy ** 2)
        X[:, 12] = cent_dist
        X[:, 13] = cent_dx
        X[:, 14] = cent_dy

        # 5. Voronoi Dominance & Area Estimate
        # The Voronoi polygon enclosing the ball strictly belongs to the nearest player!
        # Distance from ball to all players
        diff_all = frames_m[:, :, :2] - ball_xy_m_smooth[:, None, :2]  # (N, 46, 2)
        dist_all = np.sqrt(np.sum(diff_all ** 2, axis=-1))  # (N, 46)
        dist_all_nan = np.where(np.isnan(frames_m[:, :, 0]), np.inf, dist_all)

        nearest_player_idx = np.argmin(dist_all_nan, axis=1)  # (N,)
        voro_home_dom = np.where(self.team_of_track[nearest_player_idx] == HOME, 1.0, 0.0)
        voro_home_dom[~ball_valid] = 0.5

        # Voronoi cell area estimated from second-nearest player distance
        sorted_dists = np.sort(dist_all_nan, axis=1)  # (N, 46)
        d2 = sorted_dists[:, 1]
        voro_area_est = np.pi * ((d2 * 0.5) ** 2)
        voro_area_est[~ball_valid | (d2 > 50.0)] = 50.0

        X[:, 15] = voro_area_est
        X[:, 16] = voro_home_dom

        # 6. Player densities around ball
        dists_H = np.sqrt((H_pts[:, :, 0] - bx[:, None]) ** 2 + (H_pts[:, :, 1] - by[:, None]) ** 2)
        dists_A = np.sqrt((A_pts[:, :, 0] - bx[:, None]) ** 2 + (A_pts[:, :, 1] - by[:, None]) ** 2)

        dens_5m = np.sum(dists_H <= 5.0, axis=1) + np.sum(dists_A <= 5.0, axis=1)
        dens_10m = np.sum(dists_H <= 10.0, axis=1) + np.sum(dists_A <= 10.0, axis=1)
        dens_20m = np.sum(dists_H <= 20.0, axis=1) + np.sum(dists_A <= 20.0, axis=1)

        h_10m = np.sum(dists_H <= 10.0, axis=1)
        a_10m = np.sum(dists_A <= 10.0, axis=1)
        ratio_10m = (h_10m + 0.1) / (a_10m + 0.1)

        X[:, 17] = np.where(ball_valid, dens_5m, 0.0)
        X[:, 18] = np.where(ball_valid, dens_10m, 0.0)
        X[:, 19] = np.where(ball_valid, dens_20m, 0.0)
        X[:, 20] = np.where(ball_valid, ratio_10m, 1.0)

        # 7. Ball pitch location & flags
        bx_safe = np.nan_to_num(bx, nan=self.pitch_length * 0.5)
        by_safe = np.nan_to_num(by, nan=self.pitch_width * 0.5)
        bx_norm = bx_safe / self.pitch_length
        by_norm = by_safe / self.pitch_width
        dist_home_goal = np.sqrt(bx_safe ** 2 + (by_safe - self.pitch_width * 0.5) ** 2)
        dist_away_goal = np.sqrt((self.pitch_length - bx_safe) ** 2 + (by_safe - self.pitch_width * 0.5) ** 2)

        in_home_box = np.where((bx_safe <= 16.5) & (by_safe >= 13.85) & (by_safe <= 54.15), 1.0, 0.0)
        in_away_box = np.where((bx_safe >= self.pitch_length - 16.5) & (by_safe >= 13.85) & (by_safe <= 54.15), 1.0, 0.0)
        in_final_third = np.where((bx_safe >= 70.0) | (bx_safe <= 35.0), 1.0, 0.0)

        X[:, 21] = bx_norm
        X[:, 22] = by_norm
        X[:, 23] = dist_home_goal
        X[:, 24] = dist_away_goal
        X[:, 25] = in_home_box
        X[:, 26] = in_away_box
        X[:, 27] = in_final_third

        # 8. Team bounding box dimensions
        X[:, 28] = h_dx
        X[:, 29] = h_dy
        X[:, 30] = a_dx
        X[:, 31] = a_dy

        # 9. Active players
        h_active = np.sum(~np.isnan(H_pts[:, :, 0]), axis=1)
        a_active = np.sum(~np.isnan(A_pts[:, :, 0]), axis=1)
        X[:, 32] = h_active
        X[:, 33] = a_active

        # 10. Centroid dynamics (diff / 0.04)
        h_cspeed_x = np.diff(h_cx, prepend=h_cx[0]) / 0.04
        h_cspeed_y = np.diff(h_cy, prepend=h_cy[0]) / 0.04
        h_cspeed = np.sqrt(h_cspeed_x ** 2 + h_cspeed_y ** 2)

        a_cspeed_x = np.diff(a_cx, prepend=a_cx[0]) / 0.04
        a_cspeed_y = np.diff(a_cy, prepend=a_cy[0]) / 0.04
        a_cspeed = np.sqrt(a_cspeed_x ** 2 + a_cspeed_y ** 2)

        X[:, 34] = h_cspeed_x
        X[:, 35] = h_cspeed_y
        X[:, 36] = h_cspeed
        X[:, 37] = a_cspeed_x
        X[:, 38] = a_cspeed_y
        X[:, 39] = a_cspeed

        return np.nan_to_num(X, nan=0.0)
