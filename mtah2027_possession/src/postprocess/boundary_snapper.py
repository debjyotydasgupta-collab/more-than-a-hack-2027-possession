"""
boundary_snapper.py
===================
Physics-driven transition boundary snapper for football ball possession tracking.
Snaps decoded sequence transitions to exact kinematic touch impulses and pitch boundaries,
eliminating transition latency and maximizing spellF1 / LOSO temporal accuracy.
"""

import numpy as np
import pandas as pd

class BoundaryImpulseSnapper:
    def __init__(self, dt: float = 0.04, search_window: int = 6, cusum_window: int = 30,
                 accel_thresh: float = 18.0, pitch_length: float = 105.3, pitch_width: float = 68.0):
        """
        Parameters
        ----------
        dt : float
            Sampling time interval (0.04s for 25 FPS).
        search_window : int
            Window radius in frames around decoded transitions for impulse search (default: 6 frames = 0.24s).
        cusum_window : int
            Window radius in frames for CUSUM log-likelihood ratio step changepoint search (default: 30 frames = 1.2s).
        accel_thresh : float
            Ball acceleration threshold in m/s^2 indicating physical impact/kick.
        pitch_length, pitch_width : float
            Pitch dimensions in meters.
        """
        self.dt = dt
        self.search_window = search_window
        self.cusum_window = cusum_window
        self.accel_thresh = accel_thresh
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    def refine_boundary_cusum(self, llr_arr: np.ndarray, tau0: int, old_is_H: bool, w: int = 30) -> int:
        """
        Retrospective single-changepoint maximum likelihood estimation (CUSUM style).
        
        Searches tau in [tau0 - w, tau0 + w] maximizing the cumulative log-likelihood step:
            sum_{t < tau} s * llr_t - sum_{t >= tau} s * llr_t
        where s = +1 if old state is H else -1.
        Computed in O(1) per candidate via prefix sums.
        """
        s = 1.0 if old_is_H else -1.0
        a = max(0, tau0 - 2 * w)
        b = min(len(llr_arr), tau0 + 2 * w + 1)
        seg = s * llr_arr[a:b]
        c = np.concatenate([[0.0], np.cumsum(seg)])
        taus_rel = np.arange(max(a, tau0 - w), min(b - 1, tau0 + w + 1)) - a
        if len(taus_rel) == 0:
            return tau0
        scores = c[taus_rel] - (c[-1] - c[taus_rel])
        return int(taus_rel[scores.argmax()] + a)

    def snap_boundaries(self, labels: np.ndarray, ball_xy: np.ndarray, 
                        players_xy: np.ndarray, team_of_track: np.ndarray,
                        llr: np.ndarray = None) -> np.ndarray:
        """
        Snap transition boundaries in `labels` to exact physical impulse events and CUSUM MLE changepoints.

        Parameters
        ----------
        labels : np.ndarray of shape (N,)
            Decoded labels ('H', 'A', 'D', 'X' or integers 0, 1, 2, 3).
        ball_xy : np.ndarray of shape (N, 2)
            Ball coordinates in meters.
        players_xy : np.ndarray of shape (N, 46, 2)
            Player coordinates in meters.
        team_of_track : np.ndarray of shape (46,)
            Team of each track (0 for Home, 1 for Away).
        llr : np.ndarray of shape (N,), optional
            Raw ensemble log-likelihood ratio log p(H) - log p(A) for CUSUM MLE refinement.

        Returns
        -------
        snapped_labels : np.ndarray of shape (N,)
            Labels with transitions snapped to exact physical touch frames.
        """
        N = len(labels)
        snapped = labels.copy()

        # Compute ball kinematics with robust NaN handling (zero acceleration across missing frames)
        valid_b = ~np.isnan(ball_xy[:, 0]) & ~np.isnan(ball_xy[:, 1])
        bx_raw = np.nan_to_num(ball_xy[:, 0], nan=0.0)
        by_raw = np.nan_to_num(ball_xy[:, 1], nan=0.0)
        
        v_ball_xy = np.zeros((N, 2), dtype=np.float64)
        for i in range(1, N):
            if valid_b[i] and valid_b[i - 1]:
                v_ball_xy[i, 0] = (bx_raw[i] - bx_raw[i - 1]) / self.dt
                v_ball_xy[i, 1] = (by_raw[i] - by_raw[i - 1]) / self.dt
        v_ball = np.sqrt(v_ball_xy[:, 0]**2 + v_ball_xy[:, 1]**2)

        a_ball = np.zeros(N, dtype=np.float64)
        for i in range(1, N):
            if valid_b[i] and valid_b[i - 1] and valid_b[i - 2]:
                ax = (v_ball_xy[i, 0] - v_ball_xy[i - 1, 0]) / self.dt
                ay = (v_ball_xy[i, 1] - v_ball_xy[i - 1, 1]) / self.dt
                a_ball[i] = np.sqrt(ax**2 + ay**2)

        bx = np.nan_to_num(ball_xy[:, 0], nan=self.pitch_length / 2.0)
        by = np.nan_to_num(ball_xy[:, 1], nan=self.pitch_width / 2.0)

        # Compute nearest player distances
        H_mask = (team_of_track == 0)
        A_mask = (team_of_track == 1)
        
        px = np.nan_to_num(players_xy[:, :, 0], nan=self.pitch_length / 2.0)
        py = np.nan_to_num(players_xy[:, :, 1], nan=self.pitch_width / 2.0)
        
        d_all = np.sqrt((px - bx[:, None])**2 + (py - by[:, None])**2)
        dH = np.min(d_all[:, H_mask], axis=1)
        dA = np.min(d_all[:, A_mask], axis=1)

        # Distance to boundaries
        dist_x = np.minimum(bx, self.pitch_length - bx)
        dist_y = np.minimum(by, self.pitch_width - by)
        dist_bnd = np.minimum(dist_x, dist_y)

        # Find transition boundaries
        transitions = []
        for t in range(1, N):
            if snapped[t] != snapped[t - 1]:
                transitions.append((t, snapped[t - 1], snapped[t]))

        # Process each transition
        for t_trans, l_prev, l_next in transitions:
            # CASE 1: Turnover between teams (H <-> A)
            if (l_prev in ('H', 0) and l_next in ('A', 1)) or (l_prev in ('A', 1) and l_next in ('H', 0)):
                target_team = 'A' if l_next in ('A', 1) else 'H'
                d_target = dA if target_team == 'A' else dH
                old_is_H = (l_prev in ('H', 0))

                # Step 1: Retrospective CUSUM log-likelihood changepoint MLE
                if llr is not None:
                    tau_anchor = self.refine_boundary_cusum(llr, t_trans, old_is_H=old_is_H, w=self.cusum_window)
                else:
                    tau_anchor = t_trans

                # Step 2: Search for physical touch impact peak in search window around tau_anchor
                w_start = max(1, tau_anchor - self.search_window)
                w_end = min(N - 1, tau_anchor + self.search_window + 1)
                
                acc_win = a_ball[w_start:w_end]
                d_win = d_target[w_start:w_end]
                
                candidates = []
                for idx_rel, t_cand in enumerate(range(w_start, w_end)):
                    if d_win[idx_rel] < 1.4 and acc_win[idx_rel] >= self.accel_thresh:
                        score = acc_win[idx_rel] - 5.0 * d_win[idx_rel]
                        candidates.append((score, t_cand))
                
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    best_t = candidates[0][1]
                else:
                    best_t = tau_anchor
                    
                # Snap transition boundary from original t_trans to best_t
                if best_t < t_trans:
                    snapped[best_t:t_trans] = l_next
                elif best_t > t_trans:
                    snapped[t_trans:best_t] = l_prev

            # CASE 2: Live to Dead Ball (H/A -> D)
            elif l_next in ('D', 2):
                w_start = max(1, t_trans - self.search_window)
                w_end = min(N - 1, t_trans + self.search_window + 1)
                bnd_win = dist_bnd[w_start:w_end]
                out_frames = [w_start + i for i, db in enumerate(bnd_win) if db < 0.05]
                if out_frames:
                    first_out = out_frames[0]
                    if first_out < t_trans:
                        snapped[first_out:t_trans] = l_next
                    elif first_out > t_trans:
                        snapped[t_trans:first_out] = l_prev

            # CASE 3: Dead Ball to Live (D -> H/A)
            elif l_prev in ('D', 2) and l_next in ('H', 'A', 0, 1):
                w_start = max(1, t_trans - self.search_window)
                w_end = min(N - 1, t_trans + self.search_window + 1)
                v_win = v_ball[w_start:w_end]
                acc_win = a_ball[w_start:w_end]
                kick_frames = [w_start + i for i in range(len(v_win)) if acc_win[i] >= self.accel_thresh and v_win[i] > 2.0]
                if kick_frames:
                    best_kick = kick_frames[0]
                    if best_kick < t_trans:
                        snapped[best_kick:t_trans] = l_next
                    elif best_kick > t_trans:
                        snapped[t_trans:best_kick] = l_prev

        return snapped

