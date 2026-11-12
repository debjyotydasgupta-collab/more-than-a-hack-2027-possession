"""
src/preprocessing/kalman_filter.py
===================================
2D kinematic Kalman filter + RTS smoother for the ball trajectory.

State vector: [x, y, vx, vy, ax, ay]  (positions & velocities in metres,
acceleration in m/s²).

The raw ATD ball track has ~41% missing frames.  The Kalman-RTS pair
interpolates short gaps and suppresses measurement noise.  For gaps longer
than MAX_GAP_FRAMES the smoother sets the output to NaN — those regions
are genuinely unknown and should be labelled "D" or "X" by later stages.

Usage
-----
    from src.preprocessing.kalman_filter import RTSBallSmoother
    smoother = RTSBallSmoother()
    ball_xy_m = smoother.smooth_ball_trajectory(frames)   # (N, 2) metres
"""

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES  # noqa: E402

# Default constants (overridable via config.yaml)
_DEFAULT_Q_NOISE       = 0.005    # process noise (position component)
_DEFAULT_R_NOISE       = 0.5     # measurement noise (m²)
_DEFAULT_MAX_SPEED     = 38.0    # hard speed cap (m/s)
_DEFAULT_MAX_GAP       = 125     # frames; longer → NaN output
_DT                    = 0.04    # 1/25 fps


class RTSBallSmoother:
    """Forward Kalman + backward RTS smoother for the ball track.

    Parameters
    ----------
    q_noise      : float  Process noise magnitude (positions, m)
    r_noise      : float  Measurement noise variance (m²)
    max_speed    : float  Hard speed cap (m/s)
    max_gap      : int    Gaps longer than this are left as NaN
    dt           : float  Time step in seconds (default 1/25 fps = 0.04 s)
    game         : str    Key into games.GAMES for pitch dimensions
    """

    def __init__(self, q_noise=_DEFAULT_Q_NOISE, r_noise=_DEFAULT_R_NOISE,
                 max_speed=_DEFAULT_MAX_SPEED, max_gap=_DEFAULT_MAX_GAP,
                 dt=_DT, game=GAME):
        self.q_noise   = float(q_noise)
        self.r_noise   = float(r_noise)
        self.max_speed = float(max_speed)
        self.max_gap   = int(max_gap)
        self.dt        = float(dt)
        self.game      = game

        length, width = GAMES[game]["pitch"]
        self._pitch_length = float(length)
        self._pitch_width  = float(width)

        # Build system matrices once
        self._F = self._build_F(dt)
        self._H = self._build_H()
        self._Q = self._build_Q(q_noise, dt)
        self._R = self._build_R(r_noise)

    # ------------------------------------------------------------------
    # Matrix builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_F(dt):
        """6×6 constant-acceleration state transition matrix."""
        dt2 = 0.5 * dt * dt
        F = np.eye(6, dtype=np.float64)
        # x  y  vx  vy  ax  ay
        F[0, 2] = dt;  F[0, 4] = dt2
        F[1, 3] = dt;  F[1, 5] = dt2
        F[2, 4] = dt
        F[3, 5] = dt
        return F

    @staticmethod
    def _build_H():
        """2×6 observation matrix: we observe (x, y) only."""
        H = np.zeros((2, 6), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        return H

    @staticmethod
    def _build_Q(q, dt):
        """6×6 process noise covariance (discrete Wiener model)."""
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        # Block structure for one axis: [pos, vel, acc] with q scaling
        q_block = np.array([
            [dt4 / 4, dt3 / 2, dt2 / 2],
            [dt3 / 2, dt2,     dt     ],
            [dt2 / 2, dt,      1.0    ],
        ], dtype=np.float64) * q
        Q = np.zeros((6, 6), dtype=np.float64)
        Q[np.ix_([0, 2, 4], [0, 2, 4])] = q_block
        Q[np.ix_([1, 3, 5], [1, 3, 5])] = q_block
        return Q

    @staticmethod
    def _build_R(r):
        """2×2 measurement noise covariance."""
        return np.eye(2, dtype=np.float64) * r

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(self, meas, observed):
        """Kalman forward pass.

        Parameters
        ----------
        meas     : (N, 2) float64  observed (x, y) in metres; arbitrary on ~observed frames
        observed : (N,) bool       True where the measurement is valid

        Returns
        -------
        xs : (N, 6)   filtered state means
        Ps : (N, 6, 6) filtered state covariances
        """
        N  = len(meas)
        F, H, Q, R = self._F, self._H, self._Q, self._R

        xs = np.zeros((N, 6), dtype=np.float64)
        Ps = np.zeros((N, 6, 6), dtype=np.float64)

        # Initialise from first observed measurement
        first_obs = int(np.argmax(observed)) if observed.any() else 0
        x  = np.zeros(6, dtype=np.float64)
        if observed.any():
            x[0], x[1] = meas[first_obs]
        P  = np.eye(6, dtype=np.float64) * 100.0

        for i in range(N):
            # Predict
            x = F @ x
            P = F @ P @ F.T + Q

            # Update (only on observed frames)
            if observed[i]:
                y = meas[i] - H @ x           # innovation
                S = H @ P @ H.T + R
                PHt = P @ H.T
                K = np.linalg.solve(S, PHt.T).T  # Kalman gain (6, 2)
                x = x + K @ y
                P = (np.eye(6) - K @ H) @ P

                # Hard speed cap (project velocity onto sphere of radius max_speed)
                vspeed = np.sqrt(x[2] ** 2 + x[3] ** 2)
                if vspeed > self.max_speed:
                    scale = self.max_speed / vspeed
                    x[2] *= scale
                    x[3] *= scale

            xs[i] = x
            Ps[i] = P

        return xs, Ps

    # ------------------------------------------------------------------
    # Backward RTS smoother pass
    # ------------------------------------------------------------------

    @staticmethod
    def _rts_smooth(xs, Ps, F):
        """Rauch-Tung-Striebel backward smoother.

        Parameters
        ----------
        xs : (N, 6)    forward-filtered means
        Ps : (N, 6, 6) forward-filtered covariances
        F  : (6, 6)    state transition matrix

        Returns
        -------
        xs_s : (N, 6)    smoothed means
        Ps_s : (N, 6, 6) smoothed covariances
        """
        N       = len(xs)
        xs_s    = xs.copy()
        Ps_s    = Ps.copy()

        for i in range(N - 2, -1, -1):
            Pp = F @ Ps[i] @ F.T + np.eye(6) * 1e-6     # predicted covariance at i+1
            # Smoother gain: G = P[i] F^T Pp^{-1}
            G = (Ps[i] @ F.T) @ np.linalg.pinv(Pp)
            xs_s[i]  = xs[i]  + G @ (xs_s[i + 1] - F @ xs[i])
            Ps_s[i]  = Ps[i]  + G @ (Ps_s[i + 1] - Pp) @ G.T

        return xs_s, Ps_s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def smooth_ball_trajectory(self, raw_data_array):
        """Smooth the ball track.

        Parameters
        ----------
        raw_data_array : np.ndarray float32, shape (N, 47, 3)
            ATD array in **normalised** (0-1) coordinates.
            Slot 46 (last) is the ball; axis-2 = [x_norm, y_norm, speed].
            Missing values are NaN.

        Returns
        -------
        ball_xy_m : np.ndarray float64, shape (N, 2)
            Smoothed ball positions in **metres**.
            Frames inside a gap longer than self.max_gap are NaN.
        """
        N = raw_data_array.shape[0]

        # --- Convert raw ball to metres --------------------------------
        ball_norm = raw_data_array[:, 46, :2].astype(np.float64)
        observed  = ~np.isnan(ball_norm[:, 0])

        meas = ball_norm.copy()
        meas[:, 0] *= self._pitch_length
        meas[:, 1] *= self._pitch_width
        # On unobserved frames put zeros (will be ignored by filter)
        meas[~observed] = 0.0

        # --- Kalman forward pass --------------------------------------
        xs, Ps = self._forward(meas, observed)

        # --- RTS backward pass ----------------------------------------
        xs_s, _ = self._rts_smooth(xs, Ps, self._F)

        # --- Mark long gaps as NaN ------------------------------------
        ball_xy_m = xs_s[:, :2].copy()

        gap_start = None
        for i in range(N):
            if not observed[i]:
                if gap_start is None:
                    gap_start = i
            else:
                if gap_start is not None:
                    gap_len = i - gap_start
                    if gap_len > self.max_gap:
                        ball_xy_m[gap_start:i] = np.nan
                    gap_start = None

        # Trailing gap
        if gap_start is not None:
            gap_len = N - gap_start
            if gap_len > self.max_gap:
                ball_xy_m[gap_start:] = np.nan

        # Frames before the first observation (no meaningful estimate)
        if observed.any():
            first_obs = int(np.argmax(observed))
            if first_obs > 0:
                ball_xy_m[:first_obs] = np.nan
        else:
            ball_xy_m[:] = np.nan

        return ball_xy_m.astype(np.float64)

    def ball_speed_ms(self, ball_xy_m, dt=None):
        """Finite-difference speed from smoothed positions (m/s).

        Parameters
        ----------
        ball_xy_m : (N, 2) smoothed positions in metres

        Returns
        -------
        speed : (N,) float64, NaN where position is NaN
        """
        if dt is None:
            dt = self.dt
        diff  = np.diff(ball_xy_m, axis=0, prepend=ball_xy_m[[0]])
        speed = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2) / dt
        speed = np.clip(speed, 0.0, self.max_speed)
        # NaN propagation
        speed[np.isnan(ball_xy_m[:, 0])] = np.nan
        return speed


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys
    # Try to load real data
    try:
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        sys.path.insert(0, ROOT_DIR)
        sys.path.insert(0, os.path.join(ROOT_DIR, "mtah2027_possession", "src", "data"))
        import load_atd as _latd
        frames, _ = _latd.load()
        print("Loaded frames:", frames.shape)
    except Exception as e:
        print("Could not load real data ({}) — generating synthetic test.".format(e))
        N      = 1000
        frames = np.full((N, 47, 3), np.nan, dtype=np.float32)
        t      = np.linspace(0, 1, N)
        # Simulate ball moving in an arc, with 40% gaps
        rng    = np.random.default_rng(0)
        mask   = rng.random(N) > 0.4
        frames[mask, 46, 0] = (0.3 + 0.2 * np.cos(2 * np.pi * t))[mask]
        frames[mask, 46, 1] = (0.5 + 0.2 * np.sin(2 * np.pi * t))[mask]
        frames[mask, 46, 2] = 0.1

    smoother  = RTSBallSmoother()
    ball_xy_m = smoother.smooth_ball_trajectory(frames)
    n_valid   = int(np.sum(~np.isnan(ball_xy_m[:, 0])))
    print("Smoothed ball positions:", ball_xy_m.shape)
    print("Valid frames: {:,}/{:,}".format(n_valid, len(ball_xy_m)))
    if n_valid > 0:
        print("x range (m): {:.2f}..{:.2f}".format(
            float(np.nanmin(ball_xy_m[:, 0])),
            float(np.nanmax(ball_xy_m[:, 0]))))
        print("y range (m): {:.2f}..{:.2f}".format(
            float(np.nanmin(ball_xy_m[:, 1])),
            float(np.nanmax(ball_xy_m[:, 1]))))
