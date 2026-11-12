"""
src/preprocessing/coordinate_transforms.py
===========================================
Convert between the normalised 0-1 coordinate system used in the raw ATD
and real-world metres.

Conventions
-----------
* The ATD stores positions as fractions of pitch dimensions, x ∈ [0, 1]
  along the length, y ∈ [0, 1] along the width.
* Metres keep the same 0-origin (bottom-left corner = (0, 0) m).
* We do NOT centre the pitch (some models prefer the signed version — use
  `norm_to_metres_centred` for that).

Pitch dimensions come from games.py:
    GAMES["BAR-ATH"]["pitch"] = (105.3, 68.0)  # (length, width)
"""

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
# src/preprocessing -> src -> mtah2027_possession -> more-than-a-hack-2027
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES  # noqa: E402


def _pitch_dims(game=GAME):
    """Return (length_m, width_m) for the given game."""
    return GAMES[game]["pitch"]   # (105.3, 68.0)


# ---------------------------------------------------------------------------
# Public transforms
# ---------------------------------------------------------------------------

def norm_to_metres(x_norm, y_norm, game=GAME):
    """Normalised [0, 1] coordinates → metres, 0-origin.

    Parameters
    ----------
    x_norm : float or ndarray   normalised x (along pitch length)
    y_norm : float or ndarray   normalised y (along pitch width)
    game   : str                key into games.GAMES

    Returns
    -------
    (x_m, y_m) in metres, same shape as input
    """
    length, width = _pitch_dims(game)
    return np.asarray(x_norm) * length, np.asarray(y_norm) * width


def metres_to_norm(x_m, y_m, game=GAME):
    """Metres → normalised [0, 1] coordinates."""
    length, width = _pitch_dims(game)
    return np.asarray(x_m) / length, np.asarray(y_m) / width


def norm_to_metres_centred(x_norm, y_norm, game=GAME):
    """Normalised [0, 1] → centred metres (pitch centre = (0, 0)).

    Useful for symmetric features (e.g. distance from centre).
    """
    length, width = _pitch_dims(game)
    x_m = (np.asarray(x_norm) - 0.5) * length
    y_m = (np.asarray(y_norm) - 0.5) * width
    return x_m, y_m


def transform_frames(frames, game=GAME):
    """Convert an (N, K, 3) ATD array's x/y channels from norm to metres in-place.

    Parameters
    ----------
    frames : np.ndarray float32, shape (N, K, 3)
        axis-2 = [x_norm, y_norm, speed_norm]
        Modified **in place** — x and y channels are overwritten with metres.
        Speed is left as-is (raw ATD speed is not in m/s; use Kalman output
        for speed in SI units).

    Returns
    -------
    frames : same array (modified in place, also returned for convenience)
    """
    length, width = _pitch_dims(game)
    frames[:, :, 0] *= length   # x → x_m
    frames[:, :, 1] *= width    # y → y_m
    return frames


def player_positions_metres(frames, game=GAME):
    """Extract all 46 player (x, y) positions in metres from the ATD array.

    Parameters
    ----------
    frames : np.ndarray, shape (N, 47, 3), normalised coords

    Returns
    -------
    positions : np.ndarray float32, shape (N, 46, 2)
        [x_m, y_m] for each player slot; NaN where missing.
    """
    length, width = _pitch_dims(game)
    pos = frames[:, :46, :2].copy().astype(np.float32)
    pos[:, :, 0] *= length
    pos[:, :, 1] *= width
    return pos


def ball_position_metres(frames, game=GAME):
    """Extract ball (x, y) in metres.

    Parameters
    ----------
    frames : np.ndarray, shape (N, 47, 3), normalised coords

    Returns
    -------
    ball_xy : np.ndarray float32, shape (N, 2)
        NaN where the ball is missing.
    """
    length, width = _pitch_dims(game)
    ball = frames[:, 46, :2].copy().astype(np.float32)
    ball[:, 0] *= length
    ball[:, 1] *= width
    return ball


def distance_to_ball(player_xy_m, ball_xy_m):
    """Euclidean distance from each player to the ball, in metres.

    Parameters
    ----------
    player_xy_m : np.ndarray, shape (N, K, 2)   player positions (m)
    ball_xy_m   : np.ndarray, shape (N, 2)       ball position (m)

    Returns
    -------
    dist : np.ndarray float32, shape (N, K)   NaN where either is missing
    """
    bxy = ball_xy_m[:, np.newaxis, :]           # (N, 1, 2)
    diff = player_xy_m - bxy                    # (N, K, 2)
    return np.sqrt(np.sum(diff ** 2, axis=-1)).astype(np.float32)


# ---------------------------------------------------------------------------
# Pitch zone utilities (for spatial features)
# ---------------------------------------------------------------------------

def pitch_zone(x_norm, y_norm, n_x=3, n_y=2):
    """Return (zone_x, zone_y) bucket indices for a normalised position.

    Default: 3 columns × 2 rows = 6 zones (defensive third, mid, attack × left/right).
    """
    zone_x = np.clip((np.asarray(x_norm) * n_x).astype(int), 0, n_x - 1)
    zone_y = np.clip((np.asarray(y_norm) * n_y).astype(int), 0, n_y - 1)
    return zone_x, zone_y


def pitch_zone_onehot(x_norm, y_norm, n_x=3, n_y=2):
    """One-hot encoding of pitch zone, shape (..., n_x * n_y)."""
    zx, zy = pitch_zone(x_norm, y_norm, n_x, n_y)
    idx    = zx * n_y + zy
    n      = n_x * n_y
    flat   = np.asarray(idx).ravel()
    out    = np.eye(n, dtype=np.float32)[flat]
    return out.reshape(np.asarray(idx).shape + (n,))
