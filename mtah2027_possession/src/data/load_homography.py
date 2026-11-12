"""
src/data/load_homography.py
===========================
Parse the per-frame homography CSV into a dict keyed by video frame index
and also expose the full (N, 3, 3) numpy array.

The homography maps *normalised pitch coordinates* (0-1, same as the ATD)
to image pixels.  Pitch → pixel:

    px_x, px_y = project(H[frame], atd_x_norm, atd_y_norm)

NEVER multiply by pitch size before projecting — the matrix already encodes
the pitch geometry.

Output shapes
-------------
load(...)
    matrices : np.ndarray float64 (n, 3, 3) — NaN where valid=False
    reprojection_error : np.ndarray float64 (n,)
    valid : np.ndarray bool (n,)

load_as_dict(...)
    {frame_index: np.ndarray (3,3)} — only entries where valid=True
"""

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
STARTER_DIR = os.path.join(ROOT, "starter")
for _p in (ROOT, STARTER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from games import GAME, DEFAULT_SOURCE, require  # noqa: E402

# ---------------------------------------------------------------------------
# Column layout (documented in starter/load_homography.py and DATA.md)
# ---------------------------------------------------------------------------
MATRIX_COLUMNS = ["h{}{}".format(r, c) for r in range(3) for c in range(3)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """Load the homography CSV for one source of a game.

    Returns
    -------
    matrices : np.ndarray float64, shape (n_frames, 3, 3)
        Row-major 3×3 matrices.  NaN on invalid frames.
    reprojection_error : np.ndarray float64, shape (n_frames,)
        Lower is better.  Valid frames typically < 1 on the tactical camera.
    valid : np.ndarray bool, shape (n_frames,)
        False wherever the matrix is NaN (broadcast replays etc.).
    """
    try:
        path = require(["homography"], game, data_dir, source)[0]
    except IOError as err:
        raise IOError(
            "Homography file not found.\n{}\n\n"
            "Download the match bundle (see DATA.md).".format(err)
        ) from err

    # Lazy pandas import — pandas is optional but required by the starter kit
    try:
        import pandas as pd
        table = pd.read_csv(path)
        matrices = table[MATRIX_COLUMNS].values.astype(np.float64).reshape(-1, 3, 3)
        reprojection_error = table["reprojection_error"].values.astype(np.float64)
        valid = table["valid"].values.astype(bool)
    except ImportError:
        # Pure-numpy fallback (pandas not available)
        matrices, reprojection_error, valid = _load_numpy_fallback(path)

    return matrices, reprojection_error, valid


def load_as_dict(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """Load homography as {video_frame_index: 3×3 matrix} for valid frames only.

    Useful when you need fast random access by frame index without keeping
    the full (199319, 3, 3) array in memory.
    """
    matrices, _, valid = load(game, data_dir, source)
    return {i: matrices[i] for i in np.flatnonzero(valid)}


def project(matrix, x, y):
    """Normalised pitch coords (0-1) → image pixels.

    Parameters
    ----------
    matrix : (3, 3) ndarray
    x, y   : float or ndarray  — normalised pitch coords

    Returns
    -------
    (px_x, px_y) — image pixel coordinates
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    denom  = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    px_x   = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denom
    px_y   = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denom
    return px_x, px_y


def unproject(matrix, pixel_x, pixel_y):
    """Image pixels → normalised pitch coords (inverse of project)."""
    return project(np.linalg.inv(matrix), pixel_x, pixel_y)


# ---------------------------------------------------------------------------
# Pure-numpy fallback parser (no pandas required)
# ---------------------------------------------------------------------------

def _load_numpy_fallback(path):
    """Parse the homography CSV with numpy only."""
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        rows = []
        for line in fh:
            rows.append(line.strip().split(","))

    col_idx = {name: i for i, name in enumerate(header)}
    n = len(rows)

    matrices           = np.full((n, 3, 3), np.nan, dtype=np.float64)
    reprojection_error = np.full(n, np.nan, dtype=np.float64)
    valid              = np.zeros(n, dtype=bool)

    mat_col_indices = [col_idx["h{}{}".format(r, c)]
                       for r in range(3) for c in range(3)]

    for i, row in enumerate(rows):
        v = int(row[col_idx["valid"]])
        valid[i] = bool(v)
        try:
            reprojection_error[i] = float(row[col_idx["reprojection_error"]])
        except (ValueError, IndexError):
            pass
        if v:
            flat = [float(row[j]) for j in mat_col_indices]
            matrices[i] = np.array(flat, dtype=np.float64).reshape(3, 3)

    return matrices, reprojection_error, valid


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for src in ("tactical", "broadcast"):
        print("=" * 60)
        print(src)
        try:
            H, err, v = load(source=src)
        except IOError as e:
            print(" ", e)
            continue
        print("  shape:", H.shape)
        print("  valid: {}/{} ({:.1f}%)".format(int(v.sum()), len(v),
                                                  100.0 * v.mean()))
        if v.any():
            print("  reprojection error (valid) — median {:.3f}".format(
                float(np.median(err[v]))))
