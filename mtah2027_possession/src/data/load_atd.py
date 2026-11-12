"""
src/data/load_atd.py
====================
Parse the raw FIFA/EPTS ATD text file into a compact numpy array without
requiring kloppy. This is the low-level, fast loader used by the full
pipeline.

Raw line format
---------------
    <frame_id>:<p1_x,p1_y,p1_speed>;...;<p46_x,p46_y,p46_speed>:<ball_x,ball_y,ball_speed>

- frame_id counts from 1; video_frame = frame_id - 1
- 46 player slots separated by ';', then ':' separator, then ball group
- Missing observations are "NaN" or "N..." (checked on first character)
- Coordinates are normalised 0–1

Output
------
load_raw_atd(atd_txt_path) -> (frames, team_of_track)
  frames        : np.ndarray float32, shape (N, 47, 3)
                  axis-1 indices 0..45 = player slots, 46 = ball
                  axis-2 = [x_norm, y_norm, speed]
                  Missing values are np.nan
  team_of_track : np.ndarray int8, shape (46,)
                  0 = Home/Barcelona, 1 = Away/Athletic
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap so "games" is importable from any working directory
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
# src/data -> src -> mtah2027_possession -> more-than-a-hack-2027
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, DEFAULT_SOURCE, require  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_TRACKS  = 46          # player slots (tracks 1-46)
N_SLOTS   = 47          # player slots + 1 ball slot
BALL_IDX  = 46          # index of the ball in axis-1
FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING   = "N"         # first char of "NaN" or any missing marker

HOME = 0
AWAY = 1


# ---------------------------------------------------------------------------
# Team assignment from metadata XML
# ---------------------------------------------------------------------------

def team_of_each_track(meta_path):
    """Array over track slots (shape (46,)): 0 = Home, 1 = Away.

    Replicates starter/baseline.py::team_of_each_track() but kept here so
    this module is self-contained.  The first team in the XML is always Home
    (FIFA/EPTS convention).
    """
    root     = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 Teams, found {}".format(meta_path, len(team_ids)))
    home_id = team_ids[0]
    slots   = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if tid == home_id else AWAY for tid in slots], dtype=np.int8)


def team_names(meta_path):
    """(home_name, away_name) strings — for display only."""
    root = ET.parse(meta_path).getroot()
    return tuple(team.findtext("Name") for team in root.iter("Team"))


# ---------------------------------------------------------------------------
# Raw ATD parser
# ---------------------------------------------------------------------------

def _parse_group(group):
    """Parse 'x,y,speed' string -> [x, y, speed] floats or [nan, nan, nan]."""
    if not group or group[0] == MISSING:
        return [np.nan, np.nan, np.nan]
    parts = group.split(",")
    try:
        return [float(parts[0]), float(parts[1]), float(parts[2])]
    except (IndexError, ValueError):
        return [np.nan, np.nan, np.nan]


def load_raw_atd(atd_txt_path, n_frames=None):
    """Parse the raw ATD text file into a numpy array.

    Parameters
    ----------
    atd_txt_path : str
        Path to the _atd.txt file.
    n_frames : int or None
        Expected number of frames. If None, inferred from games.py (199319).

    Returns
    -------
    frames : np.ndarray, float32, shape (N, 47, 3)
        Dimension 1: indices 0..45 = player tracks, 46 = ball.
        Dimension 2: [x_norm, y_norm, speed].
        Missing values are np.nan.
    team_of_track : np.ndarray, int8, shape (46,)
        0 = Home (Barcelona), 1 = Away (Athletic Club).
        Derived from the companion _atd.xml.
    """
    if n_frames is None:
        n_frames = GAMES[GAME]["n_video_frames"]  # 199319

    # Companion XML for team assignments
    base = atd_txt_path
    if base.endswith("_atd.txt"):
        meta_path = base[:-len("_atd.txt")] + "_atd.xml"
    else:
        meta_path = base.replace(".txt", ".xml")

    if os.path.isfile(meta_path):
        team_of_track = team_of_each_track(meta_path)
    else:
        # Fallback: slots 1-24 → Home, 25-46 → Away (documented default)
        team_of_track = np.array(
            [HOME] * 24 + [AWAY] * 22, dtype=np.int8)

    frames = np.full((n_frames, N_SLOTS, 3), np.nan, dtype=np.float32)

    with open(atd_txt_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue

            frame_id  = int(parts[0])
            video_idx = frame_id - 1  # 0-based
            if video_idx < 0 or video_idx >= n_frames:
                continue

            # Player slots
            player_groups = parts[1].split(GROUP_SEP)
            for slot, group in enumerate(player_groups):
                if slot >= N_TRACKS:
                    break
                frames[video_idx, slot] = _parse_group(group)

            # Ball (last ':' group)
            frames[video_idx, BALL_IDX] = _parse_group(parts[2].strip())

    return frames, team_of_track


# ---------------------------------------------------------------------------
# Convenience: load from games.py paths
# ---------------------------------------------------------------------------

def load(game=GAME, data_dir=None, source=DEFAULT_SOURCE):
    """Load ATD using games.py path resolution.

    Returns (frames, team_of_track) — same as load_raw_atd().
    """
    try:
        meta_path, raw_path = tuple(require(["atd_meta", "atd_raw"], game, data_dir, source))
    except IOError as err:
        raise IOError(
            "ATD data files not found.\n{}\n\n"
            "Download the match bundle (see DATA.md) and unpack into ./data/".format(err)
        ) from err
    return load_raw_atd(raw_path)


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    print("Loading ATD data …")
    frames, tot = load()
    print("frames shape:", frames.shape)
    ball = frames[:, BALL_IDX, :]
    n_ball = int(np.sum(~np.isnan(ball[:, 0])))
    print("Ball present on {:,} / {:,} frames  ({:.1f}%)".format(
        n_ball, len(frames), 100.0 * n_ball / len(frames)))
    print("team_of_track:", tot)
