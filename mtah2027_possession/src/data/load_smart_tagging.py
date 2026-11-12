"""
src/data/load_smart_tagging.py
==============================
Thin wrapper around starter/load_smart_tagging.py that re-exports the
``load()`` function and adds team-name → H/A mapping.

The raw Smart Tagging XML contains team names like "FC Barcelona" and
"Athletic Club". This module maps them to the submission labels H and A
so the rest of the pipeline never sees club names.

Usage
-----
    from src.data.load_smart_tagging import load, load_with_team_labels
    events = load()          # list of dicts, same schema as starter/
    events = load_with_team_labels()  # tags["Team"] replaced with "H"/"A"/"?"
"""

import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
# src/data -> src -> mtah2027_possession -> more-than-a-hack-2027
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
STARTER_DIR = os.path.join(ROOT, "starter")

for _p in (ROOT, STARTER_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import the starter loader so we can re-export it
import load_smart_tagging as _starter_loader  # noqa: E402  (from starter/)
from games import GAME, GAMES  # noqa: E402

# ---------------------------------------------------------------------------
# Team name → submission-label mapping
# ---------------------------------------------------------------------------
# Hard-coded for BAR-ATH. If this module is extended to other games the map
# should be loaded from games.py or the XML metadata.
_TEAM_LABEL_MAP = {
    # Barcelona variations
    "fc barcelona":   "H",
    "barcelona":      "H",
    "bar":            "H",
    # Athletic Club variations
    "athletic club":  "A",
    "athletic":       "A",
    "ath":            "A",
    "athletic bilbao":"A",
}


def _map_team(raw_name):
    """Map a raw club name to H / A / '?' (unknown)."""
    if raw_name is None:
        return "?"
    return _TEAM_LABEL_MAP.get(raw_name.strip().lower(), "?")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(game=GAME, data_dir=None, fps=None):
    """Load Smart Tagging events, identical schema to starter/load_smart_tagging.load().

    Each event dict:
        {
            "code":        str,   e.g. "SET PIECES", "BUILD UP", …
            "start_s":     float, seconds from video start
            "end_s":       float
            "start_frame": int,   video frame (0-based)
            "end_frame":   int,   first frame *after* the phase (half-open)
            "tags":        dict,  e.g. {"Team": "FC Barcelona", "Half": "1st Half", …}
        }
    """
    return _starter_loader.load(game=game, data_dir=data_dir, fps=fps)


def load_with_team_labels(game=GAME, data_dir=None, fps=None):
    """Like load() but with tags["Team"] replaced by "H", "A", or "?".

    This is the preferred interface for the pipeline — no club names leak
    through to code that only knows H/A/D/X.
    """
    events = load(game=game, data_dir=data_dir, fps=fps)
    for event in events:
        raw = event["tags"].get("Team")
        event["tags"]["Team"] = _map_team(raw)
    return events


def events_as_frame_array(events, n_frames, label_key="Team"):
    """Convert a list of Smart Tagging events to a per-frame string array.

    Parameters
    ----------
    events : list of dicts from load_with_team_labels()
    n_frames : int   (199319 for BAR-ATH)
    label_key : str  which tag to read ("Team", "Half", "Type", …)

    Returns
    -------
    arr : np.ndarray of dtype '<U4', shape (n_frames,)
          Filled with event["tags"][label_key] wherever a phase is active,
          empty string elsewhere.
    """
    import numpy as np
    arr = np.full(n_frames, "", dtype="<U16")
    for event in events:
        s = max(0, event["start_frame"])
        e = min(n_frames, event["end_frame"])
        val = event["tags"].get(label_key, "")
        if val:
            arr[s:e] = val
    return arr


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    events = load_with_team_labels()
    print("Loaded {} Smart Tagging events".format(len(events)))
    codes = {}
    for ev in events:
        codes[ev["code"]] = codes.get(ev["code"], 0) + 1
    print("\nCode counts:")
    for code in sorted(codes, key=lambda c: -codes[c]):
        print("  {:40s} {:4d}".format(code, codes[code]))
    teams = sorted({ev["tags"]["Team"] for ev in events})
    print("\nMapped Team values:", teams)
