"""
src/data/__init__.py
====================
Data loaders for EPTS tracking streams, smart tagging events, and homography.
"""

from .load_atd import load_raw_atd, load as load_atd
from .load_smart_tagging import load as load_events, load_with_team_labels, load as load_smart_tagging
from .load_homography import load_as_dict as load_homography_dict, load as load_homography

__all__ = [
    "load_raw_atd",
    "load_atd",
    "load_events",
    "load_with_team_labels",
    "load_smart_tagging",
    "load_homography_dict",
    "load_homography",
]
