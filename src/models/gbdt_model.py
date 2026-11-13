"""
src/models/gbdt_model.py
========================
Root-level wrapper re-exporting mtah2027_possession/src/models/gbdt_model.py.
Directly adheres to import paths specified in `ball_possession_master_plan.md`.
"""

from mtah2027_possession.src.models.gbdt_model import (
    PossessionGBDT,
    FallbackSpatialClassifier,
    compute_transition_weights,
    encode_labels,
    HAS_LIGHTGBM,
    HAS_CATBOOST,
    CLASSES,
    CLASS_LABELS,
    LABEL_TO_INT,
    INT_TO_LABEL,
)

__all__ = [
    "PossessionGBDT",
    "FallbackSpatialClassifier",
    "compute_transition_weights",
    "encode_labels",
    "HAS_LIGHTGBM",
    "HAS_CATBOOST",
    "CLASSES",
    "CLASS_LABELS",
    "LABEL_TO_INT",
    "INT_TO_LABEL",
]

