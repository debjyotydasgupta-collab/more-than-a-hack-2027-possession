"""
src/models/gbdt_model.py
========================
Gradient Boosted Decision Tree (GBDT) model suite for ball possession classification.
Directly implements Layer 1 of the architecture described in `ball_possession_master_plan.md`:
  - Multi-class cross-entropy objective over {H, A, D, X} (Classes 0, 1, 2, 3)
  - Full LightGBM and CatBoost integration with unified interface
  - Transition boundary sample weighting (t_change +/- 5 frames at 3.0x weight)
  - Pure NumPy/SciPy FallbackSpatialClassifier for zero-dependency execution
  - Guaranteed 4-column probability alignment [H=0, A=1, D=2, X=3] even with missing training classes
  - Serialization (save/load) compatible with pipeline inference specifications
"""

from __future__ import annotations

import os
import pickle
from typing import Any
import numpy as np

# Robust import check for LightGBM
HAS_LIGHTGBM = False
lgb: Any = None
try:
    import lightgbm as lgb  # type: ignore[import-untyped]
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False

# Robust import check for CatBoost
HAS_CATBOOST = False
cb: Any = None
try:
    import catboost as cb  # type: ignore[import-untyped]
    HAS_CATBOOST = True
except ImportError:
    cb = None
    HAS_CATBOOST = False

# Global class mappings
CLASSES = np.array([0, 1, 2, 3], dtype=np.int64)
CLASS_LABELS = np.array(["H", "A", "D", "X"], dtype="<U1")
LABEL_TO_INT = {"H": 0, "A": 1, "D": 2, "X": 3}
INT_TO_LABEL = {0: "H", 1: "A", 2: "D", 3: "X"}


def encode_labels(y):
    """Convert any label collection (strings ['H','A','D','X'] or ints [0,1,2,3])
    into a contiguous 1D int64 numpy array.
    """
    if hasattr(y, "values"):
        y = y.values
    y_arr = np.asarray(y)

    # String / character / object labels
    if y_arr.dtype.kind in ("U", "S", "O"):
        flat = y_arr.ravel()
        encoded = np.empty(len(flat), dtype=np.int64)
        for i, val in enumerate(flat):
            key = str(val).strip().upper()
            encoded[i] = LABEL_TO_INT.get(key, 3)
        return encoded

    return y_arr.ravel().astype(np.int64)


def compute_transition_weights(y, boundary_window=5, boundary_multiplier=3.0):
    """Compute sample weights increasing loss weight on rare transition boundary frames.
    
    Formula from Master Blueprint Section 5.1:
      L_GBDT = - sum_{t=1}^N w_t sum_{c} y_{t,c} log(p_hat_{t,c})
      where w_t = boundary_multiplier for t in [t_change - boundary_window, t_change + boundary_window].
    
    Parameters
    ----------
    y : array-like of shape (N,)
        Frame-level target labels (integer or string).
    boundary_window : int
        Half-width window around transition frames (default: 5 frames).
    boundary_multiplier : float
        Loss weight multiplier on transition boundaries (default: 3.0).
        
    Returns
    -------
    weights : np.ndarray of shape (N,), dtype float32
        Normalized sample weights with mean 1.0.
    """
    labels = encode_labels(y)
    N = len(labels)
    weights = np.ones(N, dtype=np.float32)

    if N <= 1:
        return weights

    # Identify transition frames where label changes
    transitions = np.flatnonzero(labels[1:] != labels[:-1]) + 1

    for t in transitions:
        low = max(0, t - boundary_window)
        high = min(N, t + boundary_window + 1)
        weights[low:high] = np.maximum(weights[low:high], float(boundary_multiplier))

    # Normalize weights so average weight is 1.0
    mean_w = np.mean(weights)
    if mean_w > 0:
        weights = weights / mean_w

    return weights


class FallbackSpatialClassifier:
    """Pure NumPy/SciPy fast regularized multiclass spatial classifier.
    Used when external GBDT libraries (LightGBM/CatBoost) are not installed.
    Uses multi-class regularized logistic regression with analytical gradient updates.
    """

    def __init__(self, l2_reg=1.0, max_iter=200):
        self.l2_reg = float(l2_reg)
        self.max_iter = int(max_iter)
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None
        self.classes_ = np.array([0, 1, 2, 3], dtype=np.int64)

    def fit(self, X, y, sample_weight=None):
        """Train classifier using gradient descent with sample weights."""
        if hasattr(X, "values"):
            X = X.values
        X = np.asarray(X, dtype=np.float64)
        y = encode_labels(y)
        N, D = X.shape
        K = 4  # classes: 0, 1, 2, 3

        # Standardize features safely
        self.mean = np.nanmean(X, axis=0, keepdims=True)
        self.mean = np.nan_to_num(self.mean, nan=0.0)

        self.std = np.nanstd(X, axis=0, keepdims=True)
        self.std = np.where(np.isnan(self.std) | (self.std < 1e-6), 1.0, self.std)

        X_norm = np.nan_to_num((X - self.mean) / self.std, nan=0.0)

        # One-hot encode targets
        Y = np.zeros((N, K), dtype=np.float64)
        for k in range(K):
            Y[y == k, k] = 1.0

        if sample_weight is None:
            sw = np.ones(N, dtype=np.float64) / max(N, 1)
        else:
            sw = np.asarray(sample_weight, dtype=np.float64).ravel()
            sw_sum = np.sum(sw)
            sw = sw / sw_sum if sw_sum > 0 else np.ones(N, dtype=np.float64) / max(N, 1)

        # Initialize weights
        W = np.zeros((D, K), dtype=np.float64)
        b = np.zeros(K, dtype=np.float64)

        lr = 0.1
        for it in range(self.max_iter):
            # Logits & Softmax
            logits = X_norm @ W + b
            logits_max = np.max(logits, axis=1, keepdims=True)
            exp_l = np.exp(logits - logits_max)
            probs = exp_l / (np.sum(exp_l, axis=1, keepdims=True) + 1e-12)

            # Gradient with sample weights
            err = (probs - Y) * sw[:, None]
            grad_W = X_norm.T @ err + self.l2_reg * W
            grad_b = np.sum(err, axis=0)

            # Update with decaying learning rate
            step = lr / (1.0 + 0.01 * it)
            W -= step * grad_W
            b -= step * grad_b

        self.weights = W
        self.bias = b
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        """Return (N, 4) probability distribution."""
        if self.weights is None or self.mean is None or self.std is None or self.bias is None:
            raise ValueError("FallbackSpatialClassifier has not been fitted yet.")
        if hasattr(X, "values"):
            X = X.values
        X = np.asarray(X, dtype=np.float64)
        X_norm = np.nan_to_num((X - self.mean) / self.std, nan=0.0)
        logits = X_norm @ self.weights + self.bias
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits - logits_max)
        probs = exp_l / (np.sum(exp_l, axis=1, keepdims=True) + 1e-12)

        # Zero out classes not present in training data
        if len(self.classes_) < 4:
            mask = np.isin(np.arange(4), self.classes_)
            probs[:, ~mask] = 0.0
            probs = probs / (np.sum(probs, axis=1, keepdims=True) + 1e-12)

        return probs.astype(np.float32)


class PossessionGBDT:
    """Master Gradient Boosted Decision Tree (GBDT) model wrapper for ball possession.
    Directly aligns with Layer 1 Spatial Base Classifiers of `ball_possession_master_plan.md`.
    """

    def __init__(self, backend="auto", params=None):
        """Initialize GBDT wrapper.
        
        Parameters
        ----------
        backend : str, default='auto'
            One of 'auto', 'lightgbm', 'catboost', 'fallback'.
            'auto' uses LightGBM if installed, else CatBoost, else Fallback.
        params : dict, optional
            Hyperparameter overrides merged into default model parameters.
        """
        self.backend_choice = backend
        self.params = dict(params or {})
        self.backend = None
        self.model = None
        self.classes_ = None

        if backend == "auto":
            if HAS_LIGHTGBM:
                self.backend = "lightgbm"
            elif HAS_CATBOOST:
                self.backend = "catboost"
            else:
                self.backend = "fallback"
        else:
            self.backend = backend.lower()

    def fit(self, X, y, sample_weight=None, eval_set=None, use_transition_weights=False,
            boundary_window=5, boundary_multiplier=3.0):
        """Train the classifier on feature matrix X and target labels y.
        
        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (N, D)
            Spatial and kinematic feature matrix (e.g. 148D features).
        y : array-like of shape (N,)
            Target possession labels (integers 0..3 or strings 'H','A','D','X').
        sample_weight : array-like of shape (N,), optional
            Custom sample weights. If None and use_transition_weights=True,
            computes boundary-boosted weights according to Section 5.1.
        eval_set : list of (X_val, y_val), optional
            Validation data for early stopping.
        use_transition_weights : bool, default=False
            If True and sample_weight is None, boosts weights on boundary frames by 3.0x.
        boundary_window : int, default=5
            Window radius around state transitions for weight boosting.
        boundary_multiplier : float, default=3.0
            Multiplier for transition boundary frames.
            
        Returns
        -------
        self : PossessionGBDT
        """
        if hasattr(X, "values"):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        y = encode_labels(y)
        self.classes_ = np.unique(y)

        # Compute transition boundary weights if requested and not explicitly provided
        if sample_weight is None and use_transition_weights:
            sample_weight = compute_transition_weights(
                y, boundary_window=boundary_window, boundary_multiplier=boundary_multiplier
            )
        elif sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float32).ravel()

        # Backend 1: LightGBM
        if self.backend == "lightgbm" and HAS_LIGHTGBM and lgb is not None:
            default_params = {
                "objective": "multiclass",
                "num_class": 4,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "n_estimators": 300,
                "verbose": -1,
                "random_state": 42,
            }
            default_params.update(self.params)
            self.model = lgb.LGBMClassifier(**default_params)
            fit_kwargs: dict[str, Any] = {"sample_weight": sample_weight}
            if eval_set is not None:
                eval_set_encoded = [(np.asarray(ex), encode_labels(ey)) for ex, ey in eval_set]
                fit_kwargs["eval_set"] = eval_set_encoded
            self.model.fit(X, y, **fit_kwargs)

        # Backend 2: CatBoost
        elif self.backend == "catboost" and HAS_CATBOOST and cb is not None:
            default_params = {
                "loss_function": "MultiClass",
                "classes_count": 4,
                "iterations": 300,
                "learning_rate": 0.05,
                "depth": 6,
                "verbose": False,
                "random_seed": 42,
            }
            default_params.update(self.params)
            self.model = cb.CatBoostClassifier(**default_params)
            fit_kwargs_cb: dict[str, Any] = {"sample_weight": sample_weight}
            if eval_set is not None:
                eval_set_encoded = [(np.asarray(ex), encode_labels(ey)) for ex, ey in eval_set]
                fit_kwargs_cb["eval_set"] = eval_set_encoded[0]
            self.model.fit(X, y, **fit_kwargs_cb)

        # Backend 3: Fallback Spatial Classifier
        else:
            self.backend = "fallback"
            l2 = float(self.params.get("l2_reg", 1.0))
            max_iter = int(self.params.get("max_iter", 200))
            self.model = FallbackSpatialClassifier(l2_reg=l2, max_iter=max_iter)
            self.model.fit(X, y, sample_weight=sample_weight)

        return self

    def predict_proba(self, X):
        """Return (N, 4) probability distribution over [H, A, D, X].
        
        Guarantees strict column index alignment:
          Column 0: Home (H)
          Column 1: Away (A)
          Column 2: Dead Ball (D)
          Column 3: Unknown / Abstain (X)
        """
        if self.model is None:
            raise ValueError("Model has not been fitted or loaded.")

        if hasattr(X, "values"):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        N = len(X)

        if hasattr(self.model, "predict_proba"):
            raw_probs = self.model.predict_proba(X)
        elif hasattr(self.model, "predict") and callable(self.model.predict):
            raw_probs = self.model.predict(X, prediction_type="Probability")
        else:
            raise RuntimeError("Model backend does not support probability predictions.")

        raw_probs = np.asarray(raw_probs, dtype=np.float32)

        # Standard case: exactly (N, 4) with all 4 classes present in order
        model_classes = getattr(self.model, "classes_", self.classes_)
        if raw_probs.shape[1] == 4 and (model_classes is None or np.array_equal(model_classes, [0, 1, 2, 3])):
            return raw_probs

        # Align probability columns using model.classes_
        full_probs = np.zeros((N, 4), dtype=np.float32)
        if model_classes is not None:
            classes = model_classes
        else:
            classes = np.arange(raw_probs.shape[1])

        for col_idx, cls_label in enumerate(classes):
            try:
                cls_int = int(cls_label)
                if 0 <= cls_int < 4 and col_idx < raw_probs.shape[1]:
                    full_probs[:, cls_int] = raw_probs[:, col_idx]
            except (ValueError, TypeError):
                continue

        # Normalize rows, ensuring any all-zero row gets assigned Unknown (X=3)
        row_sums = np.sum(full_probs, axis=1, keepdims=True)
        zero_rows = (row_sums.ravel() == 0)
        if np.any(zero_rows):
            full_probs[zero_rows, 3] = 1.0
            row_sums[zero_rows] = 1.0

        return (full_probs / row_sums).astype(np.float32)

    def predict(self, X, return_strings=True):
        """Predict class labels for feature matrix X.
        
        Parameters
        ----------
        X : array-like of shape (N, D)
            Feature matrix.
        return_strings : bool, default=True
            If True, returns array of string labels ['H', 'A', 'D', 'X'].
            If False, returns array of integer codes [0, 1, 2, 3].
            
        Returns
        -------
        predictions : np.ndarray of shape (N,)
        """
        probs = self.predict_proba(X)
        pred_ints = np.argmax(probs, axis=1)

        if return_strings:
            return CLASS_LABELS[pred_ints]
        return pred_ints

    def save(self, path):
        """Serialize model to file.
        
        Parameters
        ----------
        path : str
            Filepath for saving the serialized model.
        """
        folder = os.path.dirname(os.path.abspath(path))
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        payload = {
            "backend": self.backend,
            "backend_choice": self.backend_choice,
            "params": self.params,
            "classes_": getattr(self.model, "classes_", self.classes_),
            "model": self.model,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path):
        """Load serialized model from file.
        
        Parameters
        ----------
        path : str
            Filepath of the saved model.
            
        Returns
        -------
        PossessionGBDT
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        obj = cls(backend=data.get("backend", "fallback"), params=data.get("params", {}))
        obj.model = data.get("model")
        obj.classes_ = data.get("classes_", None)
        return obj
