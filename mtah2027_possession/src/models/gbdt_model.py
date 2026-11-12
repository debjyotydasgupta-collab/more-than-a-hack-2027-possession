"""
src/models/gbdt_model.py
========================
Gradient Boosted Decision Tree (GBDT) model wrapper for ball possession classification.
Supports LightGBM and CatBoost when installed, and provides a pure-NumPy/SciPy
fallback classifier ensuring zero-dependency execution in restricted environments.

Classes:
    0: Home (H)
    1: Away (A)
    2: Dead Ball (D)
    3: Unknown / Abstain (X)
"""

import os
import pickle
import numpy as np

# Robust import check for LightGBM
HAS_LIGHTGBM = False
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    pass

# Robust import check for CatBoost
HAS_CATBOOST = False
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    pass

# Label mapping for string <-> integer conversion
LABEL_STR_TO_INT = {"H": 0, "A": 1, "D": 2, "X": 3}
LABEL_INT_TO_STR = {0: "H", 1: "A", 2: "D", 3: "X"}


def _encode_labels(y):
    """Convert any label array (string or integer) to a 1D int64 array in [0, 3]."""
    y_arr = np.asarray(y)
    if y_arr.dtype.kind in ("U", "S", "O"):
        # String / object labels
        return np.array([LABEL_STR_TO_INT.get(str(v).strip().upper(), 3) for v in y_arr.ravel()], dtype=np.int64)
    return y_arr.ravel().astype(np.int64)


class FallbackSpatialClassifier:
    """Pure NumPy/SciPy fast regularized multiclass spatial classifier.
    Used when external GBDT libraries (LightGBM/CatBoost) are not installed.
    Uses multi-class regularized logistic regression.
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
        X = np.asarray(X, dtype=np.float64)
        y = _encode_labels(y)
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
        if self.weights is None or self.mean is None:
            raise ValueError("FallbackSpatialClassifier has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        X_norm = np.nan_to_num((X - self.mean) / self.std, nan=0.0)
        logits = X_norm @ self.weights + self.bias
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits - logits_max)
        probs = exp_l / (np.sum(exp_l, axis=1, keepdims=True) + 1e-12)

        if len(self.classes_) < 4:
            mask = np.isin(np.arange(4), self.classes_)
            probs[:, ~mask] = 0.0
            probs = probs / (np.sum(probs, axis=1, keepdims=True) + 1e-12)

        return probs


class PossessionGBDT:
    """Master GBDT model wrapper for ball possession."""

    def __init__(self, backend="auto", params=None):
        self.backend = backend
        self.params = params or {}
        self.model = None

        if backend == "auto":
            if HAS_LIGHTGBM:
                self.backend = "lightgbm"
            elif HAS_CATBOOST:
                self.backend = "catboost"
            else:
                self.backend = "fallback"

    def fit(self, X, y, sample_weight=None):
        """Train the classifier on feature matrix X and labels y."""
        X = np.asarray(X, dtype=np.float32)
        y = _encode_labels(y)

        if self.backend == "lightgbm" and HAS_LIGHTGBM:
            default_params = {
                "objective": "multiclass",
                "num_class": 4,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "verbose": -1,
                "n_estimators": 300,
            }
            default_params.update(self.params)
            self.model = lgb.LGBMClassifier(**default_params)
            self.model.fit(X, y, sample_weight=sample_weight)

        elif self.backend == "catboost" and HAS_CATBOOST:
            default_params = {
                "loss_function": "MultiClass",
                "classes_count": 4,
                "iterations": 300,
                "learning_rate": 0.05,
                "depth": 6,
                "verbose": False,
            }
            default_params.update(self.params)
            self.model = cb.CatBoostClassifier(**default_params)
            self.model.fit(X, y, sample_weight=sample_weight)

        else:
            self.backend = "fallback"
            self.model = FallbackSpatialClassifier(l2_reg=1.0, max_iter=200)
            self.model.fit(X, y, sample_weight=sample_weight)

        return self

    def predict_proba(self, X):
        """Return (N, 4) probability distribution over [H, A, D, X].
        Correctly aligns probability columns using model.classes_ to ensure that
        missing training classes never cause column misalignment.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted or loaded.")

        X = np.asarray(X, dtype=np.float32)
        N = len(X)

        if hasattr(self.model, "predict_proba"):
            raw_probs = self.model.predict_proba(X)
        elif hasattr(self.model, "predict") and callable(self.model.predict):
            # Some backends or native models
            preds = self.model.predict(X, prediction_type="Probability")
            raw_probs = preds
        else:
            raise RuntimeError("Model does not support predict_proba")

        # If raw_probs is already (N, 4) and classes_ is standard [0, 1, 2, 3]
        if raw_probs.shape[1] == 4 and (not hasattr(self.model, "classes_") or np.array_equal(self.model.classes_, [0, 1, 2, 3])):
            return raw_probs.astype(np.float32)

        # Class column alignment via model.classes_
        full_probs = np.zeros((N, 4), dtype=np.float32)
        classes = getattr(self.model, "classes_", None) if hasattr(self.model, "classes_") else np.arange(raw_probs.shape[1])

        for col_idx, cls_label in enumerate(classes):
            try:
                cls_int = int(cls_label)
                if 0 <= cls_int < 4 and col_idx < raw_probs.shape[1]:
                    full_probs[:, cls_int] = raw_probs[:, col_idx]
            except (ValueError, TypeError):
                continue

        # Ensure row sums are normalized
        row_sums = np.sum(full_probs, axis=1, keepdims=True)
        zero_rows = (row_sums.ravel() == 0)
        if np.any(zero_rows):
            full_probs[zero_rows, 3] = 1.0  # Assign Unknown X to unpredicted instances
            row_sums[zero_rows] = 1.0

        return (full_probs / row_sums).astype(np.float32)

    def save(self, path):
        """Serialize model to file."""
        folder = os.path.dirname(os.path.abspath(path))
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"backend": self.backend, "model": self.model, "params": self.params}, f)

    @classmethod
    def load(cls, path):
        """Load model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(backend=data.get("backend", "fallback"), params=data.get("params", {}))
        obj.model = data.get("model")
        return obj
