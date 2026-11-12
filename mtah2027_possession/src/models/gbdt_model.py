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

# Try importing LightGBM or CatBoost
HAS_LIGHTGBM = False
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    pass

HAS_CATBOOST = False
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    pass


class FallbackSpatialClassifier:
    """Pure NumPy/SciPy fast regularized multiclass spatial classifier.
    Used when external GBDT libraries (LightGBM/CatBoost) are not installed.
    Uses multi-class regularized logistic regression with RBF / spatial kernel expansion.
    """

    def __init__(self, l2_reg=1.0, max_iter=200):
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        N, D = X.shape
        K = 4  # classes: 0, 1, 2, 3

        # Standardize features
        self.mean = np.nanmean(X, axis=0, keepdims=True)
        self.std = np.nanstd(X, axis=0, keepdims=True) + 1e-6
        X_norm = np.nan_to_num((X - self.mean) / self.std, nan=0.0)

        # One-hot encode targets
        Y = np.zeros((N, K), dtype=np.float64)
        for k in range(K):
            Y[y == k, k] = 1.0

        if sample_weight is None:
            sw = np.ones(N, dtype=np.float64) / N
        else:
            sw = sample_weight / np.sum(sample_weight)

        # Initialize weights
        W = np.zeros((D, K), dtype=np.float64)
        b = np.zeros(K, dtype=np.float64)

        lr = 0.1
        for it in range(self.max_iter):
            # Logits & Softmax
            logits = X_norm @ W + b
            logits_max = np.max(logits, axis=1, keepdims=True)
            exp_l = np.exp(logits - logits_max)
            probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

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
        return self

    def predict_proba(self, X):
        if self.weights is None or self.mean is None:
            raise ValueError("FallbackSpatialClassifier has not been fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        X_norm = np.nan_to_num((X - self.mean) / self.std, nan=0.0)
        logits = X_norm @ self.weights + self.bias
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits - logits_max)
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)


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
        """Train the classifier."""
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
            self.model = FallbackSpatialClassifier(l2_reg=1.0, max_iter=200)
            self.model.fit(X, y, sample_weight=sample_weight)

        return self

    def predict_proba(self, X):
        """Return (N, 4) probability distribution over [H, A, D, X]."""
        if self.model is None:
            raise ValueError("Model has not been fitted or loaded.")

        X = np.asarray(X, dtype=np.float32)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            # Ensure shape is (N, 4)
            if probs.shape[1] < 4:
                full_probs = np.zeros((len(X), 4), dtype=np.float32)
                full_probs[:, :probs.shape[1]] = probs
                return full_probs
            return probs
        raise RuntimeError("Model does not support predict_proba")

    def save(self, path):
        """Serialize model to file."""
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"backend": self.backend, "model": self.model}, f)

    @classmethod
    def load(cls, path):
        """Load model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(backend=data["backend"])
        obj.model = data["model"]
        return obj

