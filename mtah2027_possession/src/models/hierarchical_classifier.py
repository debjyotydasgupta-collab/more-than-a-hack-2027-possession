"""
src/models/hierarchical_classifier.py
=====================================
Two-Stage Hierarchical Ball Possession Classifier:
  Stage 1: State Classifier (Alive vs Dead Ball)
  Stage 2: Ownership Classifier (Home vs Away | Alive)
Decouples match state from team ownership, eliminating out-of-distribution confusion.
"""

import numpy as np
import lightgbm as lgb


class HierarchicalPossessionEngine:
    """
    Two-stage hierarchical model that separates game state (live/dead)
    from possession ownership (home/away).
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model_state = None
        self.model_owner = None

    def fit(self, X_all, X_inv, y, sample_weight=None):
        """
        Fit both Stage 1 and Stage 2 models.

        Parameters
        ----------
        X_all : (N, F_all) float array
            Full feature matrix (used for Stage 1 State detection).
        X_inv : (N, F_inv) float array
            Spatially invariant relative features (used for Stage 2 Ownership).
        y : (N,) int array
            Labels: 0=H, 1=A, 2=D.
        sample_weight : (N,) float array, optional
        """
        is_live = (y != 2).astype(int) # 1 if H or A, 0 if D

        # Stage 1: State Classifier (Alive vs Dead)
        state_params = {
            "objective":         "binary",
            "n_estimators":      250,
            "num_leaves":        31,
            "learning_rate":     0.06,
            "min_child_samples": 20,
            "colsample_bytree":  0.75,
            "subsample":         0.85,
            "n_jobs":            -1,
            "verbosity":         -1,
            "random_state":      self.random_state,
        }
        self.model_state = lgb.LGBMClassifier(**state_params)
        self.model_state.fit(X_all, is_live, sample_weight=sample_weight)

        # Stage 2: Ownership Classifier (Home vs Away | Alive)
        live_mask = (y != 2)
        X_live_inv = X_inv[live_mask]
        y_live = y[live_mask] # 0=H, 1=A
        w_live = sample_weight[live_mask] if sample_weight is not None else None

        # Binary label for ownership: 1 if Home, 0 if Away
        y_home = (y_live == 0).astype(int)

        owner_params = {
            "objective":         "binary",
            "n_estimators":      300,
            "num_leaves":        31,
            "learning_rate":     0.06,
            "min_child_samples": 20,
            "colsample_bytree":  0.65, # Regularize feature subsets
            "subsample":         0.85,
            "n_jobs":            -1,
            "verbosity":         -1,
            "random_state":      self.random_state,
        }
        self.model_owner = lgb.LGBMClassifier(**owner_params)
        self.model_owner.fit(X_live_inv, y_home, sample_weight=w_live)

    def predict_proba(self, X_all, X_inv):
        """
        Compute joint (N, 4) probability distribution: [P(H), P(A), P(D), P(X)].
        """
        N = len(X_all)
        if self.model_state is None or self.model_owner is None:
            raise RuntimeError("Models must be fitted before predict_proba.")
        p_live = self.model_state.predict_proba(X_all)[:, 1] # P(Alive)
        p_dead = 1.0 - p_live

        p_home_given_live = self.model_owner.predict_proba(X_inv)[:, 1] # P(Home | Alive)
        p_away_given_live = 1.0 - p_home_given_live

        probs = np.zeros((N, 4), dtype=np.float32)
        probs[:, 0] = p_live * p_home_given_live
        probs[:, 1] = p_live * p_away_given_live
        probs[:, 2] = p_dead
        probs[:, 3] = 0.0

        # Normalization guard
        row_sums = np.sum(probs, axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-6, 1.0, row_sums)
        return probs / row_sums

