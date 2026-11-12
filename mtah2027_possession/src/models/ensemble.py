"""
src/models/ensemble.py
======================
Ensemble blending layer combining predictions across multiple models.
"""

import numpy as np


class EnsembleBlender:
    """Weighted ensemble probability blender."""

    def __init__(self, weights=None):
        self.weights = weights

    def blend(self, prob_list):
        """Blend a list of (N, 4) probability distributions.

        Parameters
        ----------
        prob_list : list of np.ndarray of shape (N, 4)

        Returns
        -------
        blended : np.ndarray of shape (N, 4)
        """
        if not prob_list:
            raise ValueError("prob_list cannot be empty.")

        n_models = len(prob_list)
        if self.weights is None:
            w = np.ones(n_models, dtype=np.float32) / n_models
        else:
            w = np.array(self.weights, dtype=np.float32)
            w = w / np.sum(w)

        out = np.zeros_like(prob_list[0], dtype=np.float32)
        for i, p in enumerate(prob_list):
            out += w[i] * p

        # Re-normalize across axis 1
        sum_p = np.sum(out, axis=1, keepdims=True) + 1e-9
        return out / sum_p

