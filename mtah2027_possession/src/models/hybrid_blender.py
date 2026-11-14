"""
src/models/hybrid_blender.py
============================
Hybrid Physics-ML Uncertainty Blending Layer with Physical Consistency Gating.
More than a Hack 2027 — Phase 1: Ball Possession.
"""

import numpy as np


class HybridPhysicsBlender:
    """
    Blends ML predicted probabilities with physical Voronoi/kinematic priors
    and enforces strict physical consistency gating rules.
    """

    def __init__(self, entropy_threshold=0.35, max_fallback=0.40):
        self.entropy_threshold = float(entropy_threshold)
        self.max_fallback = float(max_fallback)

    def blend(self, probs_ml, dH, dA, v_ball, ball_valid):
        """
        Apply entropy-adaptive Voronoi blending and physical consistency gating.
        
        Parameters
        ----------
        probs_ml : (N, 4) float array
            Raw ML predicted probabilities [H, A, D, X].
        dH : (N,) float array
            Distance from ball to nearest Home player (metres).
        dA : (N,) float array
            Distance from ball to nearest Away player (metres).
        v_ball : (N,) float array
            Ball speed (m/s).
        ball_valid : (N,) bool array
            True if ball is tracked (not NaN).
        """
        N = len(probs_ml)
        p_safe = np.clip(probs_ml, 1e-6, 1.0)
        entropy = -np.sum(p_safe * np.log2(p_safe), axis=1)

        dH_clean = np.nan_to_num(dH, nan=10.0)
        dA_clean = np.nan_to_num(dA, nan=10.0)
        v_ball_clean = np.nan_to_num(v_ball, nan=0.0)
        valid = ball_valid & ~np.isnan(dH) & ~np.isnan(dA)

        # 1. Compute Physical Voronoi Inverse-Distance-Squared Prior
        prior = np.zeros((N, 4), dtype=np.float32)
        inv_H = 1.0 / (dH_clean[valid] ** 2 + 1e-3)
        inv_A = 1.0 / (dA_clean[valid] ** 2 + 1e-3)
        tot = inv_H + inv_A + 1e-6
        prior[valid, 0] = 0.95 * (inv_H / tot)
        prior[valid, 1] = 0.95 * (inv_A / tot)
        prior[valid, 2] = 0.05

        prior[~valid, 2] = 0.98
        prior[~valid, 0] = 0.01
        prior[~valid, 1] = 0.01

        # 2. Dynamic Entropy Blending
        # When model has high Shannon entropy (uncertain), blend towards physical Voronoi prior
        alpha = np.clip((entropy - self.entropy_threshold) / 0.8, 0.0, self.max_fallback)
        p_hybrid = probs_ml.copy()
        p_hybrid[valid] = (1.0 - alpha[valid, None]) * probs_ml[valid] + alpha[valid, None] * prior[valid]

        # 3. Physical Boundary Floor (Phase A):
        # When proximity dominance between Live teams is definitive during active play:
        # If dH < 2.0m and dA > 6.0m, Home clearly dominates the ball over Away
        dom_H = valid & (dH_clean < 2.0) & (dA_clean > 6.0) & (p_hybrid[:, 2] < 0.6)
        p_hybrid[dom_H, 0] = np.maximum(p_hybrid[dom_H, 0], 0.85)
        p_hybrid[dom_H, 1] = np.minimum(p_hybrid[dom_H, 1], 0.05)

        # If dA < 2.0m and dH > 6.0m, Away clearly dominates the ball over Home
        dom_A = valid & (dA_clean < 2.0) & (dH_clean > 6.0) & (p_hybrid[:, 2] < 0.6)
        p_hybrid[dom_A, 1] = np.maximum(p_hybrid[dom_A, 1], 0.85)
        p_hybrid[dom_A, 0] = np.minimum(p_hybrid[dom_A, 0], 0.05)

        # 4. Dead Ball Live Override (Phase A):
        # If ball is moving steadily (v_ball > 2.0 m/s) with a player in contact (min_dist < 1.5m)
        # and model predicts Dead Ball with low-to-moderate confidence (0.5 < P(D) < 0.85),
        # reallocate excess dead ball probability to the nearest team
        live_moving = valid & (np.minimum(dH_clean, dA_clean) < 1.5) & (v_ball_clean > 2.0) & (p_hybrid[:, 2] > 0.5) & (p_hybrid[:, 2] < 0.85)
        if np.any(live_moving):
            excess_D = p_hybrid[live_moving, 2] - 0.10
            ratio_H = prior[live_moving, 0] / (prior[live_moving, 0] + prior[live_moving, 1] + 1e-6)
            p_hybrid[live_moving, 0] += excess_D * ratio_H
            p_hybrid[live_moving, 1] += excess_D * (1.0 - ratio_H)
            p_hybrid[live_moving, 2] = 0.10

        # Re-normalize
        norm = np.sum(p_hybrid, axis=1, keepdims=True)
        norm = np.where(norm < 1e-6, 1.0, norm)
        return p_hybrid / norm
