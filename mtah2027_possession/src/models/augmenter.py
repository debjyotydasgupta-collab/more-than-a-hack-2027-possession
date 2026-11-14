"""
src/models/augmenter.py
=======================
Geometric Pitch Symmetry Augmentation with Strict Vector Invariance.
More than a Hack 2027 — Phase 1: Ball Possession.
"""

import numpy as np

PITCH_LENGTH = 105.3
PITCH_WIDTH  = 68.0


class PitchSymmetryAugmenter:
    """
    Augments football spatial-tracking features by 4x using exact geometric pitch symmetries:
      1. Original data
      2. 180° Point Rotation (Playing the opposite direction, team swap H <-> A)
      3. Lateral Pitch Mirroring (Left-right wing symmetry, lateral velocities inverted)
      4. Combined Rotation + Mirroring
    """

    def __init__(self, pitch_length=PITCH_LENGTH, pitch_width=PITCH_WIDTH):
        self.pitch_length = float(pitch_length)
        self.pitch_width = float(pitch_width)

    def augment(self, X_feats, y_labels, weights=None):
        """
        Augment feature matrix and labels by 4x.
        
        Parameters
        ----------
        X_feats : (N, 92) float array
        y_labels : (N,) int array
        weights : (N,) float array, optional
        
        Returns
        -------
        X_aug, y_aug, w_aug
        """
        X_aug = [X_feats]
        y_aug = [y_labels]
        if weights is not None:
            w_aug = [weights]
        else:
            w_aug = [np.ones(len(y_labels), dtype=np.float32)]

        # -------------------------------------------------------------
        # Transform 2: 180° Point Rotation (Team swap H <-> A)
        # -------------------------------------------------------------
        X_rot = X_feats.copy()
        # Micro features (0..31):
        X_rot[:, 0:5], X_rot[:, 5:10] = X_feats[:, 5:10], X_feats[:, 0:5]       # Top-5 dists swap
        X_rot[:, 10] = 1.0 - X_feats[:, 10]                                     # Proximity ratio invert
        X_rot[:, 11] = -X_feats[:, 11]                                          # Proximity diff invert
        X_rot[:, 12:15], X_rot[:, 15:18] = X_feats[:, 15:18], X_feats[:, 12:15] # Closing vels swap
        X_rot[:, 20], X_rot[:, 21] = X_feats[:, 21], X_feats[:, 20]            # Nearest player dist swap
        X_rot[:, 22:24], X_rot[:, 24:26] = -X_feats[:, 24:26], -X_feats[:, 22:24]# Cosines inverted & swapped
        X_rot[:, 26], X_rot[:, 27] = X_feats[:, 27], X_feats[:, 26]            # Nearest/5th ratio swap
        X_rot[:, 29] = -X_feats[:, 29]                                          # Pitch half invert
        X_rot[:, 30], X_rot[:, 31] = X_feats[:, 31], X_feats[:, 30]            # Avg dist swap

        # Spatial features (32..71):
        X_rot[:, 32], X_rot[:, 33] = X_feats[:, 33], X_feats[:, 32]            # Hull area swap
        X_rot[:, 34] = 1.0 / (X_feats[:, 34] + 1e-4)                           # Hull ratio invert
        X_rot[:, 35] = -X_feats[:, 35]                                          # Hull diff invert
        X_rot[:, 36] = self.pitch_length - X_feats[:, 39]                       # Centroid X swap & invert
        X_rot[:, 37] = self.pitch_width - X_feats[:, 40]                        # Centroid Y swap & invert
        X_rot[:, 39] = self.pitch_length - X_feats[:, 36]
        X_rot[:, 40] = self.pitch_width - X_feats[:, 37]
        X_rot[:, 38], X_rot[:, 41] = X_feats[:, 41], X_feats[:, 38]            # Centroid dist to ball swap
        X_rot[:, 42], X_rot[:, 43] = X_feats[:, 43], X_feats[:, 42]            # Compactness swap
        X_rot[:, 45] = -X_feats[:, 45]                                          # STRICT NEGATION cent_dx
        X_rot[:, 46] = -X_feats[:, 46]                                          # STRICT NEGATION cent_dy
        X_rot[:, 48] = 1.0 - X_feats[:, 48]                                     # Voronoi dominance invert
        X_rot[:, 52] = 1.0 / (X_feats[:, 52] + 1e-4)                           # 10m density ratio invert
        X_rot[:, 53] = 1.0 - X_feats[:, 53]                                     # Ball X norm invert
        X_rot[:, 54] = 1.0 - X_feats[:, 54]                                     # Ball Y norm invert
        X_rot[:, 55], X_rot[:, 56] = X_feats[:, 56], X_feats[:, 55]            # Goal dists swap
        X_rot[:, 57], X_rot[:, 58] = X_feats[:, 58], X_feats[:, 57]            # Penalty box flags swap
        X_rot[:, 60:62], X_rot[:, 62:64] = X_feats[:, 62:64], X_feats[:, 60:62] # Bounding box dx/dy swap
        X_rot[:, 64], X_rot[:, 65] = X_feats[:, 65], X_feats[:, 64]            # Active players swap
        X_rot[:, 66:69], X_rot[:, 69:72] = X_feats[:, 69:72], X_feats[:, 66:69] # Centroid speeds swap

        # Pitch Control features (72..91):
        X_rot[:, 72], X_rot[:, 73] = X_feats[:, 73], X_feats[:, 72]            # Pitch control swap
        X_rot[:, 74] = -X_feats[:, 74]                                          # STRICT NEGATION pc_diff
        X_rot[:, 75] = 1.0 / (X_feats[:, 75] + 1e-4)                           # PC ratio invert
        X_rot[:, 76], X_rot[:, 77] = X_feats[:, 77], X_feats[:, 76]            # TTI swap
        X_rot[:, 78] = -X_feats[:, 78]                                          # STRICT NEGATION tti_diff
        X_rot[:, 79], X_rot[:, 80] = X_feats[:, 80], X_feats[:, 79]            # TTI top-3 swap
        X_rot[:, 81], X_rot[:, 82] = X_feats[:, 82], X_feats[:, 81]            # P_sum swap
        X_rot[:, 84], X_rot[:, 85] = X_feats[:, 85], X_feats[:, 84]            # PC ahead swap
        X_rot[:, 86] = -X_feats[:, 86]                                          # STRICT NEGATION control_flag
        X_rot[:, 87], X_rot[:, 88] = X_feats[:, 88], X_feats[:, 87]            # PC grad swap
        X_rot[:, 89], X_rot[:, 90] = X_feats[:, 90], X_feats[:, 89]            # PC vel swap

        y_rot = y_labels.copy()
        y_rot[y_labels == 0] = 1  # Home -> Away
        y_rot[y_labels == 1] = 0  # Away -> Home
        # Dead Ball (2) stays Dead Ball (2)

        X_aug.append(X_rot)
        y_aug.append(y_rot)
        w_aug.append(w_aug[0])

        # -------------------------------------------------------------
        # Transform 3: Lateral Pitch Mirroring (Same teams, mirrored Y)
        # -------------------------------------------------------------
        X_mir = X_feats.copy()
        X_mir[:, 23] = -X_feats[:, 23]                                         # dy_H cos
        X_mir[:, 25] = -X_feats[:, 25]                                         # dy_A cos
        X_mir[:, 37] = self.pitch_width - X_feats[:, 37]                       # h_cy
        X_mir[:, 40] = self.pitch_width - X_feats[:, 40]                       # a_cy
        X_mir[:, 46] = -X_feats[:, 46]                                         # cent_dy
        X_mir[:, 54] = 1.0 - X_feats[:, 54]                                    # by_norm
        X_mir[:, 67] = -X_feats[:, 67]                                         # h_cspeed_y
        X_mir[:, 70] = -X_feats[:, 70]                                         # a_cspeed_y

        X_aug.append(X_mir)
        y_aug.append(y_labels.copy())
        w_aug.append(w_aug[0])

        # -------------------------------------------------------------
        # Transform 4: Combined Rotation + Mirror (x -> 105.3-x, y -> y)
        # -------------------------------------------------------------
        X_comb = X_rot.copy()
        X_comb[:, 23] = -X_rot[:, 23]
        X_comb[:, 25] = -X_rot[:, 25]
        X_comb[:, 37] = self.pitch_width - X_rot[:, 37]
        X_comb[:, 40] = self.pitch_width - X_rot[:, 40]
        X_comb[:, 46] = -X_rot[:, 46]
        X_comb[:, 54] = 1.0 - X_rot[:, 54]
        X_comb[:, 67] = -X_rot[:, 67]
        X_comb[:, 70] = -X_rot[:, 70]

        X_aug.append(X_comb)
        y_aug.append(y_rot.copy())
        w_aug.append(w_aug[0])

        return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0), np.concatenate(w_aug, axis=0)

