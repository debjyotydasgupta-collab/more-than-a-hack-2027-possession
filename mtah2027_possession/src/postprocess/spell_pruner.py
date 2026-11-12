"""
src/postprocess/spell_pruner.py
===============================
Sequence stabilization engine:
1. Hysteresis State Latching (Schmitt trigger logic)
2. Minimum Possession Spell Pruning & Gap Absorption
Directly maximizes segment-level spellF1 by eliminating state churn.
"""

import numpy as np


class PruneShortSpells:
    """Hysteresis filter and minimum possession spell pruner."""

    def __init__(self, min_spell_frames=15, tau_high=0.75, tau_low=0.25, k_hold=10):
        self.min_spell_frames = int(min_spell_frames)
        self.tau_high = float(tau_high)
        self.tau_low = float(tau_low)
        self.k_hold = int(k_hold)

    def apply_hysteresis(self, labels, probs):
        """Hysteresis latching: switching state requires confidence > tau_high for k_hold frames."""
        N = len(labels)
        if N <= self.k_hold or probs is None:
            return labels

        latched = labels.copy()
        current_state = latched[0]
        hold_count = 0
        candidate_state = current_state

        class_map = {"H": 0, "A": 1, "D": 2, "X": 3}

        for i in range(1, N):
            proposed = labels[i]

            if proposed != current_state:
                if proposed == candidate_state:
                    hold_count += 1
                else:
                    candidate_state = proposed
                    hold_count = 1

                c_idx = class_map.get(candidate_state, 0)
                if hold_count >= self.k_hold and probs[i, c_idx] >= self.tau_high:
                    current_state = candidate_state
                    latched[max(0, i - self.k_hold + 1):i + 1] = current_state
                    hold_count = 0
                else:
                    latched[i] = current_state
            else:
                hold_count = 0
                candidate_state = current_state
                latched[i] = current_state

        return latched

    def prune_short_spells(self, labels):
        """Merge spells shorter than min_spell_frames into neighboring spells."""
        N = len(labels)
        if N == 0:
            return labels

        cleaned = np.array(labels, dtype="<U1")

        for _ in range(10):
            # Run-length encode
            breaks = np.flatnonzero(cleaned[1:] != cleaned[:-1]) + 1
            bounds = np.concatenate(([0], breaks, [N]))
            lengths = bounds[1:] - bounds[:-1]
            seg_labels = cleaned[bounds[:-1]]

            # Find short segments (< min_spell_frames)
            short_indices = np.flatnonzero(lengths < self.min_spell_frames)
            if len(short_indices) == 0:
                break

            changed = False
            for s_idx in short_indices:
                start = bounds[s_idx]
                stop = bounds[s_idx + 1]

                # Merge with neighbor
                if s_idx > 0 and s_idx < len(seg_labels) - 1:
                    # If sandwich between same labels
                    if seg_labels[s_idx - 1] == seg_labels[s_idx + 1]:
                        target_label = seg_labels[s_idx - 1]
                    else:
                        # Longer neighbor wins
                        if lengths[s_idx - 1] >= lengths[s_idx + 1]:
                            target_label = seg_labels[s_idx - 1]
                        else:
                            target_label = seg_labels[s_idx + 1]
                elif s_idx > 0:
                    target_label = seg_labels[s_idx - 1]
                elif s_idx < len(seg_labels) - 1:
                    target_label = seg_labels[s_idx + 1]
                else:
                    continue

                cleaned[start:stop] = target_label
                changed = True

            if not changed:
                break

        return cleaned

    def prune(self, labels, probs=None):
        """Execute hysteresis stabilization and minimum spell pruning."""
        if probs is not None:
            latched = self.apply_hysteresis(labels, probs)
        else:
            latched = labels
        pruned = self.prune_short_spells(latched)
        return pruned
