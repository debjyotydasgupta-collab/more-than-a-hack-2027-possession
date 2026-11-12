"""
src/postprocess/viterbi_decoder.py
==================================
Linear-Chain Viterbi sequence decoder with learned transition penalties
and tactical Smart Tagging rule overrides.
"""

import numpy as np

# Label codes: 0=H, 1=A, 2=D, 3=X
H, A, D, X = 0, 1, 2, 3


class ViterbiSequenceDecoder:
    """Decodes frame probability distributions into an optimal sequence.
    Penalizes rapid possession toggles and integrates human tactical event constraints.
    """

    def __init__(self, trans_penalty_swap=-2.5, trans_penalty_dead=-1.0, boost_factor=1.8):
        self.trans_penalty_swap = float(trans_penalty_swap)
        self.trans_penalty_dead = float(trans_penalty_dead)
        self.boost_factor = float(boost_factor)
        self.trans_matrix = self._build_transition_matrix()

    def _build_transition_matrix(self):
        """Construct the 4x4 log-transition probability matrix.
        Rows: from state, Cols: to state.
        """
        # Base log-transition matrix (neutral diagonal)
        T = np.zeros((4, 4), dtype=np.float64)

        # High penalty for direct H <-> A transitions (turnovers usually involve a contest or ball movement)
        T[H, A] = self.trans_penalty_swap
        T[A, H] = self.trans_penalty_swap

        # Moderate penalty for transitioning to/from Dead Ball
        T[H, D] = self.trans_penalty_dead
        T[D, H] = self.trans_penalty_dead
        T[A, D] = self.trans_penalty_dead
        T[D, A] = self.trans_penalty_dead

        # Transitions involving Unknown X
        T[H, X] = -3.0
        T[A, X] = -3.0
        T[D, X] = -2.0
        T[X, H] = -1.5
        T[X, A] = -1.5
        T[X, D] = -1.0

        return T

    def apply_smart_tagging_rules(self, probs, smart_tag_events):
        """Impose deterministic bounds from the 638 tactical Smart Tagging events."""
        adjusted = probs.copy()
        N = len(adjusted)

        for event in smart_tag_events:
            start_f = event["start_frame"]
            end_f = event["end_frame"]
            if start_f >= N:
                continue
            end_f = min(N, end_f)
            duration = end_f - start_f
            if duration <= 0:
                continue

            code = (event["code"] or "").strip().upper()
            team = (event["tags"].get("Team") or "").strip()

            # Rule 1: SET PIECES -> First 75% of duration is strictly Dead Ball (D)
            if code == "SET PIECES":
                dead_limit = start_f + int(0.75 * duration)
                adjusted[start_f:dead_limit, :] = 0.001
                adjusted[start_f:dead_limit, D] = 0.997

            # Rule 2: GOALS -> Complete duration is Dead Ball (D)
            elif code == "GOALS":
                adjusted[start_f:end_f, :] = 0.001
                adjusted[start_f:end_f, D] = 0.997

            # Rule 3: High-confidence possession phases -> Boost corresponding team
            elif code in ("PROGRESSION", "BUILD UP", "BALL IN FINAL THIRD"):
                if team in ("Barcelona", "Home Team", "Home", "FC Barcelona", "H"):
                    adjusted[start_f:end_f, H] *= self.boost_factor
                elif team in ("Athletic Club", "Atletic Club", "Away Team", "Away", "Athletic", "A"):
                    adjusted[start_f:end_f, A] *= self.boost_factor

            # Rule 4: ATTACKING TRANSITION
            elif code == "ATTACKING TRANSITION":
                if team in ("Barcelona", "Home Team", "Home", "FC Barcelona", "H"):
                    adjusted[start_f:end_f, H] *= 1.4
                elif team in ("Athletic Club", "Atletic Club", "Away Team", "Away", "Athletic", "A"):
                    adjusted[start_f:end_f, A] *= 1.4

        # Re-normalize rows
        row_sums = np.sum(adjusted, axis=1, keepdims=True) + 1e-9
        return adjusted / row_sums

    def decode(self, probs, smart_tag_events=None):
        """Globally optimal Viterbi decoding over frame emissions."""
        N = len(probs)
        if N == 0:
            return np.empty(0, dtype=object)

        if smart_tag_events is not None:
            probs = self.apply_smart_tagging_rules(probs, smart_tag_events)

        log_emissions = np.log(np.clip(probs, 1e-12, 1.0))
        T = self.trans_matrix

        # Dynamic programming tables
        viterbi = np.zeros((N, 4), dtype=np.float64)
        backpointer = np.zeros((N, 4), dtype=np.int32)

        # Initial state (uniform prior)
        viterbi[0] = log_emissions[0]

        # Forward pass
        cols = np.arange(4)
        for t in range(1, N):
            # viterbi[t-1]: (4,), T: (4, 4) -> broadcast (4, 4)
            trans_scores = viterbi[t - 1, :, None] + T
            backpointer[t] = np.argmax(trans_scores, axis=0)
            viterbi[t] = trans_scores[backpointer[t], cols] + log_emissions[t]

        # Backward traceback
        best_path = np.zeros(N, dtype=np.int8)
        best_path[-1] = np.argmax(viterbi[-1])

        for t in range(N - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        label_map = np.array(["H", "A", "D", "X"], dtype="<U1")
        return label_map[best_path]

