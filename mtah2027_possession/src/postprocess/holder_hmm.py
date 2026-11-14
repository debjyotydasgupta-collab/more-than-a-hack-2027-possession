"""
mtah2027_possession/src/postprocess/holder_hmm.py
=================================================
Method 2: Latent Holder & Flight HMM with Non-Causal Forward-Backward Smoothing.

State space:
  0: Held_H    (Home controls the ball on the ground)
  1: Flight_H  (Ball in flight, kicked by Home)
  2: Held_A    (Away controls the ball on the ground)
  3: Flight_A  (Ball in flight, kicked by Away)
  4: Dead      (Ball is dead / play stopped)

Key Constraints:
1. Flight is a first-class state.
2. In flight, ball CANNOT switch teams without an interception or touch.
3. Non-causal forward-backward smoothing resolves flight phases using both past kicks
   and future receptions.
"""

import numpy as np


class LatentHolderHMM:
    def __init__(self, 
                 p_kick=0.15, 
                 p_reception=0.30, 
                 p_turnover_ground=0.01, 
                 p_dead=0.005, 
                 p_restart=0.08):
        self.p_kick = p_kick
        self.p_reception = p_reception
        self.p_turnover_ground = p_turnover_ground
        self.p_dead = p_dead
        self.p_restart = p_restart
        self.n_states = 5

    def build_transition_matrix(self, v_ball, dH, dA, touch_mask, touch_team):
        """
        Build time-varying transition matrices for the entire sequence.
        Returns:
            log_trans: (T-1, 5, 5) log-transition probabilities log P(s_t | s_{t-1})
        """
        T = len(v_ball)
        log_trans = np.full((T - 1, 5, 5), -99.0, dtype=np.float64)

        # Vectorized probability elements
        # 0: Held_H, 1: Flight_H, 2: Held_A, 3: Flight_A, 4: Dead
        for t in range(T - 1):
            v = v_ball[t + 1]
            dh = dH[t + 1]
            da = dA[t + 1]
            is_touch = touch_mask[t + 1]
            tt = touch_team[t + 1]

            # Dynamic kick transition prob
            p_k = 0.50 if (v > 3.0 and is_touch) else (0.15 if v > 4.0 else 0.02)
            # Reception prob
            p_rec_h = 0.60 if (dh < 1.2 or (is_touch and tt == 0)) else 0.05
            p_rec_a = 0.60 if (da < 1.2 or (is_touch and tt == 1)) else 0.05

            # Ground tackle prob
            p_tack_a = 0.15 if (da < 0.9 and dh > 1.2) else 0.001
            p_tack_h = 0.15 if (dh < 0.9 and da > 1.2) else 0.001

            # Transition row 0: from Held_H
            # -> Held_H, Flight_H, Held_A, Flight_A, Dead
            log_trans[t, 0, 0] = np.log(max(1e-4, 1.0 - p_k - p_tack_a - self.p_dead))
            log_trans[t, 0, 1] = np.log(max(1e-4, p_k))
            log_trans[t, 0, 2] = np.log(max(1e-6, p_tack_a))
            log_trans[t, 0, 3] = -99.0  # Impossible to kick flight for opponent
            log_trans[t, 0, 4] = np.log(max(1e-4, self.p_dead))

            # Transition row 1: from Flight_H
            # -> Held_H (reception), Flight_H (continuation), Held_A (interception), Flight_A, Dead
            log_trans[t, 1, 0] = np.log(max(1e-4, p_rec_h))
            log_trans[t, 1, 1] = np.log(max(1e-4, 1.0 - p_rec_h - p_rec_a - self.p_dead))
            log_trans[t, 1, 2] = np.log(max(1e-6, p_rec_a))
            log_trans[t, 1, 3] = -99.0  # Flight cannot switch mid-air without interception
            log_trans[t, 1, 4] = np.log(max(1e-4, self.p_dead))

            # Transition row 2: from Held_A
            # -> Held_H, Flight_H, Held_A, Flight_A, Dead
            log_trans[t, 2, 0] = np.log(max(1e-6, p_tack_h))
            log_trans[t, 2, 1] = -99.0  # Impossible to kick flight for opponent
            log_trans[t, 2, 2] = np.log(max(1e-4, 1.0 - p_k - p_tack_h - self.p_dead))
            log_trans[t, 2, 3] = np.log(max(1e-4, p_k))
            log_trans[t, 2, 4] = np.log(max(1e-4, self.p_dead))

            # Transition row 3: from Flight_A
            # -> Held_H (interception), Flight_H, Held_A (reception), Flight_A (continuation), Dead
            log_trans[t, 3, 0] = np.log(max(1e-6, p_rec_h))
            log_trans[t, 3, 1] = -99.0  # Flight cannot switch mid-air
            log_trans[t, 3, 2] = np.log(max(1e-4, p_rec_a))
            log_trans[t, 3, 3] = np.log(max(1e-4, 1.0 - p_rec_h - p_rec_a - self.p_dead))
            log_trans[t, 3, 4] = np.log(max(1e-4, self.p_dead))

            # Transition row 4: from Dead
            # -> Held_H, Flight_H, Held_A, Flight_A, Dead
            p_rest_h = self.p_restart if dh < 1.5 else 0.01
            p_rest_a = self.p_restart if da < 1.5 else 0.01
            log_trans[t, 4, 0] = np.log(max(1e-4, p_rest_h))
            log_trans[t, 4, 1] = np.log(max(1e-5, p_rest_h * 0.2))
            log_trans[t, 4, 2] = np.log(max(1e-4, p_rest_a))
            log_trans[t, 4, 3] = np.log(max(1e-5, p_rest_a * 0.2))
            log_trans[t, 4, 4] = np.log(max(1e-4, 1.0 - p_rest_h - p_rest_a))

            # Normalize rows in log space
            for s in range(5):
                max_v = np.max(log_trans[t, s])
                lse = max_v + np.log(np.sum(np.exp(log_trans[t, s] - max_v)))
                log_trans[t, s] -= lse

        return log_trans

    def forward_backward_smooth(self, probs_3class, v_ball, dH, dA, touch_mask, touch_team):
        """
        Run forward-backward offline smoothing algorithm.

        Parameters:
        -----------
        probs_3class : (T, 4) probability array with columns [P(H), P(A), P(D), P(X)]
        v_ball       : (T,) ball speed
        dH, dA       : (T,) distance to nearest Home and Away player
        touch_mask   : (T,) bool array of detected touches
        touch_team   : (T,) int array of touch attribution (0=H, 1=A, -1=none)

        Returns:
        --------
        smoothed_probs : (T, 4) smoothed marginal probabilities [P(H), P(A), P(D), P(X)]
        """
        T = len(probs_3class)
        log_trans = self.build_transition_matrix(v_ball, dH, dA, touch_mask, touch_team)

        # Map 3-class emissions to 5-state emissions:
        # States: 0=Held_H, 1=Flight_H, 2=Held_A, 3=Flight_A, 4=Dead
        # Ball in flight: weight Flight states higher
        is_high_v = (v_ball > 3.0)
        emit = np.zeros((T, 5), dtype=np.float64)

        pH = np.clip(probs_3class[:, 0], 1e-5, 0.999)
        pA = np.clip(probs_3class[:, 1], 1e-5, 0.999)
        pD = np.clip(probs_3class[:, 2], 1e-5, 0.999)

        for t in range(T):
            if is_high_v[t]:
                emit[t, 0] = pH[t] * 0.2
                emit[t, 1] = pH[t] * 0.8
                emit[t, 2] = pA[t] * 0.2
                emit[t, 3] = pA[t] * 0.8
            else:
                emit[t, 0] = pH[t] * 0.8
                emit[t, 1] = pH[t] * 0.2
                emit[t, 2] = pA[t] * 0.8
                emit[t, 3] = pA[t] * 0.2
            emit[t, 4] = pD[t]

        # Normalize emissions
        emit /= np.sum(emit, axis=1, keepdims=True)
        log_emit = np.log(np.clip(emit, 1e-7, 1.0))

        # Forward pass (in log space)
        log_alpha = np.zeros((T, 5), dtype=np.float64)
        log_alpha[0] = log_emit[0] - np.log(np.sum(np.exp(log_emit[0] - np.max(log_emit[0]))))

        for t in range(1, T):
            # log_alpha[t, j] = log_emit[t, j] + logsumexp_i(log_alpha[t-1, i] + log_trans[t-1, i, j])
            for j in range(5):
                prev_terms = log_alpha[t - 1] + log_trans[t - 1, :, j]
                max_val = np.max(prev_terms)
                log_alpha[t, j] = log_emit[t, j] + (max_val + np.log(np.sum(np.exp(prev_terms - max_val))))

            # Normalize for numerical stability
            max_a = np.max(log_alpha[t])
            log_alpha[t] -= (max_a + np.log(np.sum(np.exp(log_alpha[t] - max_a))))

        # Backward pass (in log space)
        log_beta = np.zeros((T, 5), dtype=np.float64)
        log_beta[-1] = 0.0

        for t in range(T - 2, -1, -1):
            for i in range(5):
                next_terms = log_trans[t, i, :] + log_emit[t + 1] + log_beta[t + 1]
                max_val = np.max(next_terms)
                log_beta[t, i] = max_val + np.log(np.sum(np.exp(next_terms - max_val)))

            # Normalize
            max_b = np.max(log_beta[t])
            log_beta[t] -= (max_b + np.log(np.sum(np.exp(log_beta[t] - max_b))))

        # Posterior marginals: gamma_t(s) \propto alpha_t(s) * beta_t(s)
        log_gamma = log_alpha + log_beta
        gamma = np.zeros((T, 5), dtype=np.float32)
        for t in range(T):
            max_g = np.max(log_gamma[t])
            gamma[t] = np.exp(log_gamma[t] - max_g)
            gamma[t] /= np.sum(gamma[t])

        # Marginalize 5 states back to 4 classes [H, A, D, X]
        # Home = Held_H (0) + Flight_H (1)
        # Away = Held_A (2) + Flight_A (3)
        # Dead = Dead (4)
        smoothed = np.zeros((T, 4), dtype=np.float32)
        smoothed[:, 0] = gamma[:, 0] + gamma[:, 1]
        smoothed[:, 1] = gamma[:, 2] + gamma[:, 3]
        smoothed[:, 2] = gamma[:, 4]
        smoothed[:, 3] = 0.0

        smoothed /= np.sum(smoothed, axis=1, keepdims=True)
        return smoothed

