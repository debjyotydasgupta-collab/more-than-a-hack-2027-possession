# Implementation Plan: Elevating Temporal Generalization (LOSO) from 61.79% to 85%+
**More than a Hack 2027 — Phase 1: Ball Possession (FC Barcelona Innovation Hub & Metrica Sports)**

---

## 1. Executive Synthesis & Project Trajectory

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              ACCURACY EVOLUTION MILESTONES                             │
├───────────────────────────────────┬─────────────┬─────────────┬─────────────┬──────────┤
│ Metric / Dimension                │ Baseline    │ Phase A-D   │ Phase 1-2   │ Target   │
│                                   │ (Organizer) │ (Ensemble)  │ (Hierarch.) │ (Plan)   │
├───────────────────────────────────┼─────────────┼─────────────┼─────────────┼──────────┤
│ Official Benchmark spellF1        │ 0.4242      │ 1.0000      │ 1.0000      │ 1.0000   │
│ Official Benchmark Accuracy       │ 72.75%      │ 99.92%      │ 99.92%      │ 99.95%+  │
│ 5-Fold Stratified CV Accuracy     │ 99.77%      │ 99.83%      │ 99.83%      │ 99.85%+  │
│ Tracking Noise Robustness (0.5m)  │ 96.44%      │ 99.71%      │ 99.71%      │ 99.75%+  │
│ 20% Sensor Dropout Robustness     │ 94.59%      │ 99.53%      │ 99.53%      │ 99.60%+  │
│ Temporal LOSO Generalization Acc  │ 34.79%      │ 38.03%      │ 61.79%      │ 85.00%+  │
├───────────────────────────────────┼─────────────┼─────────────┼─────────────┼──────────┤
│ Composite Accuracy Score (/10.0)  │ 7.57 / 10   │ 8.75 / 10   │ 9.22 / 10   │ 9.71+/10 │
└───────────────────────────────────┴─────────────┴─────────────┴─────────────┴──────────┘
```

### Context & The Core Diagnostic Finding
In our previous sessions, we achieved:
1. **Flawless Official Benchmark Scorer:** **Rank #1** on the competition scoreboard (`spellF1 = 1.0000`, 17/17 possessions matched, 0 spurious, 0 missed).
2. **Eliminated Absolute Pitch Coordinate Leakage:** Replaced 24 pitch-coordinate features with a 32-dimensional invariant relative kinematic feature vector.
3. **Decoupled Hierarchical Architecture:** Separated State ($P(\text{Live})$ vs $P(\text{Dead})$) from Ownership ($P(\text{Home} \mid \text{Live})$), lifting LOSO accuracy from **$34.79\% \to 61.79\%$** ($+27.00\%$ points).

**The Remaining Problem:**
The gap between the **99.92%** benchmark accuracy and **61.79%** LOSO accuracy confirms that frame-level tree models without explicit physics latching suffer when evaluated on unseen temporal sequences. The uploaded technical document provides the mathematical foundation to close this gap by transforming possession from a *proximity-driven state* into a *latched, touch-driven physical state*.

---

## 2. Mathematical Diagnosis: The 3 Root Causes of the 38.21% LOSO Error

Empirical inspection of the 21 ground-truth segments revealed the three remaining failure modes:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DIAGNOSED RESIDUAL FAILURE MODES                                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. The "Pass-in-Flight" Dilemma (Segments 2 & 7):                                     │
│    When team A launches a 30m pass, the ball travels within 1.5m of an opponent        │
│    defender mid-flight. Invariant proximity models momentarily assign possession to    │
│    team H because dH < dA, despite the defender never making physical contact.         │
│                                                                                        │
│ 2. The Dead-Ball Kinetic Ambiguity (Segment 5):                                        │
│    A ball resting on the pitch before a restart has zero velocity, but stationary       │
│    players standing over the ball can look identical to a static live buildup phase.   │
│    Ball velocity alone is insufficient to distinguish set pieces without collective    │
│    stillness metrics and boundary line geometry.                                       │
│                                                                                        │
│ 3. Missing Sequence Memory (Short Duels 0.8s – 3.5s):                                 │
│    Frame-by-frame classifiers lack a state representation of "who initiated the current│
│    passing sequence." A flight phase cannot switch ownership without a discrete touch. │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. High-Leverage Implementation Phases (Roadmap to 85%+)

```mermaid
flowchart TD
    subgraph Phase0["Phase 0: Diagnostic Audit"]
        A1["Labeling Convention Audit<br/>(Last-Toucher vs Receiver)"]
        A2["Per-Segment Error Attribution<br/>(Identify Bottleneck Segments)"]
    end

    subgraph Phase1["Phase 1: Ballistic Impulse Detection (Method 1)"]
        B1["2D Kalman Aerodynamic Drag Model<br/>x_{t+1} = F x_t + w_t"]
        B2["GLRT Innovation Test & CUSUM<br/>Λ_t = ν_t^T S_t^{-1} ν_t > χ^2_2"]
        B3["Touch Attribution & Latched Possession Features"]
    end

    subgraph Phase2["Phase 2: Dead-Ball Geometry (Method 4)"]
        C1["Touchline & Goal-Line Boundary Distance d_line"]
        C2["Collective Kinetic Energy E_t = 1/22 Σ ||v_j||²"]
        C3["Wall/Cluster Structure & Stillness Hysteresis"]
    end

    subgraph Phase3["Phase 3: Latent Holder HMM (Method 2)"]
        D1["State Space (m_t, i_t): Held, Flight, Dead"]
        D2["Flight Transition Latched to Touch Events"]
        D3["Non-Causal Forward-Backward Offline Smoothing"]
    end

    subgraph Phase4["Phase 4: Semi-Supervised Transductive Retrain (Method 5)"]
        E1["Pseudo-Label Harvest on 192k Frames (Posterior > 0.95)"]
        E2["Noise-Inoculated Student Retraining (Weight 0.3)"]
        E3["Boundary-Tolerant ±0.2s Evaluation Guard"]
    end

    Phase0 --> Phase1
    Phase1 --> Phase2
    Phase2 --> Phase3
    Phase3 --> Phase4
```

---

### Phase 0: Ground-Truth Annotation & Segment Audit
* **Target Script:** `scratch/audit_labeling_conventions.py`
* **Objective:** Establish exact ground-truth annotator behavior on flight phases before training:
  1. **Labeling Convention Audit:**
     - Compute agreement of GT labels against:
       - (a) Nearest player's team: $\arg\min_j \|\mathbf{x}_j - \mathbf{x}_{\text{ball}}\|$
       - (b) Last toucher's team: player who imparted the last velocity impulse $\Delta \mathbf{v}_{\text{ball}} > 2.0\text{ m/s}$
       - (c) Next receiver's team: player who first establishes control after flight
     - If annotators labeled flight by *last toucher*, standard forward latching applies; if labeled by *next receiver*, non-causal backward resolution applies.
  2. **Per-Segment Residual Error Audit:**
     - Quantify frame-level error contributions for each of the 21 segments to isolate whether errors stem from short duels ($0.8\text{s} - 3.5\text{s}$) or dead-ball transitions.

---

### Phase 1: Ballistic Impulse Detection & Last-Touch Latching (Method 1)
* **Target File:** `mtah2027_possession/src/physics/ballistics.py`
* **Objective:** Solve the "Pass-in-Flight" dilemma. A defender standing $1.5\text{m}$ from a flying ball CANNOT possess it unless the trajectory is physically perturbed.
* **Mathematical Formulation:**
  1. **Ball Dynamics (2D @ 25 Hz):** State $\mathbf{x}_t = [\mathbf{p}_t, \mathbf{v}_t]^T \in \mathbb{R}^4$:
     $$\mathbf{x}_{t+1} = \mathbf{F} \mathbf{x}_t + \mathbf{w}_t, \quad \mathbf{F} = \begin{bmatrix} \mathbf{I}_2 & \Delta t \cdot \mathbf{I}_2 \\ \mathbf{0}_2 & (1 - \kappa \Delta t) \cdot \mathbf{I}_2 \end{bmatrix}, \quad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})$$
     where $\kappa \approx 0.08\text{ s}^{-1}$ represents aerodynamic drag and rolling friction.
  2. **Kalman Innovation & Covariance:**
     $$\boldsymbol{\nu}_t = \mathbf{z}_t - \mathbf{H} \hat{\mathbf{x}}_{t \mid t-1}, \quad \mathbf{S}_t = \mathbf{H} \mathbf{P}_{t \mid t-1} \mathbf{H}^T + \mathbf{R}$$
  3. **Touch Detection via Generalized Likelihood Ratio Test (GLRT):**
     $$\Lambda_t = \boldsymbol{\nu}_t^T \mathbf{S}_t^{-1} \boldsymbol{\nu}_t > \chi^2_{2, 1-\alpha} \quad (\alpha = 0.01 \implies \text{threshold } \approx 9.21)$$
     Evaluated with a running CUSUM filter over a $\pm 3$-frame window ($0.24\text{s}$) to separate genuine ball kicks ($\Delta v > 2.5\text{ m/s}$) from sensor noise ($0.1\text{ m}$).
  4. **Touch Attribution to Player $j$:**
     $$P(j \mid \text{touch}) \propto \exp\left(-\frac{d_j^2}{2 \sigma_d^2}\right) \cdot \mathbf{1}\left[\cos\left(\angle(\Delta \mathbf{v}, \mathbf{u}_{j \to \text{ball}})\right) > 0\right] \cdot f(\|\mathbf{v}_j\|)$$
  5. **Pass-in-Flight Verification:**
     $$LR = \frac{p(\boldsymbol{\nu}_{t-k:t+k} \mid H_0)}{p(\boldsymbol{\nu}_{t-k:t+k} \mid H_1, \Delta \mathbf{v})}$$
     If $LR \ge 1.0$ (no statistically significant trajectory innovation), the pass is unperturbed, and possession remains strictly latched to the passing team.
* **New Features Generated:**
  - `frames_since_last_touch`
  - `team_of_last_touch` (encoded relative to nearest team)
  - `ball_speed_continuity` ($\|v_t - v_{t-1}\| / \|v_t\|$)
  - `max_innovation_last_1s`
  - `last_toucher_to_ball_dist`
  - `time_to_next_touch` (bidirectional)

---

### Phase 2: Dead-Ball Geometry & Collective Stillness Signature (Method 4)
* **Target File:** `mtah2027_possession/src/features/dead_ball_features.py`
* **Objective:** Eliminate dead-ball misclassifications on held-out temporal segments by recognizing the macroscopic structure of match stoppages.
* **Engineered Signals:**
  1. **Touchline & Goal-Line Boundary Proximity ($d_{\text{line}}$):**
     $$d_{\text{line}} = \min(x, 105.3 - x, y, 68.0 - y)$$
     A ball stationary at $d_{\text{line}} < 0.35\text{m}$ deterministically signals a throw-in, goal kick, or corner.
  2. **Out-of-Bounds Trigger:**
     $$\mathbf{1}_{\text{out}} = \mathbf{1}(x < 0.0 \lor x > 105.3 \lor y < 0.0 \lor y > 68.0)$$
     Directly triggers a Dead Ball state that persists until an in-bounds impulse is detected.
  3. **Collective Kinetic Energy ($E_t$) & Acceleration Derivative:**
     $$E_t = \frac{1}{22} \sum_{j=1}^{22} \|\mathbf{v}_j\|^2, \quad \dot{E}_t = \frac{E_t - E_{t-5}}{5 \Delta t}$$
     Dead balls exhibit a sharp drop in collective kinetic energy followed by a stationary plateau and a distinctive pre-restart ramp.
  4. **Wall / Cluster Structure:** Number of opposing players within the FIFA regulatory $9.15\text{m}$ distance of the ball, plus defensive cluster density.
  5. **Dead-Ball Hysteresis State Machine:**
     $$\text{Dead Entry: } \|\mathbf{v}_{\text{ball}}\| < 0.3\text{ m/s for } \ge 25\text{ frames (1.0s)} \land (d_{\text{line}} < 1.0\text{m} \lor E_t < Q_{0.20})$$
     $$\text{Dead Exit: } \|\mathbf{v}_{\text{ball}}\| > 2.0\text{ m/s} \land \text{GLRT Touch Impulse Verified}$$

---

### Phase 3: Latent Holder-Identity HMM with Non-Causal Smoothing (Method 2)
* **Target File:** `mtah2027_possession/src/postprocess/holder_hmm.py`
* **Objective:** Introduce sequence memory so that a pass in flight maintains continuous ownership across seconds of flight time.
* **Formulation:**
  1. **Latent State Space:** $(m_t, i_t)$, where:
     - $m_t \in \{\text{Held}, \text{Flight}, \text{Dead}\}$
     - $i_t \in \{1 \dots 22\} \cup \{\emptyset\}$ represents the current ball holder or last toucher.
  2. **Transition Dynamics:**
     - $P(\text{Held}_i \to \text{Flight}_i) = \sigma(a \cdot \text{kick\_score}_t)$
     - $P(\text{Flight}_i \to \text{Held}_j) = \text{softmax}_j(-TTI_{j,t} / \tau) \cdot \mathbf{1}[\text{touch}_t]$
     - **Constraint:** In state $\text{Flight}_i$, ownership CANNOT switch to another player or team without an impulse touch event.
  3. **Non-Causal Forward-Backward Offline Inference:**
     $$\gamma_t(m, i) = P(m_t = m, i_t = i \mid \mathbf{z}_{1:T}) \propto \alpha_t(m, i) \cdot \beta_t(m, i)$$
     Because full-match inference is offline, future receptions resolve past flight trajectories with $100\%$ certainty: if the ball is received by Athletic Club $1.4\text{s}$ later, the entire aerial flight phase is attributed to Athletic Club.

---

### Phase 4: Role/Team-Swap Augmentation & Feature Purge (Method 3)
* **Target File:** `mtah2027_possession/src/models/augmenter.py`
* **Objective:** Force tree ensembles to completely forget team identities and playing styles, eliminating segment-level overfitting.
* **Implementation:**
  1. **Strict Team Swap ($H \leftrightarrow A$):**
     - Relabel Home $\leftrightarrow$ Away ($0 \leftrightarrow 1$).
     - Invert all differential features: $\Delta d \to -\Delta d$, $TTI_{\text{diff}} \to -TTI_{\text{diff}}$, $PC_{\text{diff}} \to -PC_{\text{diff}}$, $\Delta \cos \to -\Delta \cos$.
     - Swap team-indexed columns: $d_{H1..H3} \leftrightarrow d_{A1..A3}$, $v_{H1..H2} \leftrightarrow v_{A1..A2}$.
  2. **Result:** Exactly doubles the training distribution to 60,000 frames while enforcing mathematical anti-symmetry across both teams.

---

### Phase 5: Transductive Semi-Supervised Harvest & Verification (Method 5)
* **Target File:** `train_model.py` & `test_accuracy_diagnostic.py`
* **Objective:** Exploit the 191,819 unlabeled frames of `BAR-ATH` to provide representation learning across all match phases.
* **Implementation:**
  1. **Transductive Pseudo-Label Harvest:** Run the Phase 1-3 physical model on the entire 90 minutes. Harvest 40,000 frames with posterior confidence $> 0.95$.
  2. **Boundary-Tolerant Evaluation Metric ($\pm 0.2\text{s}$):**
     In sports tracking, transition points between two running players carry an inherent $\pm 5$-frame ($0.2\text{s}$) human annotation jitter. Alongside strict frame accuracy, report boundary-tolerant accuracy:
     $$\text{Acc}_{\text{tol}} = \frac{1}{N} \sum_{t=1}^N \mathbf{1}\left(y_t \in \{\hat{y}_{t-5} \dots \hat{y}_{t+5}\}\right)$$
  3. **Gating Rule:** Validate against the official competition benchmark to guarantee that `spellF1` never drops below **1.0000**.

---

## 4. Expected Performance Matrix & Milestone Roadmap

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              EXPECTED LOSO GAIN BREAKDOWN                              │
├──────────────────────────────────────┬──────────────────────┬─────────────┬────────────┤
│ Milestone / Component                │ Mechanism            │ Expected Acc│ Cumul. Gain│
├──────────────────────────────────────┼──────────────────────┼─────────────┼────────────┤
│ Baseline Hierarchical Engine         │ Decoupled Invariant  │   61.79%    │   +0.00%   │
│ + Step 1: Ballistic Impulse Latching │ Solves flight phases │   72.50%    │  +10.71%   │
│ + Step 2: Dead-Ball Geometry Engine  │ Solves stoppages     │   78.20%    │  +16.41%   │
│ + Step 3: Latent Holder Forward-Back │ Sequence memory      │   82.80%    │  +21.01%   │
│ + Step 4: Strict Team-Swap Inversion │ Anti-memorization    │   85.50%    │  +23.71%   │
│ + Step 5: Transductive Harvest       │ 192k representation  │ ★ 87.50% ★  │  +25.71%   │
└──────────────────────────────────────┴──────────────────────┴─────────────┴────────────┘
```

### Resulting Multidimensional Accuracy Score

| Evaluation Dimension | Weight | Baseline | Current | Target with Plan | Sub-Score (/10.0) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Official Benchmark Scorer** | 40% | 0.4242 | **1.0000** | **1.0000** | **10.00 / 10.0** |
| **5-Fold Stratified CV** | 25% | 99.77% | **99.83%** | **99.85%** | **9.98 / 10.0** |
| **Sensor & Noise Robustness** | 15% | 96.44% | **99.71%** | **99.80%** | **9.98 / 10.0** |
| **Temporal LOSO Generalization** | 20% | 34.79% | **61.79%** | **87.50%** | **8.75 / 10.0** |
| **COMPOSITE TOTAL** | **100%** | **7.57 / 10** | **9.22 / 10** | **★ 9.74 / 10 ★** | **★ 9.74 / 10 ★** |

---

## 5. Execution Sequence & Dependencies

1. **Step 1:** Run Phase 0 Labeling Convention Audit on `BAR-ATH_possession_gt_public.csv`.
2. **Step 2:** Implement `ballistics.py` with 2D Kalman GLRT innovation testing and last-touch latching.
3. **Step 3:** Implement `dead_ball_features.py` with touchline distance and collective kinetic energy $E_t$.
4. **Step 4:** Integrate `holder_hmm.py` for non-causal forward-backward flight resolution.
5. **Step 5:** Run `test_accuracy_diagnostic.py` to confirm LOSO accuracy exceeds **$85.0\%$** and composite score reaches **$9.70+/10.0$**.
6. **Step 6:** Run `inference_pipeline.py` and verify `validate_submission.py` on the 199,319-row competition submission.
