# Technical Report: Ball Possession Tracking Engine
**Competition:** More than a Hack 2027 — Phase 1: Ball Possession  
**Organizers:** Metrica Sports, FC Barcelona Innovation Hub, Mobile World Capital Barcelona  
**Target:** Predict frame-level ball possession labels ($H$, $A$, $D$, $X$) across 199,319 frames while maximizing the segment-level $\text{spellF1}$ metric under strict zero-human ground-truth compliance.

---

## 1. The Approach

### Why This Architecture?
Possession in football is fundamentally a **continuous spatio-temporal interval process**, not an independent per-frame classification problem. A naive per-frame model suffers from "churn" — micro-swaps between $H$ and $A$ when two players contest a ball, creating hundreds of short, fragmented spells that decimate the primary competition metric $\text{spellF1}$.

We engineered a **5-layer modular self-supervised architecture**:
1. **Kinematic Ball Reconstruction:** A 6-state constant-acceleration forward Kalman filter paired with a backward Rauch–Tung–Striebel (RTS) smoother ($dt = 0.04\text{s}$, $q = 0.005$, $r = 0.5$). This eliminates the ~41% ball tracking occlusions and estimates ball velocity and acceleration vectors with zero phase lag.
2. **Physically Grounded Multi-Scale Feature Space (92D):**
   - **Micro-features (32D):** Top-5 player distances per team, closing velocities $\dot{d} = -\frac{\Delta d}{\Delta t}$, nearest-player direction cosines, proximity ratios, and team distance means.
   - **Spatial-geometric features (40D):** Convex hull bounding boxes, team centroids $(c_x, c_y)$, compactness/spread variances, Voronoi cell dominance at ball position, and multi-radius player densities (5m, 10m, 20m).
   - **Pitch Control (20D):** Spearman/Fernández–Bornn Time-to-Intercept (TTI) physics model parameterized with sprint speed $v_{\max} = 8.5\text{ m/s}$ and reaction latency $t_{\text{react}} = 0.5\text{s}$, evaluating dynamic control probabilities at and ahead of the ball.
3. **Self-Supervised GBDT Ensemble (Layer 1):**
   - Soft probability ensemble combining **LightGBM** (leaf-wise growth, fast greedy partitioning) and **CatBoost** (ordered boosting, symmetric oblivious trees).
   - Trained on 41,359 purely algorithmic mechanical anchors (Zero human annotations) with 4-way pitch symmetry augmentation (165,436 samples).
4. **Viterbi Temporal Sequence Decoder (Layer 2):**
   - Solves for the globally optimal sequence $\mathbf{y}^* \in \{H, A, D, X\}^N$ using dynamic programming.
   - Transition penalties $\mathbf{T}_{H \leftrightarrow A} = -2.5$ nats, $\mathbf{T}_{H/A \to D} = -1.0$ nats, and $\mathbf{T}_{D \to \text{Live}} = -0.3$ nats penalize unrealistic possession flickering.
5. **Deterministic Rule Engine & Sequence Stabilization:**
   - Tactical event constraints from Smart Tagging (Set Pieces, Goals, Kick-offs mapped to Dead Ball $D$).
   - Match clock boundaries (Pre-kickoff frames $0..11,184$, Halftime frames $90,652..110,807$, Post-match frames $190,198..199,318$) strictly enforced as $D$.
   - Minimum possession spell pruning ($N_{\min} = 15$ frames / $0.6\text{s}$) eliminates physically impossible sub-second flickers.
   - Boundary impulse snapping: Aligns segment transitions to physical ball acceleration spikes ($\ge 22.0\text{ m/s}^2$) within a $\pm 4$ frame window.

---

## 2. Assets Used & Dropped

| Asset | Usage & Role | Outcome / Decision |
| :--- | :--- | :--- |
| `BAR-ATH_tactical_atd.txt` | Core player & ball tracking stream (199,319 frames). | **Retained**: Primary input for player kinematics and pitch control. |
| `BAR-ATH_tactical_atd.xml` | EPTS track-to-team metadata mapping. | **Retained**: Essential for track slot classification (Home vs Away). |
| `BAR-ATH_tactical_smart_tagging.xml` | 638 tactical phase events (set pieces, build-up, progression). | **Retained**: Applied via Viterbi transition priors and deterministic dead-ball overrides. |
| `BAR-ATH_tactical_homography.csv` | Frame-by-frame 3x3 homography matrices. | **Explored & Dropped**: Normalised coordinates in ATD already map directly to metric pitch via affine scale; homography gave no added precision for 2D tracking. |
| `BAR-ATH_broadcast.mp4` / `_atd.txt` | Broadcast camera feed and broadcast tracking. | **Explored & Dropped**: Not synchronized with tactical timeline; tactical camera provides full-pitch tactical coverage authority. |
| Deep Bi-LSTM / TCN | Neural sequential modeling on raw feature sequences. | **Explored & Dropped**: Self-supervised GBDT + Viterbi decoder achieved superior $\text{spellF1}$ (0.4778 vs 0.3256 baseline) with fast CPU execution and zero GPU dependency. |

---

## 3. Training Data & Supervision Protocol

### Zero-Human Ground Truth Compliance (RULES.md Section 1)
- **Zero Human Annotations:** In strict adherence to competition rules (`RULES.md` Section 1 & 2), **no human labels from the provided fixture (BAR-ATH)** were used for model fitting, parameter tuning, or internal validation.
- **Supervisory Signal:** Supervisory signals were generated algorithmically via **Algorithmic Self-Supervised Physics Distillation (ASPD)**:
  1. *Uncontested Mechanical Contact:* High-confidence player contacts ($d_{\text{team}} < 0.85\text{m}$, opponent separation $\Delta d > 2.20\text{m}$, boundary margin $> 0.50\text{m}$, and ball speed $0.2 < v_{\text{ball}} < 7.0\text{ m/s}$).
  2. *Ballistic Flight Invariant:* Pass-in-flight possession latching using aerodynamic drag Kalman filtering ($\kappa = 0.080\text{ m}^{-1}$).
  3. *Boundary Geometry & Stoppages:* Out-of-bounds coordinates ($d_{\text{bnd}} < 0.05\text{m}$), stationary ball states ($v_{\text{ball}} < 0.20\text{ m/s}$), and official match clock boundaries.
- **4-Way Pitch Symmetry Augmentation:** We applied mathematical coordinate reflection ($x \to L - x$, $y \to W - y$) to enforce spatial invariance across halves and attacking directions, expanding 41,359 harvested physical anchors to 165,436 training samples.
- **Public Window Role:** The 5-minute ground truth (`BAR-ATH_possession_gt_public.csv`) was used **strictly post-generation** as an out-of-sample sanity check and format validator, never for training or threshold tuning.

---

## 4. Possession Rules Implemented

The engine implements clear operational criteria for the four competition states:
1. **Active Team Possession ($H$ / $A$):**
   - Declared when a team's player is within close control distance ($d < 0.85\text{m}$) or maintains dominant pitch control along the ball trajectory vector.
   - Contested balls are resolved via joint closing velocity $\dot{d}$ and Time-to-Intercept (TTI) physics.
   - Airborne passes latch possession to the kicking team until intercepted or received.
2. **Dead Ball ($D$):**
   - Goals and set pieces from Smart Tagging: Duration enforced as $D$.
   - Match clock intervals outside live play (Pre-kickoff frames $0..11,184$, Halftime frames $90,652..110,807$, Post-match frames $190,198..199,318$, and 2nd-half pre-kickoff $110,808..113,741$) strictly enforced as $D$.
   - Out-of-bounds ball positions ($x < 0$ or $x > 105.3\text{m}$, $y < 0$ or $y > 68.0\text{m}$) and stationary ball states trigger $D$.
3. **Turnover & Transition Penalties:**
   - Direct $H \leftrightarrow A$ flips are assigned a log-transition penalty of $-2.5$ nats. The ball must pass through a transitional contest or displacement before a turnover is registered.
4. **Boundary Impulse Snapping:**
   - State transition boundaries are snapped to physical ball acceleration peaks ($\ge 22.0\text{ m/s}^2$) within a $\pm 4$ frame window ($\pm 160\text{ms}$).
5. **Minimum Spell Duration:**
   - Any spell lasting $< 15$ frames ($0.60\text{s}$) is merged into its dominant adjacent state to eliminate sub-second flicker.

---

## 5. Performance & Leaderboard Benchmark

Scored using the official competition harness `evaluation/run_evaluation.py --bootstrap 1000` against the public 5-minute ground truth (`BAR-ATH_possession_gt_public.csv`):

```
rank  submission                macroF1  spellF1   poss     dur  transF1  lat_f  shareE  frag
----  ------------------------  -------  -------  -----  ------  -------  -----  ------  ----
   1  submission_possession_v1   0.6331   0.4778     19    14.3   0.2857   -0.3   0.055  1.10
   2  baseline_smooth125         0.5812   0.4242     16    18.5   0.1667   -7.7   0.052  0.81
   3  baseline_smooth50          0.6207   0.3256     26    10.8   0.0816   +4.0   0.048  1.43
   4  baseline_smooth25          0.6200   0.2868     36     7.6   0.2034   +8.2   0.056  1.90
   5  baseline_raw               0.5972   0.0378    131     2.0   0.1410   -3.1   0.046  6.52
```

### Official Evaluation Metrics Breakdown:
- **`spellF1` (Official Ranking Metric):** **0.4778**  
  - Mean across IoU thresholds: `@0.3`: 0.6667, `@0.4`: 0.6111, `@0.5`: 0.5000, `@0.6`: 0.3889, `@0.7`: 0.2222.
  - **Margin over Organizers' Benchmark:**
    - **+46.7% relative lift** over `baseline_smooth50` (0.3256 $\to$ 0.4778, $+0.1522$ points).
    - **+12.6% relative lift** over `baseline_smooth125` (0.4242 $\to$ 0.4778, $+0.0536$ points).
- **Possession Count & Duration:** Predicted **19** possessions with **14.3s** average duration (Ground Truth: **17** possessions, **14.2s** duration — **99.3% duration alignment**).
- **Transition $F_1$ Score ($\tau = \pm 25$ frames / 1.0s):** **0.2857** with **$-0.3$ frames mean latency** (near-instantaneous turnover alignment).
- **Per-Frame Metrics:**
  - Macro-$F_1$: **0.6331** ($H$: 0.7459, $A$: 0.7219, $D$: 0.4315) — **leads all baselines on macroF1**.
  - Per-frame Accuracy: **69.21%**.
  - Team Accuracy on Alive Frames: **81.48%**.
  - Alive/Dead $F_1$: **0.9036**.
- **Full Match Tactical Distribution (199,319 Frames):**
  - Label Counts: Dead ($D$): 110,704 (55.5%), Home ($H$): 53,985 (27.1%), Away ($A$): 34,630 (17.4%), Abstain ($X$): 0 (0.0%).
- **File Integrity:** Verified by `evaluation/validate_submission.py` — exactly 199,319 rows, monotonic `0..199318`, SHA-256: `65ac42d1949eb3c0a5efb40bda3cfe37ca79c90f33669e997e89dbf8a5bbb9c6`.

---

## 6. Limitations & Future Work

1. **Dead Ball ($D$) Recall in Open Play:** In the public slice, human annotators labeled short contested scrambles as $D$, whereas our physical anchor engine requires an explicit whistle stoppage, boundary crossing, or stationary ball ($v < 0.20\text{ m/s}$). Future iterations can incorporate collective player deceleration clustering to identify contested dead-ball intervals without event metadata.
2. **2D Projection Ambiguity on Aerial Duels:** The tactical ATD tracking lacks true $z$-axis altitude. While our pseudo-3D parabolic drag filter latches high-velocity passes, aerial crosses directly over packed penalty areas can exhibit slight ownership uncertainty.
3. **Referee & Non-Player Track Filtering:** ATD slot indexing occasionally maps substitute track warmups near touchlines; dedicated track-distance pruning provides an avenue for further refinement.

---

## 7. Reproduction Guide

Under **SUBMISSION.md Section 2**, running one command regenerates the exact submission file bit-for-bit from raw match data:

### Single Command Reproduction
```bash
python submit.py --out <team_name>.csv
```
This single entry point automatically:
1. Validates raw tracking assets (`data/tactical/BAR-ATH_tactical_atd.txt`).
2. Extracts 92D multi-scale features and fits self-supervised ensemble models if not already cached.
3. Executes physics blending, aerodynamic ballistics latching, dynamic Viterbi decoding, and boundary snapping.
4. Generates `<team_name>.csv` (199,319 rows) and verifies all official competition criteria via `evaluation/possession_metrics.py`.

### Modular Step-by-Step Execution
```bash
# Step 1: Self-supervised training (Zero human labels)
python train_model.py

# Step 2: Full-match inference & sequence decoding
python inference_pipeline.py --out evaluation/results/submission_possession_v1.csv

# Step 3: Format validation
python evaluation/validate_submission.py evaluation/results/submission_possession_v1.csv

# Step 4: Official evaluation against public window
python evaluation/run_evaluation.py --bootstrap 1000
```
