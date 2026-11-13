# FC Barcelona Hackathon "More Than A Hack 2027"
# Ball Possession Tracking — Solution README & Blueprint

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Setup Instructions](#setup-instructions)
3. [Running the Pipeline & Reproduction](#running-the-pipeline--reproduction)
4. [Official Leaderboard Benchmark](#official-leaderboard-benchmark)
5. [Metric Explanation: spellF1 vs macroF1](#metric-explanation-spellf1-vs-macrof1)
6. [Key Algorithmic Decisions](#key-algorithmic-decisions)
7. [Submission Deliverables](#submission-deliverables)

---

## Architecture Overview

The solution is a **5-layer possession tracking engine**:

```
Raw ATD Data (.txt, .xml)
       │
       ▼
┌────────────────────────────────────────────────────────────┐
│  Layer 1 — Data Ingestion & Trajectory Smoothing          │
│  • load_atd.py  →  (199319, 47, 3) raw coordinate stream   │
│  • RTSBallSmoother (Forward Kalman + RTS backward pass)    │
│    State: [x, y, vx, vy, ax, ay], dt=0.04s                 │
│    Caps speed at 38 m/s, marks >125 frame gaps as NaN      │
│  • load_smart_tagging.py → 638 tactical phase annotations  │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│  Layer 2 — Multi-Scale Feature Engineering (92D)           │
│  • micro_features.py (32D): player distances, closing      │
│    velocities, approach angles, proximity ratios           │
│  • spatial_features.py (40D): bounding box areas,         │
│    team centroids, spatial spread, Voronoi cell dominance  │
│  • pitch_control.py (20D): Spearman/Fernández TTI physics  │
│    model evaluating time-to-intercept and control fields   │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│  Layer 3 — GBDT Soft Ensemble                              │
│  • LightGBM: leaf-wise growth, fast greedy partitioning    │
│  • CatBoost: symmetric oblivious trees, robust boosting    │
│  • Transition boundary weighting (3.0x on ±5 frame delta) │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│  Layer 4 — Temporal Sequence Optimization                  │
│  • ViterbiSequenceDecoder: dynamic programming traceback   │
│    Transition penalties: H↔A: -2.5 nats, H/A↔D: -1.0 nats  │
│  • Tactical Smart Tagging priors: Set Pieces/Goals → D     │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────┐
│  Layer 5 — Boundary Enforcement & Spell Pruning            │
│  • PruneShortSpells: merges spurious flickers (<15 frames) │
│  • Deterministic Match Clock Boundary Masking:             │
│    Pre-kickoff (0..11184), Halftime (90652..110807),       │
│    Post-match (190198..199318) strictly forced to D        │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
     submission_possession_v1.csv (199,319 rows, H/A/D/X)
```

---

## Setup Instructions

```bash
# Python 3.10+ / 3.11 recommended
pip install numpy pandas scikit-learn lightgbm catboost pyyaml
```

---

## Running the Pipeline & Reproduction

### Step 1: Train Models & Compute Features
```bash
python train_model.py
```
- Smooths ball trajectory with RTS Kalman smoother.
- Generates 92D feature matrix and saves to `models/features_all.npy`.
- Trains LightGBM and CatBoost classifiers with transition sample weights.
- Evaluates 5-fold OOF cross-validation (OOF macro-F1: **0.9975**).
- Saves model artifacts to `models/lgb_model.pkl` and `models/cb_model.pkl`.

### Step 2: Run Full Ensemble Inference
```bash
python inference_pipeline.py
```
- Performs batched inference across all 199,319 frames.
- Applies Viterbi sequence decoding with transition penalties.
- Applies Smart Tagging tactical constraints and match clock boundaries.
- Executes minimum spell pruning ($N_{\min}=15$).
- Writes and validates `evaluation/results/submission_possession_v1.csv`.

### Step 3: Official Scoring & Leaderboard Verification
```bash
python evaluation/run_evaluation.py
```

---

## Official Leaderboard Benchmark

Evaluated against the competition's ground truth with `python evaluation/run_evaluation.py`:

```
scoreboard  gt/BAR-ATH_possession_gt_public.csv  199319 frames  7500 graded  21 gt segments
ground truth has 17 possessions of mean 14.2 s

rank  submission                macroF1  spellF1   poss     dur  transF1  lat_f  shareE  frag
----  ------------------------  -------  -------  -----  ------  -------  -----  ------  ----
   1  submission_possession_v1   0.9195   0.9375     15    14.7   0.8947   +0.0   0.001  0.90
   2  submission_gbdt_v1         0.9195   0.9375     15    14.7   0.8947   +0.0   0.001  0.90
   3  baseline_smooth125         0.5812   0.4242     16    18.5   0.1667   -7.7   0.052  0.81
   4  baseline_smooth50          0.6207   0.3256     26    10.8   0.0816   +4.0   0.048  1.43
   5  baseline_smooth25          0.6200   0.2868     36     7.6   0.2034   +8.2   0.056  1.90
   6  baseline_raw               0.5972   0.0378    131     2.0   0.1410   -3.1   0.046  6.52
```

### Highlights:
- **`spellF1`:** **0.9375** (#1 Rank, +0.5133 above best organizer baseline)
- **`macroF1`:** **0.9195** ($H$: 0.9552, $A$: 0.9541, $D$: 0.8491)
- **Transition $F_1$:** **0.8947** with **+0.0 frames latency** (zero temporal lag)
- **Possession Count & Duration:** 15 possessions at 14.7s average (GT: 17 possessions at 14.2s)
- **Possession Share Error:** **0.001** (0.1% error)

---

## Metric Explanation: spellF1 vs macroF1

The competition ranks on **`spellF1`**, not `macroF1`.

- `macroF1` measures per-frame accuracy over $\{H, A, D\}$. It is blind to segmentation structure and allows spurious flickers.
- `spellF1` treats each possession as a continuous temporal interval $[t_{\text{start}}, t_{\text{end}}]$ and matches predicted spells to ground truth spells using temporal Intersection-over-Union ($\text{IoU} \in \{0.3, 0.4, 0.5, 0.6, 0.7\}$).
- Rapid toggling ($H \to A \to H$) creates spurious unmatched spells, crushing precision. Our Viterbi decoder and spell pruner explicitly eliminate state churn to optimize $\text{spellF1}$.

---

## Key Algorithmic Decisions

1. **RTS Smoother over simple Kalman:** Forward-backward smoothing provides zero-phase-lag interpolation of ball coordinates.
2. **125-frame gap policy:** Occlusions $>5$ seconds are marked as `NaN` and forced to Dead Ball $D$, preventing filter drift.
3. **Transition boundary sample weighting ($3.0\times$):** Frames around turnovers receive higher training weight, creating sharp transitions.
4. **Asymmetric Viterbi transition penalties:** $H \leftrightarrow A$ turnovers cost $-2.5$ nats, while dead-ball transitions cost $-1.0$ nats.
5. **Deterministic match clock boundaries:** Applied after spell pruning to ensure zero boundary leakage.

---

## Submission Deliverables

Per `SUBMISSION.md`:
1. **Possession File:** `evaluation/results/submission_possession_v1.csv` (199,319 rows, passes `validate_submission.py`).
2. **Public Code Repository:** Clean git repository with code, configs, and reproduction scripts (no match data).
3. **Technical Document:** Detailed technical report in `TECHNICAL_REPORT.md`.
