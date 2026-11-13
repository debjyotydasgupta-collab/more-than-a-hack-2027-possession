"""
inference_pipeline.py
=====================
Full GBDT ensemble inference pipeline for Ball Possession classification.
More than a Hack 2027 — Phase 1: Ball Possession.

Steps:
  1. Load pre-trained LightGBM + CatBoost models
  2. Load pre-computed feature matrix (or recompute if missing)
  3. Run ensemble inference across ALL 199,319 frames → (N, 4) probability matrix
  4. Apply Viterbi decoder for temporal smoothing
  5. Apply smart-tagging rule overrides (SET PIECES → D, etc.)
  6. Enforce match clock boundary rules (pre-kickoff, halftime, post-match → D)
  7. Apply spell pruner (min_spell_frames=15)
  8. Output evaluation/results/submission_gbdt_v1.csv
  9. Validate and score

Usage:
    & "C:\Program Files\FreeCAD 1.0\bin\python.exe" inference_pipeline.py
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- Path setup ---
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "starter"))
sys.path.insert(0, os.path.join(REPO_ROOT, "mtah2027_possession"))

# --- Constants ---
ATD_TXT     = os.path.join(REPO_ROOT, "data", "tactical", "BAR-ATH_tactical_atd.txt")
ATD_XML     = os.path.join(REPO_ROOT, "data", "tactical", "BAR-ATH_tactical_atd.xml")
FEAT_PATH   = os.path.join(REPO_ROOT, "models", "features_all.npy")
LGB_PATH    = os.path.join(REPO_ROOT, "models", "lgb_model.pkl")
CB_PATH     = os.path.join(REPO_ROOT, "models", "cb_model.pkl")
RESULTS_DIR = os.path.join(REPO_ROOT, "evaluation", "results")
OUT_CSV     = os.path.join(RESULTS_DIR, "submission_gbdt_v1.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

PITCH_LENGTH = 105.3
PITCH_WIDTH  = 68.0
N_FRAMES     = 199_319

INT_TO_LABEL = {0: "H", 1: "A", 2: "D", 3: "X"}

# Match clock boundary frames (inclusive) → forced Dead Ball
DEAD_BALL_REGIONS = [
    (0, 11184),           # Pre-kickoff
    (90652, 110807),      # Halftime
    (190198, 199318),     # Post-match
]

# Minimum spell duration (frames)
MIN_SPELL_FRAMES = 15

# Viterbi parameters
TRANS_PENALTY_SWAP = -2.5    # penalty for H↔A swap
TRANS_PENALTY_DEAD = -1.0    # penalty for H/A ↔ D swap
TRANS_PENALTY_SAME =  0.0    # no penalty for same-class

# ===========================================================================
# STEP 1: Load models
# ===========================================================================
print("=" * 60)
print("STEP 1: Loading trained models...")

lgb_model = cb_model = None

if os.path.exists(LGB_PATH):
    with open(LGB_PATH, "rb") as f:
        lgb_model = pickle.load(f)
    print(f"  LightGBM loaded ✓  ({LGB_PATH})")
else:
    print(f"  LightGBM model NOT FOUND at {LGB_PATH}")

if os.path.exists(CB_PATH):
    with open(CB_PATH, "rb") as f:
        cb_model = pickle.load(f)
    print(f"  CatBoost  loaded ✓  ({CB_PATH})")
else:
    print(f"  CatBoost model NOT FOUND at {CB_PATH}")

if lgb_model is None and cb_model is None:
    print("ERROR: No models found. Run train_model.py first.")
    sys.exit(1)

# ===========================================================================
# STEP 2: Load feature matrix (precomputed) or recompute
# ===========================================================================
print("\nSTEP 2: Loading feature matrix...")

if os.path.exists(FEAT_PATH):
    X_all = np.load(FEAT_PATH)
    print(f"  Loaded cached features: {X_all.shape} from {FEAT_PATH}")
else:
    print("  Feature matrix not found — recomputing from ATD...")
    import time
    t0 = time.time()

    from mtah2027_possession.src.data.load_atd import load_raw_atd
    frames_raw, team_of_track = load_raw_atd(ATD_TXT, n_frames=N_FRAMES)

    # Metres conversion
    frames_m = frames_raw.copy()
    frames_m[:, :, 0] *= PITCH_LENGTH
    frames_m[:, :, 1] *= PITCH_WIDTH

    player_positions_m = frames_m[:, :46, :2]
    player_speed_ms    = frames_m[:, :46, 2]

    from mtah2027_possession.src.preprocessing.kalman_filter import RTSBallSmoother
    smoother = RTSBallSmoother()
    ball_xy_m_smooth = smoother.smooth_ball_trajectory(frames_raw)

    ball_diff    = np.diff(ball_xy_m_smooth, axis=0, prepend=ball_xy_m_smooth[[0]])
    ball_speed   = np.sqrt(ball_diff[:, 0]**2 + ball_diff[:, 1]**2) / 0.04
    ball_accel   = np.diff(ball_speed, prepend=ball_speed[0]) / 0.04
    ball_speed   = np.nan_to_num(ball_speed, nan=0.0).astype(np.float32)
    ball_accel   = np.nan_to_num(ball_accel,  nan=0.0).astype(np.float32)

    from mtah2027_possession.src.features.micro_features    import MicroFeatureExtractor
    from mtah2027_possession.src.features.spatial_features  import SpatialFeatureExtractor
    from mtah2027_possession.src.features.pitch_control     import PitchControlFeatureExtractor

    X_micro   = MicroFeatureExtractor(team_of_track, PITCH_LENGTH, PITCH_WIDTH).extract(
                    player_positions_m, ball_xy_m_smooth, ball_speed, ball_accel)
    X_spatial = SpatialFeatureExtractor(team_of_track, PITCH_LENGTH, PITCH_WIDTH).extract(
                    player_positions_m, ball_xy_m_smooth)
    X_pc      = PitchControlFeatureExtractor(team_of_track, PITCH_LENGTH, PITCH_WIDTH).extract(
                    player_positions_m, player_speed_ms, ball_xy_m_smooth)

    X_all = np.concatenate([X_micro, X_spatial, X_pc], axis=1).astype(np.float32)
    os.makedirs(os.path.dirname(FEAT_PATH), exist_ok=True)
    np.save(FEAT_PATH, X_all)
    print(f"  Feature extraction time: {time.time()-t0:.1f}s, shape: {X_all.shape}")

# ===========================================================================
# STEP 3: Ensemble inference → probability matrix (N, 4)
# ===========================================================================
print("\nSTEP 3: Running GBDT ensemble inference...")
BATCH = 50_000

def predict_proba_batched(model, X, batch_size=BATCH, model_name="model"):
    """Batched probability prediction to avoid OOM."""
    N = X.shape[0]
    probs = np.zeros((N, 4), dtype=np.float32)
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = np.arange(4)
    else:
        classes = np.asarray(classes).ravel().astype(int)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        p = model.predict_proba(X[start:end])
        for ci, c in enumerate(classes):
            if 0 <= c < 4 and ci < p.shape[1]:
                probs[start:end, c] = p[:, ci]
        pct = int(100 * end / N)
        print(f"    [{model_name}] {end:>7,}/{N:,} frames ({pct}%)", end="\r")
    print()
    return probs

probs_lgb = np.full((N_FRAMES, 4), 0.25, dtype=np.float32)
probs_cb  = np.full((N_FRAMES, 4), 0.25, dtype=np.float32)
n_models  = 0

if lgb_model is not None:
    probs_lgb = predict_proba_batched(lgb_model, X_all, model_name="LGB")
    n_models += 1
    print(f"  LGB proba sample (frame 12000): {probs_lgb[12000]}")

if cb_model is not None:
    probs_cb = predict_proba_batched(cb_model, X_all, model_name="CB")
    n_models += 1
    print(f"  CB  proba sample (frame 12000): {probs_cb[12000]}")

# Ensemble average (equal weight)
if n_models == 2:
    probs_ensemble = 0.5 * probs_lgb + 0.5 * probs_cb
elif lgb_model is not None:
    probs_ensemble = probs_lgb
else:
    probs_ensemble = probs_cb

print(f"  Ensemble proba sample (frame 12000): {probs_ensemble[12000]}")

# ===========================================================================
# STEP 4: Viterbi decoder for temporal smoothing
# ===========================================================================
print("\nSTEP 4: Viterbi decoding...")

try:
    from mtah2027_possession.src.postprocess.viterbi_decoder import ViterbiSequenceDecoder
    from mtah2027_possession.src.data.load_smart_tagging import load_with_team_labels
    events = load_with_team_labels()
    vd = ViterbiSequenceDecoder(
        trans_penalty_swap=TRANS_PENALTY_SWAP,
        trans_penalty_dead=TRANS_PENALTY_DEAD,
    )
    # decode() expects probability matrix (N,4), returns string array ['H','A','D','X']
    labels_raw = vd.decode(probs_ensemble, smart_tag_events=events)
    print("  Viterbi decoding ✓")
    if isinstance(labels_raw[0], (int, np.integer)):
        labels = np.array([INT_TO_LABEL[int(x)] for x in labels_raw], dtype="<U1")
    else:
        labels = np.array(labels_raw, dtype="<U1")
except Exception as e:
    print(f"  Viterbi failed ({e}), falling back to argmax...")
    labels_int = np.argmax(probs_ensemble, axis=1)
    labels = np.array([INT_TO_LABEL[i] for i in labels_int], dtype="<U1")

print(f"  Label distribution after Viterbi: { {l: int(np.sum(labels==l)) for l in ['H','A','D','X']} }")

# ===========================================================================
# STEP 5: Smart Tagging rule overrides
# ===========================================================================
print("\nSTEP 5: Applying Smart Tagging rule overrides...")

try:
    from mtah2027_possession.src.data.load_smart_tagging import load_with_team_labels

    events = load_with_team_labels()
    # Dead-ball event codes → force D
    DEAD_BALL_CODES = {
        "SET PIECES", "CORNER KICK", "FREE KICK", "THROW IN",
        "GOAL KICK", "PENALTY KICK", "KICK OFF", "GOAL", "FOUL", "OUT OF PLAY"
    }
    n_overridden = 0
    for ev in events:
        code = ev.get("code", "").strip().upper()
        if any(dbc in code for dbc in DEAD_BALL_CODES):
            s = max(0, ev["start_frame"])
            e = min(N_FRAMES, ev["end_frame"])
            labels[s:e] = "D"
            n_overridden += (e - s)
    print(f"  Smart Tagging: overrode {n_overridden:,} frames to D via {len(events)} events")
except Exception as e:
    print(f"  Smart Tagging override skipped: {e}")

# ===========================================================================
# STEP 6: Spell pruning (min 15 frames) — done BEFORE boundary enforcement
# ===========================================================================
print(f"\nSTEP 6: Spell pruning (min_spell_frames={MIN_SPELL_FRAMES})...")

try:
    from mtah2027_possession.src.postprocess.spell_pruner import PruneShortSpells
    pruner = PruneShortSpells(min_spell_frames=MIN_SPELL_FRAMES)
    labels = pruner.prune_short_spells(labels)
    print("  Spell pruning ✓ (via PruneShortSpells.prune_short_spells)")
except Exception as e:
    print(f"  PruneShortSpells module failed ({e}), using inline pruner...")
    # Inline spell pruner
    i = 0
    while i < N_FRAMES:
        cur = labels[i]
        j = i + 1
        while j < N_FRAMES and labels[j] == cur:
            j += 1
        spell_len = j - i
        if spell_len < MIN_SPELL_FRAMES and cur != "D":
            # Find most common neighbor label
            prev_lbl = labels[i-1] if i > 0 else "D"
            next_lbl = labels[j]   if j < N_FRAMES else "D"
            replace   = prev_lbl if prev_lbl == next_lbl else ("D" if "D" in (prev_lbl, next_lbl) else prev_lbl)
            labels[i:j] = replace
        i = j

# ===========================================================================
# STEP 7: Match clock boundary enforcement (Dead Ball) — strictly enforced last
# ===========================================================================
print("\nSTEP 7: Enforcing match clock boundaries (Dead Ball)...")

for s, e in DEAD_BALL_REGIONS:
    s = max(0, s)
    e = min(N_FRAMES - 1, e)
    labels[s:e+1] = "D"
    print(f"  Frames {s:,}..{e:,} → D ({e-s+1:,} frames)")

label_dist = {l: int(np.sum(labels == l)) for l in ["H", "A", "D", "X"]}
print(f"  Final label distribution: {label_dist}")

# ===========================================================================
# STEP 8: Write submission CSV
# ===========================================================================
print(f"\nSTEP 8: Writing submission CSV → {OUT_CSV}")

out_df = pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels})
out_df.to_csv(OUT_CSV, index=False)
print(f"  Written {len(out_df):,} rows")
print(f"  Label counts: {out_df['label'].value_counts().to_dict()}")

# ===========================================================================
# STEP 9: Validate
# ===========================================================================
print("\nSTEP 9: Validating submission...")

validate_script = os.path.join(REPO_ROOT, "evaluation", "validate_submission.py")
if os.path.exists(validate_script):
    import subprocess
    r = subprocess.run(
        ["C:\\Program Files\\FreeCAD 1.0\\bin\\python.exe", validate_script, OUT_CSV],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    print(r.stdout)
    if r.stderr:
        print("STDERR:", r.stderr[:500])
else:
    print("  validate_submission.py not found, skipping validation")

# ===========================================================================
# STEP 10: Score
# ===========================================================================
print("\nSTEP 10: Scoring...")

eval_script = os.path.join(REPO_ROOT, "evaluation", "run_evaluation.py")
if os.path.exists(eval_script):
    import subprocess
    r = subprocess.run(
        ["C:\\Program Files\\FreeCAD 1.0\\bin\\python.exe", eval_script],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    print(r.stdout)
    if r.stderr:
        print("STDERR:", r.stderr[:500])
else:
    print("  run_evaluation.py not found, skipping scoring")

print("\n" + "=" * 60)
print("INFERENCE COMPLETE")
print(f"  Output: {OUT_CSV}")
print("=" * 60)
