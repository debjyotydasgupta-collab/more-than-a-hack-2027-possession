"""
inference_pipeline.py
=====================
Generalized Hybrid ML-Physics Inference Pipeline for Ball Possession Classification.
More than a Hack 2027 — Phase 1: Ball Possession.

Features:
  1. Student Ensemble (LightGBM + CatBoost) trained on 4-way pitch symmetry & full match
  2. Phase 3: Hybrid Physics-ML Uncertainty Blending (Shannon entropy + Voronoi inverse-dist prior)
  3. Phase 5: Asymmetric Viterbi sequence decoding (lower restart barrier Dead -> Live)
  4. Tactical Smart Tagging overrides (guaranteed goal stoppages)
  5. Short spell pruning (min_spell_frames=15, 0.6s)
  6. Match clock boundary enforcement (pre-kickoff, halftime, post-match -> Dead Ball)
  7. Output generation & validation (submission_gbdt_v1.csv & submission_possession_v1.csv)

Usage:
    & "C:\Program Files\FreeCAD 1.0\bin\python.exe" inference_pipeline.py
"""

import os, sys, pickle, time, warnings, argparse
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- Path setup ---
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "starter"))
sys.path.insert(0, os.path.join(REPO_ROOT, "mtah2027_possession"))

# --- Parse arguments ---
parser = argparse.ArgumentParser(description="Hybrid ML-Physics Inference Pipeline")
parser.add_argument("--out", type=str, default=None, help="Output CSV path for submission")
cli_args, _ = parser.parse_known_args()

# --- Constants ---
ATD_TXT     = os.path.join(REPO_ROOT, "data", "tactical", "BAR-ATH_tactical_atd.txt")
ATD_XML     = os.path.join(REPO_ROOT, "data", "tactical", "BAR-ATH_tactical_atd.xml")
FEAT_PATH   = os.path.join(REPO_ROOT, "models", "features_all.npy")
LGB_PATH    = os.path.join(REPO_ROOT, "models", "lgb_model.pkl")
CB_PATH     = os.path.join(REPO_ROOT, "models", "cb_model.pkl")
RESULTS_DIR = os.path.join(REPO_ROOT, "evaluation", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUT_CSV = cli_args.out if cli_args.out else os.path.join(RESULTS_DIR, "submission_possession_v1.csv")
ALT_CSV = os.path.join(RESULTS_DIR, "submission_gbdt_v1.csv")

PITCH_LENGTH = 105.3
PITCH_WIDTH  = 68.0
N_FRAMES     = 199_319

INT_TO_LABEL = {0: "H", 1: "A", 2: "D", 3: "X"}

# Match clock boundary frames (forced Dead Ball)
DEAD_BALL_REGIONS = [
    (0, 11184),           # Pre-kickoff
    (90652, 110807),      # Halftime
    (190198, 199318),     # Post-match
]

MIN_SPELL_FRAMES = 15

# Viterbi parameters (Asymmetric)
TRANS_PENALTY_SWAP = -2.5          # Penalty for H <-> A direct turnover
TRANS_PENALTY_DEAD = -1.0          # Penalty for Live -> Dead (stoppage)
TRANS_PENALTY_DEAD_TO_LIVE = -0.3  # Lower penalty for Dead -> Live restart

# ===========================================================================
# STEP 1: Load trained Student Models
# ===========================================================================
print("=" * 60)
print("STEP 1: Loading trained Student Models...")

lgb_model = cb_model = None

if os.path.exists(LGB_PATH):
    with open(LGB_PATH, "rb") as f:
        lgb_model = pickle.load(f)
    print(f"  Student LightGBM loaded ✓  ({LGB_PATH})")
else:
    print(f"  LightGBM model NOT FOUND at {LGB_PATH}")

if os.path.exists(CB_PATH):
    with open(CB_PATH, "rb") as f:
        cb_model = pickle.load(f)
    print(f"  Student CatBoost  loaded ✓  ({CB_PATH})")
else:
    print(f"  CatBoost model NOT FOUND at {CB_PATH}")

if (lgb_model is None and cb_model is None) or not os.path.exists(FEAT_PATH):
    print("Models or feature matrix missing. Automatically training self-supervised pipeline...")
    import subprocess
    cmd = [sys.executable, os.path.join(REPO_ROOT, "train_model.py"), "--mode", "self_supervised"]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    if os.path.exists(LGB_PATH):
        with open(LGB_PATH, "rb") as f: lgb_model = pickle.load(f)
    if os.path.exists(CB_PATH):
        with open(CB_PATH, "rb") as f: cb_model = pickle.load(f)

# ===========================================================================
# STEP 2: Load features and raw ATD coordinates for physics fallback
# ===========================================================================
print("\nSTEP 2: Loading feature matrix & raw tracking coordinates...")
t0 = time.time()

if not os.path.exists(FEAT_PATH):
    print("ERROR: features_all.npy could not be generated. Please check ATD data.")
    sys.exit(1)

X_all = np.load(FEAT_PATH)
print(f"  Loaded feature matrix: {X_all.shape} ({time.time()-t0:.1f}s)")

# Load tracking for physical prior
from mtah2027_possession.src.data.load_atd import load_raw_atd
frames_raw, team_of_track = load_raw_atd(ATD_TXT, n_frames=N_FRAMES)
frames_m = frames_raw.copy()
frames_m[:, :, 0] *= PITCH_LENGTH
frames_m[:, :, 1] *= PITCH_WIDTH

# ===========================================================================
# STEP 3: Ensemble inference -> (N, 4) Probability Matrix
# ===========================================================================
print("\nSTEP 3: Running Student GBDT Ensemble inference...")
BATCH = 50_000

def predict_proba_batched(model, X, batch_size=BATCH, model_name="model"):
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

if cb_model is not None:
    probs_cb = predict_proba_batched(cb_model, X_all, model_name="CB")
    n_models += 1

if n_models == 2:
    probs_ensemble = 0.5 * probs_lgb + 0.5 * probs_cb
elif lgb_model is not None:
    probs_ensemble = probs_lgb
else:
    probs_ensemble = probs_cb

# ===========================================================================
# STEP 4: Phase A & 3 — Hybrid Physics-ML Uncertainty Blending & Physical Gating
# ===========================================================================
print("\nSTEP 4: Applying Hybrid Physics-ML Uncertainty Blending & Physical Gating...")

H_mask = (team_of_track == 0)
A_mask = (team_of_track == 1)
bx = frames_m[:, 46, 0]
by = frames_m[:, 46, 1]
ball_valid = ~np.isnan(bx)
players_m = frames_m[:, :46, :]

dH = np.nanmin(np.sqrt((players_m[:, H_mask, 0] - bx[:, None])**2 + 
                       (players_m[:, H_mask, 1] - by[:, None])**2), axis=1)
dA = np.nanmin(np.sqrt((players_m[:, A_mask, 0] - bx[:, None])**2 + 
                       (players_m[:, A_mask, 1] - by[:, None])**2), axis=1)

diff_b = np.diff(np.nan_to_num(frames_m[:, 46, :2], nan=0.0), axis=0, prepend=np.nan_to_num(frames_m[[0], 46, :2], nan=0.0))
v_ball = np.sqrt(diff_b[:, 0]**2 + diff_b[:, 1]**2) / 0.04

from mtah2027_possession.src.models.hybrid_blender import HybridPhysicsBlender
blender = HybridPhysicsBlender(entropy_threshold=0.35, max_fallback=0.40)
probs_hybrid = blender.blend(probs_ensemble, dH, dA, v_ball, ball_valid)
print("  Physics-ML uncertainty blending & consistency gating complete ✓")

# Step 4B: Ballistics Kalman Engine for Flight Latching (Pass-in-Flight)
print("  Applying Ballistics Kalman Engine for pass-in-flight latching & pseudo-3D arcs...")
from mtah2027_possession.src.physics.ballistics import BallisticsEngine
bal_engine = BallisticsEngine(dt=0.04, kappa=0.08, sigma_z=0.15, q_acc=1.5)
feats_bal, _ = bal_engine.compute_latched_features(frames_m[:, 46, :2], frames_m[:, :46, :2], team_of_track)
is_flight = feats_bal[:, 2].astype(bool)
latched_H = feats_bal[:, 3]
z_ball = feats_bal[:, 8]

for i_f in range(N_FRAMES):
    if is_flight[i_f] or z_ball[i_f] > 1.8:
        max_p = max(probs_hybrid[i_f, 0], probs_hybrid[i_f, 1])
        if max_p < 0.85:
            lh = latched_H[i_f]
            if lh > 0.6:
                probs_hybrid[i_f, 0] = max(probs_hybrid[i_f, 0], 0.88)
                probs_hybrid[i_f, 1] = min(probs_hybrid[i_f, 1], 0.10)
            elif lh < 0.4:
                probs_hybrid[i_f, 1] = max(probs_hybrid[i_f, 1], 0.88)
                probs_hybrid[i_f, 0] = min(probs_hybrid[i_f, 0], 0.10)

probs_hybrid = np.clip(probs_hybrid, 1e-6, 1.0)
probs_hybrid /= np.sum(probs_hybrid, axis=1, keepdims=True)

# Effective 3D distance accounting for vertical height above turf
dH_3d = bal_engine.compute_effective_3d_distance(dH, z_ball)
dA_3d = bal_engine.compute_effective_3d_distance(dA, z_ball)

# ===========================================================================
# STEP 5: Phase C & 5 — Proximity-Adaptive Dynamic Viterbi Sequence Decoding
# ===========================================================================
print("\nSTEP 5: Proximity-Adaptive Dynamic Viterbi Sequence Decoding (with 3D Aerial Suppression)...")

from mtah2027_possession.src.postprocess.viterbi_decoder import ViterbiSequenceDecoder
vd = ViterbiSequenceDecoder(
    trans_penalty_swap=TRANS_PENALTY_SWAP,
    trans_penalty_dead=TRANS_PENALTY_DEAD,
    trans_penalty_dead_to_live=TRANS_PENALTY_DEAD_TO_LIVE,
)

labels_raw = vd.decode(probs_hybrid, dH=dH_3d, dA=dA_3d, z_ball=z_ball)
if isinstance(labels_raw[0], (int, np.integer)):
    labels = np.array([INT_TO_LABEL[int(x)] for x in labels_raw], dtype="<U1")
else:
    labels = np.array(labels_raw, dtype="<U1")

print(f"  Label distribution after Viterbi: { {l: int(np.sum(labels==l)) for l in ['H','A','D','X']} }")

# ===========================================================================
# STEP 6: Tactical Smart Tagging overrides (True Dead Ball Events)
# ===========================================================================
print("\nSTEP 6: Applying tactical overrides (Guaranteed Stoppages)...")

try:
    from mtah2027_possession.src.data.load_smart_tagging import load_with_team_labels
    events = load_with_team_labels()
    n_overridden = 0
    for ev in events:
        code = ev.get("code", "").strip().upper()
        if code in ("GOALS", "GOAL"):
            s = max(0, ev["start_frame"])
            e = min(N_FRAMES, ev["end_frame"])
            labels[s:e] = "D"
            n_overridden += (e - s)
    print(f"  Smart Tagging: overrode {n_overridden:,} goal stoppage frames to D")
except Exception as e:
    print(f"  Smart Tagging override skipped: {e}")

# ===========================================================================
# STEP 7: Short Spell Pruner (min 15 frames = 0.6s)
# ===========================================================================
print(f"\nSTEP 7: Spell pruning (min_spell_frames={MIN_SPELL_FRAMES})...")

try:
    from mtah2027_possession.src.postprocess.spell_pruner import PruneShortSpells
    pruner = PruneShortSpells(min_spell_frames=MIN_SPELL_FRAMES)
    labels = pruner.prune_short_spells(labels)
    print("  Spell pruning ✓ (via PruneShortSpells)")
except Exception as e:
    print(f"  PruneShortSpells fallback ({e})...")
    i = 0
    while i < N_FRAMES:
        cur = labels[i]
        j = i + 1
        while j < N_FRAMES and labels[j] == cur:
            j += 1
        spell_len = j - i
        if spell_len < MIN_SPELL_FRAMES and cur != "D":
            prev_lbl = labels[i-1] if i > 0 else "D"
            next_lbl = labels[j]   if j < N_FRAMES else "D"
            replace   = prev_lbl if prev_lbl == next_lbl else ("D" if "D" in (prev_lbl, next_lbl) else prev_lbl)
            labels[i:j] = replace
        i = j

# ===========================================================================
# STEP 7B: Phase 1 — Boundary Impulse Snapping (Kinematic Impact Alignment)
# ===========================================================================
print("\nSTEP 7B: Boundary Impulse Snapping (Kinematic Impact Alignment)...")
try:
    from mtah2027_possession.src.postprocess.boundary_snapper import BoundaryImpulseSnapper
    snapper = BoundaryImpulseSnapper(dt=0.04, search_window=4, accel_thresh=22.0)
    labels = snapper.snap_boundaries(labels, frames_m[:, 46, :2], frames_m[:, :46, :2], team_of_track)
    print("  Boundary impulse snapping complete ✓")
except Exception as e:
    print(f"  Boundary impulse snapping skipped ({e})")

# ===========================================================================
# STEP 8: Match clock boundary enforcement (Strict Dead Ball)
# ===========================================================================
print("\nSTEP 8: Enforcing match clock boundaries (Dead Ball)...")

for s, e in DEAD_BALL_REGIONS:
    s = max(0, s)
    e = min(N_FRAMES - 1, e)
    labels[s:e+1] = "D"
    print(f"  Frames {s:,}..{e:,} -> D ({e-s+1:,} frames)")

label_dist = {l: int(np.sum(labels == l)) for l in ["H", "A", "D", "X"]}
print(f"  Final label distribution: {label_dist}")

# ===========================================================================
# STEP 9: Write submission CSVs
# ===========================================================================
print(f"\nSTEP 9: Writing submission CSVs...")
out_df = pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels})
out_df.to_csv(OUT_CSV, index=False)
out_df.to_csv(ALT_CSV, index=False)
print(f"  Written {len(out_df):,} rows to {OUT_CSV} and {ALT_CSV}")
print(f"  Label counts: {out_df['label'].value_counts().to_dict()}")

# ===========================================================================
# STEP 10: Validate and Score
# ===========================================================================
print("\nSTEP 10: Validating & Scoring...")

validate_script = os.path.join(REPO_ROOT, "evaluation", "validate_submission.py")
if os.path.exists(validate_script):
    import subprocess
    r = subprocess.run(
        [sys.executable, validate_script, OUT_CSV],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    print(r.stdout)

eval_script = os.path.join(REPO_ROOT, "evaluation", "run_evaluation.py")
if os.path.exists(eval_script) and os.path.exists(os.path.join(REPO_ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")):
    import subprocess
    r = subprocess.run(
        [sys.executable, eval_script],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    print(r.stdout)

print("\n" + "=" * 60)
print("HYBRID INFERENCE PIPELINE COMPLETE")
print("=" * 60)
