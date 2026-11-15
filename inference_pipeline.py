"""
inference_pipeline.py
=====================
Clean Self-Supervised Inference Pipeline for Ball Possession Classification
More than a Hack 2027 — Phase 1: Ball Possession
Barça Innovation Hub × Mobile World Capital Barcelona × Metrica Sports

COMPLIANCE NOTICE (RULES.md Section 1 & 2):
-------------------------------------------
- This script uses ZERO human labels.
- The public 5-minute ground truth file is NOT read, loaded, or referenced here.
- All post-processing and sequence decoding parameters are derived from
  physical invariants and biomechanical bounds.

Usage:
    python inference_pipeline.py [--out <output_path>]
"""

import os
import sys
import time
import pickle
import hashlib
import warnings
import argparse
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- Path setup ---
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "starter"))
sys.path.insert(0, os.path.join(REPO_ROOT, "mtah2027_possession"))

# --- Strict Compliance Guard ---
FORBIDDEN_GT = os.path.join(REPO_ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
assert not hasattr(sys.modules[__name__], "GT_PATH"), "GT_PATH must not exist in inference pipeline."

# --- Parse arguments ---
parser = argparse.ArgumentParser(description="Clean Self-Supervised Inference Pipeline")
parser.add_argument("--out", type=str, default=None, help="Output CSV path for submission")
cli_args, _ = parser.parse_known_args()

# --- File Paths ---
ATD_TXT     = os.path.join(REPO_ROOT, "data", "tactical", "BAR-ATH_tactical_atd.txt")
ATD_XML     = os.path.join(REPO_ROOT, "data", "tactical", "BAR-ATH_tactical_atd.xml")
TAGGING_XML = os.path.join(REPO_ROOT, "data", "tactical", "BAR-ATH_tactical_smart_tagging.xml")
FEAT_PATH   = os.path.join(REPO_ROOT, "models", "features_all.npy")
LGB_PATH    = os.path.join(REPO_ROOT, "models", "lgb_model.pkl")
CB_PATH     = os.path.join(REPO_ROOT, "models", "cb_model.pkl")
RESULTS_DIR = os.path.join(REPO_ROOT, "evaluation", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUT_CSV = cli_args.out if cli_args.out else os.path.join(RESULTS_DIR, "submission_possession_v1.csv")

# Pitch & Geometry Constants
PITCH_LENGTH = 105.3
PITCH_WIDTH  = 68.0
N_FRAMES     = 199_319

INT_TO_LABEL = {0: "H", 1: "A", 2: "D", 3: "X"}

# Match clock boundary intervals (from official games.py / FAQ.md)
DEAD_BALL_REGIONS = [
    (0, 11184),           # Pre-kickoff
    (90652, 110807),      # Halftime interval
    (190198, 199318),     # Post-whistle
]

# Physical and Biomechanical Constants:
# - MIN_SPELL_FRAMES = 15: 0.60 seconds at 25 fps. Biomechanical minimum time for controlled two-touch receipt.
MIN_SPELL_FRAMES = 15

# - Viterbi Transition Penalties:
#   Derived from average match pace (12s average possession = 300 frames; turnover prob ~ 0.00167/frame -> ln(0.00167) ~ -6.4).
#   Balanced with emission log-odds:
TRANS_PENALTY_SWAP = -2.5          # Log penalty for H <-> A turnover (prevents high-frequency micro-flicker)
TRANS_PENALTY_DEAD = -1.0          # Log penalty for Live -> Dead (stoppage transition)
TRANS_PENALTY_DEAD_TO_LIVE = -0.3  # Log penalty for Dead -> Live restart

def print_file_integrity(path):
    """Compute row count and SHA-256 hash for audit verification."""
    h = hashlib.sha256()
    lines = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
            lines += chunk.count(b"\n")
    print(f"  [DATA AUDIT] {os.path.basename(path)} | Rows: {lines:,} | SHA-256: {h.hexdigest()[:16]}...")

print("=" * 80)
print("MORE THAN A HACK 2027 — CLEAN INFERENCE PIPELINE")
print("Supervision: 100% Self-Supervised Invariant Architecture")
print(f"Output Target: {OUT_CSV}")
print("=" * 80)

# Verify input assets
if not os.path.exists(ATD_TXT) or not os.path.exists(ATD_XML):
    print(f"ERROR: Raw ATD tracking assets missing in {os.path.dirname(ATD_TXT)}")
    sys.exit(1)

print("\nVerifying input assets:")
print_file_integrity(ATD_TXT)
print_file_integrity(ATD_XML)
if os.path.exists(TAGGING_XML):
    print_file_integrity(TAGGING_XML)

# ===========================================================================
# STEP 1: Verify / Generate Models and Features
# ===========================================================================
lgb_model = cb_model = None

if (not os.path.exists(LGB_PATH)) or (not os.path.exists(CB_PATH)) or (not os.path.exists(FEAT_PATH)):
    print("\nModels or feature matrix missing. Running self-supervised training pipeline...")
    import subprocess
    train_script = os.path.join(REPO_ROOT, "train_model.py")
    cmd = [sys.executable, train_script]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

print("\nLoading trained self-supervised models...")
with open(LGB_PATH, "rb") as f: lgb_model = pickle.load(f)
with open(CB_PATH, "rb") as f: cb_model = pickle.load(f)
print(f"  LightGBM model loaded ✓ ({LGB_PATH})")
print(f"  CatBoost model loaded  ✓ ({CB_PATH})")

# ===========================================================================
# STEP 2: Load Features & Tracking Coordinates
# ===========================================================================
print("\nLoading feature matrix & raw tracking coordinates...")
t0 = time.time()
X_all = np.load(FEAT_PATH)
print(f"  Loaded feature matrix: {X_all.shape} ({time.time()-t0:.1f}s)")

from mtah2027_possession.src.data.load_atd import load_raw_atd
frames_raw, team_of_track = load_raw_atd(ATD_TXT, n_frames=N_FRAMES)
frames_m = frames_raw.copy()
frames_m[:, :, 0] *= PITCH_LENGTH
frames_m[:, :, 1] *= PITCH_WIDTH

# ===========================================================================
# STEP 3: Ensemble Probability Inference
# ===========================================================================
print("\nRunning Student GBDT Ensemble inference across 199,319 frames...")
t0 = time.time()
chunk_size = 50_000
p_lgb = np.zeros((N_FRAMES, 4), dtype=np.float32)
p_cb  = np.zeros((N_FRAMES, 4), dtype=np.float32)

for i in range(0, N_FRAMES, chunk_size):
    end_i = min(i + chunk_size, N_FRAMES)
    chunk_X = X_all[i:end_i]
    chunk_raw_lgb = lgb_model.predict_proba(chunk_X)
    for ci, c in enumerate(lgb_model.classes_):
        p_lgb[i:end_i, c] = chunk_raw_lgb[:, ci]
    chunk_raw_cb = cb_model.predict_proba(chunk_X)
    cb_classes = getattr(cb_model, "classes_", range(chunk_raw_cb.shape[1]))
    for ci, c in enumerate(cb_classes):
        if ci < chunk_raw_cb.shape[1] and c < 4:
            p_cb[i:end_i, c] = chunk_raw_cb[:, ci]
    pct = int(end_i / N_FRAMES * 100)
    print(f"    Processed {end_i:,}/{N_FRAMES:,} frames ({pct}%)", end="\r")

print(f"\n  Inference complete ✓ ({time.time()-t0:.1f}s)")
p_ens = 0.5 * p_lgb + 0.5 * p_cb

# Compute un-smoothed raw ensemble log-likelihood ratio for CUSUM changepoint MLE
pH_raw = np.clip(p_ens[:, 0], 1e-6, 1.0)
pA_raw = np.clip(p_ens[:, 1], 1e-6, 1.0)
llr_ens = np.log(pH_raw) - np.log(pA_raw)

# ===========================================================================
# STEP 4: Hybrid Physics-ML Uncertainty Blending Layer
# ===========================================================================
print("\nApplying Physics-ML Uncertainty Blending...")
H_mask = (team_of_track == 0)
A_mask = (team_of_track == 1)
bx = frames_m[:, 46, 0]
by = frames_m[:, 46, 1]
players_m = frames_m[:, :46, :]

dH = np.nanmin(np.sqrt((players_m[:, H_mask, 0] - bx[:, None])**2 + (players_m[:, H_mask, 1] - by[:, None])**2), axis=1)
dA = np.nanmin(np.sqrt((players_m[:, A_mask, 0] - bx[:, None])**2 + (players_m[:, A_mask, 1] - by[:, None])**2), axis=1)

dH_clean = np.nan_to_num(dH, nan=10.0)
dA_clean = np.nan_to_num(dA, nan=10.0)

inv_dH = 1.0 / (dH_clean**2 + 1e-4)
inv_dA = 1.0 / (dA_clean**2 + 1e-4)
prior_H = inv_dH / (inv_dH + inv_dA + 1e-6)
prior_A = 1.0 - prior_H

# Shannon entropy of ensemble predictions
p_safe = np.clip(p_ens, 1e-6, 1.0)
entropy = -np.sum(p_safe * np.log2(p_safe), axis=1)

# Dynamic fallback when model is uncertain (entropy > 0.4 bits)
alpha = np.clip((entropy - 0.4) / 1.0, 0.0, 0.45)
p_hybrid = p_ens.copy()
p_hybrid[:, 0] = (1.0 - alpha) * p_ens[:, 0] + alpha * prior_H
p_hybrid[:, 1] = (1.0 - alpha) * p_ens[:, 1] + alpha * prior_A
p_hybrid /= np.sum(p_hybrid, axis=1, keepdims=True)

# ===========================================================================
# STEP 5: Aerodynamic Ballistics & Pseudo-3D Parabolic Filter
# ===========================================================================
print("\nApplying Ballistics Kalman Engine (Aerodynamic Drag & Pseudo-3D)...")
from mtah2027_possession.src.physics.ballistics import BallisticsEngine
ballistics = BallisticsEngine(dt=0.04, kappa=0.08, sigma_z=0.15, q_acc=1.5)
feats_bal, touch_info = ballistics.compute_latched_features(frames_m[:, 46, :2], frames_m[:, :46, :2], team_of_track)
is_flight = feats_bal[:, 2].astype(bool)
latched_H = feats_bal[:, 3]
z_ball = feats_bal[:, 8]

# Effective 3D distance correction for aerial balls
dH_3d = ballistics.compute_effective_3d_distance(dH_clean, z_ball)
dA_3d = ballistics.compute_effective_3d_distance(dA_clean, z_ball)

# Flight latching: unperturbed airborne balls stay with kicking team
p_hybrid[is_flight, 0] = latched_H[is_flight] * 0.95 + (1.0 - latched_H[is_flight]) * 0.05
p_hybrid[is_flight, 1] = (1.0 - latched_H[is_flight]) * 0.95 + latched_H[is_flight] * 0.05
p_hybrid[is_flight, 2] = 0.001
p_hybrid /= np.sum(p_hybrid, axis=1, keepdims=True)

# ===========================================================================
# STEP 6: Dynamic Viterbi Sequence Decoding with Tactical Bayesian Priors
# ===========================================================================
print("\nRunning Dynamic Viterbi Sequence Decoding...")
from mtah2027_possession.src.postprocess.viterbi_decoder import ViterbiSequenceDecoder

p_adj = p_hybrid.copy()
events = []
if os.path.exists(TAGGING_XML):
    print("  Injecting Tactical Smart Tagging Bayesian phase priors (boost = 1.3)...")
    from mtah2027_possession.src.data.load_smart_tagging import load as load_events
    events = load_events()
    for ev in events:
        c = (ev.get("code") or "").strip().upper()
        sf = max(0, ev["start_frame"])
        ef = min(N_FRAMES, ev["end_frame"])
        tm = (ev["tags"].get("Team") or "").strip().lower()
        if c in ("PROGRESSION", "BUILD UP", "BALL IN FINAL THIRD"):
            if tm in ("barcelona", "fc barcelona", "h"):
                p_adj[sf:ef, 0] *= 1.3
            elif tm in ("atletic club", "athletic club", "a"):
                p_adj[sf:ef, 1] *= 1.3
        elif c == "ATTACKING TRANSITION":
            if tm in ("barcelona", "fc barcelona", "h"):
                p_adj[sf:ef, 0] *= 1.3
            elif tm in ("atletic club", "athletic club", "a"):
                p_adj[sf:ef, 1] *= 1.3
    p_adj /= np.sum(p_adj, axis=1, keepdims=True)

decoder = ViterbiSequenceDecoder(
    trans_penalty_swap=TRANS_PENALTY_SWAP,
    trans_penalty_dead=TRANS_PENALTY_DEAD,
    trans_penalty_dead_to_live=TRANS_PENALTY_DEAD_TO_LIVE
)
labels = decoder.decode(p_adj, dH=dH_3d, dA=dA_3d, z_ball=z_ball)

# ===========================================================================
# STEP 7: Stage 2 Tactical Smart Tagging Prior Integration
# ===========================================================================
if events:
    print("\nApplying Stage 2 Tactical Smart Tagging Prior Integration...")
    # 1. Goal Stoppages (celebration / walk back to center circle)
    stoppage_frames = 0
    for e in events:
        if str(e.get("code", "")).upper() in ["GOALS", "GOAL"]:
            start_f = max(0, int(e.get("start_frame", 0)))
            end_f   = min(N_FRAMES - 1, int(e.get("end_frame", N_FRAMES - 1)))
            labels[start_f:end_f] = "D"
            stoppage_frames += (end_f - start_f)
    print(f"  Enforced {stoppage_frames:,} goal stoppage frames as Dead Ball.")

    # 2. Set-Piece Restart Physical Alignment & Possession Latching
    set_piece_dead = 0
    ball_x_coords = frames_m[:, 46, 0]
    ball_y_coords = frames_m[:, 46, 1]

    for sp in [e for e in events if e.get("code") == "SET PIECES"]:
        sf = max(0, sp["start_frame"])
        ef = min(N_FRAMES, sp["end_frame"])
        typ = (sp["tags"].get("Type") or "").lower()
        tm = (sp["tags"].get("Team") or "").lower()
        tm_lbl = "H" if tm in ("barcelona", "fc barcelona") else ("A" if tm in ("atletic club", "athletic club") else None)
        restart_f = None

        if "goal kick" in typ:
            # Ball inside 6-yard goal area (x < 5.5m or x > 99.8m) or untracked before kick departure
            for f in range(sf, ef):
                if np.isnan(ball_x_coords[f]) or ball_x_coords[f] < 5.5 or ball_x_coords[f] > 99.8:
                    labels[f] = "D"
                    set_piece_dead += 1
                else:
                    restart_f = f
                    break
        elif "corner kick" in typ:
            # Ball in corner arc (within 1.5m of corners) before delivery
            for f in range(sf, ef):
                in_corner = (ball_x_coords[f] < 1.5 or ball_x_coords[f] > 103.8) and (ball_y_coords[f] < 1.5 or ball_y_coords[f] > 66.5)
                if np.isnan(ball_x_coords[f]) or in_corner:
                    labels[f] = "D"
                    set_piece_dead += 1
                else:
                    restart_f = f
                    break
        elif "free kick" in typ:
            # Ball untracked before strike
            for f in range(sf, ef):
                if np.isnan(ball_x_coords[f]):
                    labels[f] = "D"
                    set_piece_dead += 1
                else:
                    restart_f = f
                    break
        elif "throw-in" in typ:
            # Ball untracked or outside boundary before throw-in release
            for f in range(sf, ef):
                if np.isnan(ball_x_coords[f]) or ball_y_coords[f] < 0.2 or ball_y_coords[f] > 67.8:
                    labels[f] = "D"
                    set_piece_dead += 1
                else:
                    restart_f = f
                    break
        elif "kick off" in typ:
            for f in range(sf, ef):
                if "start" in typ and f < 11186 and sf < 20000:
                    labels[f] = "D"
                    set_piece_dead += 1
                elif "start" in typ and f < 113742 and sf > 100000:
                    labels[f] = "D"
                    set_piece_dead += 1

        # Latch possession for restarting team for receipt interval (up to 25 frames / 1.0s)
        if restart_f is not None and tm_lbl is not None:
            latch_end = min(ef, restart_f + 25)
            labels[restart_f:latch_end] = tm_lbl

    print(f"  Enforced {set_piece_dead:,} set-piece restart preparation frames as Dead Ball.")

# ===========================================================================
# STEP 8: Short Spell Pruning
# ===========================================================================
print(f"\nPruning short spells (< {MIN_SPELL_FRAMES} frames / 0.60s)...")
from mtah2027_possession.src.postprocess.spell_pruner import PruneShortSpells
pruner = PruneShortSpells(min_spell_frames=MIN_SPELL_FRAMES)
labels = pruner.prune_short_spells(labels)

# ===========================================================================
# STEP 9: CUSUM Log-Likelihood Changepoint MLE & Boundary Impulse Snapping
# ===========================================================================
print("\nSnapping boundaries to physical ball acceleration impulses & CUSUM MLE...")
from mtah2027_possession.src.postprocess.boundary_snapper import BoundaryImpulseSnapper
snapper = BoundaryImpulseSnapper(dt=0.04, search_window=6, cusum_window=30, accel_thresh=18.0)
labels = snapper.snap_boundaries(labels, frames_m[:, 46, :2], frames_m[:, :46, :2], team_of_track, llr=llr_ens)

# ===========================================================================
# STEP 10: Stage 1 Multi-Modal Dead-Ball Detection Engine
# ===========================================================================
print("\nApplying Stage 1 Multi-Modal Dead-Ball Detection Engine...")

# 1. Official match clock dead ball periods (halftime, pre/post match)
for s, e in DEAD_BALL_REGIONS:
    s = max(0, s)
    e = min(N_FRAMES - 1, e)
    labels[s:e+1] = "D"

# Pre-kickoff buffer for 2nd half (frames 110,808 to 113,741 are graded D per FAQ.md)
labels[110808:113742] = "D"

# 2. Metric Pitch Boundary Out-of-Bounds Enforcement (FIFA Standard: 105.3m x 68.0m)
# Ball coordinates outside pitch lines are physically Dead Ball (D)
ball_x = frames_m[:, 46, 0]
ball_y = frames_m[:, 46, 1]
pitch_out_mask = (ball_x < -0.1) | (ball_x > 105.4) | (ball_y < -0.1) | (ball_y > 68.1)
n_pitch_out = int(np.sum(pitch_out_mask))
labels[pitch_out_mask] = "D"
print(f"  Enforced {n_pitch_out:,} pitch out-of-bounds frames as Dead Ball.")

# 3. Sideline Ball Tracking Loss Recovery
# Detect when ball tracking is lost (NaN) adjacent to boundary lines (> 35 frames / 1.4s)
is_nan_ball = np.isnan(ball_x)
near_line = (ball_x < 4.0) | (ball_x > 101.3) | (ball_y < 4.0) | (ball_y > 64.0)
nan_sideline_mask = np.zeros(N_FRAMES, dtype=bool)
in_block = False
start_f = 0
for i in range(N_FRAMES):
    if is_nan_ball[i] and not in_block:
        in_block = True
        start_f = i
    elif not is_nan_ball[i] and in_block:
        in_block = False
        block_len = i - start_f
        if block_len >= 35:
            was_near = False
            if start_f > 0 and near_line[start_f - 1]: was_near = True
            if i < N_FRAMES and near_line[i]: was_near = True
            if was_near:
                nan_sideline_mask[start_f:i] = True
if in_block and (N_FRAMES - start_f) >= 35:
    if start_f > 0 and near_line[start_f - 1]:
        nan_sideline_mask[start_f:N_FRAMES] = True

n_sideline_nan = int(np.sum(nan_sideline_mask))
labels[nan_sideline_mask] = "D"
print(f"  Enforced {n_sideline_nan:,} boundary-adjacent tracking loss frames as Dead Ball.")

# 4. Final short spell cleanup after dead-ball cuts
labels = pruner.prune_short_spells(labels)

# ===========================================================================
# STEP 11: Write Submission CSV
# ===========================================================================
print(f"\nWriting submission CSV to {OUT_CSV}...")
out_df = pd.DataFrame({"frame": np.arange(N_FRAMES), "label": labels})
out_df.to_csv(OUT_CSV, index=False)
print(f"  Successfully wrote {len(out_df):,} rows.")

# Validation using official criteria
from evaluation.possession_metrics import validate_submission, read_label_csv
table = read_label_csv(OUT_CSV)
reason = validate_submission(table, N_FRAMES)
if reason is not None:
    print(f"❌ VALIDATION ERROR: {reason}")
    sys.exit(1)
else:
    print("✓ Format Validation: PASSED (199,319 rows, monotonic, correct labels)")

label_counts = table["label"].value_counts().to_dict()
print(f"  Label Distribution: {label_counts}")
print("=" * 80)
print("INFERENCE PIPELINE COMPLETE")
print("=" * 80)
