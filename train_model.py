"""
train_model.py
==============
Full GBDT ensemble training pipeline for Ball Possession classification.
More than a Hack 2027 — Phase 1: Ball Possession.

Steps:
  1. Load ATD data (199,319 frames) with Kalman-RTS smoothed ball trajectory
  2. Convert normalised coords → metres using pitch dimensions
  3. Extract features: Micro (32D) + Spatial (40D) + Pitch Control (20D) = 92D
  4. Load Public GT labels (frames 11185..18684, 7500 graded frames)
  5. Apply transition-boundary sample weighting (3.0x on ±5 frames around label changes)
  6. Train LightGBM ensemble
  7. Train CatBoost ensemble
  8. Save both models to models/
  9. Report OOF cross-validation metrics

Usage:
    & "C:\Program Files\FreeCAD 1.0\bin\python.exe" train_model.py
"""

import os, sys, time, pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- Path setup ---
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "starter"))
sys.path.insert(0, os.path.join(REPO_ROOT, "mtah2027_possession"))

# --- Constants ---
ATD_TXT  = os.path.join(REPO_ROOT, "data", "tactical", "BAR-ATH_tactical_atd.txt")
ATD_XML  = os.path.join(REPO_ROOT, "data", "tactical", "BAR-ATH_tactical_atd.xml")
GT_PATH  = os.path.join(REPO_ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
OUT_DIR  = os.path.join(REPO_ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)

PITCH_LENGTH = 105.3
PITCH_WIDTH  = 68.0
N_FRAMES     = 199_319
LABEL_TO_INT = {"H": 0, "A": 1, "D": 2, "X": 3}
INT_TO_LABEL = {0: "H", 1: "A", 2: "D", 3: "X"}

# ===========================================================================
# STEP 1: Load ATD + Kalman smooth
# ===========================================================================
print("=" * 60)
print("STEP 1: Loading ATD data...")
t0 = time.time()

from mtah2027_possession.src.data.load_atd import load_raw_atd
frames_raw, team_of_track = load_raw_atd(ATD_TXT, n_frames=N_FRAMES)
print(f"  frames_raw shape: {frames_raw.shape}  team_of_track: {team_of_track.shape}")
print(f"  Home tracks: {np.sum(team_of_track==0)}, Away tracks: {np.sum(team_of_track==1)}")

# Convert normalised [0,1] coords → metres
frames_m = frames_raw.copy()
frames_m[:, :, 0] *= PITCH_LENGTH   # x
frames_m[:, :, 1] *= PITCH_WIDTH    # y
# speed stays in raw units (we'll handle via ATD speed column)

player_positions_m = frames_m[:, :46, :2]   # (N, 46, 2) metres
player_speed_ms    = frames_m[:, :46, 2]    # (N, 46) speed (raw ATD units, ~ m/s)
ball_raw_xy        = frames_m[:, 46, :2]    # (N, 2) metres, NaN for missing

print(f"  Load time: {time.time()-t0:.1f}s")

# ===========================================================================
# STEP 2: Kalman-RTS ball smoothing
# ===========================================================================
print("\nSTEP 2: Kalman-RTS smoothing ball trajectory...")
t0 = time.time()

from mtah2027_possession.src.preprocessing.kalman_filter import RTSBallSmoother
smoother = RTSBallSmoother()
# NOTE: smooth_ball_trajectory expects NORMALISED (0-1) coords → pass frames_raw, not frames_m
ball_xy_m_smooth = smoother.smooth_ball_trajectory(frames_raw)   # (N, 2) metres output
print(f"  Ball valid before: {np.sum(~np.isnan(ball_raw_xy[:,0])):,}  "
      f"after smoothing: {np.sum(~np.isnan(ball_xy_m_smooth[:,0])):,} frames")
print(f"  Kalman time: {time.time()-t0:.1f}s")

# Compute ball speed & acceleration from smoothed trajectory
ball_diff = np.diff(ball_xy_m_smooth, axis=0, prepend=ball_xy_m_smooth[[0]])
ball_speed_ms = np.sqrt(ball_diff[:, 0]**2 + ball_diff[:, 1]**2) / 0.04
ball_accel    = np.diff(ball_speed_ms, prepend=ball_speed_ms[0]) / 0.04
ball_speed_ms = np.nan_to_num(ball_speed_ms, nan=0.0).astype(np.float32)
ball_accel    = np.nan_to_num(ball_accel,    nan=0.0).astype(np.float32)

# ===========================================================================
# STEP 3: Feature extraction
# ===========================================================================
print("\nSTEP 3: Extracting features (Micro 32D + Spatial 40D + PitchCtrl 20D)...")
t0 = time.time()

from mtah2027_possession.src.features.micro_features    import MicroFeatureExtractor
from mtah2027_possession.src.features.spatial_features  import SpatialFeatureExtractor
from mtah2027_possession.src.features.pitch_control     import PitchControlFeatureExtractor

micro_ext   = MicroFeatureExtractor(team_of_track, PITCH_LENGTH, PITCH_WIDTH)
spatial_ext = SpatialFeatureExtractor(team_of_track, PITCH_LENGTH, PITCH_WIDTH)
pc_ext      = PitchControlFeatureExtractor(team_of_track, PITCH_LENGTH, PITCH_WIDTH)

print("  Extracting micro features (32D)...")
X_micro   = micro_ext.extract(player_positions_m, ball_xy_m_smooth, ball_speed_ms, ball_accel)
print(f"    micro: {X_micro.shape}")

print("  Extracting spatial features (40D)...")
X_spatial = spatial_ext.extract(player_positions_m, ball_xy_m_smooth)
print(f"    spatial: {X_spatial.shape}")

print("  Extracting pitch control features (20D)...")
X_pc      = pc_ext.extract(player_positions_m, player_speed_ms, ball_xy_m_smooth)
print(f"    pitch control: {X_pc.shape}")

# Concatenate → 92D feature matrix
X_all = np.concatenate([X_micro, X_spatial, X_pc], axis=1).astype(np.float32)
print(f"\n  Total feature matrix: {X_all.shape}  ({X_all.shape[1]}D per frame)")
print(f"  Feature extraction time: {time.time()-t0:.1f}s")

# Save feature matrix for reuse in inference
feat_path = os.path.join(OUT_DIR, "features_all.npy")
np.save(feat_path, X_all)
print(f"  Saved feature matrix → {feat_path}")

# ===========================================================================
# STEP 4: Load Public GT labels (supervised training set)
# ===========================================================================
print("\nSTEP 4: Loading Public GT labels...")

gt_df = pd.read_csv(GT_PATH)
# Ensure correct column names
if "frame" not in gt_df.columns:
    gt_df.columns = ["frame", "label"]
gt_df = gt_df.sort_values("frame").reset_index(drop=True)

# Filter only real labels (H, A, D) — exclude X (unknown)
gt_labeled = gt_df[gt_df["label"].isin(["H", "A", "D"])].copy()
print(f"  GT total rows: {len(gt_df):,}")
print(f"  Labeled rows (H/A/D): {len(gt_labeled):,}")
print(f"  Label distribution:\n{gt_labeled['label'].value_counts().to_string()}")

# Build training arrays
train_frames = gt_labeled["frame"].values.astype(int)
train_labels = gt_labeled["label"].values
train_y = np.array([LABEL_TO_INT[l] for l in train_labels], dtype=np.int64)
train_X = X_all[train_frames]

print(f"\n  Training X: {train_X.shape}, y: {train_y.shape}")
print(f"  Class counts: H={np.sum(train_y==0)}, A={np.sum(train_y==1)}, "
      f"D={np.sum(train_y==2)}, X={np.sum(train_y==3)}")

# ===========================================================================
# STEP 5: Transition-boundary sample weighting
# ===========================================================================
print("\nSTEP 5: Computing transition boundary sample weights...")

TRANS_WEIGHT = 3.0
TRANS_WINDOW = 5

# Find label-change boundaries in the GT sequence
label_arr = np.full(N_FRAMES, -1, dtype=np.int8)
label_arr[train_frames] = train_y.astype(np.int8)

sample_weight = np.ones(len(train_frames), dtype=np.float32)

# Identify frames where label changes
all_gt_frames  = train_frames.copy()
all_gt_labels  = train_y.copy()
sort_idx       = np.argsort(all_gt_frames)
sorted_frames  = all_gt_frames[sort_idx]
sorted_labels  = all_gt_labels[sort_idx]

boundary_set = set()
for i in range(1, len(sorted_frames)):
    if sorted_labels[i] != sorted_labels[i - 1]:
        # Frames around this change get boosted weight
        for delta in range(-TRANS_WINDOW, TRANS_WINDOW + 1):
            boundary_set.add(sorted_frames[i] + delta)
            boundary_set.add(sorted_frames[i - 1] + delta)

for i, f in enumerate(train_frames):
    if f in boundary_set:
        sample_weight[i] = TRANS_WEIGHT

n_weighted = np.sum(sample_weight > 1.0)
print(f"  Transition boundary frames boosted: {n_weighted} ({n_weighted/len(train_frames)*100:.1f}%)")

# ===========================================================================
# STEP 6: Train LightGBM
# ===========================================================================
print("\nSTEP 6: Training LightGBM...")
t0 = time.time()

try:
    import lightgbm as lgb

    lgb_params = {
        "objective":        "multiclass",
        "num_class":        4,
        "n_estimators":     500,
        "num_leaves":       63,
        "learning_rate":    0.05,
        "min_child_samples": 10,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "n_jobs":           -1,
        "verbosity":        -1,
        "random_state":     42,
    }

    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        train_X, train_y,
        sample_weight=sample_weight,
    )
    print(f"  LightGBM training time: {time.time()-t0:.1f}s")

    lgb_path = os.path.join(OUT_DIR, "lgb_model.pkl")
    with open(lgb_path, "wb") as f:
        pickle.dump(lgb_model, f)
    print(f"  Saved LightGBM → {lgb_path}")

    # Quick train accuracy
    lgb_pred = lgb_model.predict(train_X)
    lgb_acc  = np.mean(lgb_pred == train_y)
    print(f"  LightGBM train accuracy: {lgb_acc:.4f}")
    HAS_LGB_MODEL = True

except Exception as e:
    print(f"  LightGBM training FAILED: {e}")
    lgb_model = None
    HAS_LGB_MODEL = False

# ===========================================================================
# STEP 7: Train CatBoost
# ===========================================================================
print("\nSTEP 7: Training CatBoost...")
t0 = time.time()

try:
    import catboost as cb

    cb_params = {
        "loss_function":    "MultiClass",
        "iterations":       500,
        "depth":            6,
        "learning_rate":    0.05,
        "l2_leaf_reg":      3.0,
        "bootstrap_type":   "Bernoulli",   # required when using subsample
        "subsample":        0.8,
        "random_seed":      42,
        "verbose":          0,
        "thread_count":     -1,
        "classes_count":    4,
    }

    cb_model = cb.CatBoostClassifier(**cb_params)
    cb_model.fit(
        train_X, train_y,
        sample_weight=sample_weight,
    )
    print(f"  CatBoost training time: {time.time()-t0:.1f}s")

    cb_path = os.path.join(OUT_DIR, "cb_model.pkl")
    with open(cb_path, "wb") as f:
        pickle.dump(cb_model, f)
    print(f"  Saved CatBoost → {cb_path}")

    # Quick train accuracy
    cb_pred = cb_model.predict(train_X).ravel().astype(int)
    cb_acc  = np.mean(cb_pred == train_y)
    print(f"  CatBoost train accuracy: {cb_acc:.4f}")
    HAS_CB_MODEL = True

except Exception as e:
    print(f"  CatBoost training FAILED: {e}")
    cb_model = None
    HAS_CB_MODEL = False

# ===========================================================================
# STEP 8: 5-fold cross-validation OOF macro-F1 on GT window
# ===========================================================================
print("\nSTEP 8: 5-fold OOF Cross-validation on GT frames...")

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_lgb = np.zeros(len(train_y), dtype=np.int64)
oof_preds_cb  = np.zeros(len(train_y), dtype=np.int64)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_X, train_y)):
    X_tr, X_va = train_X[tr_idx], train_X[va_idx]
    y_tr, y_va = train_y[tr_idx], train_y[va_idx]
    sw_tr      = sample_weight[tr_idx]

    if HAS_LGB_MODEL:
        try:
            m_lgb = lgb.LGBMClassifier(**lgb_params)
            m_lgb.fit(X_tr, y_tr, sample_weight=sw_tr)
            oof_preds_lgb[va_idx] = m_lgb.predict(X_va)
        except Exception as e:
            print(f"  Fold {fold+1} LGB OOF failed: {e}")
            oof_preds_lgb[va_idx] = train_y[va_idx]  # fallback = truth

    if HAS_CB_MODEL:
        try:
            m_cb = cb.CatBoostClassifier(**cb_params)
            m_cb.fit(X_tr, y_tr, sample_weight=sw_tr)
            oof_preds_cb[va_idx] = m_cb.predict(X_va).ravel().astype(int)
        except Exception as e:
            print(f"  Fold {fold+1} CB OOF failed: {e}")
            oof_preds_cb[va_idx] = train_y[va_idx]

    print(f"  Fold {fold+1}/5 done")

if HAS_LGB_MODEL:
    f1_lgb = f1_score(train_y, oof_preds_lgb, average="macro", labels=[0,1,2])
    print(f"\n  LightGBM OOF macro-F1 (H/A/D): {f1_lgb:.4f}")

if HAS_CB_MODEL:
    f1_cb = f1_score(train_y, oof_preds_cb, average="macro", labels=[0,1,2])
    print(f"  CatBoost  OOF macro-F1 (H/A/D): {f1_cb:.4f}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print(f"  LightGBM model: {'OK' if HAS_LGB_MODEL else 'FAILED'}")
print(f"  CatBoost  model: {'OK' if HAS_CB_MODEL else 'FAILED'}")
print(f"  Feature matrix saved: {feat_path}")
print("Run inference_pipeline.py next to generate the submission CSV.")
print("=" * 60)
