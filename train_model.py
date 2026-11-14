"""
train_model.py
==============
Generalized GBDT ensemble training pipeline for Ball Possession classification.
More than a Hack 2027 — Phase 1: Ball Possession.

Features 5-Phase Implementation Plan:
  Phase 1: Relative Spatial Invariance & Coordinate Decoupling
  Phase 2: Geometric Pitch Symmetry Augmentation (4x Dataset: 30,000 frames)
  Phase 4: Semi-Supervised Self-Training across 199,319 match frames
  Phase 5: Asymmetric Viterbi & Boundary Weighting

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

# Match clock boundary frames (forced Dead Ball)
DEAD_BALL_REGIONS = [
    (0, 11184),           # Pre-kickoff
    (90652, 110807),      # Halftime
    (190198, 199318),     # Post-match
]

# ===========================================================================
# STEP 1 & 2: Feature Matrix Check / Extraction
# ===========================================================================
feat_path = os.path.join(OUT_DIR, "features_all.npy")

if os.path.exists(feat_path):
    print("=" * 60)
    print("STEP 1: Loading pre-computed feature matrix...")
    t0 = time.time()
    X_all = np.load(feat_path)
    print(f"  Loaded feature matrix: {X_all.shape} ({time.time()-t0:.1f}s)")
else:
    print("=" * 60)
    print("STEP 1: Extracting features from raw ATD data...")
    t0 = time.time()
    from mtah2027_possession.src.data.load_atd import load_raw_atd
    frames_raw, team_of_track = load_raw_atd(ATD_TXT, n_frames=N_FRAMES)
    
    frames_m = frames_raw.copy()
    frames_m[:, :, 0] *= PITCH_LENGTH
    frames_m[:, :, 1] *= PITCH_WIDTH

    player_positions_m = frames_m[:, :46, :2]
    player_speed_ms    = frames_m[:, :46, 2]
    ball_raw_xy        = frames_m[:, 46, :2]

    from mtah2027_possession.src.preprocessing.kalman_filter import RTSBallSmoother
    smoother = RTSBallSmoother()
    ball_xy_m_smooth = smoother.smooth_ball_trajectory(frames_raw)

    ball_diff = np.diff(ball_xy_m_smooth, axis=0, prepend=ball_xy_m_smooth[[0]])
    ball_speed_ms = np.sqrt(ball_diff[:, 0]**2 + ball_diff[:, 1]**2) / 0.04
    ball_accel    = np.diff(ball_speed_ms, prepend=ball_speed_ms[0]) / 0.04
    ball_speed_ms = np.nan_to_num(ball_speed_ms, nan=0.0).astype(np.float32)
    ball_accel    = np.nan_to_num(ball_accel,    nan=0.0).astype(np.float32)

    from mtah2027_possession.src.features.micro_features    import MicroFeatureExtractor
    from mtah2027_possession.src.features.spatial_features  import SpatialFeatureExtractor
    from mtah2027_possession.src.features.pitch_control     import PitchControlFeatureExtractor

    micro_ext   = MicroFeatureExtractor(team_of_track, PITCH_LENGTH, PITCH_WIDTH)
    spatial_ext = SpatialFeatureExtractor(team_of_track, PITCH_LENGTH, PITCH_WIDTH)
    pc_ext      = PitchControlFeatureExtractor(team_of_track, PITCH_LENGTH, PITCH_WIDTH)

    X_micro   = micro_ext.extract(player_positions_m, ball_xy_m_smooth, ball_speed_ms, ball_accel)
    X_spatial = spatial_ext.extract(player_positions_m, ball_xy_m_smooth)
    X_pc      = pc_ext.extract(player_positions_m, player_speed_ms, ball_xy_m_smooth)

    X_all = np.concatenate([X_micro, X_spatial, X_pc], axis=1).astype(np.float32)
    np.save(feat_path, X_all)
    print(f"  Extracted and saved: {X_all.shape} ({time.time()-t0:.1f}s)")

# ===========================================================================
# STEP 3: Load Public GT labels
# ===========================================================================
print("\nSTEP 2: Loading Public Ground Truth Labels...")
gt_df = pd.read_csv(GT_PATH)
if "frame" not in gt_df.columns: gt_df.columns = ["frame", "label"]
gt_df = gt_df.sort_values("frame").reset_index(drop=True)
gt_labeled = gt_df[gt_df["label"].isin(["H", "A", "D"])].copy()

train_frames = gt_labeled["frame"].values.astype(int)
train_labels = gt_labeled["label"].values
train_y = np.array([LABEL_TO_INT[l] for l in train_labels], dtype=np.int64)
train_X = X_all[train_frames]

print(f"  Public GT: {len(train_frames)} frames (H={np.sum(train_y==0)}, A={np.sum(train_y==1)}, D={np.sum(train_y==2)})")

# Transition boundary weighting (3.0x on boundary frames)
TRANS_WEIGHT = 3.0
TRANS_WINDOW = 5
sample_weight = np.ones(len(train_frames), dtype=np.float32)

all_gt_frames  = train_frames.copy()
all_gt_labels  = train_y.copy()
sort_idx       = np.argsort(all_gt_frames)
sorted_frames  = all_gt_frames[sort_idx]
sorted_labels  = all_gt_labels[sort_idx]

boundary_set = set()
for i in range(1, len(sorted_frames)):
    if sorted_labels[i] != sorted_labels[i - 1]:
        for delta in range(-TRANS_WINDOW, TRANS_WINDOW + 1):
            boundary_set.add(sorted_frames[i] + delta)
            boundary_set.add(sorted_frames[i - 1] + delta)

for i, f in enumerate(train_frames):
    if f in boundary_set:
        sample_weight[i] = TRANS_WEIGHT

# ===========================================================================
# STEP 4: Phase B — 4-Way Pitch Symmetry Augmentation (Strict Invariance)
# ===========================================================================
print("\nSTEP 3: Applying Phase B — 4-Way Pitch Symmetry Augmentation (Strict Invariance)...")
from mtah2027_possession.src.models.augmenter import PitchSymmetryAugmenter

augmenter = PitchSymmetryAugmenter(PITCH_LENGTH, PITCH_WIDTH)
X_train_aug, y_train_aug, w_train_aug = augmenter.augment(train_X, train_y, sample_weight)
print(f"  Augmented Dataset: {X_train_aug.shape[0]:,} frames (4x expansion)")
print(f"  Class balance: H={np.sum(y_train_aug==0)}, A={np.sum(y_train_aug==1)}, D={np.sum(y_train_aug==2)}")

# ===========================================================================
# STEP 5: Phase 4 — Train Teacher & Semi-Supervised Harvest Across Full Match
# ===========================================================================
print("\nSTEP 4: Training Base Teacher Ensemble on Augmented Dataset...")
import lightgbm as lgb
import catboost as cb

lgb_params = {
    "objective":         "multiclass",
    "num_class":         4,
    "n_estimators":      350,
    "num_leaves":        45,
    "learning_rate":     0.06,
    "min_child_samples": 20,
    "subsample":         0.85,
    "colsample_bytree":  0.65,  # Anti-memorization column sampling
    "reg_alpha":         0.2,
    "reg_lambda":        1.5,
    "n_jobs":            -1,
    "verbosity":         -1,
    "random_state":      42,
}

teacher_lgb = lgb.LGBMClassifier(**lgb_params)
teacher_lgb.fit(X_train_aug, y_train_aug, sample_weight=w_train_aug)
print("  Teacher LightGBM trained successfully.")

print("\nSTEP 5: Harvesting high-confidence pseudo-labels across all 199,319 frames...")
t0 = time.time()
chunk_size = 25000
N = len(X_all)
p_teacher = np.zeros((N, 4), dtype=np.float32)

for i in range(0, N, chunk_size):
    chunk_raw = teacher_lgb.predict_proba(X_all[i:i+chunk_size])
    for ci, c in enumerate(teacher_lgb.classes_):
        p_teacher[i:i+chunk_size, c] = chunk_raw[:, ci]

# Filter dead ball boundaries
live_mask = np.ones(N, dtype=bool)
for start_f, end_f in DEAD_BALL_REGIONS:
    live_mask[start_f:end_f + 1] = False

# Exclude original public ground truth frames so teacher doesn't leak to itself
live_mask[train_frames] = False

max_p = np.max(p_teacher, axis=1)
pred_c = np.argmax(p_teacher, axis=1)

# Extract sustained high-confidence spells (duration >= 75 frames, confidence >= 0.985)
is_candidate = live_mask & (max_p >= 0.985) & (pred_c < 2) # Home or Away live play

breaks = np.flatnonzero(is_candidate[1:] != is_candidate[:-1]) + 1
run_starts = [0] + list(breaks)
run_ends = list(breaks) + [N]

selected_pseudo_indices = []
for rs, re in zip(run_starts, run_ends):
    if is_candidate[rs] and (re - rs) >= 75:  # sustained >= 3.0s
        step = 5  # take 1 frame every 0.2s
        selected_pseudo_indices.extend(range(rs, re, step))

selected_pseudo_indices = np.array(selected_pseudo_indices, dtype=np.int64)
if len(selected_pseudo_indices) > 35000:
    rng = np.random.RandomState(42)
    selected_pseudo_indices = rng.choice(selected_pseudo_indices, size=35000, replace=False)
    selected_pseudo_indices = np.sort(selected_pseudo_indices)

X_pseudo = X_all[selected_pseudo_indices]
y_pseudo = pred_c[selected_pseudo_indices]
w_pseudo = np.full(len(y_pseudo), 0.35, dtype=np.float32)

print(f"  Harvested {len(selected_pseudo_indices):,} high-confidence pseudo-frames across the full 90 minutes ({time.time()-t0:.1f}s)")
print(f"  Pseudo classes: H={np.sum(y_pseudo==0)}, A={np.sum(y_pseudo==1)}")

# Combine Augmented GT (weight 1.0) + Pseudo-GT (weight 0.35)
X_student = np.concatenate([X_train_aug, X_pseudo], axis=0)
y_student = np.concatenate([y_train_aug, y_pseudo], axis=0)
w_student = np.concatenate([w_train_aug, w_pseudo], axis=0)

# ===========================================================================
# STEP 6: Phase D — Noise-Inoculated Student Training (Jitter + Dropout)
# ===========================================================================
print("\nSTEP 6: Inoculating Student Dataset with Online Jitter (sigma=0.03) & Dropout (p=0.10)...")
rng = np.random.RandomState(42)
feat_stds = np.std(X_student, axis=0, keepdims=True)
feat_stds = np.where(feat_stds < 1e-4, 1.0, feat_stds)

# 1. Gaussian tracking jitter
noise = rng.normal(0, 0.03 * feat_stds, size=X_student.shape).astype(np.float32)
X_noisy = X_student + noise

# 2. Random feature dropout (10%)
dropout_mask = rng.rand(*X_student.shape) < 0.10
X_drop = X_student.copy()
X_drop[dropout_mask] = 0.0

X_student_train = np.concatenate([X_student, X_noisy, X_drop], axis=0)
y_student_train = np.concatenate([y_student, y_student, y_student], axis=0)
w_student_train = np.concatenate([w_student, w_student * 0.70, w_student * 0.50], axis=0)

print(f"  Total Inoculated Training Samples: {X_student_train.shape[0]:,} frames")

print("\nSTEP 7: Fitting Noise-Inoculated Student LightGBM on Full Match Representation...")
t0 = time.time()
student_lgb = lgb.LGBMClassifier(**lgb_params)
student_lgb.fit(X_student_train, y_student_train, sample_weight=w_student_train)
print(f"  Student LightGBM trained in {time.time()-t0:.1f}s")

lgb_path = os.path.join(OUT_DIR, "lgb_model.pkl")
with open(lgb_path, "wb") as f:
    pickle.dump(student_lgb, f)
print(f"  Saved Student LightGBM -> {lgb_path}")

print("\nSTEP 8: Fitting Noise-Inoculated Student CatBoost on Full Match Representation...")
t0 = time.time()
cb_params = {
    "loss_function":    "MultiClass",
    "iterations":       450,
    "depth":            6,
    "learning_rate":    0.06,
    "l2_leaf_reg":      3.0,
    "bootstrap_type":   "Bernoulli",
    "subsample":        0.85,
    "rsm":              0.65,  # Anti-memorization column sampling
    "random_seed":      42,
    "verbose":          0,
    "thread_count":     -1,
    "classes_count":    4,
}

student_cb = cb.CatBoostClassifier(**cb_params)
student_cb.fit(X_student_train, y_student_train, sample_weight=w_student_train)
print(f"  Student CatBoost trained in {time.time()-t0:.1f}s")

cb_path = os.path.join(OUT_DIR, "cb_model.pkl")
with open(cb_path, "wb") as f:
    pickle.dump(student_cb, f)
print(f"  Saved Student CatBoost -> {cb_path}")

# ===========================================================================
# STEP 8: Benchmark Check on Public Ground Truth
# ===========================================================================
print("\nSTEP 8: Evaluating Student Model on Public Ground Truth Slice...")
p_eval = student_lgb.predict(train_X)
acc = np.mean(p_eval == train_y) * 100
print(f"  Student Model Public GT Accuracy: {acc:.2f}% (Error: {100.0-acc:.2f}%)")

print("\n" + "=" * 60)
print("TRAINING PIPELINE COMPLETE (5-PHASE ARCHITECTURE APPLIED)")
print("=" * 60)
