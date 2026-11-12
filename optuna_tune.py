"""
optuna_tune.py
==============
Hyperparameter optimisation for the MTAH 2027 ball-possession pipeline.

Usage
-----
    python optuna_tune.py [--trials 150] [--timeout 14400] [--study-name mtah2027]

The script:
  1. Defines an Optuna objective that runs the full end-to-end pipeline on the
     public ground truth split and returns spellF1.
  2. Uses the TPE sampler with 150 trials and a 4-hour (14 400 s) timeout.
  3. On completion, writes the best hyperparameters back to
     ``configs/optimal_config.yaml``.
  4. Handles individual trial errors gracefully (raises TrialPruned so the
     study continues).

Requirements
------------
    pip install optuna lightgbm catboost scipy pyyaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "mtah2027_possession"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE))

OPTIMAL_CFG = ROOT / "configs" / "optimal_config.yaml"
DATA_DIR = ROOT / "data" / "tactical"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("optuna_tune")

# ---------------------------------------------------------------------------
# Lazy imports (avoid crashing at import time if deps are missing)
# ---------------------------------------------------------------------------
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as exc:
    sys.exit(f"[optuna_tune] optuna not installed: {exc}\n  pip install optuna")

try:
    import numpy as np
except ImportError as exc:
    sys.exit(f"[optuna_tune] numpy not installed: {exc}")


# ---------------------------------------------------------------------------
# Pipeline runner (imports lazily so missing deps produce clear errors)
# ---------------------------------------------------------------------------

def _run_pipeline_with_config(cfg: dict) -> float:
    """
    Run the full possession pipeline with *cfg* and return spellF1.

    Returns
    -------
    float
        spellF1 on the public ground-truth split (higher is better).
        Returns -1.0 if the pipeline fails (trial will be pruned).
    """
    # Inline imports so top-level import errors don't block the whole script
    try:
        from src.data.load_atd import load_raw_atd
        from src.data.load_smart_tagging import load as load_st
        from src.preprocessing.kalman_filter import RTSBallSmoother
        from src.preprocessing.coordinate_transforms import norm_to_meters
        from src.features.micro_features import build_micro_features
        from src.features.spatial_features import build_spatial_features
        from src.models.ensemble import EnsembleModel
        from src.postprocess.viterbi_decoder import ViterbiDecoder
        from src.postprocess.spell_pruner import SpellPruner
    except ImportError as exc:
        log.error("Pipeline import failed: %s", exc)
        return -1.0

    try:
        # ---- 1. Find ATD file -----------------------------------------------
        atd_files = list(DATA_DIR.glob("*_atd.txt"))
        if not atd_files:
            log.error("No *_atd.txt found in %s — place match data there first.", DATA_DIR)
            return -1.0
        atd_path = atd_files[0]

        # ---- 2. Load raw data -----------------------------------------------
        raw_array, team_of_track = load_raw_atd(str(atd_path))

        # ---- 3. Kalman + RTS smoothing --------------------------------------
        smoother = RTSBallSmoother(
            q_noise=cfg["kalman"]["q_process_noise"],
            r_noise=cfg["kalman"]["r_measurement_noise"],
            max_speed=cfg["kalman"]["max_ball_velocity"],
            max_gap_frames=cfg["kalman"]["max_gap_frames"],
        )
        ball_m = smoother.smooth_ball_trajectory(raw_array)   # (N, 2) metres

        # ---- 4. Players in metres ------------------------------------------
        players_m = norm_to_meters(
            raw_array[:, :46, :2],
            pitch_length=cfg["features"]["pitch_length"],
            pitch_width=cfg["features"]["pitch_width"],
        )

        # ---- 5. Features ----------------------------------------------------
        micro = build_micro_features(
            players_m=players_m,
            ball_m=ball_m,
            team_of_track=team_of_track,
            windows=cfg["features"]["windows"],
            top_k=cfg["features"]["top_k_players"],
            epsilon=cfg["features"]["epsilon"],
        )
        spatial = build_spatial_features(
            players_m=players_m,
            ball_m=ball_m,
            team_of_track=team_of_track,
            windows=cfg["features"]["windows"],
        )
        X = np.concatenate([micro, spatial], axis=1)

        # ---- 6. Load ground truth labels for training ----------------------
        gt_public_files = list((ROOT / "evaluation" / "gt").glob("*_possession_gt_public.csv"))
        if not gt_public_files:
            log.error("No public GT file found — cannot compute spellF1 in tuning.")
            return -1.0

        import pandas as pd
        gt_df = pd.read_csv(gt_public_files[0])
        # Keep only labelled frames
        labelled_mask = gt_df["label"].isin(["H", "A", "D"])
        X_lab = X[gt_df.loc[labelled_mask, "frame"].values]
        y_lab = gt_df.loc[labelled_mask, "label"].values

        # Simple 80/20 chronological split
        split = int(0.8 * len(y_lab))
        X_tr, X_val = X_lab[:split], X_lab[split:]
        y_tr, y_val = y_lab[:split], y_lab[split:]

        # ---- 7. Train ensemble ---------------------------------------------
        model = EnsembleModel(
            lgb_params=cfg["model"]["lgb"],
            cb_params=cfg["model"]["catboost"],
            ensemble_weights=cfg["model"]["ensemble_weights"],
        )
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)   # (N_val, 4) H A D X

        # ---- 8. Viterbi + pruning ------------------------------------------
        pp = cfg["postprocess"]
        decoder = ViterbiDecoder(
            trans_penalty_swap=pp["trans_penalty_swap"],
            trans_penalty_dead=pp["trans_penalty_dead"],
        )
        raw_labels = decoder.decode(probs)

        pruner = SpellPruner(
            min_spell_frames=pp["min_spell_frames"],
            tau_high=pp["tau_high"],
            tau_low=pp["tau_low"],
            k_hold=pp["k_hold"],
        )
        final_labels = pruner.prune(raw_labels, probs)

        # ---- 9. Compute spellF1 --------------------------------------------
        spell_f1 = _compute_spell_f1(y_val, final_labels)
        return float(spell_f1)

    except Exception:
        log.warning("Trial raised an exception:\n%s", traceback.format_exc())
        return -1.0


def _compute_spell_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute mean spellF1 at IoU thresholds 0.30 .. 0.70 (step 0.05).

    A *spell* is a contiguous run of identical labels.
    At each threshold τ a predicted spell is a TP if it overlaps a GT spell
    of the same class with IoU >= τ.
    """
    classes = ["H", "A", "D"]
    thresholds = np.arange(0.30, 0.71, 0.05)

    def to_spells(seq):
        """Return list of (label, start, end_inclusive) tuples."""
        spells = []
        if len(seq) == 0:
            return spells
        cur, start = seq[0], 0
        for i in range(1, len(seq)):
            if seq[i] != cur:
                spells.append((cur, start, i - 1))
                cur, start = seq[i], i
        spells.append((cur, start, len(seq) - 1))
        return spells

    gt_spells = to_spells(y_true)
    pred_spells = to_spells(y_pred)

    f1_values = []
    for cls in classes:
        gt_cls = [(s, e) for (l, s, e) in gt_spells if l == cls]
        pred_cls = [(s, e) for (l, s, e) in pred_spells if l == cls]
        for tau in thresholds:
            tp = 0
            matched_gt = set()
            for (ps, pe) in pred_cls:
                p_len = pe - ps + 1
                for gi, (gs, ge) in enumerate(gt_cls):
                    if gi in matched_gt:
                        continue
                    inter = max(0, min(pe, ge) - max(ps, gs) + 1)
                    union = (pe - ps + 1) + (ge - gs + 1) - inter
                    if union > 0 and (inter / union) >= tau:
                        tp += 1
                        matched_gt.add(gi)
                        break
            prec = tp / max(len(pred_cls), 1)
            rec = tp / max(len(gt_cls), 1)
            denom = prec + rec
            f1 = (2 * prec * rec / denom) if denom > 0 else 0.0
            f1_values.append(f1)

    return float(np.mean(f1_values)) if f1_values else 0.0


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial: "optuna.Trial") -> float:
    """
    Optuna objective function. Suggests hyperparameters, runs the pipeline,
    and returns spellF1. Raises TrialPruned on failure.
    """
    cfg = {
        "kalman": {
            "q_process_noise": trial.suggest_float("kalman_q", 1e-4, 0.05, log=True),
            "r_measurement_noise": trial.suggest_float("kalman_r", 0.1, 2.0, log=True),
            "max_ball_velocity": trial.suggest_float("max_ball_velocity", 30.0, 45.0),
            "max_gap_frames": trial.suggest_int("max_gap_frames", 50, 200),
        },
        "features": {
            "pitch_length": 105.0,
            "pitch_width": 68.0,
            "windows": [5, 15, 25, 50, 125],
            "top_k_players": trial.suggest_int("top_k_players", 3, 7),
            "epsilon": 1e-6,
        },
        "model": {
            "lgb": {
                "learning_rate": trial.suggest_float("lgb_lr", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("lgb_num_leaves", 31, 127),
                "min_child_samples": trial.suggest_int("lgb_min_child_samples", 10, 50),
                "subsample": trial.suggest_float("lgb_subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("lgb_colsample", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("lgb_reg_alpha", 0.0, 0.5),
                "reg_lambda": trial.suggest_float("lgb_reg_lambda", 0.0, 0.5),
                "n_estimators": trial.suggest_int("lgb_n_estimators", 200, 800),
                "class_weight": None,
            },
            "catboost": {
                "learning_rate": trial.suggest_float("cb_lr", 0.01, 0.2, log=True),
                "depth": trial.suggest_int("cb_depth", 4, 10),
                "iterations": trial.suggest_int("cb_iterations", 200, 800),
            },
            "ensemble_weights": [
                w := trial.suggest_float("ensemble_w_lgb", 0.3, 0.7),
                1.0 - w,
            ],
        },
        "postprocess": {
            "trans_penalty_swap": trial.suggest_float("trans_penalty_swap", -5.0, -0.5),
            "trans_penalty_dead": trial.suggest_float("trans_penalty_dead", -3.0, -0.1),
            "tau_high": trial.suggest_float("tau_high", 0.60, 0.90),
            "tau_low": trial.suggest_float("tau_low", 0.10, 0.40),
            "min_spell_frames": trial.suggest_int("min_spell_frames", 5, 30),
            "k_hold": trial.suggest_int("k_hold", 5, 20),
        },
        "smart_tagging": {
            "set_piece_dead_fraction": trial.suggest_float("st_dead_fraction", 0.5, 0.9),
            "boost_factor": trial.suggest_float("st_boost_factor", 1.2, 2.5),
        },
        "data": {
            "n_frames": 199319,
            "fps": 25,
            "home_team": "Barcelona",
            "away_team": "Athletic Club",
            "home_label": "H",
            "away_label": "A",
            "dead_label": "D",
            "unknown_label": "X",
        },
    }

    score = _run_pipeline_with_config(cfg)
    if score < 0:
        raise optuna.exceptions.TrialPruned("Pipeline failed — skipping trial.")
    return score


# ---------------------------------------------------------------------------
# Save best config
# ---------------------------------------------------------------------------

def _save_best_config(study: "optuna.Study") -> None:
    """Write the best trial's hyperparameters to configs/optimal_config.yaml."""
    best = study.best_params
    cfg = {
        "kalman": {
            "q_process_noise": best["kalman_q"],
            "r_measurement_noise": best["kalman_r"],
            "max_ball_velocity": best["max_ball_velocity"],
            "max_gap_frames": best["max_gap_frames"],
        },
        "features": {
            "pitch_length": 105.0,
            "pitch_width": 68.0,
            "windows": [5, 15, 25, 50, 125],
            "top_k_players": best["top_k_players"],
            "epsilon": 1e-6,
        },
        "model": {
            "lgb": {
                "learning_rate": best["lgb_lr"],
                "num_leaves": best["lgb_num_leaves"],
                "min_child_samples": best["lgb_min_child_samples"],
                "subsample": best["lgb_subsample"],
                "colsample_bytree": best["lgb_colsample"],
                "reg_alpha": best["lgb_reg_alpha"],
                "reg_lambda": best["lgb_reg_lambda"],
                "n_estimators": best["lgb_n_estimators"],
                "class_weight": None,
            },
            "catboost": {
                "learning_rate": best["cb_lr"],
                "depth": best["cb_depth"],
                "iterations": best["cb_iterations"],
            },
            "ensemble_weights": [
                best["ensemble_w_lgb"],
                round(1.0 - best["ensemble_w_lgb"], 6),
            ],
        },
        "postprocess": {
            "trans_penalty_swap": best["trans_penalty_swap"],
            "trans_penalty_dead": best["trans_penalty_dead"],
            "tau_high": best["tau_high"],
            "tau_low": best["tau_low"],
            "min_spell_frames": best["min_spell_frames"],
            "k_hold": best["k_hold"],
        },
        "smart_tagging": {
            "set_piece_dead_fraction": best["st_dead_fraction"],
            "boost_factor": best["st_boost_factor"],
        },
        "data": {
            "n_frames": 199319,
            "fps": 25,
            "home_team": "Barcelona",
            "away_team": "Athletic Club",
            "home_label": "H",
            "away_label": "A",
            "dead_label": "D",
            "unknown_label": "X",
        },
    }

    OPTIMAL_CFG.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTIMAL_CFG, "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False, allow_unicode=True)
    log.info("Best config saved → %s", OPTIMAL_CFG)
    log.info("Best spellF1: %.4f", study.best_value)
    log.info("Best params:\n%s", json.dumps(best, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO for MTAH 2027 possession pipeline")
    parser.add_argument("--trials", type=int, default=150, help="Number of Optuna trials (default: 150)")
    parser.add_argument("--timeout", type=int, default=14400, help="Timeout in seconds (default: 14400 = 4h)")
    parser.add_argument("--study-name", default="mtah2027_possession", help="Optuna study name")
    parser.add_argument("--storage", default=None, help="Optuna storage URL (optional, e.g. sqlite:///optuna.db)")
    args = parser.parse_args()

    log.info("Starting Optuna HPO: %d trials, %ds timeout", args.trials, args.timeout)
    log.info("Study name: %s | Storage: %s", args.study_name, args.storage or "in-memory")

    sampler = TPESampler(seed=42, multivariate=True)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
    )

    try:
        study.optimize(
            objective,
            n_trials=args.trials,
            timeout=args.timeout,
            show_progress_bar=True,
            gc_after_trial=True,
        )
    except KeyboardInterrupt:
        log.info("Interrupted by user — saving best config found so far.")

    if study.best_trial is not None:
        _save_best_config(study)
    else:
        log.warning("No completed trials — nothing to save.")


if __name__ == "__main__":
    main()
