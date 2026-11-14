"""
evaluation/sanity_check.py
==========================
Standalone Post-Generation Sanity Check Script
More than a Hack 2027 — Phase 1: Ball Possession

This is the ONLY script in the pipeline authorized to read the public ground truth
(BAR-ATH_possession_gt_public.csv). It operates strictly AFTER a submission CSV has
been generated to verify format correctness and benchmark metrics against repo baselines.

Usage:
    python evaluation/sanity_check.py <path_to_submission_csv>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import evaluation.possession_metrics as pm

GT_PATH = os.path.join(REPO_ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a submission against public 5-minute ground truth")
    parser.add_argument("submission_csv", help="Path to submission CSV to evaluate")
    args = parser.parse_args()

    sub_path = os.path.abspath(args.submission_csv)
    if not os.path.isfile(sub_path):
        print(f"ERROR: File not found: {sub_path}")
        sys.exit(1)

    if not os.path.isfile(GT_PATH):
        print(f"ERROR: Ground truth file not found: {GT_PATH}")
        sys.exit(1)

    print("=" * 80)
    print("MTAH 2027 — POST-GENERATION SANITY CHECK")
    print(f"Target Submission: {os.path.basename(sub_path)}")
    print(f"Ground Truth Ref: {os.path.basename(GT_PATH)} (7,500 graded frames / 17 spells)")
    print("=" * 80)

    # 1. Format validation
    sub_table = pm.read_label_csv(sub_path)
    err = pm.validate_submission(sub_table, 199_319)
    if err is not None:
        print(f"❌ SUBMISSION INVALID: {err}")
        sys.exit(1)
    print("✓ Format check passed (199,319 rows, monotonic, valid labels).")

    # 2. Score submission
    gt_codes = pm.label_codes(pm.read_label_csv(GT_PATH))
    pred_codes = pm.label_codes(sub_table)
    runs = pm.valid_runs(gt_codes)

    res = pm.evaluate(gt_codes, pred_codes, taus=(12, 25))

    spell_f1 = res["spell_f1_mean"]
    macro_f1 = res["macro_f1"]
    acc = res["accuracy"] * 100
    kappa = res["kappa"]
    trans_f1 = res["transitions"][25]["f1"]
    latency = res["transitions"][25]["mean_latency"]
    share_err = res["share_err"] * 100
    n_pred_spells = res["n_pred_spells"]
    mean_dur = res["mean_spell_duration_s"]

    # Compute full match possession stats
    full_runs = pm.valid_runs(np.zeros(199_319, dtype=np.int8))  # all graded
    all_spells = pm.possession_spells(pred_codes, [(0, 199_319)])
    total_possessions = len(all_spells)
    total_mean_dur = np.mean([(stop - start) * 0.04 for start, stop, code in all_spells]) if all_spells else 0.0

    print("\n" + "-" * 80)
    print("1. EVALUATION METRICS ON PUBLIC 5-MINUTE WINDOW:")
    print("-" * 80)
    print(f"  spellF1 (Rank Metric)  : {spell_f1:.4f}  (IoU 0.3..0.7 mean)")
    print(f"  macroF1 (Tie-Break)    : {macro_f1:.4f}")
    print(f"  Per-Frame Accuracy     : {acc:.2f}%")
    print(f"  Cohen's Kappa          : {kappa:.4f}")
    print(f"  Transition F1 (±1.0s)  : {trans_f1:.4f}  (Mean Latency: {latency:+.1f} frames)")
    print(f"  Possession Share Error : {share_err:.3f}%")
    print(f"  Spells in Public Window: {n_pred_spells} predicted vs 17 ground truth (mean dur: {mean_dur:.1f}s)")

    print("\n" + "-" * 80)
    print("2. FULL MATCH TACTICAL STRUCTURE (199,319 Frames):")
    print("-" * 80)
    print(f"  Total Possessions: {total_possessions} spells")
    print(f"  Mean Duration    : {total_mean_dur:.1f} seconds")
    print(f"  Label Counts     : {sub_table['label'].value_counts().to_dict()}")

    print("\n" + "-" * 80)
    print("3. BENCHMARK COMPARISON AGAINST OFFICIAL REPO BASELINES:")
    print("-" * 80)
    baselines = [
        ("Your Submission", spell_f1, macro_f1, acc, n_pred_spells, mean_dur),
        ("baseline_smooth125", 0.4242, 0.5812, 72.75, 16, 18.5),
        ("baseline_smooth50 (Floor)", 0.3256, 0.6207, 71.32, 26, 10.8),
        ("baseline_smooth25", 0.2868, 0.6200, 69.12, 36, 7.6),
        ("baseline_raw", 0.0378, 0.5972, 65.12, 131, 2.0),
    ]
    print(f"{'Method':28s} | {'spellF1':8s} | {'macroF1':8s} | {'Acc (%)':8s} | {'Spells':6s} | {'Dur (s)':7s}")
    print("-" * 75)
    for name, sf, mf, ac, sp, du in baselines:
        print(f"{name:28s} | {sf:8.4f} | {mf:8.4f} | {ac:7.2f}% | {sp:6d} | {du:6.1f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()

