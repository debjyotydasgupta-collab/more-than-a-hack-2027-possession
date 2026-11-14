"""
submit.py
=========
Authoritative End-to-End Submission Execution Entry Point
More than a Hack 2027 — Phase 1: Ball Possession
Barça Innovation Hub × Mobile World Capital Barcelona × Metrica Sports

Under SUBMISSION.md Section 2:
Running this command generates the submission CSV from raw match data,
applies the full physics-ML invariant pipeline, and validates format compliance.

Usage:
    python submit.py --out <team_name>.csv
"""

import os
import sys
import argparse
import subprocess

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

def main():
    parser = argparse.ArgumentParser(description="Generate and validate official competition submission")
    parser.add_argument("--out", type=str, default=os.path.join(REPO_ROOT, "evaluation", "results", "submission_possession_v1.csv"),
                        help="Output path for submission CSV (e.g. <team_name>.csv)")
    parser.add_argument("--mode", type=str, default="self_supervised", choices=["self_supervised", "calibrated"],
                        help="Pipeline training mode: 'self_supervised' (0 human labels) or 'calibrated'")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip scoring against public 5-minute ground truth")
    args = parser.parse_args()

    out_csv = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    print("=" * 80)
    print("MORE THAN A HACK 2027 — OFFICIAL SUBMISSION PIPELINE")
    print(f"Target Output: {out_csv}")
    print(f"Supervision Mode: {args.mode.upper()} (Zero Human Annotations)")
    print("=" * 80)

    # 1. Execute Inference Pipeline
    infer_script = os.path.join(REPO_ROOT, "inference_pipeline.py")
    cmd = [sys.executable, infer_script, "--out", out_csv]
    
    print("\n[1/2] Executing full invariant physics & sequence decoding pipeline...")
    res = subprocess.run(cmd, cwd=REPO_ROOT)
    if res.returncode != 0:
        print(f"\nERROR: Inference pipeline failed with exit code {res.returncode}")
        sys.exit(res.returncode)

    # 2. Strict Submission Validation
    print("\n[2/2] Validating submission against official competition criteria...")
    from evaluation.possession_metrics import validate_submission, read_label_csv
    
    if not os.path.isfile(out_csv):
        print(f"ERROR: Output file {out_csv} was not generated.")
        sys.exit(1)

    table = read_label_csv(out_csv)
    expected_rows = 199_319
    reason = validate_submission(table, expected_rows)

    if reason is not None:
        print(f"\n❌ VALIDATION REJECTED: {reason}")
        sys.exit(1)

    label_counts = table["label"].value_counts().to_dict()
    print(f"\n✓ Row count: exactly {len(table):,} frames (0..{expected_rows-1})")
    print(f"✓ Label distribution: {label_counts}")
    print("=" * 80)
    print("★ ALL CHECKS PASSED: File is 100% valid and ready for upload!")
    print(f"Deliverable: {out_csv}")
    print("=" * 80)

if __name__ == "__main__":
    main()
