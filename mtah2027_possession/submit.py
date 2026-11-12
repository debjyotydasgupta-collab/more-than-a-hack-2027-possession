"""
mtah2027_possession/submit.py
=============================
CLI command to generate and validate final submission.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pipeline import run_pipeline
from evaluation.possession_metrics import validate_submission, read_label_csv
from games import GAME, GAMES


def main():
    parser = argparse.ArgumentParser(description="Generate and validate official competition submission")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "optimal_config.yaml"),
                        help="Configuration YAML path")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "submission_possession_v1.csv"),
                        help="Output path for submission CSV")
    parser.add_argument("--sample", type=int, default=None,
                        help="Test on subset of frames (e.g. 5000)")
    parser.add_argument("--window", type=int, default=None,
                        help="Smoothing window half-width in frames")
    parser.add_argument("--min-spell", type=int, default=None,
                        help="Minimum spell pruning length in frames")
    args = parser.parse_args()

    success = run_pipeline(
        config_path=args.config,
        out_path=args.out,
        sample_frames=args.sample,
        window=args.window,
        min_spell=args.min_spell
    )
    if not success:
        print("Submission generation failed.")
        sys.exit(1)

    table = read_label_csv(args.out)
    expected_rows = args.sample if args.sample else GAMES[GAME]["n_video_frames"]
    reason = validate_submission(table, expected_rows)

    if reason is None:
        print(f"\nALL CHECKS PASSED: Ready for submission upload to competition platform!")
        sys.exit(0)
    else:
        print(f"\nVALIDATION FAILED: {reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()
