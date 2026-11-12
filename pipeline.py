"""
pipeline.py
===========
Root entry point redirecting to mtah2027_possession/pipeline.py.
"""

import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "mtah2027_possession"))

from mtah2027_possession.pipeline import run_pipeline

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Ball Possession Tracking Pipeline")
    parser.add_argument("--config", default="configs/optimal_config.yaml", help="Path to config file")
    parser.add_argument("--out", default=None, help="Output submission CSV path")
    parser.add_argument("--sample", type=int, default=None, help="Sample frame count for quick testing")
    parser.add_argument("--window", type=int, default=100, help="Smoothing window half-width in frames")
    args = parser.parse_args()

    run_pipeline(config_path=args.config, out_path=args.out, sample_frames=args.sample, window=args.window)
