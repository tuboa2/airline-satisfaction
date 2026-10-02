"""
CLI Runner for Strategic Ensembling & Rank Blending on Kaggle

Usage:
    python run_ensemble.py
    python run_ensemble.py --method rank
    python run_ensemble.py --method prob
"""

import argparse
import sys

from src.config import PathConfig, FeatureConfig
from src.ensemble import EnsembleOptimizer
from src.utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Strategic Ensemble Optimizer for Kaggle S6E10")
    parser.add_argument(
        "--method",
        type=str,
        default="rank",
        choices=["rank", "prob"],
        help="Blending method: 'rank' (rank-averaged, recommended) or 'prob' (linear probability)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Custom dataset directory containing train.csv and test.csv (default: auto-detect)"
    )
    return parser.parse_args()


def main():
    logger = get_logger("RunEnsemble")
    args = parse_args()

    paths = PathConfig(raw_dir=args.data_dir) if args.data_dir else PathConfig()
    feature_cfg = FeatureConfig()

    optimizer = EnsembleOptimizer(paths=paths, feature_cfg=feature_cfg)
    _, best_auc, _, weights = optimizer.optimize_blend(use_rank=(args.method == "rank"))

    logger.info(f"Ensemble optimization complete. Final Blended OOF ROC-AUC: {best_auc:.5f}")


if __name__ == "__main__":
    main()
