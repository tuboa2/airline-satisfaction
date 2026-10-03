"""
CLI Runner for Strategic Ensembling & Tail-Preserving Logit Stacking on Kaggle
Usage:
    python run_ensemble.py                  # Auto-evaluates Rank, Logit SLSQP, and Ridge Interaction Stacking
    python run_ensemble.py --method logit   # Direct Bounded SLSQP Logit Blend
    python run_ensemble.py --method ridge   # Ridge Meta-Learner with Logit Interactions
    python run_ensemble.py --method rank    # Classic Rank Averaging Baseline
"""

import argparse
import sys

from src.config import FeatureConfig, PathConfig
from src.dataset import DatasetIngestion
from src.ensemble import EnsembleOptimizer
from src.postprocess import ExactMatchPostprocessor
from src.utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Strategic Ensemble Optimizer for Kaggle S6E10")
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "logit", "ridge", "rank"],
        help="Ensemble strategy: 'auto' (selects highest OOF ROC-AUC), 'logit' (bounded SLSQP logit), 'ridge' (meta-learner), 'rank' (rank-averaging)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Custom dataset directory containing train.csv and test.csv (default: auto-detect)"
    )
    parser.add_argument(
        "--original_path",
        type=str,
        default="",
        help="Custom directory or CSV path for original host dataset"
    )
    parser.add_argument(
        "--skip_leak_override",
        action="store_true",
        help="Skip exact-match test leakage mining and overrides"
    )
    return parser.parse_args()


def main():
    logger = get_logger("RunEnsemble")
    args = parse_args()

    paths = PathConfig(raw_dir=args.data_dir, original_path=args.original_path) if (args.data_dir or args.original_path) else PathConfig()
    feature_cfg = FeatureConfig()

    optimizer = EnsembleOptimizer(paths=paths, feature_cfg=feature_cfg)
    champ_oof, best_auc, champ_test = optimizer.run_all(chosen_method=args.method)

    # Optional: Exact match test leakage mining if original dataset is available
    if not args.skip_leak_override:
        try:
            ingestion = DatasetIngestion(paths=paths, feature_cfg=feature_cfg)
            orig_files = ingestion.discover_original_files()
            if orig_files:
                import pandas as pd
                orig_parts = []
                for fpath in orig_files:
                    part_df = pd.read_csv(fpath)
                    part_df = ingestion._standardize_columns(part_df)
                    if feature_cfg.target_col in part_df.columns:
                        orig_parts.append(part_df)
                if orig_parts:
                    orig_df = pd.concat(orig_parts, axis=0).drop_duplicates().reset_index(drop=True)
                    test_df = pd.read_csv(paths.test_path)
                    test_df = ingestion._standardize_columns(test_df)
                    postprocessor = ExactMatchPostprocessor(paths=paths, feature_cfg=feature_cfg)
                    matches = postprocessor.find_exact_matches(test_df, orig_df)
                    if len(matches) > 0:
                        postprocessor.apply_overrides("submission.csv", matches)
                        postprocessor.apply_overrides(
                            f"{paths.submissions_dir}/submission_ensemble.csv", matches
                        )
        except Exception as e:
            logger.warning(f"Exact match mining check encountered: {e}")

    logger.info(f"Ensemble execution complete. Champion Blended OOF ROC-AUC: {best_auc:.5f}")


if __name__ == "__main__":
    main()
