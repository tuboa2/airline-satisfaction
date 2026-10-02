"""
Top-Level Training Runner for Kaggle and Local Environments
Execution:
    python run_training.py --model lightgbm
    python run_training.py --model catboost --device cuda
    python run_training.py --model xgboost --device cuda
"""

import argparse
import sys
from src.config import PathConfig, FeatureConfig, TrainConfig
from src.trainer import CrossValidationEngine
from src.utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Kaggle S6E10 Training Pipeline")
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "catboost", "xgboost"],
        help="Model architecture to train (default: lightgbm)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[None, "cuda", "cpu"],
        help="Hardware accelerator to use (default: auto-detect)"
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Custom directory containing train.csv and test.csv (default: auto-detect)"
    )
    return parser.parse_args()


def main():
    logger = get_logger("AirlineTraining")
    args = parse_args()

    paths = PathConfig(raw_dir=args.data_dir) if args.data_dir else PathConfig()
    feature_cfg = FeatureConfig()
    train_cfg = TrainConfig(n_splits=args.folds, random_state=args.seed)

    engine = CrossValidationEngine(
        paths=paths,
        feature_cfg=feature_cfg,
        train_cfg=train_cfg,
        model_name=args.model,
        device=args.device
    )

    oof_auc, _, _ = engine.run()
    logger.info(f"Execution finished successfully. Final OOF ROC-AUC: {oof_auc:.5f}")


if __name__ == "__main__":
    main()
