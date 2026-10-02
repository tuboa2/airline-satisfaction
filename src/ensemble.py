"""
Strategic Ensembling & Rank-Averaging Engine for Kaggle S6E10
Optimizes diverse model blend weights to maximize Out-of-Fold ROC-AUC.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from src.config import PathConfig, FeatureConfig
from src.utils import get_logger, timer


class EnsembleOptimizer:
    """
    Optimizes ensemble weights across GBDTs (LightGBM, CatBoost, XGBoost)
    and Orthogonal Tabular Neural Networks (FT-Transformer).
    """

    def __init__(self, paths: PathConfig = PathConfig(), feature_cfg: FeatureConfig = FeatureConfig()):
        self.paths = paths
        self.feature_cfg = feature_cfg
        self.logger = get_logger("EnsembleOptimizer")

    def load_ground_truth(self) -> np.ndarray:
        """Loads true binary targets from training dataset."""
        train_df = pd.read_csv(self.paths.train_path)
        if train_df[self.feature_cfg.target_col].dtype in [object, bool]:
            y_raw = train_df[self.feature_cfg.target_col].astype(str).str.lower().str.strip()
            y = (y_raw == "satisfied").astype(int).values
            if y.sum() == 0:
                y = (y_raw == "true").astype(int).values
        else:
            y = train_df[self.feature_cfg.target_col].astype(int).values
        return y

    def load_test_ids(self) -> np.ndarray:
        """Loads test passenger IDs."""
        test_df = pd.read_csv(self.paths.test_path)
        return test_df[self.feature_cfg.id_col].values

    def discover_models(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Scans outputs directory for completed model OOF and test predictions."""
        available_models = {}
        candidate_slugs = [
            "lightgbm",
            "xgboost",
            "catboost",
            "transformer",
            "ft_transformer"
        ]

        for slug in candidate_slugs:
            oof_path = os.path.join(self.paths.output_dir, f"oof_preds_{slug}.npy")
            test_path = os.path.join(self.paths.output_dir, f"test_preds_{slug}.npy")

            if os.path.exists(oof_path) and os.path.exists(test_path):
                oof = np.load(oof_path)
                test = np.load(test_path)
                available_models[slug] = (oof, test)
                self.logger.info(f"Discovered model outputs for: '{slug}'")

        return available_models

    def rank_transform(self, preds: np.ndarray) -> np.ndarray:
        """Transforms continuous predictions to normalized [0, 1] ranks."""
        return (rankdata(preds) - 1.0) / (len(preds) - 1.0)

    def optimize_blend(
        self,
        use_rank: bool = True
    ) -> Tuple[np.ndarray, float, np.ndarray, Dict[str, float]]:
        """
        Solves bounded optimization problem to maximize ROC-AUC.
        Args:
            use_rank: Whether to normalize predictions to uniform ranks before blending.
        """
        models_dict = self.discover_models()
        if not models_dict:
            raise FileNotFoundError(
                f"No model predictions found in '{self.paths.output_dir}'. "
                "Train at least two models before ensembling."
            )

        model_names = list(models_dict.keys())
        y_true = self.load_ground_truth()

        oof_list = [models_dict[m][0] for m in model_names]
        test_list = [models_dict[m][1] for m in model_names]

        # Log standalone performance
        self.logger.info("=" * 65)
        self.logger.info("STANDALONE MODEL PERFORMANCES (OOF ROC-AUC)")
        self.logger.info("=" * 65)
        for name, oof in zip(model_names, oof_list):
            auc = roc_auc_score(y_true, oof)
            self.logger.info(f"  * {name:<18}: {auc:.5f}")

        # Compute error / prediction correlations
        if len(model_names) > 1:
            self.logger.info("-" * 65)
            self.logger.info("PREDICTION CORRELATION MATRIX:")
            corr_df = pd.DataFrame(
                np.corrcoef(oof_list),
                index=model_names,
                columns=model_names
            )
            for col in corr_df.columns:
                corr_str = " | ".join([f"{corr_df.loc[row, col]:.4f}" for row in corr_df.index])
                self.logger.info(f"  {col:<16}: {corr_str}")
            self.logger.info("-" * 65)

        # Prepare matrices for blending
        if use_rank:
            oof_matrix = np.column_stack([self.rank_transform(p) for p in oof_list])
            test_matrix = np.column_stack([self.rank_transform(p) for p in test_list])
        else:
            oof_matrix = np.column_stack(oof_list)
            test_matrix = np.column_stack(test_list)

        n_models = len(model_names)

        # Objective function (Negative ROC-AUC)
        def objective(weights):
            w = weights / np.sum(weights)
            blend_oof = np.dot(oof_matrix, w)
            return -roc_auc_score(y_true, blend_oof)

        # Equal-weight initialization
        init_weights = np.ones(n_models) / n_models
        bounds = [(0.0, 1.0) for _ in range(n_models)]

        res = minimize(
            objective,
            init_weights,
            method="Nelder-Mead",
            options={"maxiter": 1000, "disp": False}
        )

        opt_weights = np.maximum(0.0, res.x)
        opt_weights = opt_weights / np.sum(opt_weights)
        best_oof_auc = -res.fun

        weight_dict = {name: float(w) for name, w in zip(model_names, opt_weights)}

        self.logger.info("=" * 65)
        self.logger.info(f"OPTIMAL ENSEMBLE ({'RANK-AVERAGED' if use_rank else 'PROBABILITY'} BLEND)")
        self.logger.info(f"Optimized OOF ROC-AUC: {best_oof_auc:.5f}")
        for name, w in weight_dict.items():
            self.logger.info(f"  - Weight for {name:<18}: {w:.4f}")
        self.logger.info("=" * 65)

        # Generate blended predictions
        final_oof = np.dot(oof_matrix, opt_weights)
        final_test = np.dot(test_matrix, opt_weights)

        # Save submissions
        test_ids = self.load_test_ids()
        sub_df = pd.DataFrame({
            self.feature_cfg.id_col: test_ids,
            self.feature_cfg.target_col: final_test
        })

        os.makedirs(self.paths.submissions_dir, exist_ok=True)
        sub_path = os.path.join(self.paths.submissions_dir, "submission_ensemble.csv")
        sub_df.to_csv(sub_path, index=False)
        sub_df.to_csv("submission.csv", index=False)
        self.logger.info(f"Generated Ensemble Submissions: '{sub_path}' & 'submission.csv'")

        return final_oof, best_oof_auc, final_test, weight_dict
