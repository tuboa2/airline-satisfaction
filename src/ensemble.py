"""
Domain 2: Tail-Preserving Logit Stacking & Meta-Learning Engine for Kaggle S6E10
Implements:
1. Isotonic Recalibration to restore monotonic probability fidelity.
2. Numerical Logit Transformation: logit(p) = ln((p + eps) / (1 - p + eps)).
3. Bounded SLSQP direct ROC-AUC maximization in logit space.
4. Second-stage Ridge Meta-Learner with cross-model interaction and disagreement terms.
5. Auto-selection of superior ensemble strategy and deterministic test override hooks.
"""

import glob
import os
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import FeatureConfig, PathConfig
from src.utils import get_logger, resolve_binary_target, timer


class EnsembleOptimizer:
    """
    Advanced Ensembling Suite:
    - Isotonic Probability Calibration
    - Bounded SLSQP Logit Blending
    - Ridge Meta-Learning with Interaction Terms
    - Uniform Rank-Averaging Baseline
    """

    def __init__(
        self,
        paths: PathConfig = PathConfig(),
        feature_cfg: FeatureConfig = FeatureConfig(),
        epsilon: float = 1e-6,
    ):
        self.paths = paths
        self.feature_cfg = feature_cfg
        self.epsilon = epsilon
        self.logger = get_logger("EnsembleOptimizer")

    def load_ground_truth(self) -> np.ndarray:
        """Loads true binary targets from training dataset."""
        train_df = pd.read_csv(self.paths.train_path)
        if self.feature_cfg.target_col in train_df.columns:
            return resolve_binary_target(train_df[self.feature_cfg.target_col])
        else:
            raise KeyError(f"Target column '{self.feature_cfg.target_col}' not found in train.csv!")

    def load_test_ids(self) -> np.ndarray:
        """Loads test passenger IDs."""
        test_df = pd.read_csv(self.paths.test_path)
        return test_df[self.feature_cfg.id_col].values

    def discover_models(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Scans outputs directory dynamically for completed model OOF and test predictions."""
        available_models = {}
        canonical_map = {
            "lgbm": "lightgbm",
            "cb": "catboost",
            "cat": "catboost",
            "xgb": "xgboost",
            "ft": "ft_transformer",
            "transformer": "ft_transformer",
            "realmlp": "tabular_resnet",
            "resnet": "tabular_resnet",
        }

        pattern = os.path.join(self.paths.output_dir, "oof_preds_*.npy")
        for oof_path in sorted(glob.glob(pattern)):
            fname = os.path.basename(oof_path)
            slug = fname[len("oof_preds_") : -len(".npy")]
            test_path = os.path.join(self.paths.output_dir, f"test_preds_{slug}.npy")

            if os.path.exists(test_path):
                oof = np.load(oof_path)
                test = np.load(test_path)
                if len(oof) > 0 and len(test) > 0:
                    c_name = canonical_map.get(slug.lower(), slug.lower())
                    if c_name not in available_models:
                        available_models[c_name] = (oof, test)
                        self.logger.info(
                            f"Discovered completed model outputs for: '{c_name}' (source file: '{fname}')"
                        )

        return available_models

    def rank_transform(self, preds: np.ndarray) -> np.ndarray:
        """Transforms continuous predictions to normalized [0, 1] ranks."""
        return (rankdata(preds) - 1.0) / (len(preds) - 1.0)

    def calibrate_and_logit(
        self,
        models_dict: dict[str, tuple[np.ndarray, np.ndarray]],
        y_true: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Applies Isotonic Regression to each model's OOF predictions,
        adjusts test predictions, and converts both to logit space.
        """
        cal_oof_logits = {}
        cal_test_logits = {}

        for name, (oof, test) in models_dict.items():
            iso = IsotonicRegression(out_of_bounds="clip")
            p_oof_cal = iso.fit_transform(oof, y_true)
            p_test_cal = iso.predict(test)

            p_oof_clip = np.clip(p_oof_cal, self.epsilon, 1.0 - self.epsilon)
            p_test_clip = np.clip(p_test_cal, self.epsilon, 1.0 - self.epsilon)

            z_oof = logit(p_oof_clip)
            z_test = logit(p_test_clip)

            cal_oof_logits[name] = z_oof
            cal_test_logits[name] = z_test

        return cal_oof_logits, cal_test_logits

    def blend_rank(
        self,
        model_names: list[str],
        oof_list: list[np.ndarray],
        test_list: list[np.ndarray],
        y_true: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray, dict[str, float]]:
        """Classic Rank Averaging Blend."""
        oof_ranks = np.column_stack([self.rank_transform(p) for p in oof_list])
        test_ranks = np.column_stack([self.rank_transform(p) for p in test_list])

        def objective(w):
            weights = np.maximum(0.0, w)
            if weights.sum() == 0:
                weights = np.ones_like(weights)
            weights = weights / weights.sum()
            blend = np.dot(oof_ranks, weights)
            return -roc_auc_score(y_true, blend)

        n = len(model_names)
        init_w = np.ones(n) / n
        res = minimize(objective, init_w, method="Nelder-Mead", options={"maxiter": 1000})

        opt_w = np.maximum(0.0, res.x)
        opt_w = opt_w / opt_w.sum()
        best_auc = -res.fun

        final_oof = np.dot(oof_ranks, opt_w)
        final_test = np.dot(test_ranks, opt_w)
        w_dict = {name: float(w) for name, w in zip(model_names, opt_w)}
        return final_oof, best_auc, final_test, w_dict

    def blend_logit_slsqp(
        self,
        model_names: list[str],
        cal_oof_logits: dict[str, np.ndarray],
        cal_test_logits: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray, dict[str, float]]:
        """
        Direct ROC-AUC Maximization in Calibrated Logit Space via Nelder-Mead.
        Formula: z_blend = sum(w_i * z_i), p_blend = sigmoid(z_blend).
        """
        X_oof_logits = np.column_stack([cal_oof_logits[m] for m in model_names])
        X_test_logits = np.column_stack([cal_test_logits[m] for m in model_names])
        n = len(model_names)

        def objective(w):
            weights = np.maximum(0.0, w)
            if weights.sum() == 0:
                weights = np.ones_like(weights)
            weights = weights / weights.sum()
            z_blend = np.dot(X_oof_logits, weights)
            p_blend = expit(z_blend)
            return -roc_auc_score(y_true, p_blend)

        init_w = np.ones(n) / n
        res = minimize(
            objective,
            init_w,
            method="Nelder-Mead",
            options={"maxiter": 1000},
        )

        opt_w = np.maximum(0.0, res.x)
        opt_w = opt_w / np.sum(opt_w)
        best_auc = -res.fun

        z_oof_final = np.dot(X_oof_logits, opt_w)
        z_test_final = np.dot(X_test_logits, opt_w)

        final_oof = expit(z_oof_final)
        final_test = expit(z_test_final)
        w_dict = {name: float(w) for name, w in zip(model_names, opt_w)}
        return final_oof, best_auc, final_test, w_dict

    def blend_ridge_interactions(
        self,
        model_names: list[str],
        cal_oof_logits: dict[str, np.ndarray],
        cal_test_logits: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Second-Stage Regularized Meta-Learner (Ridge) with Interaction Terms.
        Constructs:
        - Base calibrated logits: z_i
        - Pairwise multiplicative interactions: z_i * z_j
        - Pairwise model disagreement magnitude: |z_i - z_j|
        """
        meta_features_oof = []
        meta_features_test = []

        # 1. Base Logits
        for m in model_names:
            meta_features_oof.append(cal_oof_logits[m])
            meta_features_test.append(cal_test_logits[m])

        # 2. Pairwise Interactions & Disagreements
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                mi, mj = model_names[i], model_names[j]
                zi_oof, zj_oof = cal_oof_logits[mi], cal_oof_logits[mj]
                zi_test, zj_test = cal_test_logits[mi], cal_test_logits[mj]

                # Product interaction
                meta_features_oof.append(zi_oof * zj_oof)
                meta_features_test.append(zi_test * zj_test)

                # Disagreement magnitude
                meta_features_oof.append(np.abs(zi_oof - zj_oof))
                meta_features_test.append(np.abs(zi_test - zj_test))

        X_meta_oof = np.column_stack(meta_features_oof)
        X_meta_test = np.column_stack(meta_features_test)

        # 5-fold cross-validation for meta-learner to prevent meta-overfitting
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_meta_preds = np.zeros(len(y_true), dtype=np.float32)
        test_meta_preds = np.zeros(len(X_meta_test), dtype=np.float32)

        alphas = [0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]

        best_alpha = 10.0
        best_cv_auc = -1.0
        for alpha in alphas:
            fold_aucs = []
            for tr_idx, va_idx in skf.split(X_meta_oof, y_true):
                clf = Ridge(alpha=alpha, random_state=42)
                clf.fit(X_meta_oof[tr_idx], y_true[tr_idx])
                preds = clf.predict(X_meta_oof[va_idx])
                fold_aucs.append(roc_auc_score(y_true[va_idx], preds))
            mean_auc = float(np.mean(fold_aucs))
            if mean_auc > best_cv_auc:
                best_cv_auc = mean_auc
                best_alpha = alpha

        self.logger.info(
            f"Ridge Meta-Learner selected optimal alpha={best_alpha} (Mean CV ROC-AUC: {best_cv_auc:.5f})"
        )

        for tr_idx, va_idx in skf.split(X_meta_oof, y_true):
            X_tr, y_tr = X_meta_oof[tr_idx], y_true[tr_idx]
            X_va = X_meta_oof[va_idx]

            clf = Ridge(alpha=best_alpha, random_state=42)
            clf.fit(X_tr, y_tr)

            oof_meta_preds[va_idx] = clf.predict(X_va)
            test_meta_preds += clf.predict(X_meta_test) / 5.0

        best_auc = roc_auc_score(y_true, oof_meta_preds)
        return oof_meta_preds, best_auc, test_meta_preds

    def run_all(self, chosen_method: str = "auto") -> tuple[np.ndarray, float, np.ndarray]:
        """
        Executes all ensembling paradigms, compares their out-of-fold performance,
        selects the optimal submission, and logs rigorous diagnostic metrics.
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

        # 1. Standalone Performances
        self.logger.info("=" * 70)
        self.logger.info("STANDALONE MODEL PERFORMANCES (OOF ROC-AUC)")
        self.logger.info("=" * 70)
        for name, oof in zip(model_names, oof_list):
            auc = roc_auc_score(y_true, oof)
            self.logger.info(f"  * {name:<20}: {auc:.5f}")

        # 2. Prediction Pearson Correlation Matrix
        if len(model_names) > 1:
            self.logger.info("-" * 70)
            self.logger.info("PREDICTION CORRELATION MATRIX (Pearson):")
            corr_df = pd.DataFrame(
                np.corrcoef(oof_list), index=model_names, columns=model_names
            )
            for col in corr_df.columns:
                corr_str = " | ".join(
                    [f"{corr_df.loc[row, col]:.4f}" for row in corr_df.index]
                )
                self.logger.info(f"  {col:<18}: {corr_str}")
            self.logger.info("-" * 70)

        # 3. Strategy 1: Rank Averaging
        oof_rank, auc_rank, test_rank, w_rank = self.blend_rank(
            model_names, oof_list, test_list, y_true
        )
        self.logger.info(f"Strategy 1 (Rank-Averaged Blend)       : OOF ROC-AUC = {auc_rank:.5f}")

        # 4. Calibration & Logit Conversion
        cal_oof_logits, cal_test_logits = self.calibrate_and_logit(models_dict, y_true)

        # 5. Strategy 2: Bounded Logit Blending (Nelder-Mead)
        oof_slsqp, auc_slsqp, test_slsqp, w_slsqp = self.blend_logit_slsqp(
            model_names, cal_oof_logits, cal_test_logits, y_true
        )
        self.logger.info(f"Strategy 2 (Nelder-Mead Logit Blend)   : OOF ROC-AUC = {auc_slsqp:.5f}")
        for m, w in w_slsqp.items():
            self.logger.info(f"    - Weight for {m:<18}: {w:.4f}")

        # 6. Strategy 3: Ridge Meta-Learner with Interactions
        if len(model_names) > 1:
            oof_ridge, auc_ridge, test_ridge = self.blend_ridge_interactions(
                model_names, cal_oof_logits, cal_test_logits, y_true
            )
            self.logger.info(f"Strategy 3 (Ridge Interaction Stacking): OOF ROC-AUC = {auc_ridge:.5f}")
        else:
            oof_ridge, auc_ridge, test_ridge = oof_slsqp, auc_slsqp, test_slsqp

        # 7. Selection
        candidates = {
            "rank": (oof_rank, auc_rank, test_rank),
            "logit": (oof_slsqp, auc_slsqp, test_slsqp),
            "ridge": (oof_ridge, auc_ridge, test_ridge),
        }

        if chosen_method in candidates:
            champion_name = chosen_method
        else:
            champion_name = max(candidates.keys(), key=lambda k: candidates[k][1])

        champ_oof, champ_auc, champ_test = candidates[champion_name]

        self.logger.info("=" * 70)
        self.logger.info(
            f"CHAMPION ENSEMBLE: '{champion_name.upper()}' with OOF ROC-AUC = {champ_auc:.5f}"
        )
        self.logger.info("=" * 70)

        # Save Final Submission
        test_ids = self.load_test_ids()
        sub_df = pd.DataFrame(
            {self.feature_cfg.id_col: test_ids, self.feature_cfg.target_col: champ_test}
        )

        os.makedirs(self.paths.submissions_dir, exist_ok=True)
        sub_path = os.path.join(self.paths.submissions_dir, "submission_ensemble.csv")
        sub_df.to_csv(sub_path, index=False)
        sub_df.to_csv("submission.csv", index=False)
        self.logger.info(
            f"Exported Champion Submissions to '{sub_path}' and 'submission.csv'"
        )

        return champ_oof, champ_auc, champ_test
