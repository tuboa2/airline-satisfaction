"""
Cross-Validation Training Engine: Stratified K-Fold with OOF Scoring and Submission Generation
Integrates:
- Domain 1: Original host dataset ingestion, provenance weighting, and leakage mining
- Domain 3: Un-distilled Tabular ResNet / RealMLP with PLR embeddings
- Domain 4: Rotational SVD manifolds & Multi-way Bayesian target encoding
"""

import gc
import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import FeatureConfig, PathConfig, TrainConfig
from src.dataset import DatasetIngestion
from src.features import FeaturePipeline
from src.models import get_model
from src.postprocess import ExactMatchPostprocessor
from src.utils import detect_hardware, seed_everything, timer


class CrossValidationEngine:
    """
    Orchestrates full 5-fold cross-validation, OOF evaluation,
    and ensemble test predictions.
    """

    def __init__(
        self,
        paths: PathConfig = PathConfig(),
        feature_cfg: FeatureConfig = FeatureConfig(),
        train_cfg: TrainConfig = TrainConfig(),
        model_name: str = "lightgbm",
        device: str = None,
    ):
        self.paths = paths
        self.feature_cfg = feature_cfg
        self.train_cfg = train_cfg
        self.model_name = model_name

        if device is None:
            self.device, _ = detect_hardware()
        else:
            self.device = device

        seed_everything(self.train_cfg.random_state)
        self.pipeline = FeaturePipeline(self.feature_cfg)

    def run(self) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Executes complete training and inference pipeline:
        1. Ingests synthetic train, test, and auxiliary original dataset.
        2. Applies full feature pipeline (Domain A, C, 4).
        3. Runs 5-Fold Stratified CV on synthetic target distribution with augmented training.
        4. Logs OOF score, exports submission.csv, and checks exact-match overrides.
        """
        logging.info("=" * 70)
        logging.info(
            f"STARTING CROSS-VALIDATION PIPELINE: Model={self.model_name.upper()} | Device={self.device.upper()}"
        )
        logging.info("=" * 70)

        # 1. Load Data with Domain 1 Ingestion
        ingestion = DatasetIngestion(self.paths, self.feature_cfg, self.train_cfg)
        unified_train, test_synth, sample_weights, orig_df = ingestion.load_and_prepare()

        y_all = unified_train[self.feature_cfg.target_col].astype(int).values
        test_ids = test_synth[self.feature_cfg.id_col].values

        # 2. Transductive Feature Engineering
        X_train_all = self.pipeline.fit_transform(unified_train, test_synth)
        X_test = self.pipeline.transform(test_synth, is_train=False)

        feature_names = X_train_all.columns.tolist()
        logging.info(f"Engineered Feature Count: {len(feature_names)}")

        # Isolate synthetic indices for leak-proof CV evaluation
        synth_mask = (unified_train["is_original"] == 1).values
        synth_indices = np.where(synth_mask)[0]
        orig_indices = np.where(~synth_mask)[0]
        y_synth = y_all[synth_mask]

        logging.info(
            f"CV Evaluation Plan: {len(synth_indices)} synthetic holdout rows across {self.train_cfg.n_splits} folds."
        )
        if len(orig_indices) > 0:
            logging.info(
                f"Auxiliary Training Support: {len(orig_indices)} original dataset rows augmenting each fold."
            )

        # 3. Stratified K-Fold Training
        skf = StratifiedKFold(
            n_splits=self.train_cfg.n_splits,
            shuffle=self.train_cfg.shuffle,
            random_state=self.train_cfg.random_state,
        )

        oof_preds = np.zeros(len(synth_indices), dtype=np.float32)
        test_preds = np.zeros(len(X_test), dtype=np.float32)
        fold_scores = []

        model_name_lower = self.model_name.lower()
        if "cat" in model_name_lower or "cb" in model_name_lower:
            params = self.train_cfg.cb_params.copy()
        elif "xgb" in model_name_lower or "xgboost" in model_name_lower:
            params = self.train_cfg.xgb_params.copy()
        elif (
            "resnet" in model_name_lower
            or "realmlp" in model_name_lower
            or "tabular_resnet" in model_name_lower
            or "nn" in model_name_lower
        ):
            params = self.train_cfg.resnet_params.copy()
        elif "transformer" in model_name_lower or "ft" in model_name_lower:
            params = self.train_cfg.ft_params.copy()
        else:
            params = self.train_cfg.lgb_params.copy()

        # Check for teacher predictions if using distilled FT-Transformer
        teacher_oof = None
        teacher_test = None
        if "transformer" in model_name_lower or "ft" in model_name_lower:
            teacher_oof_files = [
                os.path.join(self.paths.output_dir, "oof_preds_lightgbm.npy"),
                os.path.join(self.paths.output_dir, "oof_preds_xgboost.npy"),
                os.path.join(self.paths.output_dir, "oof_preds_catboost.npy"),
            ]
            teacher_test_files = [
                os.path.join(self.paths.output_dir, "test_preds_lightgbm.npy"),
                os.path.join(self.paths.output_dir, "test_preds_xgboost.npy"),
                os.path.join(self.paths.output_dir, "test_preds_catboost.npy"),
            ]
            valid_oof = [np.load(f) for f in teacher_oof_files if os.path.exists(f)]
            valid_test = [np.load(f) for f in teacher_test_files if os.path.exists(f)]
            if valid_oof:
                teacher_oof = np.mean(valid_oof, axis=0)
                logging.info(f"Loaded {len(valid_oof)} GBDT teacher models for soft distillation.")
            if valid_test:
                teacher_test = np.mean(valid_test, axis=0)
                logging.info(f"Loaded {len(valid_test)} GBDT teacher models for test consistency.")

        for fold, (synth_tr_subidx, synth_va_subidx) in enumerate(skf.split(synth_indices, y_synth)):
            logging.info("-" * 50)
            logging.info(f"FOLD {fold + 1} / {self.train_cfg.n_splits}")
            logging.info("-" * 50)

            val_global_idx = synth_indices[synth_va_subidx]
            if len(orig_indices) > 0:
                train_global_idx = np.concatenate([synth_indices[synth_tr_subidx], orig_indices])
            else:
                train_global_idx = synth_indices[synth_tr_subidx]

            X_tr = X_train_all.iloc[train_global_idx]
            y_tr = y_all[train_global_idx]
            sw_tr = sample_weights[train_global_idx]

            X_va = X_train_all.iloc[val_global_idx]
            y_va = y_all[val_global_idx]

            teacher_tr = (
                teacher_oof[synth_tr_subidx]
                if teacher_oof is not None and len(teacher_oof) == len(synth_indices)
                else None
            )

            model = get_model(self.model_name, params=params.copy(), device=self.device)

            with timer(f"Fold {fold + 1} Training ({len(X_tr)} train rows, {len(X_va)} val rows)"):
                model.fit(
                    X_tr,
                    y_tr,
                    X_va,
                    y_va,
                    sample_weight=sw_tr,
                    teacher_train=teacher_tr,
                    X_test=X_test,
                    teacher_test=teacher_test,
                )

            val_preds = model.predict_proba(X_va)
            oof_preds[synth_va_subidx] = val_preds

            fold_auc = roc_auc_score(y_va, val_preds)
            fold_scores.append(fold_auc)
            logging.info(f"--> Fold {fold + 1} ROC-AUC: {fold_auc:.5f}")

            # Accumulate test predictions
            test_preds += model.predict_proba(X_test) / self.train_cfg.n_splits

            # Memory garbage collection
            del X_tr, y_tr, sw_tr, X_va, y_va, model
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            gc.collect()

        # 4. Overall Out-Of-Fold Evaluation
        overall_oof_auc = roc_auc_score(y_synth, oof_preds)
        logging.info("=" * 70)
        logging.info(f"5-FOLD CV COMPLETE: Overall Synthetic OOF ROC-AUC = {overall_oof_auc:.5f}")
        logging.info(
            f"Mean Fold AUC: {np.mean(fold_scores):.5f} | Std: {np.std(fold_scores):.5f}"
        )
        logging.info("=" * 70)

        # 5. Save Outputs & Submissions
        np.save(
            os.path.join(self.paths.output_dir, f"oof_preds_{self.model_name}.npy"),
            oof_preds,
        )
        np.save(
            os.path.join(self.paths.output_dir, f"test_preds_{self.model_name}.npy"),
            test_preds,
        )

        submission_path = os.path.join(
            self.paths.submissions_dir, f"submission_{self.model_name}.csv"
        )
        sub_df = pd.DataFrame(
            {self.feature_cfg.id_col: test_ids, self.feature_cfg.target_col: test_preds}
        )
        sub_df.to_csv(submission_path, index=False)
        sub_df.to_csv("submission.csv", index=False)
        logging.info(f"Generated Submissions: '{submission_path}' & 'submission.csv'")

        # 6. Domain 1: Safe Post-Processing Exact-Match Target Leakage Mining
        if orig_df is not None:
            postprocessor = ExactMatchPostprocessor(self.paths, self.feature_cfg)
            matches = postprocessor.find_exact_matches(test_synth, orig_df)
            if len(matches) > 0:
                postprocessor.apply_overrides(submission_path, matches)
                postprocessor.apply_overrides("submission.csv", matches)

        return overall_oof_auc, oof_preds, test_preds
