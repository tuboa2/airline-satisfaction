"""
Cross-Validation Training Engine: Stratified K-Fold with OOF Scoring and Submission Generation
"""

import gc
import os
import logging
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import PathConfig, FeatureConfig, TrainConfig
from src.features import FeaturePipeline
from src.models import get_model
from src.utils import timer, detect_hardware, seed_everything


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
        device: str = None
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

    def run(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Executes complete training and inference pipeline:
        1. Loads train and test data.
        2. Applies full feature pipeline.
        3. Runs 5-Fold Stratified CV.
        4. Logs OOF score and exports submission.csv.
        """
        logging.info("=" * 70)
        logging.info(f"STARTING CROSS-VALIDATION PIPELINE: Model={self.model_name.upper()} | Device={self.device.upper()}")
        logging.info("=" * 70)

        # 1. Load Data
        with timer("Loading Raw Datasets"):
            train_df = pd.read_csv(self.paths.train_path)
            test_df = pd.read_csv(self.paths.test_path)
            logging.info(f"Train Shape: {train_df.shape} | Test Shape: {test_df.shape}")

        # Standardize target
        if train_df[self.feature_cfg.target_col].dtype in [object, bool]:
            y_raw = train_df[self.feature_cfg.target_col].astype(str).str.lower().str.strip()
            y = (y_raw == "satisfied").astype(int).values
            if y.sum() == 0:
                y = (y_raw == "true").astype(int).values
        else:
            y = train_df[self.feature_cfg.target_col].astype(int).values

        test_ids = test_df[self.feature_cfg.id_col].values

        # 2. Feature Engineering
        X_train = self.pipeline.fit_transform(train_df)
        X_test = self.pipeline.transform(test_df, is_train=False)

        feature_names = X_train.columns.tolist()
        logging.info(f"Engineered Feature Count: {len(feature_names)}")

        # Free raw dataframes from memory
        del train_df, test_df
        gc.collect()

        # 3. Stratified K-Fold Training
        skf = StratifiedKFold(
            n_splits=self.train_cfg.n_splits,
            shuffle=self.train_cfg.shuffle,
            random_state=self.train_cfg.random_state
        )

        oof_preds = np.zeros(len(y), dtype=np.float32)
        test_preds = np.zeros(len(X_test), dtype=np.float32)
        fold_scores = []

        params = getattr(self.train_cfg, f"{self.model_name[:3]}_params", self.train_cfg.lgb_params)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
            logging.info("-" * 50)
            logging.info(f"FOLD {fold + 1} / {self.train_cfg.n_splits}")
            logging.info("-" * 50)

            X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y[val_idx]

            model = get_model(self.model_name, params=params, device=self.device)

            with timer(f"Fold {fold + 1} Training"):
                model.fit(X_tr, y_tr, X_va, y_va)

            val_preds = model.predict_proba(X_va)
            oof_preds[val_idx] = val_preds

            fold_auc = roc_auc_score(y_va, val_preds)
            fold_scores.append(fold_auc)
            logging.info(f"--> Fold {fold + 1} ROC-AUC: {fold_auc:.5f}")

            # Accumulate test predictions across folds
            test_preds += model.predict_proba(X_test) / self.train_cfg.n_splits

            # Clean memory
            del X_tr, y_tr, X_va, y_va, model
            gc.collect()

        # 4. Overall Out-Of-Fold Evaluation
        overall_oof_auc = roc_auc_score(y, oof_preds)
        logging.info("=" * 70)
        logging.info(f"5-FOLD CV COMPLETE: Overall OOF ROC-AUC = {overall_oof_auc:.5f}")
        logging.info(f"Mean Fold AUC: {np.mean(fold_scores):.5f} | Std: {np.std(fold_scores):.5f}")
        logging.info("=" * 70)

        # 5. Save Outputs & Submissions
        np.save(os.path.join(self.paths.output_dir, f"oof_preds_{self.model_name}.npy"), oof_preds)
        np.save(os.path.join(self.paths.output_dir, f"test_preds_{self.model_name}.npy"), test_preds)

        submission_path = os.path.join(self.paths.submissions_dir, f"submission_{self.model_name}.csv")
        sub_df = pd.DataFrame({
            self.feature_cfg.id_col: test_ids,
            self.feature_cfg.target_col: test_preds
        })
        sub_df.to_csv(submission_path, index=False)
        logging.info(f"Generated Submission: {submission_path}")

        # Also write standard submission.csv in root for immediate submission
        sub_df.to_csv("submission.csv", index=False)
        logging.info("Generated default 'submission.csv' ready for Kaggle submission!")

        return overall_oof_auc, oof_preds, test_preds
