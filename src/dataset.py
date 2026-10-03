"""
Domain 1: High-Leverage Dataset Ingestion & Preprocessing Alignment
Handles physical concatenation of original host dataset with synthetic train data,
provenance indicator ('is_original'), sample weight attenuation (0.5-0.75),
and adversarial density ratio weighting.
"""

import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import FeatureConfig, PathConfig, TrainConfig
from src.utils import get_logger, resolve_binary_target, timer


class DatasetIngestion:
    """
    Ingests, aligns, and weights synthetic training data and auxiliary
    original host dataset ('teejmahal20/airline-passenger-satisfaction').
    """

    def __init__(
        self,
        paths: PathConfig = PathConfig(),
        feature_cfg: FeatureConfig = FeatureConfig(),
        train_cfg: TrainConfig = TrainConfig(),
    ):
        self.paths = paths
        self.feature_cfg = feature_cfg
        self.train_cfg = train_cfg
        self.logger = get_logger("DatasetIngestion")

    def _standardize_columns(self, df: pd.DataFrame, is_original: bool = False) -> pd.DataFrame:
        """Standardizes column names and types across datasets."""
        data = df.copy()

        # Drop identifiers and row indices only from original host dataset
        cols_to_drop = ["Unnamed: 0", "id", "ID"] if is_original else ["Unnamed: 0"]
        drop_existing = [c for c in cols_to_drop if c in data.columns]
        if drop_existing:
            data = data.drop(columns=drop_existing)

        # Standardize target column if present
        target_candidates = ["satisfaction", "Satisfaction", "target", "Target"]
        for tc in target_candidates:
            if tc in data.columns:
                data[self.feature_cfg.target_col] = resolve_binary_target(data[tc])
                if tc != self.feature_cfg.target_col:
                    data = data.drop(columns=[tc])
                break

        # Standardize string values in categorical columns
        cat_mappings = {
            "Gender": {"male": "Male", "female": "Female"},
            "Customer Type": {
                "loyal customer": "Loyal Customer",
                "disloyal customer": "disloyal Customer",
                "disloyal Customer": "disloyal Customer",
            },
            "Type of Travel": {
                "business travel": "Business travel",
                "personal travel": "Personal Travel",
            },
            "Class": {
                "business": "Business",
                "eco": "Eco",
                "eco plus": "Eco Plus",
            },
        }

        for col, mapping in cat_mappings.items():
            if col in data.columns:
                data[col] = (
                    data[col]
                    .astype(str)
                    .str.strip()
                    .map(lambda s: mapping.get(s.lower(), s))
                )

        # Ensure rating columns are integers
        for rc in self.feature_cfg.rating_cols:
            if rc in data.columns:
                data[rc] = (
                    pd.to_numeric(data[rc], errors="coerce").fillna(0).astype(int)
                )

        # Align delay columns
        if (
            "Departure Delay in Minutes" in data.columns
            and "Arrival Delay in Minutes" in data.columns
        ):
            data["Arrival Delay in Minutes"] = data["Arrival Delay in Minutes"].fillna(
                data["Departure Delay in Minutes"]
            )

        return data

    def discover_original_files(self) -> list[str]:
        """Finds candidate CSV files for the original dataset."""
        candidates = []
        if self.paths.original_path and os.path.exists(self.paths.original_path):
            if os.path.isdir(self.paths.original_path):
                csvs = glob.glob(
                    os.path.join(self.paths.original_path, "*.csv")
                ) + glob.glob(
                    os.path.join(self.paths.original_path, "**", "*.csv"),
                    recursive=True,
                )
                candidates.extend(csvs)
            elif self.paths.original_path.endswith(".csv"):
                candidates.append(self.paths.original_path)

        # Also dynamically search /kaggle/input and standard local directories
        search_dirs = ["/kaggle/input", "data/original", "data/raw/original", "data/external"]
        for sdir in search_dirs:
            if os.path.exists(sdir):
                all_csvs = glob.glob(os.path.join(sdir, "**", "*.csv"), recursive=True)
                for f in all_csvs:
                    f_lower = f.lower()
                    if "playground-series-s6e10" in f_lower or "competitions" in f_lower or "sample_submission" in f_lower:
                        continue
                    if "airline" in f_lower or "satisfaction" in f_lower or "passenger" in f_lower:
                        candidates.append(f)

        # Remove synthetic train/test if accidentally matched
        filtered = []
        train_abs = os.path.abspath(self.paths.train_path) if self.paths.train_path else ""
        test_abs = os.path.abspath(self.paths.test_path) if self.paths.test_path else ""
        sub_abs = os.path.abspath(self.paths.sample_sub_path) if self.paths.sample_sub_path else ""

        for c in candidates:
            c_abs = os.path.abspath(c)
            if c_abs in [train_abs, test_abs, sub_abs]:
                continue
            c_base = os.path.basename(c).lower()
            c_dir = os.path.dirname(c).lower()
            if "playground-series-s6e10" in c_dir or "competitions" in c_dir:
                continue
            if c_base in ["sample_submission.csv"]:
                continue
            filtered.append(c)

        return sorted(list(set(filtered)))

    def compute_adversarial_weights(
        self, train_synth: pd.DataFrame, orig_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Calculates density ratio weights w(x) = p_synth(x) / (1 - p_synth(x))
        using an adversarial classifier distinguishing synthetic from original.
        """
        from lightgbm import LGBMClassifier

        self.logger.info(
            "Computing adversarial density ratio weights for original rows..."
        )

        feature_cols = [
            c
            for c in train_synth.columns
            if c in orig_df.columns
            and c not in [self.feature_cfg.target_col, "is_original", "sample_weight"]
        ]

        synth_adv = train_synth[feature_cols].copy()
        synth_adv["_adv_target"] = 1

        orig_adv = orig_df[feature_cols].copy()
        orig_adv["_adv_target"] = 0

        adv_combined = pd.concat([synth_adv, orig_adv], axis=0).reset_index(drop=True)

        # Categorical handling
        for col in self.feature_cfg.categorical_cols:
            if col in adv_combined.columns:
                adv_combined[col] = adv_combined[col].astype("category")

        X = adv_combined.drop(columns=["_adv_target"])
        y = adv_combined["_adv_target"].values

        clf = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=self.train_cfg.random_state,
            n_jobs=-1,
            verbose=-1,
        )

        skf = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=self.train_cfg.random_state
        )
        adv_preds = np.zeros(len(adv_combined), dtype=np.float32)

        for tr_idx, va_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_va = X.iloc[va_idx]
            clf.fit(X_tr, y_tr)
            adv_preds[va_idx] = clf.predict_proba(X_va)[:, 1]

        # Extract predictions for original rows
        p_synth_for_orig = adv_preds[y == 0]

        # Density ratio calculation: w(x) = p / (1 - p)
        epsilon = 1e-6
        raw_weights = p_synth_for_orig / (1.0 - p_synth_for_orig + epsilon)

        # Clip bounds to avoid extreme gradient disturbance
        clipped_weights = np.clip(
            raw_weights,
            a_min=self.train_cfg.density_ratio_clip_min,
            a_max=self.train_cfg.density_ratio_clip_max,
        )
        self.logger.info(
            f"Density ratio weights: Min={clipped_weights.min():.4f}, "
            f"Mean={clipped_weights.mean():.4f}, Max={clipped_weights.max():.4f}"
        )
        return clipped_weights

    def load_and_prepare(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame | None]:
        """
        Loads synthetic train/test, ingests and aligns original dataset if available,
        adds 'is_original' provenance column, assigns sample weights,
        and returns (unified_train, test_synth, sample_weights, orig_df_raw).
        """
        with timer("Dataset Ingestion & Alignment"):
            train_synth = pd.read_csv(self.paths.train_path)
            test_synth = pd.read_csv(self.paths.test_path)
            train_synth = self._standardize_columns(train_synth, is_original=False)
            test_synth = self._standardize_columns(test_synth, is_original=False)

            train_synth["is_original"] = 1
            test_synth["is_original"] = 1

            orig_df_raw = None
            orig_files = (
                self.discover_original_files()
                if self.train_cfg.use_original_data
                else []
            )

            if orig_files:
                self.logger.info(
                    f"Discovered {len(orig_files)} original data file(s): {orig_files}"
                )
                orig_parts = []
                for fpath in orig_files:
                    try:
                        part_df = pd.read_csv(fpath)
                        part_df = self._standardize_columns(part_df, is_original=True)
                        if self.feature_cfg.target_col in part_df.columns:
                            orig_parts.append(part_df)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to read original file {fpath}: {e}"
                        )

                if orig_parts:
                    orig_df = pd.concat(orig_parts, axis=0).reset_index(drop=True)
                    # Deduplicate original rows
                    orig_df = orig_df.drop_duplicates().reset_index(drop=True)
                    orig_df["is_original"] = 0
                    orig_df_raw = orig_df.copy()

                    self.logger.info(
                        f"Original Host Dataset loaded: {len(orig_df)} rows. "
                        f"Target Satisfaction Mean: {orig_df[self.feature_cfg.target_col].mean():.4f}"
                    )

                    # Compute sample weights
                    if self.train_cfg.use_density_ratio_weighting:
                        orig_weights = self.compute_adversarial_weights(
                            train_synth, orig_df
                        )
                    else:
                        orig_weights = np.full(
                            len(orig_df),
                            self.train_cfg.original_sample_weight,
                            dtype=np.float32,
                        )

                    orig_df["sample_weight"] = orig_weights
                    train_synth["sample_weight"] = 1.0

                    # Align columns before concatenation
                    common_cols = [
                        c for c in train_synth.columns if c in orig_df.columns
                    ]
                    unified_train = pd.concat(
                        [train_synth[common_cols], orig_df[common_cols]], axis=0
                    ).reset_index(drop=True)

                    sample_weights = unified_train["sample_weight"].values.astype(
                        np.float32
                    )
                    unified_train = unified_train.drop(columns=["sample_weight"])

                    self.logger.info(
                        f"Unified Training Set assembled: {len(unified_train)} rows "
                        f"(Synthetic: {len(train_synth)}, Original: {len(orig_df)})."
                    )
                    return unified_train, test_synth, sample_weights, orig_df_raw

            # Fallback if no original data found or disabled
            self.logger.info(
                f"Training with synthetic dataset only: {len(train_synth)} rows."
            )
            sample_weights = np.ones(len(train_synth), dtype=np.float32)
            return train_synth, test_synth, sample_weights, None
