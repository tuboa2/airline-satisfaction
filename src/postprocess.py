"""
Domain 1: Exact-Match Target Leakage Mining & Postprocessing Overrides
Mines exact duplicate feature vectors between synthetic test set and original/train datasets.
Safely overrides probabilistic test predictions with deterministic ground truth labels.
Strictly segregated from the cross-validation loop to prevent public/private shakeup.
"""

import os

import pandas as pd

from src.config import FeatureConfig, PathConfig
from src.utils import get_logger, resolve_binary_target, timer


class ExactMatchPostprocessor:
    """
    Identifies deterministic exact matches between test set feature vectors
    and historical ground-truth datasets, applying Bayesian prior target overrides.
    """

    def __init__(
        self,
        paths: PathConfig = PathConfig(),
        feature_cfg: FeatureConfig = FeatureConfig(),
    ):
        self.paths = paths
        self.feature_cfg = feature_cfg
        self.logger = get_logger("ExactMatchPostprocessor")

    def find_exact_matches(
        self,
        test_df: pd.DataFrame,
        reference_df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Locates exact feature matches between test_df and reference_df (train + original).
        Returns a DataFrame mapping test index / ID to verified target label.
        """
        with timer("Exact Duplicate Mining"):
            if feature_cols is None:
                # Columns to exclude from matching
                exclude_cols = [
                    self.feature_cfg.id_col,
                    self.feature_cfg.target_col,
                    "Unnamed: 0",
                    "id",
                    "ID",
                    "sample_weight",
                    "is_original",
                ]
                feature_cols = [
                    c
                    for c in test_df.columns
                    if c in reference_df.columns and c not in exclude_cols
                ]

            self.logger.info(
                f"Scanning {len(test_df)} test rows against {len(reference_df)} reference rows "
                f"across {len(feature_cols)} feature dimensions..."
            )

            # Standardize representation for exact matching
            test_subset = test_df[feature_cols].copy()
            ref_subset = reference_df[
                feature_cols + [self.feature_cfg.target_col]
            ].copy()
            ref_subset[self.feature_cfg.target_col] = resolve_binary_target(
                ref_subset[self.feature_cfg.target_col]
            )

            # Impute arrival delays consistently
            if (
                "Arrival Delay in Minutes" in test_subset.columns
                and "Departure Delay in Minutes" in test_subset.columns
            ):
                test_subset["Arrival Delay in Minutes"] = test_subset[
                    "Arrival Delay in Minutes"
                ].fillna(test_subset["Departure Delay in Minutes"])
                ref_subset["Arrival Delay in Minutes"] = ref_subset[
                    "Arrival Delay in Minutes"
                ].fillna(ref_subset["Departure Delay in Minutes"])

            # Collapse any conflicting reference duplicates by majority vote
            ref_grouped = (
                ref_subset.groupby(feature_cols)[self.feature_cfg.target_col]
                .mean()
                .reset_index()
            )
            # Only keep definitive (unambiguous) matches
            ref_unambiguous = ref_grouped[
                (ref_grouped[self.feature_cfg.target_col] == 0.0)
                | (ref_grouped[self.feature_cfg.target_col] == 1.0)
            ].copy()

            test_indexed = test_df[[self.feature_cfg.id_col] + feature_cols].copy()

            # Inner merge to isolate exact identical feature vectors
            matches = pd.merge(
                test_indexed,
                ref_unambiguous,
                on=feature_cols,
                how="inner",
            )

            match_count = len(matches)
            if match_count > 0:
                pos_matches = (matches[self.feature_cfg.target_col] == 1.0).sum()
                neg_matches = (matches[self.feature_cfg.target_col] == 0.0).sum()
                self.logger.info(
                    f"Found {match_count} exact test matches ({pos_matches} satisfied, {neg_matches} neutral/dissatisfied)!"
                )
            else:
                self.logger.info(
                    "No exact duplicate collisions detected between test and reference data."
                )

            return matches[[self.feature_cfg.id_col, self.feature_cfg.target_col]]

    def apply_overrides(
        self,
        submission_path: str,
        matches_df: pd.DataFrame,
        override_pos_prob: float = 0.9999,
        override_neg_prob: float = 0.0001,
        output_path: str | None = None,
    ) -> pd.DataFrame:
        """
        Applies deterministic probability overrides to the submission file for verified exact matches.
        Uses extreme logistic probabilities (0.9999 / 0.0001) to preserve ranking while bounding log-loss.
        """
        if not os.path.exists(submission_path):
            self.logger.warning(
                f"Submission file '{submission_path}' not found. Skipping override."
            )
            return None

        sub = pd.read_csv(submission_path)
        if len(matches_df) == 0:
            return sub

        match_dict = dict(
            zip(
                matches_df[self.feature_cfg.id_col],
                matches_df[self.feature_cfg.target_col],
            )
        )

        initial_preds = sub[self.feature_cfg.target_col].copy()
        replaced = 0

        for idx, row in sub.iterrows():
            row_id = row[self.feature_cfg.id_col]
            if row_id in match_dict:
                true_label = match_dict[row_id]
                new_prob = override_pos_prob if true_label == 1.0 else override_neg_prob
                sub.at[idx, self.feature_cfg.target_col] = new_prob
                replaced += 1

        self.logger.info(
            f"Successfully applied {replaced} deterministic exact-match overrides to '{submission_path}'."
        )

        out_file = output_path or submission_path
        sub.to_csv(out_file, index=False)
        self.logger.info(f"Saved leak-overridden submission to: {out_file}")
        return sub
