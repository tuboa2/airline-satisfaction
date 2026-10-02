"""
Feature Engineering Pipeline: Implements All 10 Empirically Verified Paradigms
Plus Domain A (Density & Frequency Forensics) and Domain C (Transductive Encodings)
"""

import logging
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans

from src.config import FeatureConfig
from src.utils import reduce_mem_usage, timer


class FeaturePipeline:
    """
    Production-grade Feature Engineering Pipeline.
    Implements:
    - 10 Verified Empirical Paradigms (Delay dynamics, Simpson's law, Rasch IRT, Golden/Dead zones, etc.)
    - Domain A: High-Order Categorical Frequency Encodings (sampling density proxy)
    - Domain A: Geometric Centroid & Sub-Cluster Distance Features (Manifold proximity)
    - Domain A: Local Outlier / Mode Collapse Density Score
    - Domain C: Safe Transductive Feature Statistics on concat(train, test)
    """

    def __init__(self, config: FeatureConfig = FeatureConfig()):
        self.config = config
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.freq_maps: Dict[str, Dict[str, float]] = {}
        self.count_maps: Dict[str, Dict[str, int]] = {}

        # Centroid and density parameters
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        self.global_centroid_1: Optional[np.ndarray] = None
        self.global_centroid_0: Optional[np.ndarray] = None
        self.mbk_1: Optional[MiniBatchKMeans] = None
        self.mbk_0: Optional[MiniBatchKMeans] = None
        self.mbk_anomaly: Optional[MiniBatchKMeans] = None

        self.fitted: bool = False

    def _get_frequency_keys(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Generates composite interaction keys for high-order frequency analysis."""
        age_tier = (df["Age"] // 10).astype(str)
        key1 = (
            df["Class"].astype(str) + "_" +
            df["Type of Travel"].astype(str) + "_" +
            df["Inflight wifi service"].astype(str) + "_" +
            df["Online boarding"].astype(str)
        )
        key2 = (
            df["Class"].astype(str) + "_" +
            df["Customer Type"].astype(str) + "_" +
            df["Online boarding"].astype(str) + "_" +
            df["Checkin service"].astype(str)
        )
        key3 = (
            df["Type of Travel"].astype(str) + "_" +
            df["Class"].astype(str) + "_" +
            df["Online boarding"].astype(str) + "_" +
            df["Seat comfort"].astype(str)
        )
        key4 = (
            df["Class"].astype(str) + "_" +
            df["Type of Travel"].astype(str) + "_" +
            age_tier
        )
        return {"key1": key1, "key2": key2, "key3": key3, "key4": key4}

    def fit(self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None) -> "FeaturePipeline":
        """Fits transductive frequency statistics, geometric centroids, and label encoders."""
        with timer("Fitting FeaturePipeline (Transductive Density & Geometric Forensics)"):
            # 1. Prepare combined dataframe for transductive frequency and encoder fitting
            if test_df is not None:
                full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
            else:
                full_df = train_df

            # 2. High-Order Categorical Frequency Maps
            if self.config.enable_high_order_freq:
                full_keys = self._get_frequency_keys(full_df)
                total_rows = len(full_df)
                for name, series in full_keys.items():
                    val_counts = series.value_counts()
                    self.count_maps[name] = val_counts.to_dict()
                    self.freq_maps[name] = (val_counts / total_rows).to_dict()

            # 3. Geometric Centroid & Density Modeling (Domain A)
            if self.config.enable_density_forensics:
                core_cols = self.config.core_centroid_cols
                # Pre-impute arrival delay for standardizer
                arr_delay_train = train_df["Arrival Delay in Minutes"].fillna(train_df["Departure Delay in Minutes"])
                train_core = train_df[core_cols].copy()
                train_core["Arrival Delay in Minutes"] = arr_delay_train

                self.scaler_mean = train_core.mean(axis=0).values.astype(np.float32)
                self.scaler_std = (train_core.std(axis=0) + 1e-6).values.astype(np.float32)

                X_train_scaled = ((train_core.values - self.scaler_mean) / self.scaler_std).astype(np.float32)
                y_train = (train_df[self.config.target_col] == 1).values if self.config.target_col in train_df.columns else None

                if y_train is not None:
                    # Positive and negative global centroids
                    self.global_centroid_1 = X_train_scaled[y_train].mean(axis=0)
                    self.global_centroid_0 = X_train_scaled[~y_train].mean(axis=0)

                    # Sub-cluster centroids for positive & negative manifolds
                    n_c = self.config.kmeans_clusters_per_class
                    self.mbk_1 = MiniBatchKMeans(
                        n_clusters=n_c, batch_size=4096, random_state=42, n_init=3
                    ).fit(X_train_scaled[y_train])

                    self.mbk_0 = MiniBatchKMeans(
                        n_clusters=n_c, batch_size=4096, random_state=42, n_init=3
                    ).fit(X_train_scaled[~y_train])

                # Local Outlier / Mode Collapse Anomaly Clustering on full data
                full_arr_delay = full_df["Arrival Delay in Minutes"].fillna(full_df["Departure Delay in Minutes"])
                full_core = full_df[core_cols].copy()
                full_core["Arrival Delay in Minutes"] = full_arr_delay
                X_full_scaled = ((full_core.values - self.scaler_mean) / self.scaler_std).astype(np.float32)

                self.mbk_anomaly = MiniBatchKMeans(
                    n_clusters=self.config.kmeans_anomaly_clusters, batch_size=4096, random_state=42, n_init=3
                ).fit(X_full_scaled)

            # 4. Fit LabelEncoders on full combined data
            cat_columns = [
                "Gender", "Customer Type", "Type of Travel", "Class",
                "class_x_travel_type", "gate_x_business"
            ]

            # Create synthetic composite categories on full_df for consistent encoder fit
            temp_is_bus = (full_df["Type of Travel"] == "Business travel").astype(np.int8)
            full_class_travel = full_df["Class"].astype(str) + "_" + full_df["Type of Travel"].astype(str)
            full_gate_bus = full_df["Gate location"].astype(str) + "_" + temp_is_bus.astype(str)

            col_data_map = {
                "Gender": full_df["Gender"].astype(str),
                "Customer Type": full_df["Customer Type"].astype(str),
                "Type of Travel": full_df["Type of Travel"].astype(str),
                "Class": full_df["Class"].astype(str),
                "class_x_travel_type": full_class_travel,
                "gate_x_business": full_gate_bus
            }

            for col in cat_columns:
                le = LabelEncoder()
                le.fit(col_data_map[col])
                self.label_encoders[col] = le

            self.fitted = True
            return self

    def transform(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Applies vectorized feature transformations to training or test data."""
        if not self.fitted:
            raise RuntimeError("FeaturePipeline must be fitted via fit() or fit_transform() before transform()!")

        with timer(f"Feature Engineering ({'Train' if is_train else 'Test'})"):
            data = df.copy()

            # -------------------------------------------------------------
            # 1. PHYSICAL DELAY DYNAMICS & IMPUTATION
            # -------------------------------------------------------------
            data["Arrival_Delay_is_nan"] = data["Arrival Delay in Minutes"].isna().astype(np.int8)
            data["Arrival Delay in Minutes"] = data["Arrival Delay in Minutes"].fillna(
                data["Departure Delay in Minutes"]
            )

            dep_delay = data["Departure Delay in Minutes"]
            arr_delay = data["Arrival Delay in Minutes"]

            data["total_delay"] = dep_delay + arr_delay
            data["has_delay"] = (data["total_delay"] > 0).astype(np.int8)
            data["has_severe_delay"] = (data["total_delay"] > 30).astype(np.int8)

            # Airborne Delay Recovery Delta (+11% satisfaction lift when positive)
            data["delay_recovery_delta"] = dep_delay - arr_delay
            data["worsened_in_air"] = (arr_delay > dep_delay).astype(np.int8)

            # Delay Intensity per 100 miles
            data["delay_intensity"] = data["total_delay"] / np.maximum(1.0, data["Flight Distance"] / 100.0)

            # The 15-Minute Flatline Law
            data["delay_tier"] = np.select(
                [data["total_delay"] == 0, data["total_delay"] < 15, data["total_delay"] >= 15],
                [0, 1, 2]
            ).astype(np.int8)

            # -------------------------------------------------------------
            # 2. ZERO-INFLATION & "NOT APPLICABLE" INDICATORS
            # -------------------------------------------------------------
            data["wifi_is_0"] = (data["Inflight wifi service"] == 0).astype(np.int8)
            data["booking_is_0"] = (data["Ease of Online booking"] == 0).astype(np.int8)
            data["boarding_is_0"] = (data["Online boarding"] == 0).astype(np.int8)
            data["time_convenient_is_0"] = (data["Departure/Arrival time convenient"] == 0).astype(np.int8)
            data["total_na_ratings"] = (data[self.config.rating_cols] == 0).sum(axis=1).astype(np.int8)

            # -------------------------------------------------------------
            # 3. NON-LINEAR INFLECTION THRESHOLDS & TRUMP CARDS
            # -------------------------------------------------------------
            data["wifi_is_5"] = (data["Inflight wifi service"] == 5).astype(np.int8)
            data["high_online_boarding"] = (data["Online boarding"] >= 4).astype(np.int8)
            data["high_seat_comfort"] = (data["Seat comfort"] >= 4).astype(np.int8)
            data["high_entertainment"] = (data["Inflight entertainment"] >= 4).astype(np.int8)
            data["checkin_is_acceptable"] = (data["Checkin service"] >= 3).astype(np.int8)

            # Total Digital Failure Gate (<6% satisfaction)
            data["digital_failure"] = (
                (data["Online boarding"] <= 3) & (data["Inflight wifi service"] <= 3)
            ).astype(np.int8)

            # -------------------------------------------------------------
            # 4. DOMAIN PILLAR AGGREGATIONS & BOTTLENECK LAW
            # -------------------------------------------------------------
            ratings_no_zero = data[self.config.rating_cols].replace(0, np.nan)

            data["min_service_rating"] = ratings_no_zero.min(axis=1).fillna(3).astype(np.float32)
            data["has_service_failure"] = (data["min_service_rating"] <= 2).astype(np.int8)
            data["is_all_pass_service"] = (data["min_service_rating"] >= 3).astype(np.int8)

            data["digital_score"] = ratings_no_zero[[
                "Online boarding", "Inflight wifi service", "Ease of Online booking"
            ]].mean(axis=1).fillna(3).astype(np.float32)

            data["cabin_score"] = ratings_no_zero[[
                "Seat comfort", "Leg room service", "Cleanliness", "Food and drink"
            ]].mean(axis=1).fillna(3).astype(np.float32)

            data["staff_score"] = ratings_no_zero[[
                "On-board service", "Baggage handling", "Checkin service"
            ]].mean(axis=1).fillna(3).astype(np.float32)

            data["total_service_mean"] = ratings_no_zero.mean(axis=1).fillna(3).astype(np.float32)
            data["service_rating_std"] = ratings_no_zero.std(axis=1).fillna(0).astype(np.float32)
            data["service_rating_range"] = (
                ratings_no_zero.max(axis=1) - data["min_service_rating"]
            ).fillna(0).astype(np.float32)

            # -------------------------------------------------------------
            # 5. PSYCHOMETRICS (RASCH DELIGHT & RESPONSE STYLE)
            # -------------------------------------------------------------
            rasch_score = np.zeros(len(data), dtype=np.float32)
            for item, difficulty in self.config.rasch_difficulties.items():
                rasch_score += difficulty * (data[item] >= 4).astype(np.float32)
            data["rasch_delight_score"] = rasch_score

            data["midpoint_ratio"] = (data[self.config.rating_cols] == 3).mean(axis=1).astype(np.float32)
            data["extremity_ratio"] = (
                (data[self.config.rating_cols] == 1) | (data[self.config.rating_cols] == 5)
            ).mean(axis=1).astype(np.float32)

            # -------------------------------------------------------------
            # 6. SIMPSON'S INVERSION & DEMOGRAPHIC INTERACTIONS
            # -------------------------------------------------------------
            is_business_class = (data["Class"] == "Business").astype(np.int8)
            is_business_travel = (data["Type of Travel"] == "Business travel").astype(np.int8)
            is_loyal = (data["Customer Type"] == "Loyal Customer").astype(np.int8)

            data["dist_business"] = data["Flight Distance"] * is_business_class
            data["dist_eco"] = data["Flight Distance"] * (1 - is_business_class)
            data["log_flight_distance"] = np.log1p(data["Flight Distance"]).astype(np.float32)
            data["age_x_business"] = data["Age"] * is_business_travel
            data["gate_x_business"] = data["Gate location"].astype(str) + "_" + is_business_travel.astype(str)
            data["loyal_business"] = (is_loyal & is_business_class).astype(np.int8)
            data["disloyal_economy"] = ((1 - is_loyal) & (1 - is_business_class)).astype(np.int8)

            data["premium_service_failure"] = (
                is_business_class & (
                    (data["Cleanliness"] <= 2) |
                    (data["On-board service"] <= 2) |
                    (data["Inflight entertainment"] <= 2)
                )
            ).astype(np.int8)

            # -------------------------------------------------------------
            # 7. 3-WAY MACRO-MANIFOLDS (GOLDEN SEGMENT VS DEAD ZONE)
            # -------------------------------------------------------------
            data["is_golden_segment"] = (
                is_business_class & is_business_travel & (data["Online boarding"] >= 4)
            ).astype(np.int8)

            data["is_dead_zone"] = (
                (data["Class"] == "Eco") & (data["Type of Travel"] == "Personal Travel") & (data["Online boarding"] < 4)
            ).astype(np.int8)

            data["class_x_travel_type"] = data["Class"].astype(str) + "_" + data["Type of Travel"].astype(str)

            # -------------------------------------------------------------
            # 8. DOMAIN A: HIGH-ORDER CATEGORICAL FREQUENCY FORENSICS
            # -------------------------------------------------------------
            if self.config.enable_high_order_freq and self.freq_maps:
                keys = self._get_frequency_keys(data)
                for name, col_series in keys.items():
                    freq_col = f"freq_{name}"
                    log_count_col = f"log_count_{name}"
                    data[freq_col] = col_series.map(self.freq_maps[name]).fillna(0.0).astype(np.float32)
                    data[log_count_col] = np.log1p(
                        col_series.map(self.count_maps[name]).fillna(0)
                    ).astype(np.float32)

            # -------------------------------------------------------------
            # 9. DOMAIN A: GEOMETRIC CENTROIDS & DENSITY ANOMALY SCORES
            # -------------------------------------------------------------
            if self.config.enable_density_forensics and self.scaler_mean is not None:
                core_cols = self.config.core_centroid_cols
                core_vals = data[core_cols].copy()
                core_vals["Arrival Delay in Minutes"] = data["Arrival Delay in Minutes"]
                X_norm = ((core_vals.values - self.scaler_mean) / self.scaler_std).astype(np.float32)

                if self.mbk_1 is not None and self.mbk_0 is not None:
                    # Distances to sub-cluster centroids
                    d_pos = self.mbk_1.transform(X_norm).min(axis=1)
                    d_neg = self.mbk_0.transform(X_norm).min(axis=1)

                    data["dist_to_satisfied_sub"] = d_pos.astype(np.float32)
                    data["dist_to_dissatisfied_sub"] = d_neg.astype(np.float32)
                    data["centroid_dist_ratio"] = (d_neg / (d_pos + 1e-5)).astype(np.float32)
                    data["centroid_dist_margin"] = ((d_neg - d_pos) / (d_neg + d_pos + 1e-5)).astype(np.float32)
                    data["log_dist_satisfied"] = np.log(d_pos + 1e-5).astype(np.float32)
                    data["log_dist_dissatisfied"] = np.log(d_neg + 1e-5).astype(np.float32)

                    # Global class centroid distances
                    if self.global_centroid_1 is not None and self.global_centroid_0 is not None:
                        g_pos = np.linalg.norm(X_norm - self.global_centroid_1, axis=1)
                        g_neg = np.linalg.norm(X_norm - self.global_centroid_0, axis=1)
                        data["global_dist_satisfied"] = g_pos.astype(np.float32)
                        data["global_dist_dissatisfied"] = g_neg.astype(np.float32)
                        data["global_dist_log_ratio"] = np.log((g_neg + 1e-5) / (g_pos + 1e-5)).astype(np.float32)

                if self.mbk_anomaly is not None:
                    # Anomaly density score: distance to nearest mode across full dataset
                    d_mode = self.mbk_anomaly.transform(X_norm).min(axis=1)
                    data["anomaly_density_score"] = np.log1p(d_mode).astype(np.float32)

            # -------------------------------------------------------------
            # 10. CATEGORICAL ENCODING
            # -------------------------------------------------------------
            cat_columns = [
                "Gender", "Customer Type", "Type of Travel", "Class",
                "class_x_travel_type", "gate_x_business"
            ]

            for col in cat_columns:
                le = self.label_encoders.get(col)
                if le is not None:
                    # Map known categories; unknown get -1
                    known_classes = set(le.classes_)
                    data[col] = data[col].astype(str).map(
                        lambda s: le.transform([s])[0] if s in known_classes else -1
                    ).astype(np.int16)
                else:
                    data[col] = pd.factorize(data[col])[0].astype(np.int16)

            # Drop identifier if present
            if self.config.id_col in data.columns:
                data = data.drop(columns=[self.config.id_col])

            # Drop target if present during feature matrix creation
            if self.config.target_col in data.columns:
                data = data.drop(columns=[self.config.target_col])

            # Memory optimization
            data = reduce_mem_usage(data, verbose=False)
            return data

    def fit_transform(self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Fits transductive statistics and transforms training data."""
        self.fit(train_df, test_df)
        return self.transform(train_df, is_train=True)
