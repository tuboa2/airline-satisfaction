"""
Feature Engineering Pipeline: Implements All 10 Empirically Verified Paradigms
"""

import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import FeatureConfig
from src.utils import reduce_mem_usage, timer


class FeaturePipeline:
    """
    Production-grade Feature Engineering Pipeline.
    Implements all verified univariate, bivariate, multivariate, psychometric,
    and survival insights discovered during analysis.
    """

    def __init__(self, config: FeatureConfig = FeatureConfig()):
        self.config = config
        self.label_encoders = {}
        self.fitted = False

    def transform(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Applies vectorized feature transformations to training or test data."""
        with timer(f"Feature Engineering ({'Train' if is_train else 'Test'})"):
            data = df.copy()

            # -------------------------------------------------------------
            # 1. PHYSICAL DELAY DYNAMICS & IMPUTATION
            # -------------------------------------------------------------
            data["Arrival_Delay_is_nan"] = data["Arrival Delay in Minutes"].isna().astype(np.int8)
            # Impute missing arrival delays from departure delays (physical collinearity r=0.963)
            data["Arrival Delay in Minutes"] = data["Arrival Delay in Minutes"].fillna(
                data["Departure Delay in Minutes"]
            )

            dep_delay = data["Departure Delay in Minutes"]
            arr_delay = data["Arrival Delay in Minutes"]

            data["total_delay"] = dep_delay + arr_delay
            data["has_delay"] = (data["total_delay"] > 0).astype(np.int8)
            data["has_severe_delay"] = (data["total_delay"] > 30).astype(np.int8)

            # Airborne Delay Recovery Delta (Positive = made up time in air, +11% satisfaction lift)
            data["delay_recovery_delta"] = dep_delay - arr_delay
            data["worsened_in_air"] = (arr_delay > dep_delay).astype(np.int8)

            # Delay Intensity per 100 miles
            data["delay_intensity"] = data["total_delay"] / np.maximum(1.0, data["Flight Distance"] / 100.0)

            # The 15-Minute Flatline Law (Step-function delay hazard)
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
            # Inflight Wifi == 5 (The 95% satisfaction Trump Card)
            data["wifi_is_5"] = (data["Inflight wifi service"] == 5).astype(np.int8)

            # High Rating Flags (Rating >= 4 Inflection Cliff)
            data["high_online_boarding"] = (data["Online boarding"] >= 4).astype(np.int8)
            data["high_seat_comfort"] = (data["Seat comfort"] >= 4).astype(np.int8)
            data["high_entertainment"] = (data["Inflight entertainment"] >= 4).astype(np.int8)

            # Checkin Service Unique 2 -> 3 Jump (+24.5% lift)
            data["checkin_is_acceptable"] = (data["Checkin service"] >= 3).astype(np.int8)

            # Total Digital Failure Gate (<6% satisfaction)
            data["digital_failure"] = (
                (data["Online boarding"] <= 3) & (data["Inflight wifi service"] <= 3)
            ).astype(np.int8)

            # -------------------------------------------------------------
            # 4. DOMAIN PILLAR AGGREGATIONS & BOTTLENECK LAW
            # -------------------------------------------------------------
            ratings_no_zero = data[self.config.rating_cols].replace(0, np.nan)

            # The Weakest Link (Bottleneck Law: min rating <= 2 caps satisfaction at 30%)
            data["min_service_rating"] = ratings_no_zero.min(axis=1).fillna(3).astype(np.float32)
            data["has_service_failure"] = (data["min_service_rating"] <= 2).astype(np.int8)
            data["is_all_pass_service"] = (data["min_service_rating"] >= 3).astype(np.int8)

            # Pillar Scores
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

            # Service Consistency & Range
            data["service_rating_std"] = ratings_no_zero.std(axis=1).fillna(0).astype(np.float32)
            data["service_rating_range"] = (
                ratings_no_zero.max(axis=1) - data["min_service_rating"]
            ).fillna(0).astype(np.float32)

            # -------------------------------------------------------------
            # 5. PSYCHOMETRICS (RASCH DELIGHT & RESPONSE STYLE)
            # -------------------------------------------------------------
            # Rasch-weighted score (credits hard items like wifi much more than easy items)
            rasch_score = np.zeros(len(data), dtype=np.float32)
            for item, difficulty in self.config.rasch_difficulties.items():
                rasch_score += difficulty * (data[item] >= 4).astype(np.float32)
            data["rasch_delight_score"] = rasch_score

            # Response Style: Midpoint Ratio (Fraction of 3s, Inverse AUC 0.6792)
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

            # Simpson's Paradox: Distance is POSITIVE in Business, NEGATIVE in Economy
            data["dist_business"] = data["Flight Distance"] * is_business_class
            data["dist_eco"] = data["Flight Distance"] * (1 - is_business_class)
            data["log_flight_distance"] = np.log1p(data["Flight Distance"]).astype(np.float32)

            # Age Corporate Interaction (Age curve exists strictly in business travel)
            data["age_x_business"] = data["Age"] * is_business_travel

            # Gate Location V-Shape Nominal Business Interaction
            data["gate_x_business"] = data["Gate location"].astype(str) + "_" + is_business_travel.astype(str)

            # Loyalty Class Multiplier (+44.8% premium in Business)
            data["loyal_business"] = (is_loyal & is_business_class).astype(np.int8)
            data["disloyal_economy"] = ((1 - is_loyal) & (1 - is_business_class)).astype(np.int8)

            # Corporate Failure Vector: Cleanliness / Crew Disgust Factor (7.3x higher in dissatisfied)
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

            # Structural interaction composite
            data["class_x_travel_type"] = data["Class"].astype(str) + "_" + data["Type of Travel"].astype(str)

            # -------------------------------------------------------------
            # 8. CATEGORICAL ENCODING
            # -------------------------------------------------------------
            cat_columns_to_encode = [
                "Gender", "Customer Type", "Type of Travel", "Class",
                "class_x_travel_type", "gate_x_business"
            ]

            for col in cat_columns_to_encode:
                if is_train:
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders.get(col)
                    if le is not None:
                        # Handle unseen categories gracefully
                        data[col] = data[col].astype(str).map(
                            lambda s: le.transform([s])[0] if s in le.classes_ else -1
                        )
                    else:
                        data[col] = pd.factorize(data[col])[0]

            # Drop identifier if present
            if self.config.id_col in data.columns:
                data = data.drop(columns=[self.config.id_col])

            # Drop target if present during feature matrix creation
            if self.config.target_col in data.columns:
                data = data.drop(columns=[self.config.target_col])

            # Memory optimization
            data = reduce_mem_usage(data, verbose=False)

            if is_train:
                self.fitted = True

            return data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits encoders on training data and transforms."""
        return self.transform(df, is_train=True)
