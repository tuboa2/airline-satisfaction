"""
Configuration and Global Constants for Airline Passenger Satisfaction
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class PathConfig:
    """Dynamic path resolver supporting local and Kaggle environments."""
    raw_dir: str = ""
    train_path: str = ""
    test_path: str = ""
    sample_sub_path: str = ""
    output_dir: str = "outputs"
    submissions_dir: str = "submissions"

    def __post_init__(self):
        # Kaggle input paths
        kaggle_raw = "/kaggle/input/playground-series-s6e10"
        local_raw = "data/raw"
        alt_local_raw = "../data/raw"

        if os.path.exists(kaggle_raw):
            self.raw_dir = kaggle_raw
        elif os.path.exists(local_raw):
            self.raw_dir = local_raw
        elif os.path.exists(alt_local_raw):
            self.raw_dir = alt_local_raw
        else:
            self.raw_dir = "."

        self.train_path = os.path.join(self.raw_dir, "train.csv")
        self.test_path = os.path.join(self.raw_dir, "test.csv")
        self.sample_sub_path = os.path.join(self.raw_dir, "sample_submission.csv")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.submissions_dir, exist_ok=True)


@dataclass
class FeatureConfig:
    """Feature column definitions and psychometric weights."""
    target_col: str = "satisfaction"
    id_col: str = "id"

    categorical_cols: List[str] = field(default_factory=lambda: [
        "Gender",
        "Customer Type",
        "Type of Travel",
        "Class"
    ])

    numerical_cols: List[str] = field(default_factory=lambda: [
        "Age",
        "Flight Distance",
        "Departure Delay in Minutes",
        "Arrival Delay in Minutes"
    ])

    rating_cols: List[str] = field(default_factory=lambda: [
        "Inflight wifi service",
        "Departure/Arrival time convenient",
        "Ease of Online booking",
        "Gate location",
        "Food and drink",
        "Online boarding",
        "Seat comfort",
        "Inflight entertainment",
        "On-board service",
        "Leg room service",
        "Baggage handling",
        "Checkin service",
        "Cleanliness"
    ])

    # Calibrated Rasch Difficulty Parameters (b_i) discovered empirically
    rasch_difficulties: Dict[str, float] = field(default_factory=lambda: {
        "Inflight wifi service": 0.864,
        "Ease of Online booking": 0.763,
        "Gate location": 0.519,
        "Food and drink": 0.146,
        "Departure/Arrival time convenient": 0.065,
        "Cleanliness": -0.022,
        "Checkin service": -0.042,
        "Online boarding": -0.080,
        "Leg room service": -0.210,
        "Inflight entertainment": -0.243,
        "On-board service": -0.288,
        "Seat comfort": -0.425,
        "Baggage handling": -0.784
    })


@dataclass
class TrainConfig:
    """Training, cross-validation, and optimization settings."""
    n_splits: int = 5
    random_state: int = 42
    shuffle: bool = True
    early_stopping_rounds: int = 100
    verbose_eval: int = 200

    # Default LightGBM Hyperparameters
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.04,
        "num_leaves": 63,
        "max_depth": -1,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "min_child_samples": 30,
        "n_estimators": 2500,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    })

    # Default CatBoost Hyperparameters
    cb_params: Dict[str, Any] = field(default_factory=lambda: {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 2500,
        "learning_rate": 0.05,
        "depth": 7,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "verbose": 250
    })

    # Default XGBoost Hyperparameters
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.04,
        "max_depth": 7,
        "colsample_bytree": 0.80,
        "subsample": 0.80,
        "n_estimators": 2500,
        "random_state": 42,
        "tree_method": "hist"
    })
