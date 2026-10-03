"""
Configuration and Global Constants for Airline Passenger Satisfaction
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PathConfig:
    """Dynamic path resolver supporting local and Kaggle environments."""

    raw_dir: str = ""
    train_path: str = ""
    test_path: str = ""
    sample_sub_path: str = ""
    original_path: str = ""
    output_dir: str = "outputs"
    submissions_dir: str = "submissions"

    def __post_init__(self):
        if not self.raw_dir:
            import glob

            candidates = [
                "/kaggle/input/competitions/playground-series-s6e10",
                "/kaggle/input/playground-series-s6e10",
                "data/raw",
                "../data/raw",
                "../../data/raw",
                ".",
            ]

            # Dynamic auto-discovery on Kaggle filesystem
            if os.path.exists("/kaggle/input"):
                kaggle_matches = glob.glob("/kaggle/input/**/train.csv", recursive=True)
                if kaggle_matches:
                    for km in kaggle_matches:
                        if "playground-series-s6e10" in km or "competitions" in km:
                            candidates.insert(0, os.path.dirname(km))
                            break

            for cand in candidates:
                if os.path.exists(os.path.join(cand, "train.csv")):
                    self.raw_dir = cand
                    break

        if not self.raw_dir:
            self.raw_dir = "."

        self.train_path = os.path.join(self.raw_dir, "train.csv")
        self.test_path = os.path.join(self.raw_dir, "test.csv")
        self.sample_sub_path = os.path.join(self.raw_dir, "sample_submission.csv")

        # Discover original dataset if available
        if not self.original_path:
            orig_candidates = [
                "/kaggle/input/airline-passenger-satisfaction",
                "/kaggle/input/airline-passenger-satisfaction-dataset",
                "data/original",
                "data/raw/original",
                "data/external",
            ]
            import glob
            if os.path.exists("/kaggle/input"):
                orig_matches = glob.glob("/kaggle/input/**/airline-passenger-satisfaction*/**", recursive=True)
                for om in orig_matches:
                    if os.path.isdir(om):
                        orig_candidates.insert(0, om)
                    elif om.endswith(".csv"):
                        orig_candidates.insert(0, os.path.dirname(om))

            for oc in orig_candidates:
                if os.path.exists(oc):
                    self.original_path = oc
                    break

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.submissions_dir, exist_ok=True)


@dataclass
class FeatureConfig:
    """Feature column definitions and psychometric weights."""

    target_col: str = "satisfaction"
    id_col: str = "id"

    categorical_cols: list[str] = field(
        default_factory=lambda: ["Gender", "Customer Type", "Type of Travel", "Class"]
    )

    numerical_cols: list[str] = field(
        default_factory=lambda: [
            "Age",
            "Flight Distance",
            "Departure Delay in Minutes",
            "Arrival Delay in Minutes",
        ]
    )

    rating_cols: list[str] = field(
        default_factory=lambda: [
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
            "Cleanliness",
        ]
    )

    # Calibrated Rasch Difficulty Parameters (b_i) discovered empirically
    rasch_difficulties: dict[str, float] = field(
        default_factory=lambda: {
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
            "Baggage handling": -0.784,
        }
    )

    # Domain A & C: Forensic density and transductive parameters
    enable_density_forensics: bool = True
    enable_high_order_freq: bool = True
    enable_svd_manifolds: bool = True
    n_svd_components: int = 6
    enable_multi_crosses: bool = True
    core_centroid_cols: list[str] = field(
        default_factory=lambda: [
            "Online boarding",
            "Inflight wifi service",
            "Checkin service",
            "On-board service",
            "Cleanliness",
            "Leg room service",
            "Seat comfort",
            "Arrival Delay in Minutes",
            "Flight Distance",
            "Age",
        ]
    )
    kmeans_clusters_per_class: int = 8
    kmeans_anomaly_clusters: int = 16


@dataclass
class TrainConfig:
    """Training, cross-validation, and optimization settings."""

    n_splits: int = 5
    random_state: int = 42
    shuffle: bool = True
    early_stopping_rounds: int = 100
    verbose_eval: int = 200

    # Domain 1: Original Host Dataset Ingestion & Weight Attenuation
    use_original_data: bool = True
    original_sample_weight: float = 0.65
    use_density_ratio_weighting: bool = False
    density_ratio_clip_min: float = 0.05
    density_ratio_clip_max: float = 3.0

    # Default LightGBM Hyperparameters
    lgb_params: dict[str, Any] = field(
        default_factory=lambda: {
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
            "verbose": -1,
        }
    )

    # Default CatBoost Hyperparameters
    cb_params: dict[str, Any] = field(
        default_factory=lambda: {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "iterations": 2500,
            "learning_rate": 0.05,
            "depth": 7,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "early_stopping_rounds": 100,
            "verbose": 250,
        }
    )

    # Default XGBoost Hyperparameters
    xgb_params: dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": 0.04,
            "max_depth": 7,
            "colsample_bytree": 0.80,
            "subsample": 0.80,
            "n_estimators": 2500,
            "random_state": 42,
            "tree_method": "hist",
        }
    )

    # Domain 3: Un-distilled Tabular ResNet / RealMLP with Periodic Embeddings (PLR)
    resnet_params: dict[str, Any] = field(
        default_factory=lambda: {
            "n_blocks": 3,
            "d_block": 256,
            "d_embedding": 32,
            "n_frequencies": 16,
            "dropout": 0.15,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 4096,
            "epochs": 16,
            "loss_type": "bce",
            "gamma": 15.0,
            "random_state": 42,
        }
    )

    # Domain D: Compact FT-Transformer Architecture & Training Parameters
    ft_params: dict[str, Any] = field(
        default_factory=lambda: {
            "embed_dim": 32,
            "num_layers": 3,
            "num_heads": 4,
            "ffn_ratio": 2,
            "dropout": 0.1,
            "lr": 1.5e-3,
            "weight_decay": 1e-4,
            "batch_size": 2048,
            "epochs": 7,
            "distillation_alpha": 0.5,
            "consistency_lambda": 0.15,
            "random_state": 42,
        }
    )

    @property
    def cat_params(self) -> dict[str, Any]:
        return self.cb_params

    @property
    def lightgbm_params(self) -> dict[str, Any]:
        return self.lgb_params

    @property
    def xgboost_params(self) -> dict[str, Any]:
        return self.xgb_params

    @property
    def transformer_params(self) -> dict[str, Any]:
        return self.ft_params

    @property
    def nn_params(self) -> dict[str, Any]:
        return self.resnet_params

    @property
    def realmlp_params(self) -> dict[str, Any]:
        return self.resnet_params
