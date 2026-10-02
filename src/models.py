"""
Model Wrappers: High-Performance LightGBM, CatBoost, and XGBoost with GPU/CPU Detection
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

import numpy as np


class BaseModel(ABC):
    """Abstract Model Interface."""

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val) -> None:
        pass

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        pass


class LightGBMModel(BaseModel):
    """Production LightGBM Model with Auto-Hardware Configuration."""

    def __init__(self, params: Dict[str, Any] = None, device: str = "cpu"):
        import lightgbm as lgb
        self.params = params.copy() if params else {}
        self.device = device
        self.model = None

        # Hardware optimization
        if self.device == "cuda":
            # Attempt to use GPU in LightGBM if compiled, otherwise fallback to CPU
            try:
                self.params["device"] = "gpu"
            except Exception:
                self.params["device"] = "cpu"
        else:
            self.params["device"] = "cpu"
            self.params["n_jobs"] = -1

        # Sanitize any accidental foreign hyperparameters
        for invalid_key in ["loss_function", "eval_metric", "task_type", "thread_count", "l2_leaf_reg", "iterations", "tree_method"]:
            self.params.pop(invalid_key, None)

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        import lightgbm as lgb
        trn_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=trn_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=250)
        ]

        num_boost_round = self.params.pop("n_estimators", 2500)

        try:
            self.model = lgb.train(
                self.params,
                trn_data,
                num_boost_round=num_boost_round,
                valid_sets=[trn_data, val_data],
                valid_names=["train", "valid"],
                callbacks=callbacks
            )
        except Exception as e:
            # Graceful GPU fallback if Kaggle LightGBM wheel lacks OpenCL
            if "gpu" in str(self.params.get("device", "")).lower():
                logging.warning(f"LightGBM GPU initialization failed ({e}). Falling back to multi-core CPU...")
                self.params["device"] = "cpu"
                self.params["n_jobs"] = -1
                self.model = lgb.train(
                    self.params,
                    trn_data,
                    num_boost_round=num_boost_round,
                    valid_sets=[trn_data, val_data],
                    valid_names=["train", "valid"],
                    callbacks=callbacks
                )
            else:
                raise e

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict(X, num_iteration=self.model.best_iteration)


class CatBoostModel(BaseModel):
    """Production CatBoost Model with Native GPU / Multi-threading."""

    def __init__(self, params: Dict[str, Any] = None, device: str = "cpu"):
        from catboost import CatBoostClassifier
        self.params = params.copy() if params else {}
        if device == "cuda":
            self.params["task_type"] = "GPU"
        else:
            self.params["task_type"] = "CPU"
            self.params["thread_count"] = -1

        # Prevent duplicate/conflicting parameter errors
        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", 100)
        self.verbose = self.params.pop("verbose", 250)

        # Sanitize any accidental foreign hyperparameters
        for invalid_key in ["metric", "objective", "boosting_type", "n_estimators", "num_leaves", "colsample_bytree", "subsample", "tree_method"]:
            self.params.pop(invalid_key, None)

        self.model = CatBoostClassifier(**self.params)

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=self.verbose,
            use_best_model=True
        )

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class XGBoostModel(BaseModel):
    """Production XGBoost Model with Histogram GPU / Multi-threading."""

    def __init__(self, params: Dict[str, Any] = None, device: str = "cpu"):
        from xgboost import XGBClassifier
        self.params = params.copy() if params else {}
        self.params["tree_method"] = "hist"
        if device == "cuda":
            self.params["device"] = "cuda"
        else:
            self.params["device"] = "cpu"
            self.params["n_jobs"] = -1

        # Prevent duplicate/conflicting parameter errors
        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", 100)
        self.verbose = self.params.pop("verbose", 250)

        # Sanitize any accidental foreign hyperparameters
        for invalid_key in ["metric", "loss_function", "task_type", "thread_count", "l2_leaf_reg", "num_leaves", "boosting_type"]:
            self.params.pop(invalid_key, None)

        self.model = XGBClassifier(**self.params, early_stopping_rounds=self.early_stopping_rounds)

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=self.verbose
        )

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def get_model(model_name: str, params: Dict[str, Any] = None, device: str = "cpu") -> BaseModel:
    """Factory function for model instantiation."""
    model_name_lower = model_name.lower()
    if "lgb" in model_name_lower or "lightgbm" in model_name_lower:
        return LightGBMModel(params=params, device=device)
    elif "cat" in model_name_lower or "catboost" in model_name_lower:
        return CatBoostModel(params=params, device=device)
    elif "xgb" in model_name_lower or "xgboost" in model_name_lower:
        return XGBoostModel(params=params, device=device)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from 'lightgbm', 'catboost', 'xgboost'.")
