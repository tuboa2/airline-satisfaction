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

    def fit(self, X_train, y_train, X_val, y_val, **kwargs) -> None:
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=self.verbose
        )

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class FTTransformerModel(BaseModel):
    """
    Domain D: Compact Feature Tokenizer Transformer (FT-Transformer)
    for Tabular Relational Learning & Ensemble Orthogonalization.
    Includes Domain C: Teacher-Student Soft Distillation & Test Consistency Regularization.
    """

    def __init__(self, params: Dict[str, Any] = None, device: str = "cpu"):
        self.params = params.copy() if params else {}
        self.device_str = device
        self.model = None
        self.cat_cols = []
        self.num_cols = []
        self.num_mean = None
        self.num_std = None

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        teacher_train=None,
        X_test=None,
        teacher_test=None,
        **kwargs
    ) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.metrics import roc_auc_score

        # Hardware setup
        if self.device_str == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            use_amp = True
        else:
            device = torch.device("cpu")
            use_amp = False

        logging.info(f"FT-Transformer initializing on device: {device} (AMP: {use_amp})")

        # 1. Column Segregation: Categorical vs Numerical
        cat_candidates = [
            "Gender", "Customer Type", "Type of Travel", "Class",
            "class_x_travel_type", "gate_x_business", "delay_tier"
        ]
        self.cat_cols = [c for c in cat_candidates if c in X_train.columns]
        self.num_cols = [c for c in X_train.columns if c not in self.cat_cols]

        # 2. Numerical Standardization
        X_tr_num = X_train[self.num_cols].values.astype(np.float32)
        X_va_num = X_val[self.num_cols].values.astype(np.float32)

        self.num_mean = np.nanmean(X_tr_num, axis=0)
        self.num_std = np.nanstd(X_tr_num, axis=0) + 1e-6
        X_tr_num = np.nan_to_num((X_tr_num - self.num_mean) / self.num_std)
        X_va_num = np.nan_to_num((X_va_num - self.num_mean) / self.num_std)

        # 3. Categorical Index Alignment
        cat_cardinalities = []
        if self.cat_cols:
            X_tr_cat = np.clip(X_train[self.cat_cols].values.astype(np.int64), 0, None)
            X_va_cat = np.clip(X_val[self.cat_cols].values.astype(np.int64), 0, None)
            for j in range(len(self.cat_cols)):
                max_c = max(int(X_tr_cat[:, j].max()), int(X_va_cat[:, j].max())) + 1
                cat_cardinalities.append(max_c)
        else:
            X_tr_cat = np.zeros((len(X_train), 0), dtype=np.int64)
            X_va_cat = np.zeros((len(X_val), 0), dtype=np.int64)

        # 4. Hyperparameters
        embed_dim = self.params.get("embed_dim", 32)
        num_layers = self.params.get("num_layers", 3)
        num_heads = self.params.get("num_heads", 4)
        ffn_ratio = self.params.get("ffn_ratio", 2)
        dropout = self.params.get("dropout", 0.1)
        lr = self.params.get("lr", 1.5e-3)
        weight_decay = self.params.get("weight_decay", 1e-4)
        batch_size = self.params.get("batch_size", 2048)
        epochs = self.params.get("epochs", 7)
        distill_alpha = self.params.get("distillation_alpha", 0.5)
        consist_lambda = self.params.get("consistency_lambda", 0.15)

        # 5. Define Neural Network Module
        class _FTTransformer(nn.Module):
            def __init__(self, n_num, cat_cards, d, n_layers, n_heads, ffn_r, drop):
                super().__init__()
                self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
                self.num_w = nn.Parameter(torch.randn(n_num, d) * 0.02) if n_num > 0 else None
                self.num_b = nn.Parameter(torch.zeros(n_num, d)) if n_num > 0 else None
                self.cat_embs = nn.ModuleList([
                    nn.Embedding(card, d) for card in cat_cards
                ])

                self.blocks = nn.ModuleList()
                for _ in range(n_layers):
                    self.blocks.append(nn.ModuleDict({
                        "norm1": nn.LayerNorm(d),
                        "mha": nn.MultiheadAttention(d, n_heads, dropout=drop, batch_first=True),
                        "drop1": nn.Dropout(drop),
                        "norm2": nn.LayerNorm(d),
                        "ffn": nn.Sequential(
                            nn.Linear(d, d * ffn_r),
                            nn.GELU(),
                            nn.Dropout(drop),
                            nn.Linear(d * ffn_r, d)
                        ),
                        "drop2": nn.Dropout(drop)
                    }))

                self.head_norm = nn.LayerNorm(d)
                self.head = nn.Sequential(
                    nn.Linear(d, d // 2),
                    nn.GELU(),
                    nn.Dropout(drop),
                    nn.Linear(d // 2, 1)
                )

            def forward(self, x_num, x_cat):
                B = x_num.size(0)
                tokens = [self.cls_token.expand(B, -1, -1)]

                if self.num_w is not None and x_num.size(1) > 0:
                    num_toks = x_num.unsqueeze(-1) * self.num_w + self.num_b
                    tokens.append(num_toks)

                if len(self.cat_embs) > 0 and x_cat.size(1) > 0:
                    cat_toks = torch.stack([
                        emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embs)
                    ], dim=1)
                    tokens.append(cat_toks)

                x = torch.cat(tokens, dim=1)
                for blk in self.blocks:
                    # Pre-LN Self-Attention
                    nx = blk["norm1"](x)
                    attn_out, _ = blk["mha"](nx, nx, nx)
                    x = x + blk["drop1"](attn_out)
                    # Pre-LN FeedForward
                    nx = blk["norm2"](x)
                    ffn_out = blk["ffn"](nx)
                    x = x + blk["drop2"](ffn_out)

                cls_feat = self.head_norm(x[:, 0])
                return self.head(cls_feat).squeeze(-1)

        self.model = _FTTransformer(
            n_num=len(self.num_cols),
            cat_cards=cat_cardinalities,
            d=embed_dim,
            n_layers=num_layers,
            n_heads=num_heads,
            ffn_r=ffn_ratio,
            drop=dropout
        ).to(device)

        # 6. Tensor Datasets and Loaders
        t_X_num = torch.tensor(X_tr_num, dtype=torch.float32)
        t_X_cat = torch.tensor(X_tr_cat, dtype=torch.long)
        t_y = torch.tensor(y_train, dtype=torch.float32)

        has_distill = (teacher_train is not None)
        if has_distill:
            t_teach = torch.tensor(teacher_train, dtype=torch.float32)
            train_dataset = TensorDataset(t_X_num, t_X_cat, t_y, t_teach)
        else:
            train_dataset = TensorDataset(t_X_num, t_X_cat, t_y)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        v_X_num = torch.tensor(X_va_num, dtype=torch.float32).to(device)
        v_X_cat = torch.tensor(X_va_cat, dtype=torch.long).to(device)

        # 7. Optimizer, Scaler & Scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        bce_loss_fn = nn.BCEWithLogitsLoss()

        best_auc = 0.0
        best_state = None

        # 8. Training Loop
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()

                if has_distill:
                    b_num, b_cat, b_y, b_teach = [item.to(device) for item in batch]
                else:
                    b_num, b_cat, b_y = [item.to(device) for item in batch]
                    b_teach = None

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = self.model(b_num, b_cat)
                    loss = bce_loss_fn(logits, b_y)

                    # Domain C: Soft Distillation
                    if b_teach is not None:
                        distill_loss = bce_loss_fn(logits, b_teach)
                        loss = (1.0 - distill_alpha) * loss + distill_alpha * distill_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

            scheduler.step()

            # Validation Evaluation
            self.model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    val_logits = self.model(v_X_num, v_X_cat)
                    val_probs = torch.sigmoid(val_logits).cpu().numpy()

            val_auc = roc_auc_score(y_val, val_probs)
            logging.info(f"Epoch {epoch}/{epochs} | Loss: {running_loss / len(train_loader):.4f} | Val ROC-AUC: {val_auc:.5f}")

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(device)
            logging.info(f"Loaded Best FT-Transformer State (Validation ROC-AUC: {best_auc:.5f})")

    def predict_proba(self, X) -> np.ndarray:
        import torch
        device = next(self.model.parameters()).device
        self.model.eval()

        X_num = X[self.num_cols].values.astype(np.float32)
        X_num = np.nan_to_num((X_num - self.num_mean) / self.num_std)

        if self.cat_cols:
            X_cat = np.clip(X[self.cat_cols].values.astype(np.int64), 0, None)
        else:
            X_cat = np.zeros((len(X), 0), dtype=np.int64)

        batch_size = 4096
        probs = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                b_num = torch.tensor(X_num[i:i + batch_size], dtype=torch.float32).to(device)
                b_cat = torch.tensor(X_cat[i:i + batch_size], dtype=torch.long).to(device)
                logits = self.model(b_num, b_cat)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)

        return np.concatenate(probs, axis=0)


def get_model(model_name: str, params: Dict[str, Any] = None, device: str = "cpu") -> BaseModel:
    """Factory function for model instantiation."""
    model_name_lower = model_name.lower()
    if "lgb" in model_name_lower or "lightgbm" in model_name_lower:
        return LightGBMModel(params=params, device=device)
    elif "cat" in model_name_lower or "catboost" in model_name_lower:
        return CatBoostModel(params=params, device=device)
    elif "xgb" in model_name_lower or "xgboost" in model_name_lower:
        return XGBoostModel(params=params, device=device)
    elif "transformer" in model_name_lower or "ft" in model_name_lower or "nn" in model_name_lower:
        return FTTransformerModel(params=params, device=device)
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. Choose from 'lightgbm', 'catboost', 'xgboost', 'transformer'."
        )

