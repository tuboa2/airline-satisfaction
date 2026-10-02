# ✈️ Predicting Airline Satisfaction (Kaggle Playground Series S6E10)

Production-grade Machine Learning repository implementing all 10 empirically verified analytical paradigms discovered across exhaustive exploratory data analysis.

---

## 📁 Repository Architecture

```
airline-satisfaction/
├── .gitignore               # Strict exclusion of large raw CSVs, caches, and models
├── README.md                # Documentation and Kaggle usage guide
├── requirements.txt         # Pinned production dependencies
├── run_training.py          # Unified CLI entrypoint for training and inference
├── notebooks/
│   └── kaggle_s6e10_univariate_eda.ipynb # Complete EDA notebook
└── src/
    ├── __init__.py          # Package initialization
    ├── config.py            # Dynamic paths (Kaggle vs. Local), hyperparameters & schemas
    ├── utils.py             # Memory reduction, GPU/CPU detection, and timers
    ├── features.py          # 10-paradigm production feature engineering pipeline
    ├── models.py            # LightGBM, CatBoost, and XGBoost wrappers (GPU & CPU)
    └── trainer.py           # Stratified 5-Fold Cross Validation & Submission Engine
```

---

## 🚀 Running on Kaggle Notebook (1-Line Execution)

Inside a new or existing Kaggle notebook (with GPU enabled or standard CPU):

```python
# 1. Clone or Pull repository
!git clone https://github.com/tuboa2/airline-satisfaction.git
%cd airline-satisfaction

# 2. Run 5-Fold Stratified Training (Auto-detects GPU and CPU cores)
!python run_training.py --model lightgbm

# Alternatively, train with CatBoost or XGBoost on GPU:
# !python run_training.py --model catboost --device cuda
# !python run_training.py --model xgboost --device cuda
```

The script will automatically:
1. Detect whether it is running on Kaggle (`/kaggle/input/playground-series-s6e10/`).
2. Downcast data types to prevent RAM crashes.
3. Transform features using the 10 verified paradigms.
4. Execute Stratified 5-Fold Cross-Validation.
5. Generate `submission.csv` ready to submit directly on Kaggle.

---

## 🔬 Engineered Feature Paradigms Included

1. **Physical Delay Dynamics:** Missing `Arrival Delay` imputed from `Departure Delay` ($r=0.963$), `delay_recovery_delta` (airborne time made up, $+11\%$ satisfaction lift), `delay_tier` (15-Minute Flatline Step-Function).
2. **Zero-Inflation Indicators:** Dedicated "Not Applicable" flags (`wifi_is_0`, `booking_is_0`, `boarding_is_0`) resolving the 88.7% satisfaction paradox.
3. **Thresholds & Trump Cards:** `wifi_is_5` (The 95% satisfaction Trump Card), `high_online_boarding` ($\ge 4$), `checkin_is_acceptable` ($\ge 3$), `digital_failure`.
4. **Domain Pillar Aggregations:** `Digital_Score`, `Cabin_Score`, `Staff_Score`, and `min_service_rating` (The Bottleneck Law).
5. **Simpson's Distance Inversion:** Separation of `dist_business` (positive slope) vs. `dist_eco` (negative slope).
6. **Psychometrics:** `rasch_delight_score` (difficulty-weighted ratings), `midpoint_ratio` (fraction of 3s, inverse satisfaction signal).
7. **Macro-Manifolds:** `is_golden_segment` (88.2% sat) vs. `is_dead_zone` (5.2% sat), and `premium_service_failure` (disgust & neglect vector).
