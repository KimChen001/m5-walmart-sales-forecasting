# M5 Walmart Sales Forecasting

15.071 Project 1 · Predictive AI · November 20, 2026

## Prediction task

Forecast daily units sold for each product in each store to support replenishment and sales planning. The main evaluation uses a 28-day forecast horizon. A separate next-day experiment measures performance when the latest sales are available every day.

The M5 Walmart data contains 30,490 item–store series across 10 stores and 1,941 days. The model uses item and store IDs, calendar features, weekly prices, events, state SNAP indicators, and sales history available at forecast time. The target is unit sales, a nonnegative count.

## Methods and validation

The implementation trains XGBoost regressors with a Tweedie objective. M1 uses calendar features and IDs, M2 adds price, SNAP, and events, and M3 adds lagged sales and rolling statistics. Seasonal naive and moving average forecasts provide benchmarks. The chronological split uses days 1–1,553 for training and days 1,554–1,941 for testing. Early stopping uses the final 90 training days, followed by refitting on the training period. The 28-day test uses 14 forecast windows, with 24 days scored in the last window. The batch job scores both the full panel and a 6,638-series subset defined by sales frequency in training.

| Full panel, 28-day forecasts | Item–store–day R² ↑ | MAE (units) ↓ | WAPE ↓ | M5 WRMSSE (12 levels) ↓ |
|---|---:|---:|---:|---:|
| Seasonal naive, lag 28 | 0.3057 | 1.2308 | 93.98% | 0.9378 |
| 28-day moving average | 0.5782 | 1.0204 | 77.91% | 1.0655 |
| XGBoost M1 | 0.5257 | 1.1119 | 84.89% | 0.9384 |
| XGBoost M2 | 0.5332 | 1.0968 | 83.75% | 0.9938 |
| XGBoost M3 | 0.6289 | 0.9790 | 74.75% | 0.6812 |

Source: MIT ORCD job `29323394`. WRMSSE uses 12 hierarchy levels and trailing 28-day dollar-sales weights. M3 reduces absolute forecast error by about **0.252 units per scored item–store–day** relative to seasonal naive and reduces WRMSSE by **27.4%**. Translating these gains into cost or service improvements requires a replenishment simulation with lead times, holding costs and stockout costs. This run measures forecasting accuracy.

## Reproduce on MIT ORCD

Place `calendar.csv`, `sales_train_evaluation.csv` and `sell_prices.csv` from the [M5 Forecasting Accuracy dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) in the repository root. If you already have the ZIP, extract those three files into the root. On Windows, run `Expand-Archive -LiteralPath <path-to-zip> -DestinationPath .` from the repository root. Kaggle requires sign-in and acceptance of the competition rules to obtain the data.

```bash
bash cluster/setup_env.sh  # first run only
mkdir -p logs
sbatch cluster/run_xgb_both.sbatch
```

The job trains both horizons on CPU and saves results in `outputs/xgb_ALL_ALL/` and `outputs/xgb_ALL_ALL_h1/`. Read `metrics_overall.csv` for all four full-panel metrics and `metrics_final_regular50.csv` for the regular-seller subset. In the CSV, `WRMSSE` covers all levels and `WRMSSE_L12` is the item–store level alone.

For a local smoke test, install the dependencies with `python -m pip install pandas numpy xgboost scikit-learn` and run `python xgb_forecast.py --state CA --cat FOODS --device cpu --quick`. The subset and reduced tree budget are for verification.

## Team contributions

XGBoost implementation — Zhixin. Planned additions are OLS (Kayla), the neural network (Marco), and TimesFM 3.0 (Kayla, Zhixin and Marco).
