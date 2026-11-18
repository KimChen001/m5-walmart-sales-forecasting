# Code and notebooks

[← Back to the overview](../README.md) · [Full results](results.md) · [Run and reproduce](reproduce.md) · [Archived ORCD outputs](../results/README.md)

Every model in the presentation has its code here. The XGBoost script is at the repository root, the TimesFM 3.0 scripts are covered in [Run and reproduce](reproduce.md#reproduce-timesfm-30), and all notebooks, including the XGBoost results notebook, are in [`notebooks/`](../notebooks). Open each notebook from that folder: it reads its inputs from the repository root.

| File | Model | Forecast task | Run notes |
|---|---|---|---|
| [`exploratory_analysis.ipynb`](../notebooks/exploratory_analysis.ipynb) | Data exploration behind the data slide | — | No model training |
| [`regression_ols_ridge_lasso.ipynb`](../notebooks/regression_ols_ridge_lasso.ipynb) | OLS, Ridge and LASSO, λ by 10-fold cross-validation | Next day, on 80,000 training and 20,000 test item-store-days sampled at random | 35 min on 32 ORCD cores |
| [`xgb_forecast.py`](../xgb_forecast.py) | XGBoost M1–M3 and baselines on all 30,490 series; the source of the XGBoost results | 28 days from each of 14 origins, and next day, training to `d_1553` | 3 h 15 min for both schedules on 32 ORCD cores (job `29323394`); run with [`cluster/run_xgb_both.sbatch`](../cluster/run_xgb_both.sbatch) |
| [`xgboost_full_panel.ipynb`](../notebooks/xgboost_full_panel.ipynb) | XGBoost results on the full panel: the four metrics for both schedules, WRMSSE at each of the 12 levels and by test window, daily totals, feature importance, and error by sales volume, department and store | Both schedules, from job `29323394` | Reads the archived results in [`results/xgb-29323394/`](../results/xgb-29323394), so it runs in seconds without a cluster |
| [`lstm_d1553.ipynb`](../notebooks/lstm_d1553.ipynb) | LSTM encoder–decoder, one network for all 30,490 series | 28 days from each of 14 origins, training to `d_1553` | About 5 min of training on one GPU; writes its test forecasts to `outputs/` |
| [`timesfm3_v3_v6.ipynb`](../notebooks/timesfm3_v3_v6.ipynb) | TimesFM 3.0 V3, then V4 (with LASSO), V5 (fine-tuned output head) and V6 (ensemble with recursive boosted trees) | 28 days from each of 14 origins, training to `d_1552` | Imports `timesfm3`, installed as in [Run and reproduce](reproduce.md#reproduce-timesfm-30). Reads cached forecasts from `timesfm3_v3_forecasts_all/`, `timesfm3_v5_features_all/` and `timesfm3_v6_gbm_all/`; without them the TimesFM runs take several hours on a GPU |
| [`timesfm3_improvements.py`](../timesfm3_improvements.py) | TimesFM 3.0 with price, event and zero-run features and fitted decile weights | 28 days from each of 14 origins, training to `d_1553` | 4 h 15 min on one GPU (job `28701113`) |
| [`timesfm_forecast.py`](../timesfm_forecast.py) | TimesFM 3.0 sales-only baseline | 28 days from each of 14 origins, training to `d_1553` | ORCD job `27920694` |

**Team** · Regression notebook by Kayla. XGBoost by Zhixin. LSTM notebook by Marco. TimesFM 3.0 by Kayla, Zhixin and Marco.
