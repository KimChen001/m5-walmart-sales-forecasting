# M5 Walmart Sales Forecasting

**15.071 Project 1 · Predictive AI · November 20, 2026**

<p align="center">
  <a href="docs/overview.md"><img src="docs/project-overview.png" width="49%" alt="Slide 1: the replenishment problem, the M5 data, the models and the baselines"></a>
  <a href="docs/overview.md#recommendation"><img src="docs/recommendation.png" width="49%" alt="Slide 10: why the V6 ensemble, inventory freed against today's rule, and the roll-out plan"></a>
</p>

We forecast daily unit sales for 30,490 item–store series in 10 Walmart stores to set safety stock. The best model, V6, has 8% less error than the stores' 28-day moving-average rule, which frees about **$148k** of inventory; re-forecasting XGBoost daily instead of every 4 weeks cuts its WRMSSE by **23%**.

**Contents** · [Task and data](#task-and-data) · [Models and results](#models-and-results) · [Improvement: re-forecast daily](#improvement-re-forecast-daily) · [Run the code](#run-the-code) · [Repository map](#repository-map) · [Limitations](#limitations)

**More** · [Business overview](docs/overview.md) · [Full results](docs/results.md) · [Code by model](docs/code.md) · [Cluster runs](docs/reproduce.md)

## Task and data

| | |
|---|---|
| **Data** | Kaggle M5: daily unit sales of 3,049 items in 10 stores (CA, TX, WI), Jan 2011 – May 2016, with weekly prices, events and SNAP days |
| **Target** | Units sold per item, store and day |
| **Split** | Train d_1–d_1553 (to Apr 2015) · test d_1554–d_1941 (388 days) · last 90 training days for early stopping |
| **Protocol** | Forecast 28 days ahead from 14 rolling origins, using only data known at each origin |
| **Cleaning** | Weekly prices mapped to days; days before an item's first price are dropped, since they are not real zero demand |
| **Metrics** | R², MAE and WAPE per item–store–day; M5 WRMSSE over 12 aggregation levels |

## Models and results

28-day forecasts on all 30,490 series, test year May 2015 – May 2016. Bold marks the best value in each column.

| Model | R² ↑ | MAE ↓ | WAPE ↓ | WRMSSE ↓ |
|---|---:|---:|---:|---:|
| Seasonal naive (same weekday 4 weeks earlier) | 0.306 | 1.231 | 94.0% | 0.938 |
| 28-day moving average (the stores' current rule) | 0.578 | 1.020 | 77.9% | 1.066 |
| XGBoost M3: calendar, price, events and lagged sales | 0.629 | **0.979** | **74.7%** | 0.681 |
| LSTM: one network for all series, 112 days of history | 0.636 | 0.986 | 75.4% | **0.660** |
| TimesFM 3.0 V3: pretrained, zero-shot | 0.642 | 0.988 | 75.5% | 0.744 |
| V6: TimesFM 3.0 V3 + recursive boosted trees | **0.643** | 0.984 | 75.2% | 0.716 |

For TimesFM 3.0 V3 and V6, R² is OSR² (benchmarked on the training mean) and the test starts one day earlier. The LSTM is scored on all item–days, the others on days the item is on the shelf. OLS, Ridge and LASSO answer a next-day task on sampled rows ([notebook](notebooks/regression_ols_ridge_lasso.ipynb)).

<p align="center"><img src="docs/accuracy-by-model.png" width="85%" alt="Bar charts of item-level R² and 12-level WRMSSE for each model, with XGBoost M3 re-forecast daily shown separately"></p>

- **Recent sales are the signal.** Adding lagged sales to XGBoost lifts R² from 0.533 to 0.629 and cuts WRMSSE from 0.994 to 0.681; price, SNAP and events add almost nothing on their own.
- **The trained models tie on single items and differ on totals.** They sit within 0.015 R² of each other, while the models that learn across series lead on WRMSSE.
- **Business impact.** Safety stock scales with forecast error: V6 frees about $148k of inventory at a 95% in-stock target, or $37k a year in holding cost; XGBoost M3 frees $115k and needs no GPU. Assumptions are in the [business overview](docs/overview.md#business-impact).

## Improvement: re-forecast daily

The same XGBoost M3 model and features, re-forecast every morning from yesterday's sales instead of every 4 weeks:

<p align="center"><img src="docs/improvement-daily.png" width="85%" alt="Two timelines of 28 days: one forecast every 4 weeks, so later days use older sales data, against a new forecast every morning that always uses yesterday's sales"></p>

| XGBoost M3, all 30,490 series | R² ↑ | MAE ↓ | WAPE ↓ | WRMSSE ↓ |
|---|---:|---:|---:|---:|
| Re-forecast every 4 weeks | 0.629 | 0.979 | 74.7% | 0.681 |
| **Re-forecast every day** | **0.706** | **0.911** | **69.5%** | **0.525** |

The gain is largest for totals: WRMSSE falls 41% for chain-wide sales and 7% for single item–store series.

## Run the code

Download the [M5 data ZIP (48 MB)](https://github.com/KimChen001/m5-walmart-sales-forecasting/releases/download/m5-data-v1/m5-forecasting-accuracy.zip) and unzip it into the repository root, so that `calendar.csv`, `sales_train_evaluation.csv` and `sell_prices.csv` sit next to `xgb_forecast.py`. Tested with Python 3.13.

```bash
git clone https://github.com/KimChen001/m5-walmart-sales-forecasting.git
cd m5-walmart-sales-forecasting
python -m pip install -r requirements.txt
python xgb_forecast.py --state CA --cat FOODS --device cpu --quick
jupyter notebook notebooks/
```

The quick run trains the baselines and XGBoost M1–M3 on the 5,748 California FOODS series with fewer trees and writes the four metrics to `outputs/xgb_CA_FOODS_quick/metrics_overall.csv`. From a fresh clone it took 3.6 minutes, and M3 reached R² 0.659 and WRMSSE 0.703. [`xgboost_full_panel.ipynb`](notebooks/xgboost_full_panel.ipynb) rebuilds the full-panel XGBoost results from the archived outputs in seconds, without the data. The full runs used MIT ORCD: 3 h 15 min on 32 CPU cores for XGBoost and 4 h 15 min on one GPU for TimesFM 3.0 ([cluster runs](docs/reproduce.md)). On a Mac, XGBoost also needs `brew install libomp`.

## Repository map

| Path | Contents |
|---|---|
| `xgb_forecast.py` · `wrmsse.py` · `final_metrics.py` | XGBoost training, M5 WRMSSE scoring, final metric tables |
| `timesfm_forecast.py` · `timesfm3_improvements.py` | TimesFM 3.0 baseline and adjusted runs |
| [`notebooks/`](notebooks) | Data exploration, regression, XGBoost results, LSTM, TimesFM 3.0 V3–V6 |
| [`cluster/`](cluster) | ORCD environment set-up and batch scripts |
| [`results/`](results/README.md) | Metrics archived from each ORCD job |
| [`docs/`](docs) | Business overview, full results, code by model, cluster runs |

## Limitations

- Stock-outs look like zero demand. Only TimesFM 3.0 V3 cleans likely stock-outs (0.65% of test item–days fit its rule); without inventory data, none can be confirmed.
- The dollar figures scale safety stock by each model's RMSE. They ignore lead times and how error varies across items, so they are estimates, not a replenishment simulation.
- The data cover 10 stores and one test year (2015–16); results may not transfer to other stores or periods.
- The daily re-forecasting gain applies to decisions made one day ahead.

**Team** · Regression notebook by Kayla. XGBoost by Zhixin. LSTM notebook by Marco. TimesFM 3.0 by Kayla, Zhixin and Marco.
