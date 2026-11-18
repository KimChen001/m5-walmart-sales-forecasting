# Full results

[← README](../README.md) · [Business overview](overview.md) · [Code and notebooks](code.md) · [Run and reproduce](reproduce.md) · [Archived ORCD outputs](../results/README.md)

All scores cover the full panel of 30,490 item–store series and the test period from May 2015 to May 2016.

## Full panel, 28-day forecasts

| Full panel, 28-day forecasts | R² ↑ | MAE (units) ↓ | WAPE ↓ | M5 WRMSSE (12 levels) ↓ |
|---|---:|---:|---:|---:|
| Seasonal naive, lag 28 | 0.3057 | 1.2308 | 93.98% | 0.9378 |
| 28-day moving average | 0.5782 | 1.0204 | 77.91% | 1.0655 |
| XGBoost M1 · Calendar and IDs | 0.5257 | 1.1119 | 84.89% | 0.9384 |
| XGBoost M2 · Adds price, SNAP and events | 0.5332 | 1.0968 | 83.75% | 0.9938 |
| **XGBoost M3 · Adds sales history** | 0.6289 | 0.9790 | 74.75% | **0.6812** |
| TimesFM 3.0 · Sales-only baseline | 0.6121 | **0.8983** | **68.58%** | 1.6014 |
| **TimesFM 3.0 · Fitted decile weights** | **0.6376** | 0.9669 | 73.83% | 0.8248 |

Between the two main models, TimesFM leads on R², MAE and WAPE, which score each item–store–day on its own, while XGBoost M3 leads on the 12-level WRMSSE (0.681 against 0.825), which also scores sales summed to stores, departments and states. The sales-only TimesFM run has the lowest MAE and WAPE because it forecasts the median, which minimises absolute error, but it under-forecasts total sales by 26.7%. Fitting the decile weights on training data gives up some MAE to correct that level.

MIT ORCD jobs `29323394` (XGBoost), `27920694` (TimesFM baseline), and `28701113` (TimesFM adjusted). Relative to seasonal naive, M3 reduced MAE by **0.252 units per scored item–store–day** and WRMSSE by **27.4%**. [Business impact](overview.md#business-impact) translates these errors into inventory dollars.

## TimesFM in the presentation

The slide deck reports the team's TimesFM 3.0 notebook run (V3: stock-out cleaning, discount and major-event covariates, and a mean calibrated on earlier test windows) and the recommended V6 ensemble of V3 with recursive boosted trees. Both cover the same 30,490 series, with a test period that starts one day earlier (2015-04-30).

| Team notebook, 28-day forecasts | OSR² ↑ | MAE (units) ↓ | WAPE ↓ | M5 WRMSSE (12 levels) ↓ |
|---|---:|---:|---:|---:|
| TimesFM 3.0 · V3 | 0.6415 | 0.988 | 75.5% | 0.744 |
| **V6 · V3 plus recursive boosted trees** | **0.6425** | **0.984** | **75.2%** | **0.716** |

OSR² measures error against the training-period mean rather than the test-period mean. The notebook is [`notebooks/timesfm3_v3_v6.ipynb`](../notebooks/timesfm3_v3_v6.ipynb).

## Next-day forecasts

The same XGBoost job reran each model with sales history updated every day, which matches the setting of the regression experiment. Seasonal naive, M1 and M2 do not use recent sales, so their scores are unchanged and are omitted.

| Full panel, next-day forecasts | R² ↑ | MAE (units) ↓ | WAPE ↓ | M5 WRMSSE (12 levels) ↓ |
|---|---:|---:|---:|---:|
| Same as yesterday | 0.4666 | 1.1233 | 85.76% | 1.1184 |
| 28-day moving average | 0.6304 | 0.9757 | 74.50% | 1.0486 |
| **XGBoost M3 · Adds sales history** | **0.7058** | **0.9108** | **69.54%** | **0.5249** |

Moving M3 from a 28-day to a next-day schedule raises R² from 0.629 to 0.706 and lowers MAE from 0.979 to 0.911, WAPE from 74.75% to 69.54% and WRMSSE from 0.681 to 0.525. The [XGBoost results notebook](../notebooks/xgboost_full_panel.ipynb) breaks both schedules down by hierarchy level, test window, department, store and sales volume.

## How the metrics are computed

R², MAE, and WAPE are computed per item–store–day. [WRMSSE scoring](../wrmsse.py) uses trailing 28-day dollar-sales weights across all 12 levels. In the CSV files, `WRMSSE` is the 12-level score and `WRMSSE_L12` is the item–store score.

Both XGBoost and TimesFM report rolling WRMSSE across 12 M5 hierarchy levels. XGBoost zero-fills positions outside its scored rows when constructing the WRMSSE panel. The adjusted TimesFM run scores its complete forecast panel, so the WRMSSE comparison retains that implementation difference.
