# Run and reproduce

[← Back to the overview](../README.md) · [Full results](results.md) · [Code and notebooks](code.md) · [Archived ORCD outputs](../results/README.md)

## Run locally

Clone the repository and download [`m5-forecasting-accuracy.zip`](https://github.com/KimChen001/m5-walmart-sales-forecasting/releases/download/m5-data-v1/m5-forecasting-accuracy.zip). In PowerShell, replace the ZIP path if your browser saved it elsewhere.

```powershell
git clone https://github.com/KimChen001/m5-walmart-sales-forecasting.git
cd m5-walmart-sales-forecasting
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\m5-forecasting-accuracy.zip" -DestinationPath .
python -m pip install "xgboost==3.4.1" "pandas==3.0.6" "numpy==2.5.3" "scikit-learn==1.9.1"
python xgb_forecast.py --state CA --cat FOODS --device cpu --quick
Import-Csv outputs/xgb_CA_FOODS_quick/metrics_overall.csv | Format-Table R2,MAE,WAPE,WRMSSE
```

The quick run checks the training and scoring pipeline on **5,748 California FOODS series** with a reduced tree budget. It completed in approximately **3 minutes** on the development machine. The scripts read `calendar.csv`, `sales_train_evaluation.csv`, and `sell_prices.csv` from the repository root. The [official Kaggle data page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) documents the files.

## Reproduce the full XGBoost benchmark

On MIT ORCD, place the three CSVs next to `xgb_forecast.py` and submit the batch job from the repository root.

```bash
bash cluster/setup_env.sh  # first run only
mkdir -p logs
sbatch cluster/run_xgb_both.sbatch
```

The [training script](../xgb_forecast.py) fits XGBoost regressors with Tweedie loss. It trains on days 1–1,553 and tests on days 1,554–1,941. Early stopping uses the final 90 training days before refitting. The 28-day backtest uses 14 rolling windows and scores 24 days in the last window. Sales-history features (lags of 1 to 364 days and 7- to 180-day rolling means) are measured at each window's forecast origin, so every day in a window sees the same information, and a `days_ahead` feature tells the model how far ahead it is forecasting. The next-day experiment (`--horizon 1`) moves the origin to the previous day, updates sales history daily, and writes to its own output directory.

| Output | Location |
|---|---|
| 28-day forecasts | `outputs/xgb_ALL_ALL/` |
| Next-day forecasts | `outputs/xgb_ALL_ALL_h1/` |
| Full-panel scores | `metrics_overall.csv` in either directory |
| Regular-seller scores | `metrics_final_regular50.csv` in either directory |

Job `29323394` took 3 h 15 min on 32 cores. Its exported results are archived in [`results/xgb-29323394/`](../results/xgb-29323394), which the [XGBoost results notebook](../notebooks/xgboost_full_panel.ipynb) reads.

## Reproduce TimesFM 3.0

The **sales-only baseline** (ORCD job `27920694`) used the same frozen TimesFM 3.0 checkpoint and 28-day backtest. It used sales history for prediction, kept symmetric averaging off, and took the default median forecast. Weekly prices identified which rows to score. WAPE and 12-level WRMSSE were calculated from the archived predictions. Baseline WRMSSE sets positions without saved forecasts to zero using the same scoring script as XGBoost.

After installing TimesFM below, run the [baseline script](../timesfm_forecast.py) from the repository root on an ORCD GPU compute node:

```bash
module load miniforge
eval "$(conda shell.bash hook)"
conda activate m5
PYTHONPATH="$PWD/timesfm3_lib${PYTHONPATH:+:$PYTHONPATH}" python timesfm_forecast.py --model 3.0 --state ALL --cat ALL
```

Baseline outputs go to `outputs/timesfm3_ALL_ALL/`. The adjusted run uses the script and batch job below.

The [TimesFM 3.0 evaluation script](../timesfm3_improvements.py) uses the same three M5 CSVs, 30,490 series, and chronological 80/20 split as the XGBoost benchmark. It holds the pretrained model weights fixed, adds price, event and historical zero-run features, and fits nine nonnegative decile weights on four windows before the test period. The job compares median forecasts with symmetry off and on, equal decile weights, and fitted decile weights.

On MIT ORCD, start from the existing `m5` environment with CUDA-enabled PyTorch and SciPy. Install the TimesFM version used for the original run into a separate directory to keep its imports isolated.

```bash
module load miniforge
conda run -n m5 python -m pip install --no-deps --target timesfm3_lib 'timesfm==3.0.2'
mkdir -p logs
sbatch cluster/run_timesfm3_gpu.sbatch
```

The job first checks 16 series and then runs the full backtest, downloading the [Google checkpoint](https://huggingface.co/google/timesfm-3.0-pytorch) if needed. Outputs are under `outputs/timesfm3_<job_id>/full/`. Read `metrics_overall.csv` for the four configurations, `wrmsse_by_window.csv` for the 14 window scores, and `decile_weights.csv` for the fitted weights. Job `28701113` completed in 4 h 15 min on one GPU.
