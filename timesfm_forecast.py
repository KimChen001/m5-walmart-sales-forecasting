"""Zero-shot M5 backtest with Google TimesFM 2.5 or 3.0."""
import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--state", default="ALL")
parser.add_argument("--cat", default="ALL")
parser.add_argument("--model", choices=["2.5", "3.0"], default="2.5")
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()

root = Path(__file__).resolve().parent
last_day, train_end, horizon, context, batch = 1941, 1553, 28, 1024, 32
calendar = pd.read_csv(root / "calendar.csv", parse_dates=["date"]).iloc[:last_day]
sales = pd.read_csv(root / "sales_train_evaluation.csv")
if args.state != "ALL":
    sales = sales[sales.state_id == args.state]
if args.cat != "ALL":
    sales = sales[sales.cat_id == args.cat]
if args.smoke:
    sales = sales.head(batch)
if sales.empty:
    raise SystemExit("No series selected")
ids = sales.id.to_numpy()
values = sales[[f"d_{i}" for i in range(1, last_day + 1)]].to_numpy(np.float32)
keys = pd.MultiIndex.from_frame(sales[["store_id", "item_id"]])
prices = pd.read_csv(root / "sell_prices.csv")
rows = keys.get_indexer(pd.MultiIndex.from_frame(prices[["store_id", "item_id"]]))
week_ids, day_weeks = np.unique(calendar.wm_yr_wk.to_numpy(), return_inverse=True)
weeks = pd.Index(week_ids).get_indexer(prices.wm_yr_wk)
valid_price = (rows >= 0) & (weeks >= 0) & prices.sell_price.notna().to_numpy()
listed = np.zeros((len(ids), len(week_ids)), dtype=bool)
listed[rows[valid_price], weeks[valid_price]] = True
del sales, prices, rows, weeks

if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU unavailable")
print(f"GPU: {torch.cuda.get_device_name(0)}; series: {len(ids):,}", flush=True)
torch.set_float32_matmul_precision("high")
if args.model == "3.0":
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    model = TimesFM3Evaluator(ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch",
        per_core_batch_size=batch,
        device="cuda",
    ))
    prefix = "timesfm3"
else:
    import timesfm

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch", torch_compile=False
    )
    model.compile(timesfm.ForecastConfig(
        max_context=context, max_horizon=horizon, per_core_batch_size=batch,
        normalize_inputs=True, infer_is_positive=True,
    ))
    prefix = "timesfm"

out = root / "outputs" / f"{prefix}_{args.state}_{args.cat}{'_smoke' if args.smoke else ''}"
out.mkdir(parents=True, exist_ok=True)
predictions = out / "test_predictions.csv.gz"
temporary = out / "test_predictions.partial.csv.gz"
daily_parts = []
squared = absolute = signed = units = 0.0
count = positive_count = 0
actual_squared = percentage_error = 0.0
with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
    stop = min(train_end + horizon, last_day) if args.smoke else last_day
    for origin in range(train_end, stop, horizon):
        end = min(origin + horizon, stop)
        start = max(0, origin - context)
        days = np.arange(origin, end)
        daily_actual = np.zeros(len(days))
        daily_forecast = np.zeros(len(days))
        for left in range(0, len(ids), batch):
            right = min(left + batch, len(ids))
            history = values[left:right, start:origin].copy()
            history[~listed[left:right, day_weeks[start:origin]]] = np.nan
            inputs = []
            for row in history:
                observed = np.flatnonzero(np.isfinite(row))
                inputs.append(row[observed[0]:] if observed.size else np.zeros(1, np.float32))
            with torch.inference_mode():
                if args.model == "3.0":
                    point = np.stack([
                        result.forecast for result in model.predict_batch(
                            inputs, horizon=horizon, return_quantiles=False,
                            use_symmetric_averaging=False, use_znorm=True,
                        )
                    ])
                else:
                    point, _ = model.forecast(horizon=horizon, inputs=inputs)
            point = np.maximum(point[:, :len(days)], 0)
            if not np.isfinite(point).all():
                raise RuntimeError(f"Non-finite predictions for origin {origin}, batch {left}")
            item, day = np.nonzero(listed[left:right, day_weeks[origin:end]])
            actual = values[left + item, origin + day]
            estimate = point[item, day]
            error = estimate.astype(np.float64) - actual
            squared += np.square(error).sum()
            absolute += np.abs(error).sum()
            signed += error.sum()
            units += actual.sum(dtype=np.float64)
            count += len(actual)
            actual_squared += np.square(actual.astype(np.float64)).sum()
            positive = actual > 0
            positive_count += positive.sum()
            percentage_error += (np.abs(error[positive]) / actual[positive]).sum()
            daily_actual += np.bincount(day, weights=actual, minlength=len(days))
            daily_forecast += np.bincount(day, weights=estimate, minlength=len(days))
            pd.DataFrame({
                "id": ids[left + item], "d_num": days[day] + 1,
                "y": actual, "timesfm": estimate, "date": calendar.date.to_numpy()[days[day]],
            }).to_csv(handle, index=False, header=(origin == train_end and left == 0))
        daily_parts.append(pd.DataFrame({
            "date": calendar.date.to_numpy()[days], "y": daily_actual, "timesfm": daily_forecast,
        }))
        print(f"Finished d_{origin + 1} to d_{end}; {count:,} scored rows", flush=True)

daily = pd.concat(daily_parts, ignore_index=True)
daily_error = daily.timesfm - daily.y
xmas = (daily.date.dt.month == 12) & (daily.date.dt.day == 25)
open_days = ~xmas & (daily.y > 0)
item_sst = actual_squared - units ** 2 / count if count else 0.0
daily_sst = np.square(daily.y - daily.y.mean()).sum()
metrics = pd.DataFrame([{
    "RMSE": np.sqrt(squared / count) if count else np.nan,
    "MAE": absolute / count if count else np.nan,
    "Item_MAPE_positive_actuals": percentage_error / positive_count if positive_count else np.nan,
    "Item_R2": 1 - squared / item_sst if item_sst > 0 else np.nan,
    "Daily_total_R2": 1 - np.square(daily_error).sum() / daily_sst if daily_sst > 0 else np.nan,
    "Bias": signed / units if units else np.nan,
    "Daily_total_RMSE": np.sqrt(np.square(daily_error).mean()),
    "Daily_total_MAPE_exXmas": (daily_error[open_days].abs() / daily.y[open_days]).mean(),
}], index=["timesfm"])
metrics.to_csv(out / "metrics_overall.csv")
daily.to_csv(out / "daily_totals.csv", index=False)
temporary.replace(predictions)
print(metrics.to_string(), flush=True)
