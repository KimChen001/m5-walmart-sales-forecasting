"""TimesFM 3.0 rolling M5 evaluation with covariates and decile calibration."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from timesfm3 import ModelConfig, TimesFM3Evaluator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)

split, last_day, horizon, context, batch = 1553, 1941, 28, 1024, 8
calendar = pd.read_csv("calendar.csv", parse_dates=["date"]).iloc[:last_day].reset_index(drop=True)
sales = pd.read_csv("sales_train_evaluation.csv", nrows=16 if args.smoke else None)
values = sales[[f"d_{d}" for d in range(1, last_day + 1)]].to_numpy(np.float32)
keys = pd.MultiIndex.from_frame(sales[["store_id", "item_id"]])
prices = pd.read_csv("sell_prices.csv")
week_ids, day_weeks = np.unique(calendar.wm_yr_wk, return_inverse=True)
rows = keys.get_indexer(pd.MultiIndex.from_frame(prices[["store_id", "item_id"]]))
weeks = pd.Index(week_ids).get_indexer(prices.wm_yr_wk)
valid = (rows >= 0) & (weeks >= 0) & prices.sell_price.notna().to_numpy()
weekly = np.full((len(keys), len(week_ids)), np.nan, dtype=np.float32)
weekly[rows[valid], weeks[valid]] = prices.sell_price.to_numpy()[valid]
del prices

events = ["SuperBowl", "ValentinesDay", "Easter", "Mother's day",
          "IndependenceDay", "Thanksgiving", "Christmas", "NewYear"]
event_flags = np.stack([
    (calendar.event_name_1.eq(event) | calendar.event_name_2.eq(event)).to_numpy(np.float32)
    for event in events
])

torch.set_float32_matmul_precision("high")
model = TimesFM3Evaluator(ModelConfig(
    checkpoint_path="google/timesfm-3.0-pytorch", per_core_batch_size=batch, device="cuda",
))


def hierarchy_matrix(series):
    levels = [[], ["state_id"], ["store_id"], ["cat_id"], ["dept_id"],
              ["state_id", "cat_id"], ["state_id", "dept_id"],
              ["store_id", "cat_id"], ["store_id", "dept_id"], ["item_id"],
              ["state_id", "item_id"], ["store_id", "item_id"]]
    row_indices, groups = [], []
    total = 0
    for columns in levels:
        if columns:
            codes = series.groupby(columns, sort=False).ngroup().to_numpy()
        else:
            codes = np.zeros(len(series), dtype=int)
        size = int(codes.max()) + 1
        row_indices.append(codes + total)
        groups.append(slice(total, total + size))
        total += size
    matrix = csr_matrix((np.ones(len(series) * len(levels), dtype=np.float32),
                         (np.concatenate(row_indices), np.tile(np.arange(len(series)), len(levels)))),
                        shape=(total, len(series)))
    return matrix, groups


def window_wrmsse(bottom_forecasts, origin, end):
    history = hierarchy @ values[:, :origin]
    positive = history > 0
    first_sale = np.argmax(positive, axis=1)
    differences = np.diff(history, axis=1)
    differences *= np.arange(origin - 1)[None, :] >= first_sale[:, None]
    scale = np.square(differences).sum(axis=1, dtype=np.float64) / np.maximum(
        origin - first_sale - 1, 1
    )
    del differences, history
    prices = np.nan_to_num(weekly[:, day_weeks[origin - horizon:origin]], nan=0)
    dollars = hierarchy @ np.sum(values[:, origin - horizon:origin] * prices,
                                axis=1, dtype=np.float64)
    eligible = (scale > 0) & (dollars > 0)
    truth = hierarchy @ values[:, origin:end]
    results = []
    for forecast in bottom_forecasts:
        mse = np.mean(np.square(hierarchy @ forecast - truth), axis=1, dtype=np.float64)
        rmsse = np.sqrt(np.divide(mse, scale, out=np.zeros_like(mse), where=eligible))
        level_scores = []
        for group in hierarchy_levels:
            weights = np.where(eligible[group], dollars[group], 0)
            if weights.sum() > 0:
                level_scores.append(np.average(rmsse[group], weights=weights))
        if not level_scores:
            raise ValueError("No eligible series for WRMSSE")
        results.append(np.mean(level_scores))
    return results


hierarchy, hierarchy_levels = hierarchy_matrix(sales)
del sales


def forecast_batches(origin, symmetric):
    start = max(0, origin - context)
    end = min(last_day, origin + horizon)
    history_length = origin - start
    historical_weeks = day_weeks[start:origin]
    dates = np.arange(origin, end)
    flags = np.pad(event_flags[:, start:end], ((0, 0), (0, origin + horizon - end)))
    for left in range(0, len(keys), batch):
        right = min(left + batch, len(keys))
        price_history = weekly[left:right, historical_weeks]
        history = values[left:right, start:origin].copy()
        history[~np.isfinite(price_history)] = np.nan
        inputs, zero_flags, known = [], [], []
        for row, past_prices in zip(history, price_history):
            observed = np.flatnonzero(np.isfinite(row))
            first = int(observed[0]) if observed.size else history_length - 1
            recent = row[max(0, history_length - 56):]
            # Use only observations preceding the forecast origin.
            normally_sells = (recent > 0).sum() >= 28
            zero = row == 0
            streak = np.arange(history_length) - np.maximum.accumulate(
                np.where(zero, -1, np.arange(history_length))
            )
            flag = (normally_sells & (streak >= 7)).astype(np.float32)
            recent_prices = past_prices[-56:]
            observed_prices = recent_prices[np.isfinite(recent_prices)]
            # Carry the last observed price forward as the forecast-date price assumption.
            discount = np.zeros(history_length + horizon, dtype=np.float32)
            if observed_prices.size:
                reference = float(np.median(observed_prices))
                if reference > 0:
                    discount[:history_length] = np.where(
                        np.isfinite(past_prices), 1 - past_prices / reference, 0
                    )
                    discount[history_length:] = 1 - observed_prices[-1] / reference
            inputs.append(row[first:] if observed.size else np.zeros(1, np.float32))
            zero_flags.append(flag[None, first:])
            known.append(np.vstack((discount[None, first:], flags[:, first:])))
        with torch.inference_mode():
            results = list(model.predict_batch(
                inputs, horizon=horizon, return_quantiles=True,
                past_only_covariates=zero_flags, past_future_covariates=known,
                use_symmetric_averaging=symmetric, use_znorm=True,
            ))
        listed = np.isfinite(weekly[left:right, day_weeks[origin:end]])
        quantiles = np.maximum(np.stack([r.quantiles for r in results])[:, :len(dates)], 0)
        if quantiles.shape != (*listed.shape, 9) or not np.isfinite(quantiles).all():
            raise ValueError(f"Invalid decile forecasts at origin {origin}, batch {left}")
        yield slice(left, right), dates, listed, quantiles



def fit_decile_weights():
    gram, cross, count = np.zeros((9, 9)), np.zeros(9), 0
    origins = range(split - (1 if args.smoke else 4) * horizon, split, horizon)
    for origin in origins:
        for rows, dates, listed, quantiles in forecast_batches(origin, symmetric=True):
            actual = values[rows, dates][listed].astype(np.float64)
            quantiles = quantiles[listed].astype(np.float64)
            gram += quantiles.T @ quantiles
            cross += quantiles.T @ actual
            count += len(actual)
        print(f"Calibration window ending d_{origin + horizon}", flush=True)
    if not count:
        raise ValueError("No calibration observations")
    gram /= count
    cross /= count
    result = minimize(
        lambda w: w @ gram @ w - 2 * w @ cross,
        np.full(9, 1 / 9), jac=lambda w: 2 * (gram @ w - cross),
        bounds=[(0, 1)] * 9, constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        method="SLSQP", options={"ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


weights = fit_decile_weights()
pd.Series(weights, index=np.arange(1, 10) / 10, name="weight").to_csv(args.out / "decile_weights.csv")
# The two medians isolate the effect of symmetry with identical input features.
names = ["Covariates / symmetry off / median", "Covariates / symmetry on / median",
         "Covariates / symmetry on / equal deciles", "Covariates / symmetry on / fitted deciles"]
scores = np.zeros((4, 7))
actual_daily = np.zeros(last_day)
forecast_daily = np.zeros((4, last_day))


def accumulate(index, dates, actual, point):
    error = point - actual
    positive = actual > 0
    scores[index] += [len(actual), actual.sum(), actual @ actual, np.abs(error).sum(),
                      error @ error, np.abs(error[positive] / actual[positive]).sum(),
                      positive.sum()]
    np.add.at(forecast_daily[index], dates, point)


stop = split + horizon if args.smoke else last_day
window_scores = []
for origin in range(split, stop, horizon):
    end = min(origin + horizon, stop)
    bottom = np.zeros((len(names), len(keys), end - origin), dtype=np.float32)
    for symmetric in (False, True):
        for rows, dates, listed, q in forecast_batches(origin, symmetric=symmetric):
            actual = values[rows, dates][listed].astype(np.float64)
            scored_dates = np.broadcast_to(dates, listed.shape)[listed]
            variants = [(0, q[..., 4])]
            if symmetric:
                variants = [(1, q[..., 4]), (2, q.mean(axis=-1)), (3, q @ weights)]
            else:
                np.add.at(actual_daily, scored_dates, actual)
            for index, point in variants:
                bottom[index, rows, :] = point
                accumulate(index, scored_dates, actual, point[listed].astype(np.float64))
    window_scores.append(window_wrmsse(bottom, origin, end))
    print(f"Scored test window through d_{end}; WRMSSE {window_scores[-1]}", flush=True)
pd.DataFrame(window_scores, index=range(split + 1, stop + 1, horizon), columns=names).to_csv(
    args.out / "wrmsse_by_window.csv", index_label="first_forecast_day"
)

days = calendar.date.iloc[split:stop]
actual = actual_daily[split:stop]
mape_days = (actual > 0) & ~((days.dt.month == 12) & (days.dt.day == 25)).to_numpy()
records = []
for name, score, forecast in zip(names, scores, forecast_daily[:, split:stop]):
    count, units, squares, absolute, sse, ape, positive_count = score
    error = forecast - actual
    records.append({
        "Variant": name, "Item MAE": absolute / count, "Item RMSE": np.sqrt(sse / count),
        "Item WAPE": absolute / units, "Item MAPE (positive actuals)": ape / positive_count,
        "Item R2": 1 - sse / (squares - units**2 / count),
        "Daily RMSE": np.sqrt(np.mean(error**2)), "Daily WAPE": np.abs(error).sum() / units,
        "Daily MAPE (excl. Christmas)": np.mean(np.abs(error[mape_days] / actual[mape_days])),
        "Daily R2": 1 - (error @ error) / np.sum((actual - actual.mean())**2),
        "Total sales bias": error.sum() / units,
    })
metrics = pd.DataFrame(records).set_index("Variant")
metrics["Mean rolling WRMSSE"] = np.mean(window_scores, axis=0)
metrics.to_csv(args.out / "metrics_overall.csv")
pd.DataFrame({"date": days.to_numpy(), "actual": actual,
              **dict(zip(names, forecast_daily[:, split:stop]))}).to_csv(
    args.out / "daily_totals.csv", index=False,
)
lines = ["TimesFM 3.0 Forecast Evaluation Results".center(104), "=" * 104,
         "28-day rolling backtest | Historical zero-run flag | Event and price covariates",
         "Future discount uses last observed price as a proxy for planned price.",
         "WRMSSE averages the window scores (the final window has 24 days in the full run).",
         "It uses the full item-store panel and trailing 28-day dollar-sales weights.", "=" * 104]
for name, row in metrics.iterrows():
    lines.append(name)
    items = list(row.items())
    formatted = [(label, format(value, ".2%" if any(
        token in label for token in ("APE", "bias")) else ",.4f")) for label, value in items]
    for left, right in zip(formatted[::2], formatted[1::2]):
        lines.append(f"{left[0]:<34}{left[1]:>15}    {right[0]:<34}{right[1]:>15}")
    if len(formatted) % 2:
        label, value = formatted[-1]
        lines.append(f"{label:<34}{value:>15}")
    lines.append("=" * 104)
report = "\n".join(lines) + "\n"
(args.out / "timesfm_summary.txt").write_text(report, encoding="utf-8")
print(report)
