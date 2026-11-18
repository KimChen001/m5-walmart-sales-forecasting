"""
Compute all metrics for both panels in one go and print them as one table.

Two panels:
    FULL      all 30,490 item-store series
    REGULAR   only regular sellers with "sales on more than 50% of days" (6,638 series) —— aligned with Kayla's setup

Metrics:
    R2 / RMSE / MAE / WAPE / Bias        item-store-day level, rows on the shelf only
    Daily_R2 / Daily_MAPE                daily totals, excluding Christmas (stores closed)
    WRMSSE / WRMSSE_L12                  read from the results wrmsse.py has already written

Usage (from ~/m5-project):
    python final_metrics.py outputs/xgb_ALL_ALL

Prerequisite: these two have already run; their outputs are read directly, not recomputed
    python wrmsse.py outputs/xgb_ALL_ALL
    python wrmsse.py outputs/xgb_ALL_ALL --min-selling-share 0.5
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAST_DAY = 1941
SKIP = {"id", "d_num", "y", "date", "month", "window", "snap", "is_event",
        "store", "dept", "cat", "state", "volume"}
ORDER = ["naive_lag28", "naive_last", "ma_28",
         "xgb_M1_calendar", "xgb_M2_price_event", "xgb_M3_full"]


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def point_metrics(te, models, cal):
    """item-store-day metrics + daily-total metrics"""
    daily = te.groupby("d_num")[["y"] + models].sum()
    d = pd.to_datetime(cal.date.to_numpy()[daily.index - 1])
    open_ = ~((d.month == 12) & (d.day == 25))          # stores close on Christmas; totals ~0 would blow up MAPE
    rows = {}
    for m in models:
        e = te[m] - te.y
        rows[m] = {
            "R2": r2(te.y, te[m]),
            "RMSE": np.sqrt((e ** 2).mean()),
            "MAE": e.abs().mean(),
            "WAPE": e.abs().sum() / te.y.sum(),
            "Bias": te[m].sum() / te.y.sum() - 1,
            "Daily_R2": r2(daily.y[open_], daily[m][open_]),
            "Daily_MAPE": ((daily[m][open_] - daily.y[open_]).abs() / daily.y[open_]).mean(),
        }
    return pd.DataFrame(rows).T


def add_wrmsse(df, path):
    """merge in the results wrmsse.py has written; if missing, leave blank and do not recompute"""
    if path.exists():
        w = pd.read_csv(path, index_col=0)
        for c in ("WRMSSE", "WRMSSE_L12"):
            if c in w.columns:
                df[c] = w[c]
    else:
        print(f"  ({path.name} not found; the two WRMSSE columns are left blank)")
    return df


def show(title, df, n_series, n_rows):
    print()
    print("=" * 104)
    print(f"{title}   |   {n_series:,} series   |   {n_rows:,} scored item-days")
    print("=" * 104)
    cols = [c for c in ["R2", "Daily_R2", "RMSE", "MAE", "WAPE", "Bias",
                        "Daily_MAPE", "WRMSSE", "WRMSSE_L12"] if c in df.columns]
    out = df.reindex([m for m in ORDER if m in df.index]).loc[:, cols]
    print(out.round(4).to_string())


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/xgb_ALL_ALL")
    data_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")

    day_cols = [f"d_{i}" for i in range(1, LAST_DAY + 1)]
    print("reading sales_train_evaluation.csv ...")
    sales = pd.read_csv(data_dir / "sales_train_evaluation.csv",
                        usecols=["id"] + day_cols, dtype={c: np.int16 for c in day_cols})
    share = (sales[day_cols].to_numpy() > 0).mean(axis=1)
    regular = set(sales.id[share > 0.5])
    print(f"  regular sellers (sales on >50% of days): {len(regular):,} / {len(sales):,}")
    del sales

    cal = pd.read_csv(data_dir / "calendar.csv", parse_dates=["date"]).iloc[:LAST_DAY]

    print("reading test_predictions.csv.gz ...")
    head = pd.read_csv(out_dir / "test_predictions.csv.gz", nrows=0)
    models = [c for c in head.columns if c not in SKIP]
    te = pd.read_csv(out_dir / "test_predictions.csv.gz",
                     usecols=["id", "d_num", "y"] + models,
                     dtype={"d_num": np.int32, "y": np.float32,
                            **{m: np.float32 for m in models}})
    print(f"  {len(te):,} rows, model columns: {', '.join(models)}")

    full = add_wrmsse(point_metrics(te, models, cal), out_dir / "metrics_wrmsse.csv")
    full.to_csv(out_dir / "metrics_final_full.csv")

    sub = te[te.id.isin(regular)]
    reg = add_wrmsse(point_metrics(sub, models, cal),
                     out_dir / "metrics_wrmsse_regular50.csv")
    reg.to_csv(out_dir / "metrics_final_regular50.csv")

    pd.set_option("display.width", 250, "display.max_columns", 30)
    show("PANEL A — FULL: all series", full, te.id.nunique(), len(te))
    show("PANEL B — REGULAR SELLERS: the 6,638 subset aligned with Kayla", reg, sub.id.nunique(), len(sub))

    print()
    print("=" * 104)
    print("Difference between the panels (REGULAR - FULL); negative = better after dropping slow sellers")
    print("=" * 104)
    common = [c for c in full.columns if c in reg.columns]
    print((reg[common] - full[common]).round(4).to_string())
    print()
    print("Note: same models, same predictions, only the scored series differ. The whole gap comes from the panel, not the model.")
    print(f"written to {out_dir}/metrics_final_full.csv and metrics_final_regular50.csv")


if __name__ == "__main__":
    main()
