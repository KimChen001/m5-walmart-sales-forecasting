"""
一次算完两套面板的全部指标, 打印成一张表。

两套面板:
    FULL      全部 30,490 条 item-store 序列
    REGULAR   只保留"有销量的天数 > 50%"的常销品 (6,638 条) —— 对齐 Kayla 的口径

指标:
    R2 / RMSE / MAE / WAPE / Bias        item-store-day 层面, 只算已上架的行
    Daily_R2 / Daily_MAPE                每日总量, 排除圣诞 (闭店)
    WRMSSE / WRMSSE_L12                  从 wrmsse.py 已经写好的结果里读进来

用法 (在 ~/m5-project 下):
    python final_metrics.py outputs/xgb_ALL_ALL

前提: 这两条已经跑过, 它们的输出会被直接读取, 不重算
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
    """item-store-day 指标 + 每日总量指标"""
    daily = te.groupby("d_num")[["y"] + models].sum()
    d = pd.to_datetime(cal.date.to_numpy()[daily.index - 1])
    open_ = ~((d.month == 12) & (d.day == 25))          # 圣诞闭店, 总量 ~0, 会让 MAPE 爆掉
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
    """把 wrmsse.py 写好的结果并进来; 没有就留空, 不重算"""
    if path.exists():
        w = pd.read_csv(path, index_col=0)
        for c in ("WRMSSE", "WRMSSE_L12"):
            if c in w.columns:
                df[c] = w[c]
    else:
        print(f"  (没找到 {path.name}, WRMSSE 这两列留空)")
    return df


def show(title, df, n_series, n_rows):
    print()
    print("=" * 104)
    print(f"{title}   |   {n_series:,} 条序列   |   {n_rows:,} 个计分的 item-day")
    print("=" * 104)
    cols = [c for c in ["R2", "Daily_R2", "RMSE", "MAE", "WAPE", "Bias",
                        "Daily_MAPE", "WRMSSE", "WRMSSE_L12"] if c in df.columns]
    out = df.reindex([m for m in ORDER if m in df.index]).loc[:, cols]
    print(out.round(4).to_string())


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/xgb_ALL_ALL")
    data_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")

    day_cols = [f"d_{i}" for i in range(1, LAST_DAY + 1)]
    print("读 sales_train_evaluation.csv ...")
    sales = pd.read_csv(data_dir / "sales_train_evaluation.csv",
                        usecols=["id"] + day_cols, dtype={c: np.int16 for c in day_cols})
    share = (sales[day_cols].to_numpy() > 0).mean(axis=1)
    regular = set(sales.id[share > 0.5])
    print(f"  常销品 (>50% 的天数有销量): {len(regular):,} / {len(sales):,}")
    del sales

    cal = pd.read_csv(data_dir / "calendar.csv", parse_dates=["date"]).iloc[:LAST_DAY]

    print("读 test_predictions.csv.gz ...")
    head = pd.read_csv(out_dir / "test_predictions.csv.gz", nrows=0)
    models = [c for c in head.columns if c not in SKIP]
    te = pd.read_csv(out_dir / "test_predictions.csv.gz",
                     usecols=["id", "d_num", "y"] + models,
                     dtype={"d_num": np.int32, "y": np.float32,
                            **{m: np.float32 for m in models}})
    print(f"  {len(te):,} 行, 模型列: {', '.join(models)}")

    full = add_wrmsse(point_metrics(te, models, cal), out_dir / "metrics_wrmsse.csv")
    full.to_csv(out_dir / "metrics_final_full.csv")

    sub = te[te.id.isin(regular)]
    reg = add_wrmsse(point_metrics(sub, models, cal),
                     out_dir / "metrics_wrmsse_regular50.csv")
    reg.to_csv(out_dir / "metrics_final_regular50.csv")

    pd.set_option("display.width", 250, "display.max_columns", 30)
    show("PANEL A — FULL: 全部序列", full, te.id.nunique(), len(te))
    show("PANEL B — REGULAR SELLERS: 对齐 Kayla 的 6,638 子集", reg, sub.id.nunique(), len(sub))

    print()
    print("=" * 104)
    print("两套面板的差距 (REGULAR - FULL), 负数 = 筛掉慢销品之后变好")
    print("=" * 104)
    common = [c for c in full.columns if c in reg.columns]
    print((reg[common] - full[common]).round(4).to_string())
    print()
    print("注: 同一个模型、同一份预测, 只是计分的序列不同。差距全部来自面板, 不是模型。")
    print(f"已写入 {out_dir}/metrics_final_full.csv 和 metrics_final_regular50.csv")


if __name__ == "__main__":
    main()
