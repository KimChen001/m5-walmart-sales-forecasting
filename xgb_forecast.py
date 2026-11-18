"""
XGBoost sales forecasting on M5 (Walmart) data; by default the CA state, FOODS category subset.

Time split (by time, no shuffling):
    train  d_1    – d_1553  (80%, 2011-01-29 ~ 2015-04-30)
    test   d_1554 – d_1941  (20%, 2015-05-01 ~ 2016-05-22)
    The last 90 days of the training set (d_1464–d_1553) are used for early stopping to choose the number
    of trees; the model is then refit on the whole training set.

Forecast setting: rolling 28-day forecasts. Every historical sales feature is lagged by >= 28 days,
    so everything used for any test day is known before its 28-day window starts, and the whole test
    period can be predicted in one pass.

Ablation (features added layer by layer):
    M1 = calendar + IDs
    M2 = M1 + price / SNAP / events
    M3 = M2 + lagged sales / rolling statistics

Metrics:
    RMSE / MAE / WAPE / R2  —— every item-store series weighted equally
    WRMSSE (official M5 metric) —— weighted by "dollar sales in the 28 days before the origin", scaled by
                               each series' own volatility, averaged over 12 aggregation levels;
                               implemented in wrmsse.py

Usage:
    python xgb_forecast.py                 # default: CA FOODS
    python xgb_forecast.py --quick         # few trees, only checks the pipeline
    python xgb_forecast.py --state TX --cat HOBBIES
    python xgb_forecast.py --state ALL --cat ALL --device cuda   # full panel, on a cluster GPU
"""
import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from wrmsse import WRMSSE, build_dollars

try:
    import psutil                     # only used for the memory guard; skipped if not installed on the cluster
except ImportError:
    psutil = None

ap = argparse.ArgumentParser()
ap.add_argument("--state", default="CA", help="CA / TX / WI / ALL")
ap.add_argument("--cat", default="FOODS", help="FOODS / HOBBIES / HOUSEHOLD / ALL")
ap.add_argument("--device", default="cpu", help="cpu or cuda")
ap.add_argument("--data", default=str(Path(__file__).resolve().parent), help="folder holding the three csv files")
ap.add_argument("--quick", action="store_true", help="few trees, only to check the pipeline")
ap.add_argument("--horizon", type=int, default=28, choices=[1, 28],
                help="28 = one forecast origin every 28 days (M5 convention); 1 = re-forecast every day (matches OLS)")
args = ap.parse_args()

DATA = Path(args.data)
_hz = "" if args.horizon == 28 else f"_h{args.horizon}"
OUT = Path(__file__).resolve().parent / "outputs" / f"xgb_{args.state}_{args.cat}{_hz}{'_quick' if args.quick else ''}"
OUT.mkdir(parents=True, exist_ok=True)

D = 1941            # last day with sales (d_1942–d_1969 have calendar data only, no sales)
TRAIN_END = 1553    # 80% / 20%; must match our own TimesFM runs (split=1553)
VALID_START = 1464  # last 90 days of the training set, for early stopping
H = 28              # WRMSSE scoring window, fixed at 28 days (M5 definition)
HZ = args.horizon   # forecast step: re-forecast every HZ days

ROUNDS, ES = (40, 10) if args.quick else (3000, 100)
PARAMS = dict(
    objective="reg:tweedie", tweedie_variance_power=1.2,  # count data with many zeros
    eta=0.08, max_depth=8, subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", eval_metric="rmse", seed=42, device=args.device,
    nthread=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())),
)

t0 = time.time()
def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

# ---------------------------------------------------------------- 1. Load data
day_cols = [f"d_{i}" for i in range(1, D + 1)]
sales = pd.read_csv(DATA / "sales_train_evaluation.csv", dtype={c: np.int16 for c in day_cols})
if args.state != "ALL":
    sales = sales[sales.state_id == args.state]
if args.cat != "ALL":
    sales = sales[sales.cat_id == args.cat]
sales = sales.reset_index(drop=True)
n = len(sales)
S = sales[day_cols].to_numpy(np.float32)                    # (series, day)
sales = sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]]

cal = pd.read_csv(DATA / "calendar.csv").iloc[:D]
assert (cal.d.to_numpy() == np.array(day_cols)).all()
date = pd.to_datetime(cal.date)

prices = pd.read_csv(DATA / "sell_prices.csv")
prices = prices[prices.store_id.isin(sales.store_id.unique()) & prices.item_id.isin(sales.item_id.unique())]
log(f"{n} series x {D} days, {len(prices)} price rows")

# weekly prices -> daily price matrix (series, day)
weeks = np.sort(cal.wm_yr_wk.unique())
prices = prices[prices.wm_yr_wk <= weeks[-1]]
row_of = pd.Series(np.arange(n), index=sales.store_id + "|" + sales.item_id)
pr_row = row_of.reindex(prices.store_id + "|" + prices.item_id).to_numpy().astype(int)
Pw = np.full((n, len(weeks)), np.nan, np.float32)
Pw[pr_row, np.searchsorted(weeks, prices.wm_yr_wk.to_numpy())] = prices.sell_price.to_numpy()
P = Pw[:, np.searchsorted(weeks, cal.wm_yr_wk.to_numpy())]
del prices, Pw

valid = ~np.isnan(P)                      # has a price = on the shelf; zeros before launch are not real sales
Sn = np.where(valid, S, np.nan).astype(np.float32)

# ---------------------------------------------------------------- 2. Long table
# rows ordered by (day, series), so train / validation / test are contiguous slices and no data is copied
mask = valid.T.ravel()
N = int(mask.sum())
d_num = np.repeat(np.arange(1, D + 1), n)[mask]
sid = np.tile(np.arange(n), D)[mask]
y = S.T.ravel()[mask]
log(f"long table: {N:,} rows (rows before launch removed)")

M1 = ["item_id", "dept_id", "store_id", "wday", "month", "year", "dom", "week"]
M2 = M1 + ["sell_price", "price_rel_max", "price_rel_mean", "price_chg_wk",
           "snap", "is_event", "event_name", "event_type"]
M3 = M2 + ["days_ahead", "lag_28",
           "olag_1", "olag_2", "olag_7", "olag_14", "olag_28", "olag_364",
           "rmean_7", "rmean_28", "rmean_56", "rmean_180", "rstd_28"]
CATS = {"item_id", "dept_id", "store_id", "event_name", "event_type"}
col = {f: j for j, f in enumerate(M3)}

# Memory guard: X is the largest object in the pipeline. When a laptop runs out of memory, Windows does not
# raise an error; it falls back to the page file and the machine freezes for hours instead of failing,
# so we stop here before allocating.
# Skipped on the cluster (SLURM): psutil sees the whole node's memory, not this job's allocation.
x_gb = N * len(M3) * 4 / 2 ** 30
peak_gb = x_gb * 2.4                      # X + float64 temporaries for rolling statistics + DMatrix + evaluation table
if psutil is not None:
    free_gb = psutil.virtual_memory().available / 2 ** 30
    log(f"X needs {x_gb:.1f} GB, pipeline peak about {peak_gb:.0f} GB, {free_gb:.0f} GB available now")
    if free_gb < peak_gb and not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit(
            f"\nNot enough memory: the peak needs about {peak_gb:.0f} GB, only {free_gb:.0f} GB is available.\n"
            f"Running anyway would make Windows fall back to the page file and freeze the machine for hours.\n"
            f"Close large programs such as the browser, run a subset (--state CA --cat FOODS), "
            f"or submit to the cluster (sbatch cluster/run_xgb_cpu.sbatch).")

X = np.empty((N, len(M3)), np.float32)    # M1/M2 are prefixes of M3, so slicing columns copies nothing

def put(name, a, kind):
    """kind: 'sd' = (series, day) matrix, 's' = one value per series, 'd' = one value per day"""
    a = np.asarray(a, np.float32)
    if kind == "sd":
        X[:, col[name]] = a.T.ravel()[mask]
    elif kind == "s":
        X[:, col[name]] = np.tile(a, D)[mask]
    else:
        X[:, col[name]] = np.repeat(a, n)[mask]

def shift(A, L):
    out = np.full_like(A, np.nan)
    out[:, L:] = A[:, :-L]
    return out

def codes(s):
    c = pd.Categorical(s).codes.astype(np.float32)
    c[c < 0] = np.nan                     # missing -> NaN, XGBoost handles it
    return c

# ---------------------------------------------------------------- 3. Features
# IDs and calendar
for c in ["item_id", "dept_id", "store_id"]:
    put(c, codes(sales[c]), "s")
put("wday", cal.wday, "d")                # note: 1 = Saturday
put("month", cal.month, "d")
put("year", cal.year, "d")
put("dom", date.dt.day, "d")
put("week", date.dt.isocalendar().week.astype(int), "d")

# prices (only the current and earlier prices, no future information)
with np.errstate(invalid="ignore", divide="ignore"):
    put("sell_price", P, "sd")
    put("price_rel_max", P / np.fmax.accumulate(P, axis=1), "sd")
    pmean = np.nancumsum(P, axis=1) / np.maximum(np.cumsum(valid, axis=1), 1)
    put("price_rel_mean", P / pmean, "sd")
    put("price_chg_wk", P / shift(P, 7) - 1, "sd")
del pmean

# SNAP: each series takes only the SNAP flag of its own state
snap_cols = cal[["snap_CA", "snap_TX", "snap_WI"]].to_numpy(np.float32)       # (day, 3)
put("snap", snap_cols[:, sales.state_id.map({"CA": 0, "TX": 1, "WI": 2}).to_numpy()].T, "sd")
put("is_event", cal.event_name_1.notna(), "d")
put("event_name", codes(cal.event_name_1), "d")
put("event_type", codes(cal.event_type_1), "d")

# ---- forecast origins
# Every day in the window [o+1, o+HZ] may only use sales up to and including day o.
#   HZ = 28: one origin every 28 days —— the official M5 convention, consistent with TimesFM and Kayla's notebook
#   HZ = 1 : re-forecast every day —— consistent with Kayla's OLS/LASSO (rolling means end yesterday)
# orig_of_day[c] = origin of the window that column c (0-based) belongs to; its value = number of days of history.
cols = np.arange(D)
if HZ == 1:
    orig_of_day = cols.copy()                    # history ends yesterday
else:
    orig_of_day = TRAIN_END + HZ * np.floor_divide(cols - TRAIN_END, HZ)
put("days_ahead", cols - orig_of_day + 1, "d")   # 1..HZ: days since the origin

def pick(A, back):
    """A[:, origin - back]. back=1 is the origin day itself; NaN if that day is before the data start"""
    j = orig_of_day - back
    out = A[:, np.clip(j, 0, D - 1)].copy()
    out[:, j < 0] = np.nan
    return out

# Lagged sales. lag_28 is kept as is: it is the M5 naive baseline (t-28 <= origin, valid for any horizon).
# olag_j = sales j days before the origin; all days in the window share the same value.
for j in [1, 2, 7, 14, 28, 364]:
    put(f"olag_{j}", pick(Sn, j), "sd")
put("lag_28", shift(Sn, 28), "sd")

# Rolling statistics: windows end at the origin; computed from cumulative sums, NaNs before launch excluded
ok = ~np.isnan(Sn)
v = np.where(ok, Sn, 0).astype(np.float64)
z = np.zeros((n, 1))
cs = np.concatenate([z, v.cumsum(1)], 1)
cs2 = np.concatenate([z, (v * v).cumsum(1)], 1)
cc = np.concatenate([z, ok.cumsum(1)], 1)
del v, ok

def roll(w):
    s, s2, c = (np.full((n, D), np.nan) for _ in range(3))
    s[:, w - 1:] = cs[:, w:] - cs[:, :-w]
    s2[:, w - 1:] = cs2[:, w:] - cs2[:, :-w]
    c[:, w - 1:] = cc[:, w:] - cc[:, :-w]
    c[c < max(1, w // 2)] = np.nan        # skip windows with too few valid days
    mean = s / c
    std = np.sqrt(np.clip(s2 / c - mean ** 2, 0, None))
    return mean, std

for w in [7, 28, 56, 180]:
    mean, std = roll(w)
    put(f"rmean_{w}", pick(mean, 1), "sd")       # window ends at the origin
    if w == 28:
        put("rstd_28", pick(std, 1), "sd")
del cs, cs2, cc, mean, std
log("features done")

# ---------------------------------------------------------------- 4. Split
i_va = int(np.searchsorted(d_num, VALID_START))
i_te = int(np.searchsorted(d_num, TRAIN_END + 1))
log(f"train {i_va:,} rows | validation {i_te - i_va:,} rows | test {N - i_te:,} rows")

# ---------------------------------------------------------------- 5. Training
def fit(feats):
    k = len(feats)
    types = ["c" if f in CATS else "q" for f in feats]
    def qdm(a, b, ref=None):
        return xgb.QuantileDMatrix(X[a:b, :k], y[a:b], feature_names=feats, feature_types=types,
                                   enable_categorical=True, ref=ref)
    dtr = qdm(0, i_va)
    dva = qdm(i_va, i_te, dtr)
    b = xgb.train(PARAMS, dtr, ROUNDS, evals=[(dva, "valid")], early_stopping_rounds=ES, verbose_eval=200)
    n_best = b.best_iteration + 1
    del dtr, dva, b
    final = xgb.train(PARAMS, qdm(0, i_te), int(n_best * 1.05))   # refit on the whole training set
    dte = xgb.DMatrix(X[i_te:, :k], feature_names=feats, feature_types=types, enable_categorical=True)
    return final, final.predict(dte), n_best

preds = {
    "naive_lag28": np.nan_to_num(X[i_te:, col["lag_28"]]),     # same day 4 weeks earlier (valid for both horizons)
    "naive_last": np.nan_to_num(X[i_te:, col["olag_1"]]),      # sales on the origin day; with HZ=1 this is "yesterday"
    "ma_28": np.nan_to_num(X[i_te:, col["rmean_28"]]),         # mean of the 28 days up to the origin
}
n_trees = {}
for name, feats in [("xgb_M3_full", M3), ("xgb_M2_price_event", M2), ("xgb_M1_calendar", M1)]:
    log(f"training {name} ({len(feats)} features)")
    model, preds[name], n_trees[name] = fit(feats)
    log(f"  {name}: {n_trees[name]} trees")
    if name == "xgb_M3_full":
        gain = pd.Series(model.get_score(importance_type="gain"))
        (gain / gain.sum()).sort_values(ascending=False).rename("gain_share").to_csv(OUT / "feature_importance.csv")
        model.save_model(OUT / "xgb_M3_full.json")

# ---------------------------------------------------------------- 6. Evaluation
te = pd.DataFrame({"id": sales.id.to_numpy()[sid[i_te:]], "d_num": d_num[i_te:], "y": y[i_te:], **preds})
te["date"] = date.to_numpy()[te.d_num - 1]
te["month"] = te.date.dt.to_period("M").astype(str)
te["window"] = (te.d_num - TRAIN_END - 1) // H + 1
te["snap"] = X[i_te:, col["snap"]].astype(int)
te["is_event"] = X[i_te:, col["is_event"]].astype(int)
te["store"] = sales.store_id.to_numpy()[sid[i_te:]]
te["dept"] = sales.dept_id.to_numpy()[sid[i_te:]]
te["cat"] = sales.cat_id.to_numpy()[sid[i_te:]]
te["state"] = sales.state_id.to_numpy()[sid[i_te:]]
# volume buckets: three tiers by mean daily sales in the training period
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)   # items launched only in the test period are all NaN
    train_mean = np.nanmean(Sn[:, :TRAIN_END], axis=1)
q = np.nanquantile(train_mean, [1 / 3, 2 / 3])
bucket = np.where(np.isnan(train_mean), "new_item",
                  np.where(train_mean < q[0], "low", np.where(train_mean < q[1], "mid", "high")))
te["volume"] = bucket[sid[i_te:]]

models = list(preds)

def scores(g):
    out = {}
    for m in models:
        e = g[m] - g.y
        out[m] = pd.Series({"RMSE": np.sqrt((e ** 2).mean()), "MAE": e.abs().mean(),
                            "WAPE": e.abs().sum() / g.y.sum(), "Bias": g[m].sum() / g.y.sum() - 1,
                            "R2": 1 - (e ** 2).sum() / ((g.y - g.y.mean()) ** 2).sum()})
    return pd.DataFrame(out)

overall = scores(te)
daily = te.groupby("date")[["y"] + models].sum()
open_days = daily[~((daily.index.month == 12) & (daily.index.day == 25))]   # stores close on Christmas; sales ≈ 0 would blow up MAPE
for m in models:
    e = daily[m] - daily.y
    overall.loc["Daily_total_RMSE", m] = np.sqrt((e ** 2).mean())
    overall.loc["Daily_total_MAPE_exXmas", m] = ((open_days[m] - open_days.y).abs() / open_days.y).mean()
    eo = open_days[m] - open_days.y
    overall.loc["Daily_total_R2_exXmas", m] = 1 - (eo ** 2).sum() / ((open_days.y - open_days.y.mean()) ** 2).sum()
overall = overall.T

# WRMSSE (official M5 metric): RMSE / WAPE treat every series alike, but a series that sells 300 units a day
# matters far more to the business than one that sells 0.3. WRMSSE first scales each series by its own
# historical volatility (the one-step error of a naive random walk), then weights it by "dollar sales in the
# 28 days before the forecast origin", and finally averages over 12 aggregation levels
# (total / state / store / category / department / ... / item-store).
# Weights and scaling denominators are recomputed for each 28-day rolling window, using only data before the window starts.
log("computing WRMSSE")
origins = list(range(TRAIN_END, D, H))

def as_matrix(v):
    """prediction vector on the long table -> (series, test days) matrix; cells before launch are 0 (actual sales are 0 too)"""
    M = np.zeros((n, D - TRAIN_END), np.float32)
    M[sid[i_te:], d_num[i_te:] - 1 - TRAIN_END] = np.nan_to_num(np.asarray(v, np.float32))
    return M

ev = WRMSSE(sales, S, build_dollars(S, P), origins, H)
wrmsse, wrmsse_level, wrmsse_window = ev.report({m: as_matrix(preds[m]) for m in models})
wrmsse_level.to_csv(OUT / "wrmsse_by_level.csv")
wrmsse_window.to_csv(OUT / "wrmsse_by_window.csv")
wrmsse_weights = ev.weight_table()
wrmsse_weights.to_csv(OUT / "wrmsse_weights.csv")
wrmsse_note = ev.note()
for c in wrmsse.columns:                        # WRMSSE / WRMSSE_distinct / WRMSSE_L12
    overall[c] = wrmsse[c]
del ev

overall["n_trees"] = pd.Series(n_trees)
overall.to_csv(OUT / "metrics_overall.csv")
daily.to_csv(OUT / "daily_totals.csv")

def wape_by(key):
    g = te.groupby(key)
    return pd.DataFrame({m: g.apply(lambda d: (d[m] - d.y).abs().sum() / d.y.sum(), include_groups=False)
                         for m in models})

breakdowns = {k: wape_by(k) for k in ["month", "window", "snap", "is_event", "state", "store", "cat", "dept", "volume"]}
for k, df in breakdowns.items():
    df.to_csv(OUT / f"wape_by_{k}.csv")

# descriptive statistics of the training period (for the findings)
trn = pd.DataFrame({"y": y[:i_te], "snap": X[:i_te, col["snap"]], "is_event": X[:i_te, col["is_event"]],
                    "wday": X[:i_te, col["wday"]], "dept": sales.dept_id.to_numpy()[sid[:i_te]]})
desc = {
    "snap_lift_by_dept": (trn.groupby(["dept", "snap"]).y.mean().unstack()
                          .pipe(lambda t: t[1] / t[0] - 1)).round(4).to_dict(),
    "event_lift": float(trn[trn.is_event == 1].y.mean() / trn[trn.is_event == 0].y.mean() - 1),
    "mean_by_wday(1=Sat)": trn.groupby("wday").y.mean().round(3).to_dict(),
    "share_zero_sales_test": float((te.y == 0).mean()),
}
(OUT / "train_descriptives.json").write_text(json.dumps(desc, indent=2, ensure_ascii=False))
te.drop(columns=["date"]).to_csv(OUT / "test_predictions.csv.gz", index=False)

pd.set_option("display.width", 160, "display.max_columns", 20)
print("\n===== Overall test-set metrics =====")
print(overall.round(4))
print("\n----- WRMSSE by level (dollar-weighted RMSSE) -----")
print(wrmsse_level.round(4))
print("\n----- WRMSSE by window (one 28-day rolling window each) -----")
print(wrmsse_window.round(4))
print("\n----- Number of series and weight concentration by level -----")
print(wrmsse_weights)
print("\nLower WRMSSE is better; 1 means the error equals a historical one-step naive random-walk forecast")
if wrmsse_note:
    print(wrmsse_note)
for k in ["volume", "snap", "is_event", "cat", "dept", "state", "store", "month"]:
    print(f"\n----- WAPE by {k} -----")
    print(breakdowns[k].round(4))
print("\n----- Feature importance (M3, share of gain) -----")
print(pd.read_csv(OUT / "feature_importance.csv", index_col=0).head(15).round(4))
print("\n----- Training-period descriptive statistics -----")
print(json.dumps(desc, indent=2, ensure_ascii=False))
log(f"done, results in {OUT}")
