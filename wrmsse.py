"""
WRMSSE —— the official M5 metric: RMSE weighted by dollar sales (Weighted) and scaled by each series' own volatility (Scaled)

Why it is needed (where RMSE / WAPE fall short):
  RMSE and WAPE treat every item-store series alike: a series that sells 300 units a day and one that sells
  0.3 a day count in the metric only through their absolute errors. But Walmart cares about money, and
  replenishment decisions are made at several levels: store, department, item. WRMSSE does two things:

    1) Scaled —— each series' error is first divided by its own historical "day-to-day volatility"
       (the one-step error of a naive random walk), which removes units so that big items and long-tail
       items can be added up; RMSSE = 1 means the forecast error equals the historical error of the naive
       "today equals yesterday" forecast.

    2) Weighted —— weight = the series' share of dollar sales (units x price) in the 28 days before the
       forecast origin. Series that sell more, or at higher prices, weigh more; this is where the
       "weight of sales" comes from. Weights are normalized to 1 within each level.

  The score is then computed at each of 12 aggregation levels and averaged (total / state / store /
  category / department / state x category / state x department / store x category / store x department /
  item / item x state / item x store), so a model can neither look good only on totals nor only on
  single items. Lower WRMSSE is better.

Two ways to use this file:
  1) As a module, computed right after training in xgb_forecast.py:
         from wrmsse import WRMSSE, build_dollars
         ev = WRMSSE(sales_meta, S, build_dollars(S, P), origins, horizon=28)
         table = ev.report({name: yhat_matrix for name, ...})

  2) As a script, to score a finished results folder afterwards (no retraining):
         python wrmsse.py                                  # default outputs/xgb_CA_FOODS
         python wrmsse.py outputs/xgb_ALL_ALL
         python wrmsse.py outputs/timesfm_ALL_ALL --weight units

Backtest convention: the test period rolls in 28-day windows (matching the window column of xgb_forecast.py).
Each window's weights and scaling denominators are recomputed from data before that window starts only, so
there is no future information, matching a real "re-forecast every 4 weeks" process. Final WRMSSE = the
mean of the windows' WRMSSE.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# 12 aggregation levels: (name, grouping keys). None = everything summed into one series
LEVELS = [
    ("L1_total", None),
    ("L2_state", ["state_id"]),
    ("L3_store", ["store_id"]),
    ("L4_cat", ["cat_id"]),
    ("L5_dept", ["dept_id"]),
    ("L6_state_cat", ["state_id", "cat_id"]),
    ("L7_state_dept", ["state_id", "dept_id"]),
    ("L8_store_cat", ["store_id", "cat_id"]),
    ("L9_store_dept", ["store_id", "dept_id"]),
    ("L10_item", ["item_id"]),
    ("L11_item_state", ["item_id", "state_id"]),
    ("L12_item_store", ["item_id", "store_id"]),
]
SERIES_LEVEL = "L12_item_store"      # the finest level, reported on its own for comparison with WAPE
ROWS = 2048                          # rows per chunk, bounds the memory of intermediate results


def build_dollars(S, P):
    """dollar sales matrix = units x that day's price; cells without a price (not on the shelf) are 0"""
    return np.where(np.isnan(P), 0.0, S * np.nan_to_num(P)).astype(np.float32)


def _group_codes(meta, keys):
    """returns (group index of each row, number of groups)"""
    if keys is None:
        return np.zeros(len(meta), np.int64), 1
    key = meta[keys[0]].astype(str)
    for k in keys[1:]:
        key = key + "|" + meta[k].astype(str)
    codes, uniques = pd.factorize(key, sort=True)
    return codes.astype(np.int64), len(uniques)


def _aggregate(A, codes, n_groups):
    """sum a (series, day) matrix by group into (group, day)"""
    if n_groups == len(A) and np.array_equal(codes, np.arange(len(A))):
        return A                                   # finest level, nothing to do
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(n_groups))
    return np.add.reduceat(A[order], starts, axis=0)


def _cum_diff2(A, origins):
    """C[:, k] = sum_{t=1}^{origins[k]-1} (A_t - A_{t-1})^2, 0-based day index, accumulated in chunks to save memory"""
    n = len(A)
    out = np.zeros((n, len(origins)), np.float64)
    run = np.zeros(n)
    lo = 1
    for k, o in enumerate(origins):
        if o > lo:
            for r in range(0, n, ROWS):
                d = np.diff(A[r:r + ROWS, lo - 1:o], axis=1).astype(np.float64)
                run[r:r + ROWS] += np.einsum("ij,ij->i", d, d)
        out[:, k] = run
        lo = max(lo, o)
    return out


class WRMSSE:
    """
    meta     : DataFrame, one row per series, with item_id / dept_id / cat_id / store_id / state_id
    sales    : (series, day) units matrix covering the whole history
    dollars  : (series, day) dollar sales matrix (from build_dollars); may be None when weight="units"
    origins  : start of each forecast window, 0-based day index (d_1554 -> 1553)
    horizon  : window length, default 28
    weight   : "dollars" (official M5) or "units"
    """

    def __init__(self, meta, sales, dollars, origins, horizon=28, weight="dollars", weight_days=28):
        self.origins = [int(o) for o in origins]
        self.horizon = int(horizon)
        self.t0 = self.origins[0]                  # first test day; column 0 of the scoring matrices
        self.D = sales.shape[1]
        self.weight = weight
        self.window_names = [f"W{k + 1}" for k in range(len(self.origins))]
        src = sales if weight == "units" else dollars
        if src is None:
            raise ValueError("weight='dollars' needs the dollars matrix")

        self.levels = {}
        self.duplicates = {}          # a subset (e.g. only CA x FOODS) can make some levels identical
        for name, keys in LEVELS:
            codes, n_g = _group_codes(meta, keys)
            for other, L in self.levels.items():
                if np.array_equal(codes, L["codes"]):
                    self.duplicates[name] = self.duplicates.get(other, other)
                    break
            A = _aggregate(sales, codes, n_g).astype(np.float32)
            V = _aggregate(src, codes, n_g)

            # --- scaling denominator: historical mean squared error of the naive one-step forecast, from the series' first sale on
            first = np.argmax(A > 0, axis=1)
            alive = A.any(axis=1)
            # the jump on the launch day (0 -> first sale) is left out of the denominator; with first == 0 there is no such difference
            first_sq = np.where(alive & (first > 0), A[np.arange(n_g), first].astype(np.float64) ** 2, 0.0)
            C = _cum_diff2(A, self.origins)
            counts = np.asarray(self.origins)[None, :] - first[:, None] - 1   # number of usable differences
            with np.errstate(invalid="ignore", divide="ignore"):
                scale = (C - first_sq[:, None]) / counts
                scale[(counts <= 0) | ~alive[:, None] | (scale <= 0)] = np.nan

            # --- weights: share of dollar sales (or units) in the weight_days days before the window origin, normalized within the level
            w = np.zeros((n_g, len(self.origins)))
            for k, o in enumerate(self.origins):
                w[:, k] = V[:, max(0, o - weight_days):o].sum(axis=1)
            w[np.isnan(scale)] = 0.0                       # series with an invalid denominator are left out
            totals = w.sum(axis=0, keepdims=True)
            w = np.divide(w, totals, out=np.zeros_like(w), where=totals > 0)

            self.levels[name] = {
                "codes": codes, "n_groups": n_g, "scale": scale, "weight": w,
                "Y": np.ascontiguousarray(A[:, self.t0:]),  # actual values in the test period
                "dropped": int((np.isnan(scale)).all(axis=1).sum()),
            }
            del A, V, C

    # ------------------------------------------------------------------ scoring
    def _rmsse(self, level, F):
        """returns RMSSE by (group, window)"""
        L = self.levels[level]
        Y = L["Y"]
        out = np.full((L["n_groups"], len(self.origins)), np.nan)
        for k, o in enumerate(self.origins):
            a, b = o - self.t0, min(o + self.horizon, self.D) - self.t0
            e = (F[:, a:b] - Y[:, a:b]).astype(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[:, k] = np.sqrt(np.einsum("ij,ij->i", e, e) / (b - a) / L["scale"][:, k])
        return out

    def by_level_window(self, yhat):
        """one model -> weighted RMSSE by (12 levels x windows)"""
        rows = {}
        for name in self.levels:
            L = self.levels[name]
            F = _aggregate(np.asarray(yhat, np.float32), L["codes"], L["n_groups"])
            r = self._rmsse(name, F)
            rows[name] = np.nansum(np.where(L["weight"] > 0, r * L["weight"], 0.0), axis=0)
        return pd.DataFrame(rows, index=self.window_names).T

    def score(self, yhat):
        """full results for one model"""
        lw = self.by_level_window(yhat)
        distinct = [l for l in lw.index if l not in self.duplicates]
        return {
            "by_level_window": lw,
            "by_window": lw.mean(axis=0),                       # mean over the 12 levels
            "by_level": lw.mean(axis=1),                        # mean over the windows
            "WRMSSE": float(lw.mean(axis=0).mean()),            # official M5 convention: always 12 levels
            # on a subset some levels collapse into the same one (e.g. with CA only, state = total), and the
            # official convention counts them twice; this column averages distinct levels only (equal on ALL/ALL)
            "WRMSSE_distinct": float(lw.loc[distinct].mean(axis=0).mean()),
            "WRMSSE_L12": float(lw.loc[SERIES_LEVEL].mean()),   # item-store level only
        }

    def report(self, preds):
        """
        preds: {model name: (series, test days) prediction matrix}
        returns (summary, by_level, by_window), three tables ready to save as csv
        """
        scores = {name: self.score(yhat) for name, yhat in preds.items()}
        summary = pd.DataFrame(
            {n: {k: v for k, v in s.items() if isinstance(v, float)} for n, s in scores.items()}
        ).T.rename_axis("model")
        by_level = pd.DataFrame({n: s["by_level"] for n, s in scores.items()}).rename_axis("level")
        by_window = pd.DataFrame({n: s["by_window"] for n, s in scores.items()}).rename_axis("window")
        return summary, by_level, by_window

    def weight_table(self):
        """for each level: number of series / number scored / weight concentration (weight share of the top 10% of series)"""
        rows = {}
        for name, L in self.levels.items():
            w = L["weight"].mean(axis=1)
            top = np.sort(w)[::-1]
            k = max(1, int(np.ceil(0.1 * len(top))))
            rows[name] = {
                "series": L["n_groups"],
                "scored": int((w > 0).sum()),
                "dropped_no_scale": L["dropped"],
                "weight_share_top10pct": round(float(top[:k].sum()), 4),
                "same_as": self.duplicates.get(name, ""),
            }
        return pd.DataFrame(rows).T.rename_axis("level")

    def note(self):
        """warning for subsets: collapsed levels make one level count twice"""
        if not self.duplicates:
            return ""
        dup = ", ".join(f"{k} = {v}" for k, v in self.duplicates.items())
        return (f"Note: in this subset {len(self.duplicates)} level(s) collapse into other levels ({dup}); "
                f"they are counted twice in the mean over 12 levels. This does not happen on the full ALL/ALL run.")


# ====================================================================== script entry point
def _load_run(out_dir, data_dir, weight, train_end=1553, last_day=1941, horizon=28,
              min_selling_share=None):
    """score WRMSSE for a finished results folder"""
    # read only the header first to find the model columns, then read what is needed —— the full panel has
    # 9 million rows, and reading string columns such as store/dept/cat as well would take several GB more
    head = pd.read_csv(out_dir / "test_predictions.csv.gz", nrows=0)
    models = [c for c in head.columns
              if c not in {"id", "d_num", "y", "date", "month", "window", "snap", "is_event",
                           "store", "dept", "cat", "state", "volume"}]
    if not models:
        raise SystemExit(f"no prediction columns found in {out_dir}")
    te = pd.read_csv(out_dir / "test_predictions.csv.gz", usecols=["id", "d_num", "y"] + models,
                     dtype={"d_num": np.int32, "y": np.float32,
                            **{m: np.float32 for m in models}})
    ids = pd.Index(te.id.unique())

    day_cols = [f"d_{i}" for i in range(1, last_day + 1)]
    sales = pd.read_csv(data_dir / "sales_train_evaluation.csv",
                        dtype={c: np.int16 for c in day_cols})
    sales = sales[sales.id.isin(ids)].reset_index(drop=True)
    if len(sales) != len(ids):
        raise SystemExit(f"the predictions have {len(ids)} series, but only {len(sales)} match the raw data")

    # Keep only "regular sellers": items with sales on more than min_selling_share of all last_day days.
    # Used to align with results that were run on a regular-seller subset only —— filtering changes the
    # level structure and the weights, so it has to happen here, not as an adjustment to the results afterwards.
    if min_selling_share is not None:
        share = (sales[day_cols].to_numpy() > 0).mean(axis=1)
        keep = share > min_selling_share
        print(f"regular-seller filter (sales on >{min_selling_share:.0%} of days): "
              f"keeping {int(keep.sum()):,} / {len(sales):,} series")
        sales = sales[keep].reset_index(drop=True)
        te = te[te.id.isin(set(sales.id))].reset_index(drop=True)

    S = sales[day_cols].to_numpy(np.float32)
    meta = sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]]

    cal = pd.read_csv(data_dir / "calendar.csv").iloc[:last_day]
    weeks = np.sort(cal.wm_yr_wk.unique())
    prices = pd.read_csv(data_dir / "sell_prices.csv")
    prices = prices[prices.wm_yr_wk <= weeks[-1]]
    # match on the combined (store, item) key, not on store and item separately with isin ——
    # filtering them separately lets in combinations that do not exist, especially after the regular-seller filter
    row_of = pd.Series(np.arange(len(meta)), index=meta.store_id + "|" + meta.item_id)
    pr_row = row_of.reindex(prices.store_id + "|" + prices.item_id).to_numpy()
    hit = ~np.isnan(pr_row)
    pr_row = pr_row[hit].astype(int)
    prices = prices[hit]
    Pw = np.full((len(meta), len(weeks)), np.nan, np.float32)
    Pw[pr_row, np.searchsorted(weeks, prices.wm_yr_wk.to_numpy())] = prices.sell_price.to_numpy()
    P = Pw[:, np.searchsorted(weeks, cal.wm_yr_wk.to_numpy())]
    del prices, Pw

    # long table -> (series, test days) matrix; missing cells = not on the shelf, where actual sales are 0 anyway
    row = pd.Index(meta.id).get_indexer(te.id)
    cols = te.d_num.to_numpy() - 1 - train_end
    T = last_day - train_end
    preds = {}
    for m in models:
        M = np.zeros((len(meta), T), np.float32)
        M[row, cols] = np.nan_to_num(te[m].to_numpy(np.float32))
        preds[m] = M

    origins = list(range(train_end, last_day, horizon))
    ev = WRMSSE(meta, S, build_dollars(S, P), origins, horizon, weight=weight)
    return ev, preds, models


def main(argv=None):
    import argparse

    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="score WRMSSE for existing prediction results")
    ap.add_argument("out_dir", nargs="?", default=str(here / "outputs" / "xgb_CA_FOODS"),
                    help="results folder; must contain test_predictions.csv.gz")
    ap.add_argument("--data", default=str(here), help="folder holding the three raw csv files")
    ap.add_argument("--weight", default="dollars", choices=["dollars", "units"])
    ap.add_argument("--min-selling-share", type=float, default=None,
                    help="regular sellers only: share of days with sales must exceed this value (0.5 to align with the regular-seller subset)")
    a = ap.parse_args(argv)

    out_dir, data_dir = Path(a.out_dir), Path(a.data)
    ev, preds, models = _load_run(out_dir, data_dir, a.weight,
                                  min_selling_share=a.min_selling_share)
    summary, by_level, by_window = ev.report(preds)
    weights = ev.weight_table()

    tag = "" if a.min_selling_share is None else f"_regular{int(100 * a.min_selling_share)}"
    summary.to_csv(out_dir / f"metrics_wrmsse{tag}.csv")
    by_level.to_csv(out_dir / f"wrmsse_by_level{tag}.csv")
    by_window.to_csv(out_dir / f"wrmsse_by_window{tag}.csv")
    weights.to_csv(out_dir / f"wrmsse_weights{tag}.csv")

    # also merge WRMSSE back into the overall metrics table
    mpath = out_dir / "metrics_overall.csv"
    if mpath.exists() and a.min_selling_share is None:   # subset results do not overwrite the full metrics table
        overall = pd.read_csv(mpath, index_col=0)
        for c in summary.columns:
            overall[c] = summary[c]
        overall.to_csv(mpath)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    order = summary.WRMSSE.sort_values().index
    print(f"===== WRMSSE ({out_dir.name}, weight = {a.weight}) =====")
    print(summary.loc[order].round(4).to_string())
    print("\n----- By level (weighted RMSSE at each of the 12 levels) -----")
    print(by_level[order].round(4).to_string())
    print("\n----- By window (one 28-day rolling window each) -----")
    print(by_window[order].round(4).to_string())
    print("\n----- Number of series and weight concentration by level -----")
    print(weights.round(4).to_string())
    print("\nWRMSSE      = mean over 12 levels x windows; 1.0 = same as the historical error of a naive random walk")
    print("WRMSSE_L12  = the finest item-store level only")
    if ev.note():
        print(ev.note())
    print(f"results written to {out_dir}")
    return summary


if __name__ == "__main__":
    main()
