"""
WRMSSE —— M5 官方指标: 按销售额加权 (Weighted) + 按序列自身波动缩放 (Scaled) 的 RMSE

为什么需要它 (RMSE / WAPE 不够用的地方):
  RMSE 和 WAPE 把每条 item-store 序列一视同仁, 一条日均卖 300 件的序列和一条日均
  卖 0.3 件的序列在指标里的分量只取决于它们的绝对误差。但 Walmart 关心的是钱,
  而且补货决策在门店 / 部门 / 单品等不同层级上都要做。WRMSSE 做两件事:

    1) Scaled —— 每条序列的误差先除以它自己历史上的"天到天波动"
       (naive 随机游走的一步预测误差), 量纲被消掉, 大单品和长尾单品可以相加;
       RMSSE = 1 表示预测误差和"今天等于昨天"这种朴素预测在历史上的误差一样大。

    2) Weighted —— 权重 = 该序列在预测起点前 28 天的销售额 (销量 x 售价) 占比。
       卖得多 / 卖得贵的序列权重大, 这就是 "weight of sales" 的来源。
       权重在每个层级内部各自归一化到 1。

  再在 12 个聚合层级上各算一遍取平均 (总体 / 州 / 门店 / 品类 / 部门 / 州x品类 /
  州x部门 / 店x品类 / 店x部门 / 单品 / 单品x州 / 单品x店), 所以模型既不能只顾
  总量好看, 也不能只顾单品。WRMSSE 越小越好。

本文件的两种用法:
  1) 作为模块, 在 xgb_forecast.py 里训练完直接算:
         from wrmsse import WRMSSE, build_dollars
         ev = WRMSSE(sales_meta, S, build_dollars(S, P), origins, horizon=28)
         table = ev.report({name: yhat_matrix for name, ...})

  2) 作为脚本, 给已经跑完的结果目录补算 (不用重新训练):
         python wrmsse.py                                  # 默认 outputs/xgb_CA_FOODS
         python wrmsse.py outputs/xgb_ALL_ALL
         python wrmsse.py outputs/timesfm_ALL_ALL --weight units

回测口径: 测试期按 28 天一个窗口滚动 (和 xgb_forecast.py 的 window 列一致)。
每个窗口的权重和缩放分母都只用该窗口起点之前的数据重新算一遍, 所以没有未来信息,
和真实的"每 4 周重新预测一次"一致。最终 WRMSSE = 各窗口 WRMSSE 的平均。
"""
from pathlib import Path

import numpy as np
import pandas as pd

# 12 个聚合层级: (名字, 分组键). None = 全部加总成一条序列
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
SERIES_LEVEL = "L12_item_store"      # 最细的那一层, 单独报一个数方便和 WAPE 对照
ROWS = 2048                          # 分块行数, 控制中间结果的内存


def build_dollars(S, P):
    """销售额矩阵 = 销量 x 当日售价; 没有价格 (未上架) 的格子记 0"""
    return np.where(np.isnan(P), 0.0, S * np.nan_to_num(P)).astype(np.float32)


def _group_codes(meta, keys):
    """返回 (每行所属组的编号, 组数)"""
    if keys is None:
        return np.zeros(len(meta), np.int64), 1
    key = meta[keys[0]].astype(str)
    for k in keys[1:]:
        key = key + "|" + meta[k].astype(str)
    codes, uniques = pd.factorize(key, sort=True)
    return codes.astype(np.int64), len(uniques)


def _aggregate(A, codes, n_groups):
    """把 (序列, 天) 矩阵按组加总成 (组, 天)"""
    if n_groups == len(A) and np.array_equal(codes, np.arange(len(A))):
        return A                                   # 最细层级, 不用动
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(n_groups))
    return np.add.reduceat(A[order], starts, axis=0)


def _cum_diff2(A, origins):
    """C[:, k] = sum_{t=1}^{origins[k]-1} (A_t - A_{t-1})^2, 0 基的日索引, 分块累加省内存"""
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
    meta     : DataFrame, 每行一条序列, 需要 item_id / dept_id / cat_id / store_id / state_id
    sales    : (序列, 天) 销量矩阵, 覆盖全部历史
    dollars  : (序列, 天) 销售额矩阵 (build_dollars 的结果); weight="units" 时可以传 None
    origins  : 每个预测窗口的起点, 0 基日索引 (d_1554 -> 1553)
    horizon  : 窗口长度, 默认 28
    weight   : "dollars" (M5 官方) 或 "units"
    """

    def __init__(self, meta, sales, dollars, origins, horizon=28, weight="dollars", weight_days=28):
        self.origins = [int(o) for o in origins]
        self.horizon = int(horizon)
        self.t0 = self.origins[0]                  # 测试期第一天, 评分矩阵以它为第 0 列
        self.D = sales.shape[1]
        self.weight = weight
        self.window_names = [f"W{k + 1}" for k in range(len(self.origins))]
        src = sales if weight == "units" else dollars
        if src is None:
            raise ValueError("weight='dollars' 需要传入 dollars 矩阵")

        self.levels = {}
        self.duplicates = {}          # 子集 (比如只跑 CA x FOODS) 会让某些层级退化成同一个
        for name, keys in LEVELS:
            codes, n_g = _group_codes(meta, keys)
            for other, L in self.levels.items():
                if np.array_equal(codes, L["codes"]):
                    self.duplicates[name] = self.duplicates.get(other, other)
                    break
            A = _aggregate(sales, codes, n_g).astype(np.float32)
            V = _aggregate(src, codes, n_g)

            # --- 缩放分母: 历史上 naive 一步预测的均方误差, 从该序列第一次有销量算起
            first = np.argmax(A > 0, axis=1)
            alive = A.any(axis=1)
            # 上架那天的跳变 (0 -> 第一次销量) 不算进分母; first == 0 时本来就没有这个差分
            first_sq = np.where(alive & (first > 0), A[np.arange(n_g), first].astype(np.float64) ** 2, 0.0)
            C = _cum_diff2(A, self.origins)
            counts = np.asarray(self.origins)[None, :] - first[:, None] - 1   # 可用的差分个数
            with np.errstate(invalid="ignore", divide="ignore"):
                scale = (C - first_sq[:, None]) / counts
                scale[(counts <= 0) | ~alive[:, None] | (scale <= 0)] = np.nan

            # --- 权重: 窗口起点前 weight_days 天的销售额 (或销量) 占比, 层级内归一化
            w = np.zeros((n_g, len(self.origins)))
            for k, o in enumerate(self.origins):
                w[:, k] = V[:, max(0, o - weight_days):o].sum(axis=1)
            w[np.isnan(scale)] = 0.0                       # 分母无效的序列不参与
            totals = w.sum(axis=0, keepdims=True)
            w = np.divide(w, totals, out=np.zeros_like(w), where=totals > 0)

            self.levels[name] = {
                "codes": codes, "n_groups": n_g, "scale": scale, "weight": w,
                "Y": np.ascontiguousarray(A[:, self.t0:]),  # 测试期实际值
                "dropped": int((np.isnan(scale)).all(axis=1).sum()),
            }
            del A, V, C

    # ------------------------------------------------------------------ 打分
    def _rmsse(self, level, F):
        """返回 (组, 窗口) 的 RMSSE"""
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
        """一个模型 -> (12 层级 x 窗口) 的加权 RMSSE"""
        rows = {}
        for name in self.levels:
            L = self.levels[name]
            F = _aggregate(np.asarray(yhat, np.float32), L["codes"], L["n_groups"])
            r = self._rmsse(name, F)
            rows[name] = np.nansum(np.where(L["weight"] > 0, r * L["weight"], 0.0), axis=0)
        return pd.DataFrame(rows, index=self.window_names).T

    def score(self, yhat):
        """一个模型的完整结果"""
        lw = self.by_level_window(yhat)
        distinct = [l for l in lw.index if l not in self.duplicates]
        return {
            "by_level_window": lw,
            "by_window": lw.mean(axis=0),                       # 12 个层级平均
            "by_level": lw.mean(axis=1),                        # 各窗口平均
            "WRMSSE": float(lw.mean(axis=0).mean()),            # M5 官方口径: 固定 12 个层级
            # 跑子集时有些层级会退化成同一个 (比如只有 CA 时 state = total), 官方口径会把它重复计数;
            # 这一列只对不重复的层级取平均, 全量 ALL/ALL 时两者相同
            "WRMSSE_distinct": float(lw.loc[distinct].mean(axis=0).mean()),
            "WRMSSE_L12": float(lw.loc[SERIES_LEVEL].mean()),   # 只看 item-store 层
        }

    def report(self, preds):
        """
        preds: {模型名: (序列, 测试期天数) 的预测矩阵}
        返回 (summary, by_level, by_window), 三张可以直接存 csv 的表
        """
        scores = {name: self.score(yhat) for name, yhat in preds.items()}
        summary = pd.DataFrame(
            {n: {k: v for k, v in s.items() if isinstance(v, float)} for n, s in scores.items()}
        ).T.rename_axis("model")
        by_level = pd.DataFrame({n: s["by_level"] for n, s in scores.items()}).rename_axis("level")
        by_window = pd.DataFrame({n: s["by_window"] for n, s in scores.items()}).rename_axis("window")
        return summary, by_level, by_window

    def weight_table(self):
        """每个层级的序列条数 / 有效条数 / 权重集中度 (最大的 10% 序列占多少权重)"""
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
        """跑子集时的提醒: 退化掉的层级会让某一层被重复计算"""
        if not self.duplicates:
            return ""
        dup = ", ".join(f"{k} = {v}" for k, v in self.duplicates.items())
        return (f"注意: 当前子集下有 {len(self.duplicates)} 个层级退化成了别的层级 ({dup}), "
                f"它们在 12 个层级的平均里被重复计了一次; 跑全量 ALL/ALL 时不会出现。")


# ====================================================================== 脚本入口
def _load_run(out_dir, data_dir, weight, train_end=1553, last_day=1941, horizon=28,
              min_selling_share=None):
    """给一个已经跑完的结果目录补算 WRMSSE"""
    # 先只读表头认出模型列, 再按需读 —— 全量有 900 万行, 把 store/dept/cat 这些
    # 字符串列一起读进来会多吃好几 GB
    head = pd.read_csv(out_dir / "test_predictions.csv.gz", nrows=0)
    models = [c for c in head.columns
              if c not in {"id", "d_num", "y", "date", "month", "window", "snap", "is_event",
                           "store", "dept", "cat", "state", "volume"}]
    if not models:
        raise SystemExit(f"{out_dir} 里没找到预测列")
    te = pd.read_csv(out_dir / "test_predictions.csv.gz", usecols=["id", "d_num", "y"] + models,
                     dtype={"d_num": np.int32, "y": np.float32,
                            **{m: np.float32 for m in models}})
    ids = pd.Index(te.id.unique())

    day_cols = [f"d_{i}" for i in range(1, last_day + 1)]
    sales = pd.read_csv(data_dir / "sales_train_evaluation.csv",
                        dtype={c: np.int16 for c in day_cols})
    sales = sales[sales.id.isin(ids)].reset_index(drop=True)
    if len(sales) != len(ids):
        raise SystemExit(f"预测里有 {len(ids)} 条序列, 原始数据里只匹配到 {len(sales)} 条")

    # 只保留"常销品": 在全部 last_day 天里有超过 min_selling_share 比例的天数卖出过东西。
    # 用来对齐别人只跑常销品子集的结果 —— 筛完之后层级结构和权重都会跟着变,
    # 所以必须在这里筛, 不能事后对结果做加减。
    if min_selling_share is not None:
        share = (sales[day_cols].to_numpy() > 0).mean(axis=1)
        keep = share > min_selling_share
        print(f"常销品筛选 (>{min_selling_share:.0%} 的天数有销量): "
              f"保留 {int(keep.sum()):,} / {len(sales):,} 条序列")
        sales = sales[keep].reset_index(drop=True)
        te = te[te.id.isin(set(sales.id))].reset_index(drop=True)

    S = sales[day_cols].to_numpy(np.float32)
    meta = sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]]

    cal = pd.read_csv(data_dir / "calendar.csv").iloc[:last_day]
    weeks = np.sort(cal.wm_yr_wk.unique())
    prices = pd.read_csv(data_dir / "sell_prices.csv")
    prices = prices[prices.wm_yr_wk <= weeks[-1]]
    # 按 (门店, 单品) 组合键匹配, 不能对 store 和 item 分别 isin ——
    # 分开筛会放进一些实际不存在的组合, 筛过常销品之后尤其明显
    row_of = pd.Series(np.arange(len(meta)), index=meta.store_id + "|" + meta.item_id)
    pr_row = row_of.reindex(prices.store_id + "|" + prices.item_id).to_numpy()
    hit = ~np.isnan(pr_row)
    pr_row = pr_row[hit].astype(int)
    prices = prices[hit]
    Pw = np.full((len(meta), len(weeks)), np.nan, np.float32)
    Pw[pr_row, np.searchsorted(weeks, prices.wm_yr_wk.to_numpy())] = prices.sell_price.to_numpy()
    P = Pw[:, np.searchsorted(weeks, cal.wm_yr_wk.to_numpy())]
    del prices, Pw

    # 长表 -> (序列, 测试期天数) 矩阵; 没出现的格子 = 未上架, 实际销量本来就是 0
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
    ap = argparse.ArgumentParser(description="给已有的预测结果补算 WRMSSE")
    ap.add_argument("out_dir", nargs="?", default=str(here / "outputs" / "xgb_CA_FOODS"),
                    help="结果目录, 里面要有 test_predictions.csv.gz")
    ap.add_argument("--data", default=str(here), help="三个原始 csv 所在目录")
    ap.add_argument("--weight", default="dollars", choices=["dollars", "units"])
    ap.add_argument("--min-selling-share", type=float, default=None,
                    help="只算常销品: 有销量的天数占比要高于这个值 (对齐别人的 0.5 子集时用)")
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

    # 顺手把 WRMSSE 并回总指标表
    mpath = out_dir / "metrics_overall.csv"
    if mpath.exists() and a.min_selling_share is None:   # 子集结果不回写全量指标表
        overall = pd.read_csv(mpath, index_col=0)
        for c in summary.columns:
            overall[c] = summary[c]
        overall.to_csv(mpath)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    order = summary.WRMSSE.sort_values().index
    print(f"===== WRMSSE ({out_dir.name}, 权重 = {a.weight}) =====")
    print(summary.loc[order].round(4).to_string())
    print("\n----- 分层级 (12 个层级的加权 RMSSE) -----")
    print(by_level[order].round(4).to_string())
    print("\n----- 分窗口 (28 天一个滚动窗口) -----")
    print(by_window[order].round(4).to_string())
    print("\n----- 各层级的序列数与权重集中度 -----")
    print(weights.round(4).to_string())
    print("\nWRMSSE      = 12 个层级 x 各窗口的平均; 1.0 = 和朴素随机游走的历史误差相当")
    print("WRMSSE_L12  = 只看最细的 item-store 层级")
    if ev.note():
        print(ev.note())
    print(f"结果已写入 {out_dir}")
    return summary


if __name__ == "__main__":
    main()
