"""
XGBoost 销量预测 —— M5 (Walmart) 数据, 默认 CA 州 FOODS 子集

时间切分 (按时间, 不打乱):
    训练  d_1    – d_1553  (80%, 2011-01-29 ~ 2015-04-30)
    测试  d_1554 – d_1941  (20%, 2015-05-01 ~ 2016-05-22)
    训练集最后 90 天 (d_1464–d_1553) 用来 early stopping 选树的数量, 选好后用全部训练集重训

预测设定: 滚动 28 天预测。所有历史销量特征都滞后 >= 28 天,
    所以测试期任意一天用到的信息在它所在的 28 天窗口开始前都已知, 整个测试期可以一次性预测。

消融实验 (特征一层层加):
    M1 = 日历 + ID
    M2 = M1 + 价格 / SNAP / 事件
    M3 = M2 + 滞后销量 / 滚动统计

评估指标:
    RMSE / MAE / WAPE / R2  —— 每条 item-store 序列一视同仁
    WRMSSE (M5 官方指标)    —— 按"起点前 28 天销售额"加权 + 按序列自身波动缩放,
                               在 12 个聚合层级上取平均; 实现在 wrmsse.py

用法:
    python xgb_forecast.py                 # 默认 CA FOODS
    python xgb_forecast.py --quick         # 少量树, 只检查流程
    python xgb_forecast.py --state TX --cat HOBBIES
    python xgb_forecast.py --state ALL --cat ALL --device cuda   # 全量, 集群 GPU 上跑
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
    import psutil                     # 只用来做内存护栏, 集群上没装就跳过
except ImportError:
    psutil = None

ap = argparse.ArgumentParser()
ap.add_argument("--state", default="CA", help="CA / TX / WI / ALL")
ap.add_argument("--cat", default="FOODS", help="FOODS / HOBBIES / HOUSEHOLD / ALL")
ap.add_argument("--device", default="cpu", help="cpu 或 cuda")
ap.add_argument("--data", default=str(Path(__file__).resolve().parent), help="三个 csv 所在的目录")
ap.add_argument("--quick", action="store_true", help="少量树, 只用来检查流程")
ap.add_argument("--horizon", type=int, default=28, choices=[1, 28],
                help="28 = 每 28 天一个预测起点 (M5 口径); 1 = 每天重新预测 (对齐 OLS)")
args = ap.parse_args()

DATA = Path(args.data)
_hz = "" if args.horizon == 28 else f"_h{args.horizon}"
OUT = Path(__file__).resolve().parent / "outputs" / f"xgb_{args.state}_{args.cat}{_hz}{'_quick' if args.quick else ''}"
OUT.mkdir(parents=True, exist_ok=True)

D = 1941            # 有销量的最后一天 (d_1942–d_1969 只有日历, 没有销量)
TRAIN_END = 1553    # 80% / 20%; 必须和我们自己的 TimesFM 跑法 (split=1553) 一致
VALID_START = 1464  # 训练集最后 90 天做 early stopping
H = 28              # WRMSSE 的评分窗口长度, 固定 28 天 (M5 定义)
HZ = args.horizon   # 预测步长: 每隔 HZ 天重新预测一次

ROUNDS, ES = (40, 10) if args.quick else (3000, 100)
PARAMS = dict(
    objective="reg:tweedie", tweedie_variance_power=1.2,  # 大量 0 的计数数据
    eta=0.08, max_depth=8, subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", eval_metric="rmse", seed=42, device=args.device,
    nthread=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())),
)

t0 = time.time()
def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

# ---------------------------------------------------------------- 1. 读数据
day_cols = [f"d_{i}" for i in range(1, D + 1)]
sales = pd.read_csv(DATA / "sales_train_evaluation.csv", dtype={c: np.int16 for c in day_cols})
if args.state != "ALL":
    sales = sales[sales.state_id == args.state]
if args.cat != "ALL":
    sales = sales[sales.cat_id == args.cat]
sales = sales.reset_index(drop=True)
n = len(sales)
S = sales[day_cols].to_numpy(np.float32)                    # (序列, 天)
sales = sales[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]]

cal = pd.read_csv(DATA / "calendar.csv").iloc[:D]
assert (cal.d.to_numpy() == np.array(day_cols)).all()
date = pd.to_datetime(cal.date)

prices = pd.read_csv(DATA / "sell_prices.csv")
prices = prices[prices.store_id.isin(sales.store_id.unique()) & prices.item_id.isin(sales.item_id.unique())]
log(f"{n} 条序列 x {D} 天, 价格记录 {len(prices)} 行")

# 每周价格 -> 每日价格矩阵 (序列, 天)
weeks = np.sort(cal.wm_yr_wk.unique())
prices = prices[prices.wm_yr_wk <= weeks[-1]]
row_of = pd.Series(np.arange(n), index=sales.store_id + "|" + sales.item_id)
pr_row = row_of.reindex(prices.store_id + "|" + prices.item_id).to_numpy().astype(int)
Pw = np.full((n, len(weeks)), np.nan, np.float32)
Pw[pr_row, np.searchsorted(weeks, prices.wm_yr_wk.to_numpy())] = prices.sell_price.to_numpy()
P = Pw[:, np.searchsorted(weeks, cal.wm_yr_wk.to_numpy())]
del prices, Pw

valid = ~np.isnan(P)                      # 有价格 = 已上架; 上架前的 0 不是真实销量
Sn = np.where(valid, S, np.nan).astype(np.float32)

# ---------------------------------------------------------------- 2. 转成长表
# 行按 (天, 序列) 排列, 这样训练 / 验证 / 测试都是连续切片, 不用复制数据
mask = valid.T.ravel()
N = int(mask.sum())
d_num = np.repeat(np.arange(1, D + 1), n)[mask]
sid = np.tile(np.arange(n), D)[mask]
y = S.T.ravel()[mask]
log(f"长表 {N:,} 行 (已删除上架前的行)")

M1 = ["item_id", "dept_id", "store_id", "wday", "month", "year", "dom", "week"]
M2 = M1 + ["sell_price", "price_rel_max", "price_rel_mean", "price_chg_wk",
           "snap", "is_event", "event_name", "event_type"]
M3 = M2 + ["days_ahead", "lag_28",
           "olag_1", "olag_2", "olag_7", "olag_14", "olag_28", "olag_364",
           "rmean_7", "rmean_28", "rmean_56", "rmean_180", "rstd_28"]
CATS = {"item_id", "dept_id", "store_id", "event_name", "event_type"}
col = {f: j for j, f in enumerate(M3)}

# 内存护栏: X 是整个流程最大的一块。本机内存不够时 Windows 不会报错, 而是改用页面文件,
# 结果是整机卡住几小时而不是干脆失败 —— 所以在分配之前先拦一道。
# 集群上 (SLURM) 跳过: psutil 看到的是整个节点的内存, 不是这个 job 的配额。
x_gb = N * len(M3) * 4 / 2 ** 30
peak_gb = x_gb * 2.4                      # X + 滚动统计的 float64 临时量 + DMatrix + 评估表
if psutil is not None:
    free_gb = psutil.virtual_memory().available / 2 ** 30
    log(f"X 需要 {x_gb:.1f} GB, 整个流程峰值约 {peak_gb:.0f} GB, 当前可用 {free_gb:.0f} GB")
    if free_gb < peak_gb and not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit(
            f"\n内存不够: 峰值约需 {peak_gb:.0f} GB, 现在只有 {free_gb:.0f} GB 可用。\n"
            f"硬跑下去 Windows 会改用页面文件, 整机卡住几小时而不是报错。\n"
            f"先关掉浏览器等大进程, 或者跑子集 (--state CA --cat FOODS), "
            f"或者提交到集群 (sbatch cluster/run_xgb_cpu.sbatch)。")

X = np.empty((N, len(M3)), np.float32)    # M1/M2 是 M3 的前缀, 切列不用复制

def put(name, a, kind):
    """kind: 'sd' = (序列, 天) 矩阵, 's' = 每条序列一个值, 'd' = 每天一个值"""
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
    c[c < 0] = np.nan                     # 缺失 -> NaN, XGBoost 自己处理
    return c

# ---------------------------------------------------------------- 3. 特征
# ID 和日历
for c in ["item_id", "dept_id", "store_id"]:
    put(c, codes(sales[c]), "s")
put("wday", cal.wday, "d")                # 注意: 1 = 周六
put("month", cal.month, "d")
put("year", cal.year, "d")
put("dom", date.dt.day, "d")
put("week", date.dt.isocalendar().week.astype(int), "d")

# 价格 (只用当天及之前的价格, 没有未来信息)
with np.errstate(invalid="ignore", divide="ignore"):
    put("sell_price", P, "sd")
    put("price_rel_max", P / np.fmax.accumulate(P, axis=1), "sd")
    pmean = np.nancumsum(P, axis=1) / np.maximum(np.cumsum(valid, axis=1), 1)
    put("price_rel_mean", P / pmean, "sd")
    put("price_chg_wk", P / shift(P, 7) - 1, "sd")
del pmean

# SNAP: 每条序列只取它所在州的 SNAP
snap_cols = cal[["snap_CA", "snap_TX", "snap_WI"]].to_numpy(np.float32)       # (天, 3)
put("snap", snap_cols[:, sales.state_id.map({"CA": 0, "TX": 1, "WI": 2}).to_numpy()].T, "sd")
put("is_event", cal.event_name_1.notna(), "d")
put("event_name", codes(cal.event_name_1), "d")
put("event_type", codes(cal.event_type_1), "d")

# ---- 预测起点
# 窗口 [o+1, o+HZ] 里的每一天都只能用第 o 天及之前的销量。
#   HZ = 28: 每 28 天一个起点 —— M5 官方口径, 和 TimesFM / Kayla 笔记本一致
#   HZ = 1 : 每天重新预测 —— 和 Kayla 的 OLS/LASSO 一致 (滚动均值截止昨天)
# orig_of_day[c] = 第 c 列(0 基)那天所属窗口的起点, 数值 = 可用历史的天数。
cols = np.arange(D)
if HZ == 1:
    orig_of_day = cols.copy()                    # 历史截止昨天
else:
    orig_of_day = TRAIN_END + HZ * np.floor_divide(cols - TRAIN_END, HZ)
put("days_ahead", cols - orig_of_day + 1, "d")   # 1..HZ: 距起点多少天

def pick(A, back):
    """A[:, 起点 - back]。back=1 即起点当天; 起点前没有那一天就记 NaN"""
    j = orig_of_day - back
    out = A[:, np.clip(j, 0, D - 1)].copy()
    out[:, j < 0] = np.nan
    return out

# 滞后销量。lag_28 原样保留: 它是 M5 的朴素基线 (t-28 <= 起点, 任何 horizon 都合法)。
# olag_j = 起点往前第 j 天的销量, 窗口内所有天共用同一个值。
for j in [1, 2, 7, 14, 28, 364]:
    put(f"olag_{j}", pick(Sn, j), "sd")
put("lag_28", shift(Sn, 28), "sd")

# 滚动统计: 窗口止于起点; 用累加和计算, 上架前的 NaN 不算进去
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
    c[c < max(1, w // 2)] = np.nan        # 窗口里有效天数太少就不算
    mean = s / c
    std = np.sqrt(np.clip(s2 / c - mean ** 2, 0, None))
    return mean, std

for w in [7, 28, 56, 180]:
    mean, std = roll(w)
    put(f"rmean_{w}", pick(mean, 1), "sd")       # 窗口止于起点
    if w == 28:
        put("rstd_28", pick(std, 1), "sd")
del cs, cs2, cc, mean, std
log("特征完成")

# ---------------------------------------------------------------- 4. 切分
i_va = int(np.searchsorted(d_num, VALID_START))
i_te = int(np.searchsorted(d_num, TRAIN_END + 1))
log(f"训练 {i_va:,} 行 | 验证 {i_te - i_va:,} 行 | 测试 {N - i_te:,} 行")

# ---------------------------------------------------------------- 5. 训练
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
    final = xgb.train(PARAMS, qdm(0, i_te), int(n_best * 1.05))   # 用全部训练集重训
    dte = xgb.DMatrix(X[i_te:, :k], feature_names=feats, feature_types=types, enable_categorical=True)
    return final, final.predict(dte), n_best

preds = {
    "naive_lag28": np.nan_to_num(X[i_te:, col["lag_28"]]),     # 4 周前同一天 (两种 horizon 都合法)
    "naive_last": np.nan_to_num(X[i_te:, col["olag_1"]]),      # 起点那天的销量; HZ=1 时就是"昨天"
    "ma_28": np.nan_to_num(X[i_te:, col["rmean_28"]]),         # 截止起点的前 28 天均值
}
n_trees = {}
for name, feats in [("xgb_M3_full", M3), ("xgb_M2_price_event", M2), ("xgb_M1_calendar", M1)]:
    log(f"训练 {name} ({len(feats)} 个特征)")
    model, preds[name], n_trees[name] = fit(feats)
    log(f"  {name}: {n_trees[name]} 棵树")
    if name == "xgb_M3_full":
        gain = pd.Series(model.get_score(importance_type="gain"))
        (gain / gain.sum()).sort_values(ascending=False).rename("gain_share").to_csv(OUT / "feature_importance.csv")
        model.save_model(OUT / "xgb_M3_full.json")

# ---------------------------------------------------------------- 6. 评估
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
# 销量分层: 按训练期日均销量分三档
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)   # 测试期才上架的商品整行是 NaN
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
open_days = daily[~((daily.index.month == 12) & (daily.index.day == 25))]   # 圣诞节闭店, 销量≈0, 会让 MAPE 爆掉
for m in models:
    e = daily[m] - daily.y
    overall.loc["Daily_total_RMSE", m] = np.sqrt((e ** 2).mean())
    overall.loc["Daily_total_MAPE_exXmas", m] = ((open_days[m] - open_days.y).abs() / open_days.y).mean()
    eo = open_days[m] - open_days.y
    overall.loc["Daily_total_R2_exXmas", m] = 1 - (eo ** 2).sum() / ((open_days.y - open_days.y.mean()) ** 2).sum()
overall = overall.T

# WRMSSE (M5 官方指标): RMSE / WAPE 把每条序列一视同仁, 但一条日均卖 300 件的序列和
# 一条日均卖 0.3 件的序列对生意的意义完全不同。WRMSSE 先用每条序列自己的历史波动
# (naive 随机游走的一步误差) 做缩放, 再按"预测起点前 28 天的销售额"加权, 最后在 12 个
# 聚合层级 (总体 / 州 / 门店 / 品类 / 部门 / ... / item-store) 上取平均。
# 权重和缩放分母都按 28 天的滚动窗口逐个重算, 只用窗口起点之前的数据。
log("计算 WRMSSE")
origins = list(range(TRAIN_END, D, H))

def as_matrix(v):
    """长表上的预测向量 -> (序列, 测试期天数) 矩阵; 未上架的格子记 0 (实际销量也是 0)"""
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

# 训练期的描述统计 (给 findings 用)
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
print("\n===== 测试集总体指标 =====")
print(overall.round(4))
print("\n----- WRMSSE 分层级 (按销售额加权的 RMSSE) -----")
print(wrmsse_level.round(4))
print("\n----- WRMSSE 分窗口 (28 天一个滚动窗口) -----")
print(wrmsse_window.round(4))
print("\n----- 各层级的序列数与权重集中度 -----")
print(wrmsse_weights)
print("\nWRMSSE 越小越好; = 1 表示误差和历史上的朴素随机游走一步预测相当")
if wrmsse_note:
    print(wrmsse_note)
for k in ["volume", "snap", "is_event", "cat", "dept", "state", "store", "month"]:
    print(f"\n----- WAPE by {k} -----")
    print(breakdowns[k].round(4))
print("\n----- 特征重要性 (M3, gain 占比) -----")
print(pd.read_csv(OUT / "feature_importance.csv", index_col=0).head(15).round(4))
print("\n----- 训练期描述统计 -----")
print(json.dumps(desc, indent=2, ensure_ascii=False))
log(f"完成, 结果在 {OUT}")
