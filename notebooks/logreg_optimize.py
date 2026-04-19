# -*- coding: utf-8 -*-
# ---
# # MVP バックテスト 重み最適化 (Logistic Regression)
#
# mvp_backtest.py と同じ11特徴量を train (2025-08-01 〜 2025-12-31) でLR学習し、
# test (2026-01-01 〜 2026-04-18) で再バックテスト。叩き台重みとの精度比較も出力する。
# ---

# %% imports
import sqlite3
import math
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.linear_model import LogisticRegression

BASE = Path(__file__).resolve().parent.parent
DB   = BASE / "boatrace.db"
OUTD = BASE / "notebooks" / "output"
OUTD.mkdir(parents=True, exist_ok=True)

# プロジェクト root を sys.path に
sys.path.insert(0, str(BASE))

TRAIN_START, TRAIN_END = "2025-08-01", "2025-12-31"
TEST_START,  TEST_END  = "2026-01-01", "2026-04-18"

# 叩き台重み (比較用)
BASELINE_WEIGHTS = {
    "f1_global":  1.0, "f2_local":  0.4, "f3_ST":      0.6,
    "f4_disp":    1.0, "f5_motor":  0.5, "f6_form":    0.3,
    "f6_nomatch": 0.0, "f7_lane_L": 1.0, "f8_V":       1.0,
    "f9_W":       1.0, "f10_H":     1.0,
}
FEATURES = list(BASELINE_WEIGHTS.keys())

# 枠番 L
LANE_L = {1: +2.06, 2: +0.12, 3: 0.0, 4: -0.26, 5: -0.82, 6: -1.53}

# 風補正 W
KERNEL = {
    "differential":    [-1.00, +0.35, +0.35, +0.15, +0.10, +0.05],
    "dashout":         [-1.00, +0.10, +0.10, +0.30, +0.30, +0.20],
    "even":            [-1.00, +0.20, +0.20, +0.20, +0.20, +0.20],
    "sacrifice_2":     [+1.00, -1.00,   0.0,   0.0,   0.0,   0.0],
    "sacrifice_outer": [+1.00,   0.0,   0.0,   0.0, -0.50, -0.50],
}
WIND_MAP = {
    (2, 6):   ("differential",    -6.78), (17, 7):  ("differential",    -6.61),
    (24, 7):  ("differential",   -13.87), (6, 17):  ("differential",   -10.62),
    (22, 12): ("differential",    -9.14), (20, 6):  ("differential",    -6.76),
    (11, 13): ("dashout",         -9.76), (14, 11): ("dashout",         -8.07),
    (18, 15): ("dashout",        -13.04), (4, 13):  ("even",            -4.41),
    (20, 14): ("even",            -8.48), (24, 1):  ("sacrifice_2",     +5.84),
    (23, 12): ("sacrifice_2",     +7.17), (17, 17): ("sacrifice_outer", +7.07),
    (4, 5):   ("sacrifice_outer", +5.98),
}
P_BASE_LANE1 = {
    1:0.4839, 2:0.4440, 3:0.4827, 4:0.4547, 5:0.5339, 6:0.5180,
    7:0.5878, 8:0.5662, 9:0.5598, 10:0.5193, 11:0.5413, 12:0.5878,
    13:0.5952, 14:0.4807, 15:0.5592, 16:0.5686, 17:0.5737, 18:0.6261,
    19:0.6073, 20:0.5944, 21:0.6024, 22:0.5668, 23:0.5483, 24:0.6336,
}

def logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p))


# %% 1. 特徴量再計算 (mvp_backtest.pyと同じロジック)
print("[1] 特徴量再計算中...")
conn = sqlite3.connect(DB)
period_filter = f"date >= '{TRAIN_START}' AND date <= '{TEST_END}'"

rc = pd.read_sql_query(f"""
    SELECT date, stadium, race_number, lane, racerid,
           global_win_pt, local_win_pt, aveST, motor_in2nd
    FROM race_cards WHERE {period_filter}
""", conn)
rcond = pd.read_sql_query(f"""
    SELECT date, stadium, race_number,
           wind_direction, wind_speed, wave_height,
           display_time_1, display_time_2, display_time_3,
           display_time_4, display_time_5, display_time_6
    FROM race_conditions WHERE {period_filter}
""", conn)
rres = pd.read_sql_query(f"""
    SELECT date, stadium, race_number, boat, rank
    FROM race_results WHERE {period_filter} AND rank IS NOT NULL
""", conn)
cs = pd.read_sql_query(f"""
    SELECT date, stadium, racerid, race_number, rank FROM current_series
    WHERE date >= '2025-07-15' AND date <= '{TEST_END}' AND rank IS NOT NULL
""", conn)
rres_all = pd.read_sql_query("""
    SELECT stadium, boat, rank FROM race_results
    WHERE rank IS NOT NULL AND boat BETWEEN 1 AND 6 AND rank BETWEEN 1 AND 6
""", conn)
conn.close()

rc = rc.dropna(subset=["racerid"]).copy()
rc["racerid"] = rc["racerid"].astype("int64")
rc["stadium"] = rc["stadium"].astype("int64")
cs = cs.dropna(subset=["racerid"]).copy()
cs["racerid"] = cs["racerid"].astype("int64")
cs["stadium"] = cs["stadium"].astype("int64")

# A層
rc["f1_global"] = (rc["global_win_pt"] - 5.3) / 1.3
f2_raw = (rc["local_win_pt"] - 5.4) / 1.3
rc["f2_local"] = np.where(rc["local_win_pt"] >= 2.4, f2_raw, rc["f1_global"])
rc["f3_ST"] = -(rc["aveST"] - 0.16) / 0.023

# B層
# f4: 展示 z-score
dt_long = rcond.melt(
    id_vars=["date","stadium","race_number"],
    value_vars=[f"display_time_{i}" for i in range(1,7)],
    var_name="lane", value_name="dt")
dt_long["lane"] = dt_long["lane"].str.extract(r"(\d)").astype(int)
dt_v = dt_long.dropna(subset=["dt"])
dt_v = dt_v[dt_v["dt"] > 0].copy()
g = dt_v.groupby(["date","stadium","race_number"])["dt"]
dt_v["race_mean"] = g.transform("mean")
dt_v["race_std"]  = g.transform("std")
dt_v["sigma_eff"] = dt_v["race_std"].fillna(0.025).clip(lower=0.025)
dt_v["f4_disp_raw"] = -(dt_v["dt"] - dt_v["race_mean"]) / dt_v["sigma_eff"]
dt_v["f4_disp"]     = dt_v["f4_disp_raw"].clip(-2.5, 2.5)
rc = rc.merge(dt_v[["date","stadium","race_number","lane","f4_disp"]],
              on=["date","stadium","race_number","lane"], how="left")

# f5
rc["f5_motor"] = (rc["motor_in2nd"] - 33) / 11

# f6: EWMA
cs = cs.sort_values(["stadium","racerid","date","race_number"]).reset_index(drop=True)
cs["ewma_here"] = cs.groupby(["stadium","racerid"])["rank"].transform(
    lambda s: s.ewm(alpha=0.35, adjust=False).mean())
epoch = pd.Timestamp("2020-01-01")
rc["rc_date"] = pd.to_datetime(rc["date"])
cs["cs_date"] = pd.to_datetime(cs["date"])
rc["pos"] = (rc["rc_date"] - epoch).dt.days.astype("int64") * 13 + rc["race_number"].astype("int64")
cs["pos"] = (cs["cs_date"] - epoch).dt.days.astype("int64") * 13 + cs["race_number"].astype("int64")
cs_keep = cs[["stadium","racerid","pos","cs_date","ewma_here"]].rename(columns={"ewma_here":"cs_ewma"})
rc_sorted = rc.sort_values("pos")
cs_sorted = cs_keep.sort_values("pos")
merged = pd.merge_asof(rc_sorted, cs_sorted, on="pos", by=["stadium","racerid"],
                       direction="backward", allow_exact_matches=False)
merged["days_diff"] = (merged["rc_date"] - merged["cs_date"]).dt.days
valid = merged["days_diff"].notna() & (merged["days_diff"] >= 0) & (merged["days_diff"] <= 7)
merged.loc[~valid, "cs_ewma"] = np.nan
merged["f6_form_raw"] = -(merged["cs_ewma"] - 3.5) / 1.0
merged["f6_form"]    = merged["f6_form_raw"].fillna(0.0)
merged["f6_nomatch"] = merged["f6_form_raw"].isna().astype(int)
rc = merged.sort_values(["date","stadium","race_number","lane"]).reset_index(drop=True)

# C層: L, V, W, H
rc["f7_lane_L"] = rc["lane"].map(LANE_L)

# V
g_sl = rres_all.groupby(["stadium","boat"])
win_rate = g_sl.apply(lambda df: (df["rank"]==1).mean())
nat_lane = rres_all.groupby("boat").apply(lambda df: (df["rank"]==1).mean())
V_table = {}
for (s, l), p in win_rate.items():
    V_table[(int(s), int(l))] = float(logit(p) - logit(nat_lane[l]))
rc["f8_V"] = [V_table.get((int(s), int(l)), 0.0) for s, l in zip(rc["stadium"], rc["lane"])]

# W
cond_keep = rcond[["date","stadium","race_number","wind_direction","wind_speed","wave_height"]]
rc = rc.merge(cond_keep, on=["date","stadium","race_number"], how="left")
def compute_W(s, l, wd, ws):
    if pd.isna(wd):
        return 0.0
    key = (int(s), int(wd))
    if key not in WIND_MAP:
        return 0.0
    if int(wd) != 17 and (pd.isna(ws) or ws < 2):
        return 0.0
    kern_name, dpct = WIND_MAP[key]
    kern = KERNEL[kern_name]
    pb = P_BASE_LANE1[int(s)]
    W1 = logit(pb + dpct/100.0) - logit(pb)
    return float(W1 * kern[int(l)-1] / kern[0])
rc["f9_W"] = [compute_W(s,l,wd,ws) for s,l,wd,ws in zip(rc["stadium"], rc["lane"], rc["wind_direction"], rc["wind_speed"])]

# H
def h_coef(h):
    if pd.isna(h): return 0.0
    if h <= 3: return 0.0
    if h <= 5: return 0.3
    if h <= 9: return 0.7
    return 1.0
K_LANE_WAVE = {1:0.0, 2:-0.05, 3:-0.10, 4:-0.15, 5:-0.20, 6:-0.25}
rc["h_coef"] = rc["wave_height"].apply(h_coef)
rc["f10_H"]  = rc["h_coef"] * rc["lane"].map(K_LANE_WAVE)

for f in FEATURES:
    rc[f] = rc[f].fillna(0.0)

# actual rank 紐付け (y=1着フラグ)
res = rres.rename(columns={"boat":"lane","rank":"actual_rank"})
rc = rc.merge(res[["date","stadium","race_number","lane","actual_rank"]],
              on=["date","stadium","race_number","lane"], how="left")
rc["y"] = (rc["actual_rank"] == 1).astype(int)

# 期間split
rc_train = rc[(rc["date"] >= TRAIN_START) & (rc["date"] <= TRAIN_END)].copy()
rc_test  = rc[(rc["date"] >= TEST_START)  & (rc["date"] <= TEST_END)].copy()
print(f"  train: {len(rc_train):,} 行 / test: {len(rc_test):,} 行")

X_train = rc_train[FEATURES].to_numpy(dtype=float)
y_train = rc_train["y"].to_numpy()
X_test  = rc_test[FEATURES].to_numpy(dtype=float)
y_test  = rc_test["y"].to_numpy()


# %% 2. Logistic Regression fit
print("\n[2] Logistic Regression 学習中...")
model = LogisticRegression(
    penalty="l2", C=1.0, max_iter=1000,
    solver="lbfgs", fit_intercept=False,
)
model.fit(X_train, y_train)
beta = model.coef_.flatten()

# SE 推定 (Fisher情報量近似)
p_hat = model.predict_proba(X_train)[:, 1]
W_diag = p_hat * (1 - p_hat)
H = X_train.T @ (X_train * W_diag.reshape(-1, 1))  # Hessian
# L2 正則化項: (1/C) * I
H += (1.0 / model.C) * np.eye(len(beta))
try:
    cov = np.linalg.inv(H)
    se = np.sqrt(np.diag(cov))
except np.linalg.LinAlgError:
    se = np.full_like(beta, np.nan)

# 重要度 (|β| × std(X))
X_std = X_train.std(axis=0)
importance = np.abs(beta) * X_std

# レポート
print("\n## 推定された重み")
print()
print("| 特徴量 | 叩き台 w₀ | 最適化 β | SE | β/SE (z) | 符号一致 | |β|×std(X) |")
print("|---|---:|---:|---:|---:|:---:|---:|")
for i, f in enumerate(FEATURES):
    w0 = BASELINE_WEIGHTS[f]
    b = beta[i]
    s = se[i]
    z = b / s if s > 0 else float('nan')
    # 符号一致: 叩き台の正負と推定の正負が一致
    if w0 == 0.0:
        match = "—"
    else:
        match = "○" if np.sign(b) == np.sign(w0) else "✗"
    print(f"| {f} | {w0:+.2f} | {b:+.4f} | {s:.4f} | {z:+.2f} | {match} | {importance[i]:.3f} |")

# 最重要 3 特徴量
imp_rank = sorted(zip(FEATURES, importance), key=lambda x: -x[1])
print("\n**重要度トップ3 (|β|×std(X)):**")
for f, imp in imp_rank[:3]:
    print(f"  - {f}: {imp:.3f}")


# %% 3. 予想＆評価ユーティリティ
def backtest(rc_df, weights, features, label):
    """重みベクトルで S を計算し、レース内softmax → 上位3予想 → 的中評価"""
    X = rc_df[features].to_numpy(dtype=float)
    if hasattr(weights, "__iter__"):
        w = np.array(weights, dtype=float)
    else:
        w = weights
    rc_df = rc_df.copy()
    rc_df["S"] = X @ w

    def race_softmax(s):
        s_arr = np.asarray(s, dtype=float)
        m = s_arr.max()
        e = np.exp(s_arr - m)
        return e / e.sum()

    rc_df["P"] = rc_df.groupby(["date","stadium","race_number"])["S"].transform(race_softmax)
    rc_df["pred_rank"] = rc_df.groupby(["date","stadium","race_number"])["S"].rank(
        method="min", ascending=False).astype(int)

    def agg_race(g):
        p1 = g[g["pred_rank"]==1]["lane"].iloc[0] if (g["pred_rank"]==1).any() else None
        p2 = g[g["pred_rank"]==2]["lane"].iloc[0] if (g["pred_rank"]==2).any() else None
        p3 = g[g["pred_rank"]==3]["lane"].iloc[0] if (g["pred_rank"]==3).any() else None
        a1 = g[g["actual_rank"]==1]["lane"].iloc[0] if (g["actual_rank"]==1).any() else None
        a2 = g[g["actual_rank"]==2]["lane"].iloc[0] if (g["actual_rank"]==2).any() else None
        a3 = g[g["actual_rank"]==3]["lane"].iloc[0] if (g["actual_rank"]==3).any() else None
        if a1 is not None:
            p_actual = g[g["lane"]==a1]["P"].iloc[0]
            ll = -math.log(max(p_actual, 1e-9))
        else:
            ll = None
        return pd.Series({
            "pred_1": p1, "pred_2": p2, "pred_3": p3,
            "actual_1": a1, "actual_2": a2, "actual_3": a3,
            "log_loss": ll,
            "hit_1st":  (p1 == a1) if (p1 and a1) else False,
            "hit_top2": any(p in {a1, a2} for p in [p1, p2] if p) if a1 else False,
            "hit_top3": any(p in {a1, a2, a3} for p in [p1, p2, p3] if p) if a1 else False,
        })

    race_log = rc_df.groupby(["date","stadium","race_number"]).apply(agg_race).reset_index()
    v = race_log.dropna(subset=["actual_1"])
    n = len(v)
    hit1 = v["hit_1st"].mean() * 100
    hit2 = v["hit_top2"].mean() * 100
    hit3 = v["hit_top3"].mean() * 100
    ll = v["log_loss"].mean()
    print(f"\n### {label} (N={n:,})")
    print(f"  1着 hit:    {hit1:.2f}%")
    print(f"  2連対 hit:  {hit2:.2f}%")
    print(f"  3連対 hit:  {hit3:.2f}%")
    print(f"  Log-loss:   {ll:.4f}")
    return {"N": n, "hit1": hit1, "hit2": hit2, "hit3": hit3, "ll": ll, "race_log": race_log}


# %% 4. バックテスト比較
print("\n[3] バックテスト (train)")
baseline_w = np.array([BASELINE_WEIGHTS[f] for f in FEATURES])
t_base_train = backtest(rc_train, baseline_w, FEATURES, "Train 叩き台重み")
t_opt_train  = backtest(rc_train, beta,        FEATURES, "Train 最適化重み")

print("\n[3] バックテスト (test)")
t_base_test = backtest(rc_test, baseline_w, FEATURES, "Test 叩き台重み")
t_opt_test  = backtest(rc_test, beta,        FEATURES, "Test 最適化重み")

# %% 5. 比較表
print("\n## 叩き台 vs 最適化 精度比較")
print()
print("| データ | 指標 | 叩き台 | 最適化 | 差分 |")
print("|---|---|---:|---:|---:|")
for lbl, base, opt in [("Train", t_base_train, t_opt_train),
                       ("Test",  t_base_test,  t_opt_test)]:
    for metric, key, fmt in [
        ("1着 hit (%)",    "hit1", "{:.2f}"),
        ("2連対 hit (%)",  "hit2", "{:.2f}"),
        ("3連対 hit (%)",  "hit3", "{:.2f}"),
        ("Log-loss",       "ll",   "{:.4f}"),
    ]:
        b = base[key]
        o = opt[key]
        if key == "ll":
            diff = f"{o - b:+.4f}"
        else:
            diff = f"{o - b:+.2f} pt"
        print(f"| {lbl} | {metric} | {fmt.format(b)} | {fmt.format(o)} | {diff} |")

# %% 6. 会場別 (Test、最適化のみ)
from scripts.stadiums import STADIUMS
print("\n## 会場別 (Test、最適化重み)")
test_log_opt = t_opt_test["race_log"].dropna(subset=["actual_1"])
rows = []
for s, gdf in test_log_opt.groupby("stadium"):
    rows.append({
        "stadium": int(s), "name": STADIUMS.get(int(s), "?"),
        "N": len(gdf),
        "hit_1st": gdf["hit_1st"].mean() * 100,
        "hit_top2": gdf["hit_top2"].mean() * 100,
        "hit_top3": gdf["hit_top3"].mean() * 100,
        "log_loss": gdf["log_loss"].mean(),
    })
out = pd.DataFrame(rows).sort_values("stadium")
print(out.to_string(index=False))

# CSV出力
t_opt_test["race_log"].to_csv(OUTD / "test_log_optimized.csv", index=False, encoding="utf-8-sig")
t_opt_train["race_log"].to_csv(OUTD / "train_log_optimized.csv", index=False, encoding="utf-8-sig")

# 重みもCSV化
pd.DataFrame({
    "feature": FEATURES,
    "baseline": [BASELINE_WEIGHTS[f] for f in FEATURES],
    "optimized": beta.tolist(),
    "se": se.tolist(),
    "z": (beta/se).tolist() if np.all(se>0) else [np.nan]*len(FEATURES),
    "importance_abs_beta_std": importance.tolist(),
}).to_csv(OUTD / "logreg_weights.csv", index=False, encoding="utf-8-sig")

print(f"\nCSV 書込: {OUTD}/logreg_weights.csv, test_log_optimized.csv, train_log_optimized.csv")
print("\n最適化完了")
