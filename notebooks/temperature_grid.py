# -*- coding: utf-8 -*-
# ---
# # Softmax 温度パラメータ τ のグリッドサーチ
#
# logreg_optimize.py で学習した β をそのまま使い、softmax の温度 τ のみを変える。
#
# **重要な事前確認**: top-k 予想は τ の単調変換で不変のため、
# 1着/2連対/3連対の的中率は **τ に依存しない**。
# 変化するのは Log-loss (確率値の絶対値) のみ。
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

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

BASE = Path(__file__).resolve().parent.parent
DB   = BASE / "boatrace.db"
OUTD = BASE / "notebooks" / "output"
OUTD.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE))

TRAIN_START, TRAIN_END = "2025-08-01", "2025-12-31"
TEST_START,  TEST_END  = "2026-01-01", "2026-04-18"
TAU_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

FEATURES = [
    "f1_global", "f2_local", "f3_ST",
    "f4_disp", "f5_motor", "f6_form", "f6_nomatch",
    "f7_lane_L", "f8_V", "f9_W", "f10_H",
]

LANE_L = {1: +2.06, 2: +0.12, 3: 0.0, 4: -0.26, 5: -0.82, 6: -1.53}
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


# %% 1. 特徴量構築 (logreg_optimize.py と同じ)
print("[1] 特徴量構築中...")
conn = sqlite3.connect(DB)
pf = f"date >= '{TRAIN_START}' AND date <= '{TEST_END}'"
rc = pd.read_sql_query(f"""
    SELECT date, stadium, race_number, lane, racerid,
           global_win_pt, local_win_pt, aveST, motor_in2nd
    FROM race_cards WHERE {pf}
""", conn)
rcond = pd.read_sql_query(f"""
    SELECT date, stadium, race_number,
           wind_direction, wind_speed, wave_height,
           display_time_1, display_time_2, display_time_3,
           display_time_4, display_time_5, display_time_6
    FROM race_conditions WHERE {pf}
""", conn)
rres = pd.read_sql_query(f"""
    SELECT date, stadium, race_number, boat, rank
    FROM race_results WHERE {pf} AND rank IS NOT NULL
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

rc["f1_global"] = (rc["global_win_pt"] - 5.3) / 1.3
f2_raw = (rc["local_win_pt"] - 5.4) / 1.3
rc["f2_local"] = np.where(rc["local_win_pt"] >= 2.4, f2_raw, rc["f1_global"])
rc["f3_ST"] = -(rc["aveST"] - 0.16) / 0.023

dt_long = rcond.melt(id_vars=["date","stadium","race_number"],
    value_vars=[f"display_time_{i}" for i in range(1,7)], var_name="lane", value_name="dt")
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
rc["f5_motor"] = (rc["motor_in2nd"] - 33) / 11

cs = cs.sort_values(["stadium","racerid","date","race_number"]).reset_index(drop=True)
cs["ewma_here"] = cs.groupby(["stadium","racerid"])["rank"].transform(
    lambda s: s.ewm(alpha=0.35, adjust=False).mean())
epoch = pd.Timestamp("2020-01-01")
rc["rc_date"] = pd.to_datetime(rc["date"])
cs["cs_date"] = pd.to_datetime(cs["date"])
rc["pos"] = (rc["rc_date"] - epoch).dt.days.astype("int64") * 13 + rc["race_number"].astype("int64")
cs["pos"] = (cs["cs_date"] - epoch).dt.days.astype("int64") * 13 + cs["race_number"].astype("int64")
cs_keep = cs[["stadium","racerid","pos","cs_date","ewma_here"]].rename(columns={"ewma_here":"cs_ewma"})
merged = pd.merge_asof(rc.sort_values("pos"), cs_keep.sort_values("pos"),
                       on="pos", by=["stadium","racerid"],
                       direction="backward", allow_exact_matches=False)
merged["days_diff"] = (merged["rc_date"] - merged["cs_date"]).dt.days
vm = merged["days_diff"].notna() & (merged["days_diff"] >= 0) & (merged["days_diff"] <= 7)
merged.loc[~vm, "cs_ewma"] = np.nan
merged["f6_form_raw"] = -(merged["cs_ewma"] - 3.5) / 1.0
merged["f6_form"]    = merged["f6_form_raw"].fillna(0.0)
merged["f6_nomatch"] = merged["f6_form_raw"].isna().astype(int)
rc = merged.sort_values(["date","stadium","race_number","lane"]).reset_index(drop=True)

rc["f7_lane_L"] = rc["lane"].map(LANE_L)

g_sl = rres_all.groupby(["stadium","boat"])
win_rate = g_sl.apply(lambda df: (df["rank"]==1).mean())
nat_lane = rres_all.groupby("boat").apply(lambda df: (df["rank"]==1).mean())
V_table = {(int(s), int(l)): float(logit(p) - logit(nat_lane[l])) for (s,l), p in win_rate.items()}
rc["f8_V"] = [V_table.get((int(s), int(l)), 0.0) for s, l in zip(rc["stadium"], rc["lane"])]

cond_keep = rcond[["date","stadium","race_number","wind_direction","wind_speed","wave_height"]]
rc = rc.merge(cond_keep, on=["date","stadium","race_number"], how="left")

def compute_W(s, l, wd, ws):
    if pd.isna(wd): return 0.0
    key = (int(s), int(wd))
    if key not in WIND_MAP: return 0.0
    if int(wd) != 17 and (pd.isna(ws) or ws < 2): return 0.0
    kn, dp = WIND_MAP[key]
    kern = KERNEL[kn]
    pb = P_BASE_LANE1[int(s)]
    W1 = logit(pb + dp/100.0) - logit(pb)
    return float(W1 * kern[int(l)-1] / kern[0])

rc["f9_W"] = [compute_W(s,l,wd,ws) for s,l,wd,ws in
              zip(rc["stadium"], rc["lane"], rc["wind_direction"], rc["wind_speed"])]

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

res = rres.rename(columns={"boat":"lane","rank":"actual_rank"})
rc = rc.merge(res[["date","stadium","race_number","lane","actual_rank"]],
              on=["date","stadium","race_number","lane"], how="left")
rc["y"] = (rc["actual_rank"] == 1).astype(int)

rc_train = rc[(rc["date"] >= TRAIN_START) & (rc["date"] <= TRAIN_END)].copy()
rc_test  = rc[(rc["date"] >= TEST_START)  & (rc["date"] <= TEST_END)].copy()
print(f"  train: {len(rc_train):,} 行 / test: {len(rc_test):,} 行")


# %% 2. LR 学習 (logreg_optimize と同じ設定)
print("\n[2] LR 学習中...")
X_train = rc_train[FEATURES].to_numpy(dtype=float)
y_train = rc_train["y"].to_numpy()
model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000,
                            solver="lbfgs", fit_intercept=False)
model.fit(X_train, y_train)
beta = model.coef_.flatten()
print(f"  β = {np.array2string(beta, precision=3)}")


# %% 3. Test で S = β^T X を一発計算
X_test = rc_test[FEATURES].to_numpy(dtype=float)
S_test = X_test @ beta
rc_test["S"] = S_test


# %% 4. 温度グリッドサーチ
print("\n[3] τ グリッドサーチ中...")

def race_softmax_with_tau(s, tau):
    s_arr = np.asarray(s, dtype=float) / tau
    m = s_arr.max()
    e = np.exp(s_arr - m)
    return e / e.sum()

grid_results = []
for tau in TAU_GRID:
    # レースごとに softmax を計算
    rc_test[f"P_tau"] = (
        rc_test
        .groupby(["date","stadium","race_number"])["S"]
        .transform(lambda s: race_softmax_with_tau(s.values, tau))
    )
    # top-k ranking は S で決まる (τ で不変) → 一度だけ計算
    if "pred_rank" not in rc_test.columns:
        rc_test["pred_rank"] = rc_test.groupby(
            ["date","stadium","race_number"])["S"].rank(
                method="min", ascending=False).astype(int)

    # レース集計
    def agg(g):
        p1 = g[g["pred_rank"]==1]["lane"].iloc[0] if (g["pred_rank"]==1).any() else None
        p2 = g[g["pred_rank"]==2]["lane"].iloc[0] if (g["pred_rank"]==2).any() else None
        p3 = g[g["pred_rank"]==3]["lane"].iloc[0] if (g["pred_rank"]==3).any() else None
        a1 = g[g["actual_rank"]==1]["lane"].iloc[0] if (g["actual_rank"]==1).any() else None
        a2 = g[g["actual_rank"]==2]["lane"].iloc[0] if (g["actual_rank"]==2).any() else None
        a3 = g[g["actual_rank"]==3]["lane"].iloc[0] if (g["actual_rank"]==3).any() else None
        if a1 is None:
            return pd.Series({"hit_1st": False, "hit_top2": False, "hit_top3": False, "ll": None})
        p_act = g[g["lane"]==a1]["P_tau"].iloc[0]
        ll = -math.log(max(p_act, 1e-9))
        return pd.Series({
            "hit_1st":  p1 == a1,
            "hit_top2": any(p in {a1,a2} for p in [p1,p2] if p is not None),
            "hit_top3": any(p in {a1,a2,a3} for p in [p1,p2,p3] if p is not None),
            "ll": ll,
        })

    race_log = rc_test.groupby(["date","stadium","race_number"]).apply(agg).reset_index()
    v = race_log.dropna(subset=["ll"])
    n = len(v)
    h1 = v["hit_1st"].mean() * 100
    h2 = v["hit_top2"].mean() * 100
    h3 = v["hit_top3"].mean() * 100
    ll_mean = v["ll"].mean()
    grid_results.append({"tau": tau, "N": n, "hit1": h1, "hit2": h2, "hit3": h3, "ll": ll_mean})
    print(f"  τ={tau:.1f}: hit1={h1:.2f}%  hit2={h2:.2f}%  hit3={h3:.2f}%  log-loss={ll_mean:.4f}")

grid_df = pd.DataFrame(grid_results)

# %% 5. 表とベスト τ
print("\n## τ × 4指標グリッド (Test)")
print()
print("| τ | 1着 hit (%) | 2連対 hit (%) | 3連対 hit (%) | Log-loss |")
print("|---:|---:|---:|---:|---:|")
for r in grid_results:
    print(f"| {r['tau']:.1f} | {r['hit1']:.2f} | {r['hit2']:.2f} | {r['hit3']:.2f} | {r['ll']:.4f} |")

# 最適 τ
best_hit1_idx = grid_df["hit1"].idxmax()
best_ll_idx   = grid_df["ll"].idxmin()
best_hit1 = grid_df.loc[best_hit1_idx]
best_ll   = grid_df.loc[best_ll_idx]

# バランス基準: Log-loss が最小値+0.05以内でhit1が最大のもの
ll_budget = grid_df["ll"].min() + 0.05
balanced_candidates = grid_df[grid_df["ll"] <= ll_budget]
best_balanced = balanced_candidates.loc[balanced_candidates["hit1"].idxmax()]

print(f"\n## 最適 τ 提案")
print()
print(f"- **(a) 1着的中率最大**: τ = {best_hit1['tau']:.1f} (hit1={best_hit1['hit1']:.2f}%, ll={best_hit1['ll']:.4f})")
print(f"- **(b) Log-loss最小**:  τ = {best_ll['tau']:.1f} (hit1={best_ll['hit1']:.2f}%, ll={best_ll['ll']:.4f})")
print(f"- **(c) バランス型** (Log-loss ≤ min+0.05): τ = {best_balanced['tau']:.1f} "
      f"(hit1={best_balanced['hit1']:.2f}%, ll={best_balanced['ll']:.4f})")

# %% 6. グラフ
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(grid_df["tau"], grid_df["hit1"], "o-", color="tab:blue", label="1着 hit (%)")
ax1.set_xlabel("τ (temperature)")
ax1.set_ylabel("1着 hit rate (%)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.axhline(55, color="tab:blue", linestyle=":", alpha=0.3, label="目標 55%")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(grid_df["tau"], grid_df["ll"], "s-", color="tab:red", label="Log-loss")
ax2.set_ylabel("Log-loss", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax2.axhline(1.30, color="tab:red", linestyle=":", alpha=0.3, label="目標 1.30")

ax1.set_title("Temperature τ Grid Search (Test)")
fig.tight_layout()
out_png = OUTD / "temperature_grid.png"
fig.savefig(out_png, dpi=100, bbox_inches="tight")
print(f"\nプロット保存: {out_png}")

# CSV出力
grid_df.to_csv(OUTD / "temperature_grid.csv", index=False, encoding="utf-8-sig")
print(f"CSV 保存: {OUTD}/temperature_grid.csv")

print("\n⚠️  予想どおり hit_1st / hit_top2 / hit_top3 は τ に対して不変。"
      "\n   top-k は S の順位で決まり、softmax (単調変換) は順位を変えないため。"
      "\n   Log-loss のみが τ で変化。確率校正の用途で活用を。")
