# -*- coding: utf-8 -*-
# ---
# # VIF による多重共線性分析
#
# 元の 11特徴量 (Model 1) と v2 の 10特徴量 (Model 2) を比較。
# VIF_k = 1 / (1 - R²_k) で、R²_k は他の全特徴量で k を回帰した決定係数。
#
# 判定: <5 健全、5-10 グレー、>10 問題あり、>30 深刻
# ---

# %% imports
import sqlite3
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy import stats as sstats

BASE = Path(__file__).resolve().parent.parent
DB   = BASE / "boatrace.db"
OUTD = BASE / "notebooks" / "output"
OUTD.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE))

TRAIN_START, TRAIN_END = "2025-08-01", "2025-12-31"
TEST_END   = "2026-04-18"

FEATS_M1 = [
    "f1_global","f2_local","f3_ST",
    "f4_disp","f5_motor","f6_form","f6_nomatch",
    "f7_lane_L","f8_V","f9_W","f10_H",
]
FEATS_M2 = [
    "theta_ability", "f3_ST",
    "f4_disp","f5_motor","f6_form","f6_nomatch",
    "f7_lane_L","f8_V","f9_W","f10_H",
]

# --- 定数 ---
LANE_L = {1:+2.06, 2:+0.12, 3:0.0, 4:-0.26, 5:-0.82, 6:-1.53}
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


# %% 1. 特徴量構築 (train のみ、他と同じロジック)
print("[1] 特徴量構築中 (train 期間のみ)...")
conn = sqlite3.connect(DB)
pf = f"date >= '{TRAIN_START}' AND date <= '{TRAIN_END}'"
rc = pd.read_sql_query(f"SELECT date, stadium, race_number, lane, racerid, global_win_pt, local_win_pt, aveST, motor_in2nd FROM race_cards WHERE {pf}", conn)
rcond = pd.read_sql_query(f"SELECT date, stadium, race_number, wind_direction, wind_speed, wave_height, display_time_1, display_time_2, display_time_3, display_time_4, display_time_5, display_time_6 FROM race_conditions WHERE {pf}", conn)
cs = pd.read_sql_query(f"SELECT date, stadium, racerid, race_number, rank FROM current_series WHERE date >= '2025-07-15' AND date <= '{TRAIN_END}' AND rank IS NOT NULL", conn)
rres_all = pd.read_sql_query("SELECT stadium, boat, rank FROM race_results WHERE rank IS NOT NULL AND boat BETWEEN 1 AND 6 AND rank BETWEEN 1 AND 6", conn)
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
dt_v["f4_disp"]   = (-(dt_v["dt"] - dt_v["race_mean"]) / dt_v["sigma_eff"]).clip(-2.5, 2.5)
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

for f in FEATS_M1:
    rc[f] = rc[f].fillna(0.0)

# PCA for M2
pca = PCA(n_components=1)
pca.fit(rc[["f1_global","f2_local"]].to_numpy())
loadings = pca.components_[0]
if loadings.sum() < 0:
    loadings = -loadings
    pca.components_ = -pca.components_
rc["theta_ability"] = pca.transform(rc[["f1_global","f2_local"]].to_numpy()).flatten()

print(f"  train N = {len(rc):,}")


# %% 2. VIF 計算関数
def compute_vif(X, cols):
    """X: (N, K) numpy, cols: list of names."""
    results = []
    for k, col in enumerate(cols):
        others = [i for i in range(len(cols)) if i != k]
        if len(others) == 0:
            results.append((col, np.nan, 0.0))
            continue
        X_oth = X[:, others]
        y = X[:, k]
        lr = LinearRegression()
        lr.fit(X_oth, y)
        r2 = lr.score(X_oth, y)
        r2 = min(max(r2, 0), 1 - 1e-12)
        vif = 1 / (1 - r2)
        results.append((col, vif, r2))
    return results

def vif_judgement(v):
    if v < 5:   return "✅ 健全"
    if v < 10:  return "⚠️ グレー"
    if v < 30:  return "🟥 多重共線性"
    return "🟥🟥 深刻"


# %% 3. Model 1 VIF
print("\n[2] Model 1 (11特徴量) の VIF 計算中...")
X_m1 = rc[FEATS_M1].to_numpy(dtype=float)
vif_m1 = compute_vif(X_m1, FEATS_M1)

print("\n## Model 1 VIF")
print("| 特徴量 | VIF | R² | 判定 |")
print("|---|---:|---:|:---|")
for col, v, r2 in vif_m1:
    print(f"| {col} | {v:.3f} | {r2:.4f} | {vif_judgement(v)} |")


# %% 4. Model 2 VIF
print("\n[3] Model 2 (v2, 10特徴量) の VIF 計算中...")
X_m2 = rc[FEATS_M2].to_numpy(dtype=float)
vif_m2 = compute_vif(X_m2, FEATS_M2)

print("\n## Model 2 VIF (v2)")
print("| 特徴量 | VIF | R² | 判定 |")
print("|---|---:|---:|:---|")
for col, v, r2 in vif_m2:
    print(f"| {col} | {v:.3f} | {r2:.4f} | {vif_judgement(v)} |")


# %% 5. ペアワイズ相関行列
print("\n[4] ペアワイズ相関行列 ...")
corr_m1 = rc[FEATS_M1].corr()
corr_m2 = rc[FEATS_M2].corr()

def plot_corr(corr, title, path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            v = corr.iloc[i, j]
            color = "white" if abs(v) > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)

plot_corr(corr_m1, "Model 1: 11特徴量 相関行列", OUTD / "correlation_matrix_model1.png")
plot_corr(corr_m2, "Model 2 (v2): 10特徴量 相関行列", OUTD / "correlation_matrix_model2.png")
print(f"  保存: correlation_matrix_model1.png, correlation_matrix_model2.png")


# %% 6. 重複ペア検証
print("\n[5] 事前仮説: 重複ペア相関の統計検証...")
pairs = [
    ("Model 1", "f1_global",    "f2_local"),
    ("Model 1", "f1_global",    "f8_V"),
    ("Model 1", "f7_lane_L",    "f8_V"),
    ("Model 1", "f5_motor",     "f1_global"),
    ("Model 1", "f6_form",      "f6_nomatch"),
    ("Model 2", "theta_ability","f8_V"),
    ("Model 2", "theta_ability","f5_motor"),
    ("Model 2", "f7_lane_L",    "f8_V"),
    ("Model 2", "theta_ability","f3_ST"),
]
print("\n| モデル | ペア | Pearson r | p-value | 判定 |")
print("|---|---|---:|---:|:---|")
dp_rows = []
for model, a, b in pairs:
    r, p = sstats.pearsonr(rc[a], rc[b])
    abs_r = abs(r)
    if abs_r < 0.3:
        j = "✅ 弱い"
    elif abs_r < 0.6:
        j = "⚠️ 中"
    elif abs_r < 0.8:
        j = "🟥 強い"
    else:
        j = "🟥🟥 非常に強い"
    print(f"| {model} | {a} ↔ {b} | {r:+.4f} | {p:.2e} | {j} |")
    dp_rows.append({"model": model, "feat1": a, "feat2": b, "pearson_r": r, "p_value": p})


# %% 7. 推奨アクション
print("\n## 推奨アクション (VIF + 相関からの解釈)")
high_vif_m1 = [(c, v) for c, v, _ in vif_m1 if v >= 5]
high_vif_m2 = [(c, v) for c, v, _ in vif_m2 if v >= 5]
print(f"\nModel 1 で VIF ≥ 5: {len(high_vif_m1)} 個")
for c, v in high_vif_m1:
    print(f"  - {c}: VIF={v:.2f}")
print(f"\nModel 2 で VIF ≥ 5: {len(high_vif_m2)} 個")
for c, v in high_vif_m2:
    print(f"  - {c}: VIF={v:.2f}")


# %% 8. CSV 出力
pd.DataFrame([{"feature":c,"vif":v,"r_squared":r2,
               "judgement":vif_judgement(v)} for c,v,r2 in vif_m1]
).to_csv(OUTD / "vif_model1.csv", index=False, encoding="utf-8-sig")
pd.DataFrame([{"feature":c,"vif":v,"r_squared":r2,
               "judgement":vif_judgement(v)} for c,v,r2 in vif_m2]
).to_csv(OUTD / "vif_model2.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(dp_rows).to_csv(OUTD / "duplicate_pair_analysis.csv",
                              index=False, encoding="utf-8-sig")
print(f"\nCSV 書込: vif_model1.csv, vif_model2.csv, duplicate_pair_analysis.csv")
print("完了")
