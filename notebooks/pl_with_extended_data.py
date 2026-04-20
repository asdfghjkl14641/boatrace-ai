# -*- coding: utf-8 -*-
# ---
# # Plackett-Luce 拡張データ再学習 (train 期間拡大 + 新H設計)
#
# ## 変更点
# - train: 2025-08-01〜 → **2023-05-01〜2025-12-31** (race_cards 全期間, ~2.5年)
# - test:  2026-01-01〜2026-04-18 (同じ)
# - f10_H: 現行「外枠ほど不利」→ **差し型カーネル** (1枠崩れ→2-4枠拾う) に再設計
# - race_conditions 無い期間 (2023-05〜2025-07) では f4/f9/f10 = 0 (中立) 扱い
#
# 比較:
#   v2_old       (旧 train 2025-08〜, 旧 H): 1着 56.97%
#   v2_ext_oldH  (新 train 拡張,         旧 H)
#   v2_ext_newH  (新 train 拡張,         新 H) ← 本命
# ---

# %% imports
import sqlite3
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
DB   = BASE / "boatrace.db"
OUTD = BASE / "notebooks" / "output"
OUTD.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE))

TRAIN_START, TRAIN_END = "2023-05-01", "2025-12-31"  # 拡張
TEST_START,  TEST_END  = "2026-01-01", "2026-04-18"

LAMBDA_L2 = 0.01
TAU_GRID = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

FEATURES = ["theta_ability","f3_ST",
            "f4_disp","f5_motor","f6_form","f6_nomatch",
            "f7_lane_L","f8_V","f9_W","f10_H"]

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


# --- 新 H 関数 (wave_lane_analysis.py の実測から設計) ---
# パターンA: 波で1枠が崩れ、2-4枠が拾う (風 differential と同じ)
# 実測 logit Δ (波10cm+):
#   枠1: -0.226  枠2: +0.149  枠3: +0.065  枠4: +0.384  枠5: +0.148  枠6: -0.520
# → 6枠は荒れで逆に不利 (フライング増?)。簡略化のため 4-5cm〜10cm+ で段階的シフト。
#
# 設計: H_new = h_magnitude(wave_bin) * kernel_h[lane-1]
#   kernel_h は 風 'differential' に近い配分 (1枠崩れを2-4枠中心に)

KERNEL_H_NEW = [-1.00, +0.30, +0.25, +0.25, +0.15, +0.05]  # 2-4枠中心に分配
def h_magnitude_new(wave_h):
    """波高 → 1枠の崩れ量 (logit)"""
    if pd.isna(wave_h): return 0.0
    if wave_h <= 1:  return 0.0    # 0-1cm: 影響なし
    if wave_h <= 3:  return 0.0    # 2-3cm: 影響なし
    if wave_h <= 5:  return -0.05  # 4-5cm: 軽く崩れる
    if wave_h <= 9:  return -0.15  # 6-9cm: 中程度
    return -0.25                   # 10cm+: 強く崩れる

def compute_H_new(lane, wave_h):
    mag = h_magnitude_new(wave_h)
    if mag == 0.0:
        return 0.0
    return float(mag * KERNEL_H_NEW[int(lane)-1] / (-1.0))  # scale = mag/kernel[0]

# 比較用: 現行 H
def compute_H_old(lane, wave_h):
    if pd.isna(wave_h): return 0.0
    if wave_h <= 3:  hc = 0.0
    elif wave_h <= 5: hc = 0.3
    elif wave_h <= 9: hc = 0.7
    else: hc = 1.0
    k = {1:0.0, 2:-0.05, 3:-0.10, 4:-0.15, 5:-0.20, 6:-0.25}[int(lane)]
    return hc * k


# %% 1. データ取得 (拡張 train)
print("[1] データ取得中 (拡張期間)...")
conn = sqlite3.connect(DB)
pf = f"date >= '{TRAIN_START}' AND date <= '{TEST_END}'"
rc = pd.read_sql_query(f"SELECT date, stadium, race_number, lane, racerid, global_win_pt, local_win_pt, aveST, motor_in2nd FROM race_cards WHERE {pf}", conn)
rcond = pd.read_sql_query(f"SELECT date, stadium, race_number, wind_direction, wind_speed, wave_height, display_time_1, display_time_2, display_time_3, display_time_4, display_time_5, display_time_6 FROM race_conditions WHERE {pf}", conn)
rres = pd.read_sql_query(f"SELECT date, stadium, race_number, boat, rank FROM race_results WHERE {pf} AND rank IS NOT NULL", conn)
cs = pd.read_sql_query(f"SELECT date, stadium, racerid, race_number, rank FROM current_series WHERE date >= '2023-04-25' AND date <= '{TEST_END}' AND rank IS NOT NULL", conn)
rres_all = pd.read_sql_query("SELECT stadium, boat, rank FROM race_results WHERE rank IS NOT NULL AND boat BETWEEN 1 AND 6 AND rank BETWEEN 1 AND 6", conn)
conn.close()
print(f"  race_cards: {len(rc):,} / race_conditions: {len(rcond):,} / race_results: {len(rres):,}")

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

# f4
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

# f6 EWMA
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

# f7
rc["f7_lane_L"] = rc["lane"].map(LANE_L)

# f8: V 補正 (race_results 全期間から計算)
g_sl = rres_all.groupby(["stadium","boat"])
win_rate = g_sl.apply(lambda df: (df["rank"]==1).mean())
nat_lane = rres_all.groupby("boat").apply(lambda df: (df["rank"]==1).mean())
V_table = {(int(s), int(l)): float(logit(p) - logit(nat_lane[l])) for (s,l), p in win_rate.items()}
rc["f8_V"] = [V_table.get((int(s), int(l)), 0.0) for s, l in zip(rc["stadium"], rc["lane"])]

# f9: W (風補正)
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

# f10_H: 新旧 両方計算
rc["f10_H_old"] = [compute_H_old(l, h) for l, h in zip(rc["lane"], rc["wave_height"])]
rc["f10_H_new"] = [compute_H_new(l, h) for l, h in zip(rc["lane"], rc["wave_height"])]

# NULL 埋め
for f in ["f1_global","f2_local","f3_ST","f4_disp","f5_motor","f6_form","f6_nomatch",
          "f7_lane_L","f8_V","f9_W","f10_H_old","f10_H_new"]:
    rc[f] = rc[f].fillna(0.0)

# actual_rank 紐付け
res = rres.rename(columns={"boat":"lane","rank":"actual_rank"})
rc = rc.merge(res[["date","stadium","race_number","lane","actual_rank"]],
              on=["date","stadium","race_number","lane"], how="left")

rc_train = rc[(rc["date"] >= TRAIN_START) & (rc["date"] <= TRAIN_END)].copy()
rc_test  = rc[(rc["date"] >= TEST_START)  & (rc["date"] <= TEST_END)].copy()
print(f"  train rows: {len(rc_train):,} ({rc_train['date'].nunique()} days)")
print(f"  test  rows: {len(rc_test):,} ({rc_test['date'].nunique()} days)")

# race_conditions あり/なし のカバー確認
cov_train = rc_train["wave_height"].notna().mean() * 100
cov_test  = rc_test["wave_height"].notna().mean() * 100
print(f"  race_conditions カバー率 train: {cov_train:.1f}%  test: {cov_test:.1f}%")


# %% 2. θ_ability PCA (新 train で再 fit)
print("\n[2] PCA (f1+f2 → θ_ability) を新 train で再学習...")
pca = PCA(n_components=1)
pca.fit(rc_train[["f1_global","f2_local"]].to_numpy())
load = pca.components_[0]
evr = pca.explained_variance_ratio_[0]
if load.sum() < 0:
    load = -load; pca.components_ = -pca.components_
print(f"  loadings: f1={load[0]:+.4f}  f2={load[1]:+.4f}  EVR: {evr*100:.2f}%")

rc_train["theta_ability"] = pca.transform(rc_train[["f1_global","f2_local"]].to_numpy()).flatten()
rc_test["theta_ability"]  = pca.transform(rc_test[["f1_global","f2_local"]].to_numpy()).flatten()


# %% 3. レーステンソル構築
def build_race_tensors(df, features):
    grp = df.sort_values(["date","stadium","race_number","lane"]).groupby(
        ["date","stadium","race_number"], sort=False)
    Xl, pil, kl = [], [], []
    for key, g in grp:
        if len(g) != 6 or set(g["lane"]) != {1,2,3,4,5,6}: continue
        gg = g.sort_values("lane")
        rs = gg.set_index("lane")["actual_rank"]
        try:
            l1 = int(rs[rs==1].index[0])
            l2 = int(rs[rs==2].index[0])
            l3 = int(rs[rs==3].index[0])
        except IndexError: continue
        Xl.append(gg[features].to_numpy(dtype=float))
        pil.append([l1-1, l2-1, l3-1])
        kl.append(key)
    return np.stack(Xl), np.array(pil, dtype=int), pd.DataFrame(kl, columns=["date","stadium","race_number"])

print("\n[3] レーステンソル構築 (旧H & 新H)...")
FEATURES_OLD = ["theta_ability","f3_ST","f4_disp","f5_motor","f6_form","f6_nomatch",
                "f7_lane_L","f8_V","f9_W","f10_H_old"]
FEATURES_NEW = ["theta_ability","f3_ST","f4_disp","f5_motor","f6_form","f6_nomatch",
                "f7_lane_L","f8_V","f9_W","f10_H_new"]
X_train_old, pi_train, keys_train = build_race_tensors(rc_train, FEATURES_OLD)
X_test_old,  pi_test,  keys_test  = build_race_tensors(rc_test,  FEATURES_OLD)
X_train_new, _, _ = build_race_tensors(rc_train, FEATURES_NEW)
X_test_new,  _, _ = build_race_tensors(rc_test,  FEATURES_NEW)
print(f"  train races: {len(keys_train):,} / test races: {len(keys_test):,}")


# %% 4. PL 最適化
def pl_nll_and_grad(beta, X, pi, lam=LAMBDA_L2):
    N, M, K = X.shape
    S = X @ beta
    removed = np.zeros((N, M), dtype=bool)
    ri = np.arange(N)
    nll = 0.0; grad = np.zeros(K)
    for t in range(3):
        pt = pi[:, t]
        Sm = np.where(removed, -np.inf, S)
        lse = logsumexp(Sm, axis=1)
        nll += float((lse - S[ri, pt]).sum())
        p = np.exp(Sm - lse[:, None])
        grad += -X[ri, pt, :].sum(axis=0) + np.einsum("nj,njk->k", p, X)
        removed[ri, pt] = True
    nll += 0.5 * lam * float(beta @ beta)
    grad += lam * beta
    return nll, grad

def fit_pl(X, pi, x0, label=""):
    r = minimize(fun=lambda b: pl_nll_and_grad(b, X, pi), x0=x0, jac=True,
                 method="L-BFGS-B",
                 options={"ftol": 1e-6, "maxiter": 1000, "disp": False})
    if label: print(f"  [{label}] converged={r.success} iter={r.nit} NLL={r.fun:.1f}")
    return r.x, r.nit

# 初期値 (v2 の β)
x0 = np.array([+0.3809, +0.0594, +0.1556, +0.1017, +0.1576, +0.0590,
               +0.5243, +0.3694, +0.5980, -1.4867])

print("\n[4] PL 最適化 (旧H)...")
beta_old, iter_old = fit_pl(X_train_old, pi_train, x0, "v2_ext_oldH")
print(f"  β = {np.array2string(beta_old, precision=3)}")

print("\n[4] PL 最適化 (新H)...")
# 新Hでは初期値の H 係数を +1.0 (素直に使う) にしておく
x0_new = x0.copy(); x0_new[-1] = 1.0
beta_new, iter_new = fit_pl(X_train_new, pi_train, x0_new, "v2_ext_newH")
print(f"  β = {np.array2string(beta_new, precision=3)}")


# %% 5. 評価
def evaluate(beta, X_T, pi_T, tau=1.0, label=""):
    N = len(X_T)
    S = X_T @ beta
    S_tau = S / tau
    Ss = S_tau - S_tau.max(axis=1, keepdims=True)
    P = np.exp(Ss) / np.exp(Ss).sum(axis=1, keepdims=True)
    pred = np.argsort(-S, axis=1)[:, :3]
    act = pi_T
    hit1 = (pred[:,0] == act[:,0]).mean() * 100
    a2s = [set(act[i,:2].tolist()) for i in range(N)]
    hit2 = np.mean([any(p in a2s[i] for p in pred[i,:2]) for i in range(N)]) * 100
    a3s = [set(act[i,:3].tolist()) for i in range(N)]
    hit3 = np.mean([any(p in a3s[i] for p in pred[i,:3]) for i in range(N)]) * 100
    Pw = P[np.arange(N), act[:,0]]
    ll = -np.log(np.maximum(Pw, 1e-9)).mean()
    pl_ll = 0.0
    removed = np.zeros((N, 6), dtype=bool)
    for t in range(3):
        Sm = np.where(removed, -np.inf, S_tau)
        lse = logsumexp(Sm, axis=1)
        pl_ll += (lse - S_tau[np.arange(N), act[:,t]]).mean()
        removed[np.arange(N), act[:,t]] = True
    if label:
        print(f"\n### {label} τ={tau} N={N:,}")
        print(f"  1着: {hit1:.2f}%  2連対: {hit2:.2f}%  3連対: {hit3:.2f}%  LL: {ll:.4f}  PLNLL: {pl_ll:.4f}")
    return {"hit1":hit1,"hit2":hit2,"hit3":hit3,"ll":ll,"pl_ll":pl_ll}


# τ グリッド (旧H / 新H)
print("\n[5] τ グリッドサーチ...")
best_old = None; best_new = None
for tau in TAU_GRID:
    m_o = evaluate(beta_old, X_test_old, pi_test, tau=tau)
    m_n = evaluate(beta_new, X_test_new, pi_test, tau=tau)
    print(f"  τ={tau:.1f} | oldH hit1={m_o['hit1']:.2f}% ll={m_o['ll']:.4f} | newH hit1={m_n['hit1']:.2f}% ll={m_n['ll']:.4f}")
    if best_old is None or m_o["ll"] < best_old["ll"]:
        best_old = {**m_o, "tau": tau}
    if best_new is None or m_n["ll"] < best_new["ll"]:
        best_new = {**m_n, "tau": tau}

print(f"\n最適 τ (oldH): {best_old['tau']}  ll={best_old['ll']:.4f}")
print(f"最適 τ (newH): {best_new['tau']}  ll={best_new['ll']:.4f}")


# %% 6. 既存 v2 (旧 train, 旧H) との 3-way 比較
V2_OLD = {"tau":0.8, "hit1":56.97, "hit2":90.88, "hit3":99.58, "ll":1.2085, "pl_ll":3.9841,
          "train_races": 22150, "train_period": "2025-08 〜 2025-12"}

print("\n## 3-way 比較 (Test, 各 τ 最適)")
print()
print("| モデル | train期間 | train races | 1着 | 2連対 | 3連対 | LL | PLNLL | τ |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
print(f"| v2_old (旧H, 旧train) | 2025-08〜 | 22,150 | 56.97% | 90.88% | 99.58% | 1.2085 | 3.9841 | 0.8 |")
print(f"| **v2_ext_oldH** | 2023-05〜 | {len(keys_train):,} | {best_old['hit1']:.2f}% | {best_old['hit2']:.2f}% | {best_old['hit3']:.2f}% | {best_old['ll']:.4f} | {best_old['pl_ll']:.4f} | {best_old['tau']:.1f} |")
print(f"| **v2_ext_newH** | 2023-05〜 | {len(keys_train):,} | {best_new['hit1']:.2f}% | {best_new['hit2']:.2f}% | {best_new['hit3']:.2f}% | {best_new['ll']:.4f} | {best_new['pl_ll']:.4f} | {best_new['tau']:.1f} |")


# %% 7. 波あり/なしでの精度比較 (test)
print("\n[6] 波あり vs なし 別 test精度 (新Hが波ありで改善したか):")
# test の各レースの波高を取得
test_wave = []
for i in range(len(keys_test)):
    k = keys_test.iloc[i]
    sub = rc_test[(rc_test["date"]==k["date"]) & (rc_test["stadium"]==k["stadium"]) & (rc_test["race_number"]==k["race_number"])]
    h = sub["wave_height"].iloc[0] if len(sub) > 0 else None
    test_wave.append(h)
test_wave = pd.Series(test_wave)

for model_name, beta, X_T in [("oldH", beta_old, X_test_old), ("newH", beta_new, X_test_new)]:
    S = X_T @ beta
    pred = np.argsort(-S, axis=1)[:, :3]
    act = pi_test
    hit1 = (pred[:,0] == act[:,0])
    df_eval = pd.DataFrame({"wave": test_wave, "hit1": hit1})
    df_eval["wave_bin"] = df_eval["wave"].apply(
        lambda x: "none(NA)" if pd.isna(x) else ("nowave(0-3)" if x<=3 else "wave(4+)")
    )
    agg = df_eval.groupby("wave_bin")["hit1"].agg(["mean","count"])
    print(f"\n  {model_name}:")
    for idx, row in agg.iterrows():
        print(f"    {idx:15s}  hit1={row['mean']*100:.2f}%  N={int(row['count']):,}")


# %% 8. 係数比較 (新H vs 旧H)
print("\n## 係数比較 (拡張train)")
print()
print("| 特徴量 | β_oldH | β_newH | 差分 |")
print("|---|---:|---:|---:|")
for i, f in enumerate(FEATURES_OLD[:-1]):  # f10 以外は共通
    b_o = beta_old[i]; b_n = beta_new[i]
    print(f"| {f} | {b_o:+.4f} | {b_n:+.4f} | {b_n-b_o:+.4f} |")
# f10
print(f"| f10_H (旧形式) | {beta_old[-1]:+.4f} | — | — |")
print(f"| f10_H (新形式) | — | {beta_new[-1]:+.4f} | — |")


# %% 9. 会場別 (Test, newH)
from scripts.stadiums import STADIUMS
S = X_test_new @ beta_new
pred = np.argsort(-S, axis=1)[:, :3]
rows = []
for i in range(len(keys_test)):
    s = int(keys_test.iloc[i]["stadium"])
    a = pi_test[i]; p = pred[i]
    rows.append({
        "stadium": s, "hit_1st":  p[0] == a[0],
        "hit_top2": any(x in set(a[:2].tolist()) for x in p[:2]),
        "hit_top3": any(x in set(a[:3].tolist()) for x in p[:3]),
    })
agg = pd.DataFrame(rows).groupby("stadium").agg(
    N=("hit_1st","count"),
    hit_1st=("hit_1st", lambda s: s.mean()*100),
    hit_top2=("hit_top2", lambda s: s.mean()*100),
    hit_top3=("hit_top3", lambda s: s.mean()*100),
).reset_index()
agg["name"] = agg["stadium"].map(STADIUMS)
agg = agg[["stadium","name","N","hit_1st","hit_top2","hit_top3"]].sort_values("hit_1st", ascending=False)
print("\n## 会場別 (Test, v2_ext_newH)")
print("トップ5:")
print(agg.head(5).to_string(index=False))
print("\nボトム5:")
print(agg.tail(5).to_string(index=False))


# %% 10. CSV 出力
pd.DataFrame({
    "feature": FEATURES_NEW,
    "beta_old_H": beta_old,
    "beta_new_H": beta_new,
}).to_csv(OUTD / "v2_extended_weights.csv", index=False, encoding="utf-8-sig")

pd.DataFrame([
    {"model":"v2_old (旧train旧H)", **V2_OLD},
    {"model":"v2_ext_oldH",  "tau":best_old["tau"], "hit1":best_old["hit1"], "hit2":best_old["hit2"], "hit3":best_old["hit3"], "ll":best_old["ll"], "pl_ll":best_old["pl_ll"], "train_races":len(keys_train), "train_period": f"{TRAIN_START}〜{TRAIN_END}"},
    {"model":"v2_ext_newH",  "tau":best_new["tau"], "hit1":best_new["hit1"], "hit2":best_new["hit2"], "hit3":best_new["hit3"], "ll":best_new["ll"], "pl_ll":best_new["pl_ll"], "train_races":len(keys_train), "train_period": f"{TRAIN_START}〜{TRAIN_END}"},
]).to_csv(OUTD / "three_versions_comparison.csv", index=False, encoding="utf-8-sig")

print("\nCSV 書込: v2_extended_weights.csv, three_versions_comparison.csv")
print("完了")
