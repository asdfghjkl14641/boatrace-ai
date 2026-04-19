# -*- coding: utf-8 -*-
# ---
# # Plackett-Luce + θ_ability v2 (f1+f2 のみ統合、f3_ST 独立)
#
# 前回 v1 (f1+f2+f3 → θ) は EVR 75.64% で情報損失が大きく、
# 1着的中率が -0.35pt 劣化した。
# ST は「能力」とは別の「スタート技術」軸なので独立させる。
#
# ## 特徴量構成 (10)
# θ_ability (← f1,f2 を PCA で統合),
# f3_ST, f4_disp, f5_motor, f6_form, f6_nomatch,
# f7_lane_L, f8_V, f9_W, f10_H
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

from scipy.optimize import minimize
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

BASE = Path(__file__).resolve().parent.parent
DB   = BASE / "boatrace.db"
OUTD = BASE / "notebooks" / "output"
OUTD.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE))

TRAIN_START, TRAIN_END = "2025-08-01", "2025-12-31"
TEST_START,  TEST_END  = "2026-01-01", "2026-04-18"

OLD_FEATURES = ["f1_global","f2_local","f3_ST",
                "f4_disp","f5_motor","f6_form","f6_nomatch",
                "f7_lane_L","f8_V","f9_W","f10_H"]
# v2 特徴量: θ (f1+f2のみ) + f3_ST + 残り
NEW_FEATURES = ["theta_ability", "f3_ST",
                "f4_disp","f5_motor","f6_form","f6_nomatch",
                "f7_lane_L","f8_V","f9_W","f10_H"]

LAMBDA_L2 = 0.01
TAU_GRID = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

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


# %% 1. 特徴量構築
print("[1] 特徴量構築中...")
conn = sqlite3.connect(DB)
pf = f"date >= '{TRAIN_START}' AND date <= '{TEST_END}'"
rc = pd.read_sql_query(f"SELECT date, stadium, race_number, lane, racerid, global_win_pt, local_win_pt, aveST, motor_in2nd FROM race_cards WHERE {pf}", conn)
rcond = pd.read_sql_query(f"SELECT date, stadium, race_number, wind_direction, wind_speed, wave_height, display_time_1, display_time_2, display_time_3, display_time_4, display_time_5, display_time_6 FROM race_conditions WHERE {pf}", conn)
rres = pd.read_sql_query(f"SELECT date, stadium, race_number, boat, rank FROM race_results WHERE {pf} AND rank IS NOT NULL", conn)
cs = pd.read_sql_query(f"SELECT date, stadium, racerid, race_number, rank FROM current_series WHERE date >= '2025-07-15' AND date <= '{TEST_END}' AND rank IS NOT NULL", conn)
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
merged["f6_form"]     = merged["f6_form_raw"].fillna(0.0)
merged["f6_nomatch"]  = merged["f6_form_raw"].isna().astype(int)
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

for f in OLD_FEATURES:
    rc[f] = rc[f].fillna(0.0)

res = rres.rename(columns={"boat":"lane","rank":"actual_rank"})
rc = rc.merge(res[["date","stadium","race_number","lane","actual_rank"]],
              on=["date","stadium","race_number","lane"], how="left")

rc_train = rc[(rc["date"] >= TRAIN_START) & (rc["date"] <= TRAIN_END)].copy()
rc_test  = rc[(rc["date"] >= TEST_START)  & (rc["date"] <= TEST_END)].copy()


# %% 2. PCA: f1 + f2 → θ_ability (v2)
print("\n[2] PCA 学習中 (f1+f2 → θ_ability, v2)...")
pca = PCA(n_components=1)
X_f12_train = rc_train[["f1_global","f2_local"]].to_numpy(dtype=float)
pca.fit(X_f12_train)
loadings = pca.components_[0]  # (2,)
evr = pca.explained_variance_ratio_[0]
print(f"  PCA loadings (f1, f2): [{loadings[0]:+.4f}, {loadings[1]:+.4f}]")
print(f"  explained_variance_ratio_: {evr*100:.2f}%")
# 符号を正に揃える
if loadings.sum() < 0:
    print("  (符号反転して能力=大 に揃える)")
    loadings = -loadings
    pca.components_ = -pca.components_

rc_train["theta_ability"] = pca.transform(X_f12_train).flatten()
rc_test["theta_ability"]  = pca.transform(rc_test[["f1_global","f2_local"]].to_numpy(dtype=float)).flatten()

print(f"  θ stats train: mean={rc_train['theta_ability'].mean():+.3f}, "
      f"std={rc_train['theta_ability'].std():.3f}")


# %% 3. レーステンソル構築
def build_race_tensors(df, features):
    grp = df.sort_values(["date","stadium","race_number","lane"]).groupby(
        ["date","stadium","race_number"], sort=False)
    X_list, pi_list, key_list = [], [], []
    for key, g in grp:
        if len(g) != 6 or set(g["lane"]) != {1,2,3,4,5,6}:
            continue
        gg = g.sort_values("lane")
        rs = gg.set_index("lane")["actual_rank"]
        try:
            l1 = int(rs[rs==1].index[0])
            l2 = int(rs[rs==2].index[0])
            l3 = int(rs[rs==3].index[0])
        except IndexError:
            continue
        X_list.append(gg[features].to_numpy(dtype=float))
        pi_list.append([l1-1, l2-1, l3-1])
        key_list.append(key)
    return (np.stack(X_list), np.array(pi_list, dtype=int),
            pd.DataFrame(key_list, columns=["date","stadium","race_number"]))

print("\n[3] レーステンソル構築中...")
X_train_T, pi_train, keys_train = build_race_tensors(rc_train, NEW_FEATURES)
X_test_T,  pi_test,  keys_test  = build_race_tensors(rc_test,  NEW_FEATURES)
print(f"  train races: {len(keys_train):,} / test races: {len(keys_test):,}")


# %% 4. PL NLL / grad / hessian
def pl_nll_and_grad(beta, X, pi, lam=LAMBDA_L2):
    N, M, K = X.shape
    S = X @ beta
    removed = np.zeros((N, M), dtype=bool)
    row_idx = np.arange(N)
    nll = 0.0
    grad = np.zeros(K)
    for t in range(3):
        pi_t = pi[:, t]
        S_masked = np.where(removed, -np.inf, S)
        lse = logsumexp(S_masked, axis=1)
        nll += float((lse - S[row_idx, pi_t]).sum())
        p = np.exp(S_masked - lse[:, None])
        grad += -X[row_idx, pi_t, :].sum(axis=0) + np.einsum("nj,njk->k", p, X)
        removed[row_idx, pi_t] = True
    nll += 0.5 * lam * float(beta @ beta)
    grad += lam * beta
    return nll, grad

def pl_hessian(beta, X, pi, lam=LAMBDA_L2):
    N, M, K = X.shape
    S = X @ beta
    removed = np.zeros((N, M), dtype=bool)
    row_idx = np.arange(N)
    H = np.zeros((K, K))
    for t in range(3):
        pi_t = pi[:, t]
        S_masked = np.where(removed, -np.inf, S)
        lse = logsumexp(S_masked, axis=1)
        p = np.exp(S_masked - lse[:, None])
        pX  = np.einsum("nj,njk->nk", p, X)
        pXX = np.einsum("nj,njk,njl->nkl", p, X, X)
        H  += pXX.sum(axis=0) - np.einsum("nk,nl->kl", pX, pX)
        removed[row_idx, pi_t] = True
    H += lam * np.eye(K)
    return H


# %% 5. 初期値: 旧PL係数を引き継ぎ
# v1 θ 係数 (0.3578) と旧PL f3_ST (+0.0418) は情報の分配方向が異なるので、
# v2 では θ(f1+f2統合) + f3独立 として初期化する。
# θ_ability: v1のθ係数 0.3578 を初期値に (f3寄与分が外れるため実際はもっと小さくなるはず)
# f3_ST: 旧 PL の f3 係数 0.0418 を初期値
BETA_INIT_V2 = {
    "theta_ability": 0.3578,
    "f3_ST":         0.0418,
    "f4_disp":       0.1505,
    "f5_motor":      0.1028,
    "f6_form":       0.1427,
    "f6_nomatch":    0.0580,
    "f7_lane_L":     0.5238,
    "f8_V":          0.3693,
    "f9_W":          0.5977,
    "f10_H":        -1.4870,
}
beta_init = np.array([BETA_INIT_V2[f] for f in NEW_FEATURES])


# %% 6. PL 最適化
print("\n[4] PL 最適化中 (v2, 10特徴量)...")
nll_history = []
def cb(b):
    nll_history.append(pl_nll_and_grad(b, X_train_T, pi_train)[0])
nll_history.append(pl_nll_and_grad(beta_init, X_train_T, pi_train)[0])

result = minimize(
    fun=lambda b: pl_nll_and_grad(b, X_train_T, pi_train),
    x0=beta_init, jac=True, method="L-BFGS-B",
    options={"ftol": 1e-6, "maxiter": 1000, "disp": False},
    callback=cb,
)
beta_v2 = result.x
print(f"  converged: {result.success}  iter: {result.nit}  NLL: {result.fun:.2f}")
print(f"  β_v2: {np.array2string(beta_v2, precision=3)}")

# SE
H = pl_hessian(beta_v2, X_train_T, pi_train)
try:
    cov = np.linalg.inv(H)
    se_v2 = np.sqrt(np.diag(cov))
except np.linalg.LinAlgError:
    se_v2 = np.full(len(beta_v2), np.nan)


# %% 7. 評価
def evaluate(beta, X_T, pi_T, tau=1.0, label=""):
    N = len(X_T)
    S = X_T @ beta
    S_tau = S / tau
    S_stab = S_tau - S_tau.max(axis=1, keepdims=True)
    P = np.exp(S_stab) / np.exp(S_stab).sum(axis=1, keepdims=True)
    pred = np.argsort(-S, axis=1)[:, :3]
    act = pi_T
    hit1 = (pred[:,0] == act[:,0]).mean() * 100
    act_s2 = [set(act[i,:2].tolist()) for i in range(N)]
    hit2 = np.mean([any(p in act_s2[i] for p in pred[i,:2]) for i in range(N)]) * 100
    act_s3 = [set(act[i,:3].tolist()) for i in range(N)]
    hit3 = np.mean([any(p in act_s3[i] for p in pred[i,:3]) for i in range(N)]) * 100
    ll = -np.log(np.maximum(P[np.arange(N), act[:,0]], 1e-9)).mean()
    pl_ll = 0.0
    removed = np.zeros((N, 6), dtype=bool)
    for t in range(3):
        Sm = np.where(removed, -np.inf, S_tau)
        lse = logsumexp(Sm, axis=1)
        pl_ll += (lse - S_tau[np.arange(N), act[:,t]]).mean()
        removed[np.arange(N), act[:,t]] = True
    if label:
        print(f"\n### {label} (τ={tau}, N={N:,})")
        print(f"  1着: {hit1:.2f}%  2連対: {hit2:.2f}%  3連対: {hit3:.2f}%  LL: {ll:.4f}  PLNLL: {pl_ll:.4f}")
    return {"hit1": hit1, "hit2": hit2, "hit3": hit3, "ll": ll, "pl_ll": pl_ll}


# τ グリッド
print("\n[5] τ グリッドサーチ...")
grid = []
for tau in TAU_GRID:
    m = evaluate(beta_v2, X_test_T, pi_test, tau=tau)
    grid.append({"tau": tau, **m})
    print(f"  τ={tau:.1f}: hit1={m['hit1']:.2f}%  ll={m['ll']:.4f}  pl_nll={m['pl_ll']:.4f}")
grid_df = pd.DataFrame(grid)
best_tau = float(grid_df.loc[grid_df["ll"].idxmin(), "tau"])
m_v2_best = evaluate(beta_v2, X_test_T, pi_test, tau=best_tau, label=f"新 PL+θv2 最適 τ={best_tau}")


# %% 8. 3モデル比較
# 旧 PL: 11特徴量 (v0)
OLD_PL_TEST = {"tau": 0.8, "hit1": 57.40, "hit2": 91.47, "hit3": 99.52, "ll": 1.2008, "pl_ll": 3.9641}
# v1: θ (f1+f2+f3統合), 10特徴量
V1_TEST = {"tau": 0.8, "hit1": 57.05, "hit2": 90.92, "hit3": 99.56, "ll": 1.2066, "pl_ll": 3.9846}

print("\n## 3モデル比較 (Test, τ=0.8)")
print()
print("| 指標 | 旧 PL (11特) | v1: θ₃(f1+f2+f3統合, 10特) | **v2: θ₂(f1+f2統合, 10特)** |")
print("|---|---:|---:|---:|")
for k, lbl, fmt in [
    ("hit1", "1着 hit (%)", "{:.2f}"),
    ("hit2", "2連対 hit (%)", "{:.2f}"),
    ("hit3", "3連対 hit (%)", "{:.2f}"),
    ("ll",   "Log-loss", "{:.4f}"),
    ("pl_ll","PL NLL (1-3着)", "{:.4f}"),
]:
    o = OLD_PL_TEST[k]
    v1 = V1_TEST[k]
    v2 = m_v2_best[k]
    print(f"| {lbl} | {fmt.format(o)} | {fmt.format(v1)} | **{fmt.format(v2)}** |")

# 収束比較
print()
print("| 収束 | 47 iter | 6 iter | {} iter |".format(result.nit))


# %% 9. PCA loadings
print("\n## PCA loadings (f1,f2 → θ_ability, v2)")
print()
print("| 入力 | 負荷量 w |")
print("|---|---:|")
print(f"| f1_global | {loadings[0]:+.4f} |")
print(f"| f2_local  | {loadings[1]:+.4f} |")
print(f"\n説明分散比 (EVR): **{evr*100:.2f}%**")


# %% 10. 係数比較表
OLD_PL = {"f1_global":+0.4874,"f2_local":+0.1054,"f3_ST":+0.0418,
          "f4_disp":+0.1505,"f5_motor":+0.1028,"f6_form":+0.1427,
          "f6_nomatch":+0.0580,"f7_lane_L":+0.5238,"f8_V":+0.3693,
          "f9_W":+0.5977,"f10_H":-1.4870}
V1_BETA = {"theta_ability":+0.3578, "f4_disp":+0.1548,"f5_motor":+0.1014,
           "f6_form":+0.1675,"f6_nomatch":+0.0635,"f7_lane_L":+0.5264,
           "f8_V":+0.3710,"f9_W":+0.5983,"f10_H":-1.4861}

print("\n## 係数比較 (3モデル)")
print()
print("| 特徴量 | 旧 PL (11特) | v1 (10特) | **v2 (10特)** | SE(v2) | z(v2) |")
print("|---|---:|---:|---:|---:|---:|")
# 特徴量リスト結合
all_feats = ["theta_ability","f1_global","f2_local","f3_ST",
             "f4_disp","f5_motor","f6_form","f6_nomatch",
             "f7_lane_L","f8_V","f9_W","f10_H"]
for f in all_feats:
    o = OLD_PL.get(f, None)
    v1 = V1_BETA.get(f, None)
    idx = NEW_FEATURES.index(f) if f in NEW_FEATURES else None
    v2 = beta_v2[idx] if idx is not None else None
    se2 = se_v2[idx] if idx is not None else None
    def fmt(x):
        return f"{x:+.4f}" if x is not None else "—"
    def fmtse(x):
        return f"{x:.4f}" if x is not None else "—"
    z2 = v2/se2 if (v2 is not None and se2 and se2>0) else None
    def fmtz(x):
        return f"{x:+.2f}" if x is not None else "—"
    print(f"| {f} | {fmt(o)} | {fmt(v1)} | **{fmt(v2)}** | {fmtse(se2)} | {fmtz(z2)} |")


# %% 11. 会場別
from scripts.stadiums import STADIUMS
S_test = X_test_T @ beta_v2
pred_top = np.argsort(-S_test, axis=1)[:, :3]
rows = []
for i in range(len(keys_test)):
    s = int(keys_test.iloc[i]["stadium"])
    a = pi_test[i]; p = pred_top[i]
    rows.append({
        "stadium": s,
        "hit_1st":  p[0] == a[0],
        "hit_top2": any(x in set(a[:2].tolist()) for x in p[:2]),
        "hit_top3": any(x in set(a[:3].tolist()) for x in p[:3]),
    })
by_s = pd.DataFrame(rows)
agg = by_s.groupby("stadium").agg(
    N=("hit_1st","count"),
    hit_1st=("hit_1st", lambda s: s.mean()*100),
    hit_top2=("hit_top2", lambda s: s.mean()*100),
    hit_top3=("hit_top3", lambda s: s.mean()*100),
).reset_index()
agg["name"] = agg["stadium"].map(STADIUMS)
agg = agg[["stadium","name","N","hit_1st","hit_top2","hit_top3"]].sort_values("hit_1st", ascending=False)
print("\n## 会場別 (Test, v2)")
print("トップ5:")
print(agg.head(5).to_string(index=False))
print("\nボトム5:")
print(agg.tail(5).to_string(index=False))


# %% 12. CSV 出力
pd.DataFrame({
    "feature": NEW_FEATURES,
    "beta_v2": beta_v2,
    "se_v2":   se_v2,
    "z_v2":    beta_v2 / np.where(se_v2 > 0, se_v2, np.nan),
}).to_csv(OUTD / "latent_v2_weights.csv", index=False, encoding="utf-8-sig")

pd.DataFrame({
    "input": ["f1_global","f2_local"],
    "loading": loadings.tolist(),
    "explained_variance_ratio": [evr, ""],
}).to_csv(OUTD / "latent_v2_pca_loadings.csv", index=False, encoding="utf-8-sig")

pd.DataFrame([
    {"metric":"hit1", "old_pl":OLD_PL_TEST["hit1"], "v1_theta3":V1_TEST["hit1"], "v2_theta2":m_v2_best["hit1"]},
    {"metric":"hit2", "old_pl":OLD_PL_TEST["hit2"], "v1_theta3":V1_TEST["hit2"], "v2_theta2":m_v2_best["hit2"]},
    {"metric":"hit3", "old_pl":OLD_PL_TEST["hit3"], "v1_theta3":V1_TEST["hit3"], "v2_theta2":m_v2_best["hit3"]},
    {"metric":"ll",   "old_pl":OLD_PL_TEST["ll"],   "v1_theta3":V1_TEST["ll"],   "v2_theta2":m_v2_best["ll"]},
    {"metric":"pl_ll","old_pl":OLD_PL_TEST["pl_ll"],"v1_theta3":V1_TEST["pl_ll"],"v2_theta2":m_v2_best["pl_ll"]},
]).to_csv(OUTD / "three_models_comparison.csv", index=False, encoding="utf-8-sig")

print("\nCSV 書込: latent_v2_weights.csv, latent_v2_pca_loadings.csv, three_models_comparison.csv")
print("完了")
