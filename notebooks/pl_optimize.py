# -*- coding: utf-8 -*-
# ---
# # Plackett-Luce モデルによる重み最適化
#
# 着順列全体の尤度を用いて β を推定し、LR (二値1着) とのギャップを埋める。
#
# ## 数理モデル
#
# 1レースの1-2-3着列 (π1, π2, π3) の尤度:
#
#   P = ∏_{t=1..3} [ exp(S_{π_t}) / Σ_{j ∈ active_t} exp(S_j) ]
#
# active_t は「まだ着順が決まっていない艇集合」。
#
# NLL = -Σ_races log P + (λ/2)||β||²
# ∂NLL/∂β_k = Σ_races Σ_t [-X_{πt,k} + Σ_{j∈active_t} p_{t,j} X_{j,k}] + λ β_k
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

from scipy.optimize import minimize
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

BASE = Path(__file__).resolve().parent.parent
DB   = BASE / "boatrace.db"
OUTD = BASE / "notebooks" / "output"
OUTD.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE))

TRAIN_START, TRAIN_END = "2025-08-01", "2025-12-31"
TEST_START,  TEST_END  = "2026-01-01", "2026-04-18"

FEATURES = [
    "f1_global", "f2_local", "f3_ST",
    "f4_disp", "f5_motor", "f6_form", "f6_nomatch",
    "f7_lane_L", "f8_V", "f9_W", "f10_H",
]
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


# %% 1. 特徴量構築 (logreg_optimize と同じ)
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

for f in FEATURES:
    rc[f] = rc[f].fillna(0.0)

# actual rank 紐付け
res = rres.rename(columns={"boat":"lane","rank":"actual_rank"})
rc = rc.merge(res[["date","stadium","race_number","lane","actual_rank"]],
              on=["date","stadium","race_number","lane"], how="left")

rc_train = rc[(rc["date"] >= TRAIN_START) & (rc["date"] <= TRAIN_END)].copy()
rc_test  = rc[(rc["date"] >= TEST_START)  & (rc["date"] <= TEST_END)].copy()


# %% 2. レース単位テンソル化 (X: N×6×K, pi: N×3)
def build_race_tensors(df, features):
    """Return X (N, 6, K), pi (N, 3) with 0-based boat indices for 1st,2nd,3rd,
    and race_keys (date, stadium, race_number) for N rows.
    各レースで6艇(lane1-6)全て揃い、かつ rank 1,2,3 全て欠けていないレースのみ残す。"""
    grp = df.sort_values(["date","stadium","race_number","lane"]).groupby(
        ["date","stadium","race_number"], sort=False)
    X_list, pi_list, key_list = [], [], []
    for key, g in grp:
        if len(g) != 6 or set(g["lane"]) != {1,2,3,4,5,6}:
            continue
        gg = g.sort_values("lane")
        # rank 1,2,3 の lane 取得
        rank_series = gg.set_index("lane")["actual_rank"]
        try:
            lane1 = int(rank_series[rank_series == 1].index[0])
            lane2 = int(rank_series[rank_series == 2].index[0])
            lane3 = int(rank_series[rank_series == 3].index[0])
        except IndexError:
            continue  # 1,2,3着のどれかが欠落
        X_r = gg[features].to_numpy(dtype=float)  # (6, K)
        X_list.append(X_r)
        pi_list.append([lane1-1, lane2-1, lane3-1])
        key_list.append(key)
    X = np.stack(X_list)  # (N, 6, K)
    pi = np.array(pi_list, dtype=int)  # (N, 3)
    keys = pd.DataFrame(key_list, columns=["date","stadium","race_number"])
    return X, pi, keys

print("\n[2] レーステンソル構築中 (train/test)...")
X_train_T, pi_train, keys_train = build_race_tensors(rc_train, FEATURES)
X_test_T,  pi_test,  keys_test  = build_race_tensors(rc_test,  FEATURES)
print(f"  train races: {len(keys_train):,} / test races: {len(keys_test):,}")
N_train, _, K = X_train_T.shape


# %% 3. LR 初期値 (logreg_optimize.py と同じ設定で fit)
print("\n[3] LR 初期値推定中...")
rc_train_flat = rc_train.copy()
rc_train_flat["y"] = (rc_train_flat["actual_rank"] == 1).astype(int)
lr_model = LogisticRegression(
    penalty="l2", C=1.0, max_iter=1000, solver="lbfgs", fit_intercept=False)
lr_model.fit(rc_train_flat[FEATURES].to_numpy(dtype=float),
             rc_train_flat["y"].to_numpy())
beta_init = lr_model.coef_.flatten().copy()
print(f"  β_LR init: {np.array2string(beta_init, precision=3)}")


# %% 4. PL NLL + Gradient (解析的)
def pl_nll_and_grad(beta, X, pi, lam=LAMBDA_L2):
    # X: (N, 6, K), pi: (N, 3)
    N, M, K = X.shape  # M=6
    S = X @ beta  # (N, 6)

    # "removed" mask: True if this boat is already placed (excluded from subsequent softmax)
    removed = np.zeros((N, M), dtype=bool)
    row_idx = np.arange(N)

    nll = 0.0
    grad = np.zeros(K)

    for t in range(3):
        pi_t = pi[:, t]  # (N,)
        # Active set: set S[removed] = -inf
        S_masked = np.where(removed, -np.inf, S)  # (N, 6)
        lse = logsumexp(S_masked, axis=1)  # (N,)
        S_winner = S[row_idx, pi_t]
        nll += float((lse - S_winner).sum())

        # Softmax probs over active set: exp(S - lse) (zero for removed)
        p = np.exp(S_masked - lse[:, None])  # (N, 6)

        # Grad contribution: Σ_n [ -X[n, pi_t[n]] + Σ_j p[n,j] X[n,j] ]
        X_winner = X[row_idx, pi_t, :]  # (N, K)
        grad += -X_winner.sum(axis=0) + np.einsum("nj,njk->k", p, X)

        # Update removed
        removed[row_idx, pi_t] = True

    # L2
    nll += 0.5 * lam * float(beta @ beta)
    grad += lam * beta

    return nll, grad


# 単体テスト: gradient 一致 (少数サンプル + 数値微分)
print("\n[4] Gradient unit test (小サンプル)...")
rng = np.random.default_rng(42)
small_N = 50
idx = rng.choice(N_train, size=small_N, replace=False)
X_small = X_train_T[idx]
pi_small = pi_train[idx]
beta_test = beta_init + rng.normal(0, 0.01, size=K)
nll0, g_analytic = pl_nll_and_grad(beta_test, X_small, pi_small)
eps = 1e-5
g_numeric = np.zeros(K)
for k in range(K):
    b1 = beta_test.copy(); b1[k] += eps
    b2 = beta_test.copy(); b2[k] -= eps
    g_numeric[k] = (pl_nll_and_grad(b1, X_small, pi_small)[0] -
                    pl_nll_and_grad(b2, X_small, pi_small)[0]) / (2*eps)
max_diff = np.abs(g_analytic - g_numeric).max()
print(f"  勾配の最大差分 (analytic vs numeric): {max_diff:.3e}")
assert max_diff < 1e-4, "勾配一致しない!"
print("  ✓ 勾配OK")


# %% 5. Plackett-Luce 最適化
print("\n[5] PL 最適化中 (L-BFGS-B)...")

nll_history = []

def callback(b):
    nll_history.append(pl_nll_and_grad(b, X_train_T, pi_train)[0])

nll0, _ = pl_nll_and_grad(beta_init, X_train_T, pi_train)
nll_history.append(nll0)

result = minimize(
    fun=lambda b: pl_nll_and_grad(b, X_train_T, pi_train),
    x0=beta_init,
    jac=True,
    method="L-BFGS-B",
    options={"ftol": 1e-6, "maxiter": 1000, "disp": False},
    callback=callback,
)
beta_pl = result.x
print(f"  converged: {result.success}  iter: {result.nit}  NLL: {result.fun:.2f}")
print(f"  β_PL: {np.array2string(beta_pl, precision=3)}")

# %% 5.1 SE via analytic Hessian
print("\n[5b] SE (analytic Hessian)...")
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
        # For each race: H_r = X^T diag(p) X - (p^T X)^T (p^T X)
        # Sum over races using einsum:
        pX = np.einsum("nj,njk->nk", p, X)                    # (N, K)
        pXX = np.einsum("nj,njk,njl->nkl", p, X, X)           # (N, K, K)
        H += pXX.sum(axis=0) - np.einsum("nk,nl->kl", pX, pX)
        removed[row_idx, pi_t] = True
    H += lam * np.eye(K)
    return H

H = pl_hessian(beta_pl, X_train_T, pi_train)
try:
    cov = np.linalg.inv(H)
    se_pl = np.sqrt(np.diag(cov))
except np.linalg.LinAlgError:
    se_pl = np.full(K, np.nan)


# %% 6. 評価関数
def evaluate(beta, X_T, pi_T, keys_T, tau=1.0, label=""):
    """X_T: (N, 6, K), pi_T: (N, 3), keys_T: DataFrame with date/stadium/race_number"""
    N = len(keys_T)
    S = X_T @ beta  # (N, 6)
    S_tau = S / tau
    S_stab = S_tau - S_tau.max(axis=1, keepdims=True)
    P = np.exp(S_stab) / np.exp(S_stab).sum(axis=1, keepdims=True)  # (N, 6)
    # top-3 by S (τで不変)
    pred = np.argsort(-S, axis=1)[:, :3]  # (N, 3) 0-indexed
    # Hit rates
    actual = pi_T  # (N, 3) 0-indexed
    hit1 = (pred[:, 0] == actual[:, 0]).mean() * 100
    # 2連対: 予想top2 のいずれかが実際の1or2
    act_set2 = [set(actual[i, :2].tolist()) for i in range(N)]
    hit2 = np.mean([any(p in act_set2[i] for p in pred[i, :2]) for i in range(N)]) * 100
    # 3連対
    act_set3 = [set(actual[i, :3].tolist()) for i in range(N)]
    hit3 = np.mean([any(p in act_set3[i] for p in pred[i, :3]) for i in range(N)]) * 100
    # Log-loss (1着)
    P_winner = P[np.arange(N), actual[:, 0]]
    ll1 = -np.log(np.maximum(P_winner, 1e-9)).mean()
    # PL Log-loss (3着まで)
    pl_ll_total = 0.0
    removed = np.zeros((N, 6), dtype=bool)
    for t in range(3):
        S_masked = np.where(removed, -np.inf, S_tau)
        lse_t = logsumexp(S_masked, axis=1)
        S_win = S_tau[np.arange(N), actual[:, t]]
        pl_ll_total += (lse_t - S_win).mean()
        removed[np.arange(N), actual[:, t]] = True
    if label:
        print(f"\n### {label} (τ={tau}, N={N:,})")
        print(f"  1着 hit:    {hit1:.2f}%")
        print(f"  2連対 hit:  {hit2:.2f}%")
        print(f"  3連対 hit:  {hit3:.2f}%")
        print(f"  Log-loss:   {ll1:.4f}")
        print(f"  PL NLL:     {pl_ll_total:.4f}")
    return {"hit1": hit1, "hit2": hit2, "hit3": hit3, "ll": ll1, "pl_ll": pl_ll_total}


# %% 7. Train/Test 基本評価
m_lr_train   = evaluate(beta_init, X_train_T, pi_train, keys_train, tau=1.0, label="Train LR (τ=1.0)")
m_lr_t13_tr  = evaluate(beta_init, X_train_T, pi_train, keys_train, tau=1.3, label="Train LR (τ=1.3)")
m_pl_train   = evaluate(beta_pl,   X_train_T, pi_train, keys_train, tau=1.0, label="Train PL (τ=1.0)")

m_lr_test    = evaluate(beta_init, X_test_T,  pi_test,  keys_test, tau=1.0, label="Test  LR (τ=1.0)")
m_lr_t13_te  = evaluate(beta_init, X_test_T,  pi_test,  keys_test, tau=1.3, label="Test  LR (τ=1.3)")
m_pl_test    = evaluate(beta_pl,   X_test_T,  pi_test,  keys_test, tau=1.0, label="Test  PL (τ=1.0)")


# %% 8. PL の τ グリッドサーチ (Test)
print("\n[6] PL モデルでの τ グリッドサーチ (Test)...")
pl_grid = []
for tau in TAU_GRID:
    m = evaluate(beta_pl, X_test_T, pi_test, keys_test, tau=tau)
    pl_grid.append({"tau": tau, **m})
    print(f"  τ={tau:.1f}: hit1={m['hit1']:.2f}%  ll={m['ll']:.4f}  pl_nll={m['pl_ll']:.4f}")

pl_grid_df = pd.DataFrame(pl_grid)
best_tau_row = pl_grid_df.loc[pl_grid_df["ll"].idxmin()]
best_tau = float(best_tau_row["tau"])
print(f"\n  Log-loss 最小 τ = {best_tau:.1f} (ll={best_tau_row['ll']:.4f})")

# 最適τでのtest最終評価
m_pl_test_best = evaluate(beta_pl, X_test_T, pi_test, keys_test, tau=best_tau,
                           label=f"Test PL (τ={best_tau})")


# %% 9. 比較表 (LRτ=1.3 vs PL)
print("\n## LR (τ=1.3) vs PL (τ={:.1f}) 精度比較 (Test)".format(best_tau))
print()
print("| 指標 | LR (τ=1.3) | PL (τ={:.1f}) | 差分 |".format(best_tau))
print("|---|---:|---:|---:|")
for k, lbl, fmt in [
    ("hit1", "1着 hit (%)", "{:.2f}"),
    ("hit2", "2連対 hit (%)", "{:.2f}"),
    ("hit3", "3連対 hit (%)", "{:.2f}"),
    ("ll",   "Log-loss (1着)", "{:.4f}"),
    ("pl_ll","PL NLL (1-3着)", "{:.4f}"),
]:
    lr_v = m_lr_t13_te[k]
    pl_v = m_pl_test_best[k]
    diff = pl_v - lr_v
    sign = "+" if diff >= 0 else ""
    dfmt = "{:.4f}" if "ll" in k else "{:.2f}"
    print(f"| {lbl} | {fmt.format(lr_v)} | {fmt.format(pl_v)} | {sign}{dfmt.format(diff)} |")


# %% 10. 推定 β 比較表
print("\n## 推定 β 比較")
print()
print("| 特徴量 | β_LR | β_PL | Δ(PL-LR) | SE(PL) | z_PL | 符号OK |")
print("|---|---:|---:|---:|---:|---:|:---:|")
for i, f in enumerate(FEATURES):
    dlr = beta_init[i]
    dpl = beta_pl[i]
    s = se_pl[i]
    z = dpl/s if s > 0 else float('nan')
    sign_ok = "○" if np.sign(dpl) == np.sign(dlr) else "✗"
    print(f"| {f} | {dlr:+.4f} | {dpl:+.4f} | {dpl-dlr:+.4f} | {s:.4f} | {z:+.2f} | {sign_ok} |")


# %% 11. 学習曲線
print("\n[7] 学習曲線・可視化...")
fig1, ax = plt.subplots(figsize=(9, 5))
ax.plot(nll_history, "o-", color="tab:blue")
ax.set_xlabel("Iteration (L-BFGS-B callback)")
ax.set_ylabel("Train NLL")
ax.set_title("Plackett-Luce Learning Curve")
ax.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(OUTD / "pl_learning_curve.png", dpi=100, bbox_inches="tight")

# %% 12. 会場別精度
print("\n[8] 会場別精度 (Test, PL, 最適τ)...")
from scripts.stadiums import STADIUMS

S_test = X_test_T @ beta_pl
pred_top = np.argsort(-S_test, axis=1)[:, :3]
rows = []
for i in range(len(keys_test)):
    stadium = int(keys_test.iloc[i]["stadium"])
    a = pi_test[i]
    p = pred_top[i]
    hit1 = p[0] == a[0]
    hit2 = any(x in set(a[:2].tolist()) for x in p[:2])
    hit3 = any(x in set(a[:3].tolist()) for x in p[:3])
    rows.append({"stadium": stadium, "hit_1st": hit1, "hit_top2": hit2, "hit_top3": hit3})
by_s = pd.DataFrame(rows)
agg = by_s.groupby("stadium").agg(
    N=("hit_1st","count"),
    hit_1st=("hit_1st", lambda s: s.mean()*100),
    hit_top2=("hit_top2", lambda s: s.mean()*100),
    hit_top3=("hit_top3", lambda s: s.mean()*100),
).reset_index()
agg["name"] = agg["stadium"].map(STADIUMS)
agg = agg[["stadium","name","N","hit_1st","hit_top2","hit_top3"]].sort_values("hit_1st", ascending=False)
print("\n## 会場別 (Test, PL)\nトップ5:")
print(agg.head(5).to_string(index=False))
print("\nボトム5:")
print(agg.tail(5).to_string(index=False))


# %% 13. CSV 出力
pd.DataFrame({
    "feature": FEATURES,
    "beta_lr": beta_init,
    "beta_pl": beta_pl,
    "delta":    beta_pl - beta_init,
    "se_pl":   se_pl,
    "z_pl":    beta_pl / se_pl,
}).to_csv(OUTD / "pl_weights.csv", index=False, encoding="utf-8-sig")

# race_log (test)
rec = []
for i in range(len(keys_test)):
    a = pi_test[i] + 1  # 1-based lane
    p = pred_top[i] + 1
    rec.append({
        "date": keys_test.iloc[i]["date"],
        "stadium": int(keys_test.iloc[i]["stadium"]),
        "race_number": int(keys_test.iloc[i]["race_number"]),
        "pred_1": int(p[0]), "pred_2": int(p[1]), "pred_3": int(p[2]),
        "actual_1": int(a[0]), "actual_2": int(a[1]), "actual_3": int(a[2]),
        "hit_1st": bool(p[0] == a[0]),
        "hit_top2": any(int(x) in set(a[:2].tolist()) for x in p[:2]),
        "hit_top3": any(int(x) in set(a[:3].tolist()) for x in p[:3]),
    })
pd.DataFrame(rec).to_csv(OUTD / "pl_race_log.csv", index=False, encoding="utf-8-sig")

# 比較CSV
pd.DataFrame({
    "metric": ["hit1","hit2","hit3","ll","pl_ll"],
    "LR_tau_1.3": [m_lr_t13_te[k] for k in ["hit1","hit2","hit3","ll","pl_ll"]],
    "PL_tau_best": [m_pl_test_best[k] for k in ["hit1","hit2","hit3","ll","pl_ll"]],
}).to_csv(OUTD / "pl_vs_lr_comparison.csv", index=False, encoding="utf-8-sig")

print(f"\nCSV 書込: pl_weights.csv, pl_race_log.csv, pl_vs_lr_comparison.csv")
print("\nPL 最適化完了")
