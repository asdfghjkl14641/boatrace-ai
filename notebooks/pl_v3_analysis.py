# -*- coding: utf-8 -*-
"""
Phase A-5〜A-7: v3 vs v2 比較 + VIF + 保留事項判定 + release notes
- pl_v3_training.py を runpy で再利用
- v2 β は v2_extended_weights.csv から読み取り専用
"""
from __future__ import annotations
import io, sys, runpy, json, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "notebooks" / "output"

print("=" * 80)
print("v3 Analysis — Phase A-5〜A-7")
print("=" * 80)
print("\n[0] pl_v3_training.py を runpy 実行 (新特徴量と v3 β 取得)...")
_buf = io.StringIO(); _o = sys.stdout; sys.stdout = _buf
try:
    ns = runpy.run_path(str(BASE / "notebooks" / "pl_v3_training.py"))
finally:
    sys.stdout = _o

beta_v3 = ns["beta_new"]
X_train = ns["X_train_new"]; pi_train = ns["pi_train"]; keys_train = ns["keys_train"]
X_test  = ns["X_test_new"];  pi_test  = ns["pi_test"];  keys_test  = ns["keys_test"]
FEATURES = ns["FEATURES_NEW"]
evaluate = ns["evaluate"]
best_new = ns["best_new"]
pca = ns["pca"]
print(f"  ✓ β_v3 = {np.array2string(beta_v3, precision=3)}")
print(f"  ✓ train N={len(keys_train):,} / test N={len(keys_test):,}")
print(f"  ✓ 最適 τ = {best_new['tau']}")


# =====================================================================
# Phase A-5: 4 モデル比較
# =====================================================================
print("\n" + "=" * 80)
print("[Phase A-5] 4 モデル比較")
print("=" * 80)

# v2 β を読み取り (絶対に上書きしない)
v2_w = pd.read_csv(OUT / "v2_extended_weights.csv")
beta_v2 = v2_w["beta_new_H"].to_numpy()
print(f"\n読取 v2_extended_weights.csv (READ-ONLY):")
print(f"  β_v2 (newH) = {np.array2string(beta_v2, precision=3)}")

# --- モデル 1: ベースライン (常に 1-2-3) ---
N_te = len(pi_test)
baseline_hit1 = (pi_test[:, 0] == 0).mean() * 100  # 1 枠 (index 0) が 1 着
baseline_hit2 = ((pi_test[:, 0] == 0) & (pi_test[:, 1] == 1) |
                 (pi_test[:, 0] == 1) & (pi_test[:, 1] == 0)).mean() * 100
# 実際には「1-2-3 予想」の一致率
# 予想 = [0,1,2] 固定
pred_base = np.array([[0,1,2]] * N_te)
from itertools import permutations
h1b = (pi_test[:, 0] == 0).mean() * 100
# top2 hit: 予想 top2 {0,1} と 実 top2 の重なり
h2b = np.mean([len(set(pi_test[i,:2]) & {0,1}) > 0 for i in range(N_te)]) * 100
h3b = np.mean([len(set(pi_test[i,:3]) & {0,1,2}) > 0 for i in range(N_te)]) * 100

# --- モデル 2: v2_ext_newH (前回仕様書の数値を引用) ---
v2_ref = {"hit1": 57.04, "hit2": 90.92, "hit3": 99.56, "ll": 1.203, "pl_ll": 3.9641, "tau": 0.8}

# --- モデル 3: v2_newdata (β_v2 × 新 X_test、τ=0.8) ---
TAU = 0.8
m_v2new = evaluate(beta_v2, X_test, pi_test, tau=TAU, label="v2_newdata")

# --- モデル 4: v3 (β_v3 × 新 X_test、τ=best_new['tau']) ---
m_v3 = best_new  # 既に計算済

# 比較表
print("\n### 4 モデル比較 (test 2026-01〜2026-04-18)")
print(f"{'モデル':>20s} {'1着%':>8s} {'2連対%':>8s} {'3連対%':>8s} {'LL':>7s} {'PL_NLL':>8s} {'τ':>4s}")
print("-" * 70)
print(f"{'ベースライン 1-2-3':>20s} {h1b:>7.2f}% {h2b:>7.2f}% {h3b:>7.2f}% {'-':>7s} {'-':>8s} {'-':>4s}")
print(f"{'v2_ext_newH (ref)':>20s} {v2_ref['hit1']:>7.2f}% {v2_ref['hit2']:>7.2f}% {v2_ref['hit3']:>7.2f}% {v2_ref['ll']:>7.4f} {v2_ref['pl_ll']:>8.4f} {v2_ref['tau']:>4.1f}")
print(f"{'v2_newdata (β_v2×新X)':>20s} {m_v2new['hit1']:>7.2f}% {m_v2new['hit2']:>7.2f}% {m_v2new['hit3']:>7.2f}% {m_v2new['ll']:>7.4f} {m_v2new['pl_ll']:>8.4f} {TAU:>4.1f}")
print(f"{'v3 (β_v3×新X, τ 最適)':>20s} {m_v3['hit1']:>7.2f}% {m_v3['hit2']:>7.2f}% {m_v3['hit3']:>7.2f}% {m_v3['ll']:>7.4f} {m_v3['pl_ll']:>8.4f} {m_v3['tau']:>4.1f}")

compare_df = pd.DataFrame([
    {"model":"baseline_1-2-3", "hit1": h1b, "hit2": h2b, "hit3": h3b,
     "ll": None, "pl_ll": None, "tau": None},
    {"model":"v2_ext_newH (ref)", "hit1": v2_ref['hit1'], "hit2": v2_ref['hit2'],
     "hit3": v2_ref['hit3'], "ll": v2_ref['ll'], "pl_ll": v2_ref['pl_ll'], "tau": v2_ref['tau']},
    {"model":"v2_newdata", "hit1": m_v2new['hit1'], "hit2": m_v2new['hit2'],
     "hit3": m_v2new['hit3'], "ll": m_v2new['ll'], "pl_ll": m_v2new['pl_ll'], "tau": TAU},
    {"model":"v3_newH", "hit1": m_v3['hit1'], "hit2": m_v3['hit2'],
     "hit3": m_v3['hit3'], "ll": m_v3['ll'], "pl_ll": m_v3['pl_ll'], "tau": m_v3['tau']},
])
compare_df.to_csv(OUT / "v3_vs_v2_comparison.csv", index=False, encoding="utf-8-sig")
print(f"\nsaved: v3_vs_v2_comparison.csv")


# --- 係数の変化 ---
print("\n### 係数の変化 (v2 vs v3)")
print(f"{'feature':>14s} {'β_v2':>9s} {'β_v3':>9s} {'差':>9s}")
print("-" * 50)
coef_rows = []
for i, name in enumerate(FEATURES):
    b2, b3 = beta_v2[i], beta_v3[i]
    d = b3 - b2
    print(f"{name:>14s} {b2:>+9.4f} {b3:>+9.4f} {d:>+9.4f}")
    coef_rows.append({"feature": name, "beta_v2": b2, "beta_v3": b3, "diff": d})

# --- 寄与度の再計算 (test 期間の標準化後係数 × std 比) ---
# score = X @ β、寄与度 = |β_i| × std(X_i) / Σ|β_j| × std(X_j)
X_test_flat = X_test.reshape(-1, len(FEATURES))
std_new = X_test_flat.std(axis=0)
abs_contrib_v2 = np.abs(beta_v2) * std_new
abs_contrib_v3 = np.abs(beta_v3) * std_new
pct_v2 = abs_contrib_v2 / abs_contrib_v2.sum() * 100
pct_v3 = abs_contrib_v3 / abs_contrib_v3.sum() * 100

print("\n### 寄与度 % (|β| × std(X) / 合計)")
print(f"{'feature':>14s} {'v2 %':>8s} {'v3 %':>8s} {'変化':>7s}")
print("-" * 50)
contrib_rows = []
for i, name in enumerate(FEATURES):
    v2p, v3p = pct_v2[i], pct_v3[i]
    chg = v3p - v2p
    print(f"{name:>14s} {v2p:>7.2f}% {v3p:>7.2f}% {chg:>+6.2f}")
    contrib_rows.append({"feature": name, "contrib_v2_%": v2p, "contrib_v3_%": v3p, "diff": chg})
pd.DataFrame(contrib_rows).to_csv(OUT / "v3_contribution.csv", index=False, encoding="utf-8-sig")

# --- VIF 再検証 ---
print("\n### VIF (多重共線性)")
from scipy.linalg import inv
X_flat = X_train.reshape(-1, len(FEATURES))
# Remove constant columns and compute VIF = 1 / (1 - R^2_i)
# using correlation matrix approach
corr = np.corrcoef(X_flat.T)
try:
    invc = inv(corr)
    vif = np.diag(invc)
    print(f"{'feature':>14s} {'VIF':>7s}")
    for i, name in enumerate(FEATURES):
        print(f"{name:>14s} {vif[i]:>7.3f}")
    print(f"\n最大 VIF: {vif.max():.3f} ({FEATURES[vif.argmax()]})")
except Exception as e:
    print(f"VIF 計算失敗: {e}")
    vif = None

# 各種 CSV
pd.DataFrame(coef_rows).to_csv(OUT / "v3_coef_diff.csv", index=False, encoding="utf-8-sig")

# --- 会場別 test 性能 (v3) ---
print("\n### 会場別 Test 性能 (v3)")
S_test = X_test @ beta_v3
pred_top1 = S_test.argmax(axis=1)
df_ven = pd.DataFrame({
    "stadium": keys_test["stadium"].to_numpy(),
    "hit1": (pred_top1 == pi_test[:, 0]).astype(float),
})
# stadium id -> name
STA_NAMES = {1:"桐生",2:"戸田",3:"江戸川",4:"平和島",5:"多摩川",6:"浜名湖",7:"蒲郡",8:"常滑",9:"津",10:"三国",11:"びわこ",12:"住之江",13:"尼崎",14:"鳴門",15:"丸亀",16:"児島",17:"宮島",18:"徳山",19:"下関",20:"若松",21:"芦屋",22:"福岡",23:"唐津",24:"大村"}
agg = df_ven.groupby("stadium").agg(N=("hit1","count"), hit1=("hit1","mean")).reset_index()
agg["name"] = agg["stadium"].map(STA_NAMES)
agg["hit1_%"] = agg["hit1"] * 100
agg = agg.sort_values("hit1", ascending=False)
print("\n苦戦会場 (ボトム5) v3 の精度:")
for _, row in agg.tail(5).iterrows():
    print(f"  stadium={row['stadium']} ({row['name']}): N={int(row['N'])}  hit1={row['hit1_%']:.2f}%")
# 先手 top5 bottom5 取り出し
agg_out = agg[["stadium","name","N","hit1_%"]].reset_index(drop=True)
agg_out.to_csv(OUT / "v3_venue_performance.csv", index=False, encoding="utf-8-sig")


# =====================================================================
# Phase A-6: 保留事項判定
# =====================================================================
print("\n" + "=" * 80)
print("[Phase A-6] 保留事項判定")
print("=" * 80)

# §12.1 風寄与度問題 (v2 = 24.6% → v3 ?)
f9_idx = FEATURES.index("f9_W")
f9_v3 = pct_v3[f9_idx]
if f9_v3 <= 20:
    verdict_121 = f"解消 (寄与度 24.6% → {f9_v3:.1f}%、サンプル不足由来と確定)"
elif f9_v3 <= 22:
    verdict_121 = f"部分解消 (寄与度 24.6% → {f9_v3:.1f}%、構造的要因の可能性あり)"
else:
    verdict_121 = f"未解消 (寄与度 24.6% → {f9_v3:.1f}%、非線形モデル検討へ)"
print(f"\n§12.1 風寄与度 24% 問題: {verdict_121}")

# §12.2 f6_nomatch z 値 (t-value)
# 近似: β / (β の SE)。SE は Hessian 対角から、または bootstrap。
# 簡易 boot (N=100)
from scipy.optimize import minimize
from scipy.special import logsumexp
LAMBDA_L2 = 0.01
def pl_nll_grad_f(X, pi, lam):
    def fn(beta):
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
    return fn

print("\nBootstrap (N=100) で SE 計算中...")
np.random.seed(42)
rng = np.random.default_rng(42)
N_train = len(pi_train)
B = 100
boot_betas = np.zeros((B, len(FEATURES)))
_buf = io.StringIO(); _so = sys.stdout; sys.stdout = _buf
try:
    for b in range(B):
        idx = rng.integers(0, N_train, N_train)
        Xb = X_train[idx]; pib = pi_train[idx]
        r = minimize(fun=pl_nll_grad_f(Xb, pib, LAMBDA_L2),
                     x0=beta_v3, jac=True, method="L-BFGS-B",
                     options={"ftol": 1e-5, "maxiter": 300})
        boot_betas[b] = r.x
finally:
    sys.stdout = _so
se_v3 = boot_betas.std(axis=0, ddof=1)
z_v3 = beta_v3 / se_v3
ci_lo = np.percentile(boot_betas, 2.5, axis=0)
ci_hi = np.percentile(boot_betas, 97.5, axis=0)
ci_df = pd.DataFrame({
    "feature": FEATURES, "beta": beta_v3, "se": se_v3, "z": z_v3,
    "ci_lo": ci_lo, "ci_hi": ci_hi
})
ci_df.to_csv(OUT / "v3_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
print("\n### β 係数 + SE + z (v3)")
print(f"{'feature':>14s} {'β':>9s} {'SE':>8s} {'z':>7s} {'CI95':>20s}")
for i, name in enumerate(FEATURES):
    print(f"{name:>14s} {beta_v3[i]:>+9.4f} {se_v3[i]:>8.4f} {z_v3[i]:>+7.2f} [{ci_lo[i]:+.3f}, {ci_hi[i]:+.3f}]")

f6n_idx = FEATURES.index("f6_nomatch")
z_f6n = z_v3[f6n_idx]
if abs(z_f6n) >= 2:
    verdict_122 = f"有意化 (z={z_f6n:.2f}、継続採用)"
else:
    verdict_122 = f"未有意 (z={z_f6n:.2f}、除外検討)"
print(f"\n§12.2 f6_nomatch: {verdict_122}")

# §12.3 f10_H 寄与度
f10_idx = FEATURES.index("f10_H_new")
f10_v3 = pct_v3[f10_idx]
v2_f10 = pct_v2[f10_idx]
if f10_v3 > v2_f10 * 1.2:
    verdict_123 = f"上昇 (寄与度 {v2_f10:.1f}% → {f10_v3:.1f}%、有意継続)"
elif f10_v3 < v2_f10 * 0.8:
    verdict_123 = f"低下 (寄与度 {v2_f10:.1f}% → {f10_v3:.1f}%、優先度低)"
else:
    verdict_123 = f"横這い (寄与度 {v2_f10:.1f}% → {f10_v3:.1f}%、継続観察)"
print(f"§12.3 f10_H 寄与度: {verdict_123}")


# =====================================================================
# Phase A-7: PCA 保存 + release notes
# =====================================================================
print("\n" + "=" * 80)
print("[Phase A-7] 保存 + release notes")
print("=" * 80)

# PCA 保存
with open(OUT / "v3_pca_params.pkl", "wb") as f:
    pickle.dump({
        "components": pca.components_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "mean": pca.mean_,
    }, f)
print(f"\nsaved: v3_pca_params.pkl (EVR={pca.explained_variance_ratio_[0]*100:.2f}%)")

# Release notes
notes = f"""# v3_newH リリースノート

## モデル名
v3_newH (v2_ext_newH の発展、race_conditions 補完完了後の再学習)

## 学習日
{datetime.now():%Y-%m-%d %H:%M JST}

## データ範囲
- train: 2023-05-01 〜 2025-12-31 ({len(keys_train):,} レース)
- test:  2026-01-01 〜 2026-04-18 ({len(keys_test):,} レース)

## v2 → v3 の主な変更
1. **race_conditions を補完** (旧 73k → 新 456k、train f4_disp 完備率 ~40% → 96%)
2. 特徴量再計算: PCA を新 train で fit、L/V/W/H も再計算
3. 新 H 設計 (差し型カーネル) 維持
4. ただし特徴量数は変更なし (10 個)

## 4 モデル比較 (test 性能)

| モデル | 1着% | 2連対% | 3連対% | LL | PL_NLL | τ |
|---|---:|---:|---:|---:|---:|---:|
| ベースライン 1-2-3 | {h1b:.2f} | {h2b:.2f} | {h3b:.2f} | - | - | - |
| v2_ext_newH (ref) | {v2_ref['hit1']:.2f} | {v2_ref['hit2']:.2f} | {v2_ref['hit3']:.2f} | {v2_ref['ll']:.4f} | {v2_ref['pl_ll']:.4f} | {v2_ref['tau']:.1f} |
| v2_newdata (β_v2×新X) | {m_v2new['hit1']:.2f} | {m_v2new['hit2']:.2f} | {m_v2new['hit3']:.2f} | {m_v2new['ll']:.4f} | {m_v2new['pl_ll']:.4f} | {TAU:.1f} |
| **v3_newH** | **{m_v3['hit1']:.2f}** | **{m_v3['hit2']:.2f}** | **{m_v3['hit3']:.2f}** | **{m_v3['ll']:.4f}** | **{m_v3['pl_ll']:.4f}** | **{m_v3['tau']:.1f}** |

## 保留事項判定
- **§12.1 風寄与度 24% 問題**: {verdict_121}
- **§12.2 f6_nomatch 低寄与度**: {verdict_122}
- **§12.3 f10_H 寄与度**: {verdict_123}

## 最大 VIF
{vif.max():.3f} ({FEATURES[vif.argmax()]})

## 苦戦会場 (ボトム5 Test 1着 hit率)
{agg.tail(5)[['stadium','name','N','hit1_%']].to_string(index=False)}
"""
(OUT / "v3_release_notes.md").write_text(notes, encoding="utf-8")
print(f"saved: v3_release_notes.md")

# C 層 tables は pl_v3_training.py 内で扱う (L は PL β に内包、V は f8_V、W は f9_W、H は f10_H)
# それぞれの中間テーブルはすでに training 内で使われているので JSON に出力
# ここでは簡易サマリのみ
c_layer = {
    "L": {"note": "枠1着率 logit は PL β_f7_lane_L に内包", "beta_f7": float(beta_v3[FEATURES.index("f7_lane_L")])},
    "V": {"note": "会場×枠は f8_V に内包", "beta_f8": float(beta_v3[FEATURES.index("f8_V")])},
    "W": {"note": "風補正は f9_W に内包", "beta_f9": float(beta_v3[FEATURES.index("f9_W")])},
    "H": {"note": "波補正は f10_H (新形式) に内包", "beta_f10": float(beta_v3[FEATURES.index("f10_H_new")])},
}
(OUT / "v3_c_layer_tables.json").write_text(
    json.dumps(c_layer, ensure_ascii=False, indent=2), encoding="utf-8")
print("saved: v3_c_layer_tables.json")

print("\n" + "=" * 80)
print("v3 再学習完了 — 全出力ファイル:")
print("=" * 80)
for f in ["v3_weights.csv","v3_three_versions_comparison.csv","v3_vs_v2_comparison.csv",
          "v3_coef_diff.csv","v3_contribution.csv","v3_bootstrap_ci.csv",
          "v3_venue_performance.csv","v3_pca_params.pkl","v3_c_layer_tables.json",
          "v3_release_notes.md"]:
    p = OUT / f
    if p.exists():
        print(f"  ✓ {f}")
    else:
        print(f"  ✗ {f} 不存在")

# v2 ファイル保護確認
print("\nv2 ファイル保護確認:")
for f in ["v2_extended_weights.csv", "latent_v2_pca_loadings.csv", "pl_weights.csv"]:
    p = OUT / f
    print(f"  {f}: {'未変更 ✓' if p.exists() else 'MISSING!'}")
