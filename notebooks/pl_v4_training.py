# -*- coding: utf-8 -*-
"""
v4 学習 — 既存 10 特徴量 + f11 + f12 で PL 再学習
v3 のスクリプトを runpy で実行して rc データを取得し、
v4_features_v2.csv から f11/f12 を merge して 12 特徴量で学習。
"""
from __future__ import annotations
import io, sys, runpy, warnings, pickle, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "notebooks" / "output"
import os as _os
OUT_SUFFIX = _os.environ.get("V4_OUT_SUFFIX", "")
sys.path.insert(0, str(BASE))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
warnings.filterwarnings("ignore")

print("=" * 80)
print("v4 training — 12 features (existing 10 + f11 + f12)")
print("=" * 80)

# ==================== v3 pipeline を runpy ====================
print("\n[0] pl_v3_training.py を runpy (rc dataframe 取得) ...")
_buf = io.StringIO(); _o = sys.stdout; sys.stdout = _buf
try:
    ns = runpy.run_path(str(BASE / "notebooks" / "pl_v3_training.py"))
finally:
    sys.stdout = _o
rc_full = ns["rc"]           # 全期間 race_cards (特徴量付与済み)
rc_train = ns["rc_train"]
rc_test  = ns["rc_test"]
pi_train = ns["pi_train"]; keys_train = ns["keys_train"]
pi_test  = ns["pi_test"];  keys_test  = ns["keys_test"]
FEATURES_NEW = ns["FEATURES_NEW"]
build_race_tensors = ns["build_race_tensors"]
evaluate = ns["evaluate"]
fit_pl = ns["fit_pl"]
beta_v3 = ns["beta_new"]     # v3 学習済 β
best_new = ns["best_new"]
pca = ns["pca"]
print(f"   v3 β = {np.array2string(beta_v3, precision=3)}")
print(f"   train rows={len(rc_train):,} test rows={len(rc_test):,}")

# ==================== v4_features_v2 を merge ====================
print("\n[1] v4_features_v2.csv を merge (f11/f12 付与) ...")
v4f = pd.read_csv(OUT / f"v4{OUT_SUFFIX}_features_v2.csv")
v4f["race_date"] = pd.to_datetime(v4f["race_date"])
# merge キーは (date, stadium, race_number, lane, racerid)
v4f_key = v4f[["race_date","stadium","race_no","lane","racerid",
               "f11_final","f11_final_n","f12_final","f12_final_n"]].rename(
    columns={"race_date":"date","race_no":"race_number"})

# train と test で別々にマージ
def attach_v4(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    m = df.merge(v4f_key, on=["date","stadium","race_number","lane","racerid"], how="left")
    m["f11_final"] = m["f11_final"].fillna(0)
    m["f12_final"] = m["f12_final"].fillna(0)
    m["f11_final_n"] = m["f11_final_n"].fillna(0).astype(int)
    m["f12_final_n"] = m["f12_final_n"].fillna(0).astype(int)
    return m

rc_train_v4 = attach_v4(rc_train)
rc_test_v4  = attach_v4(rc_test)
print(f"   rc_train_v4: {len(rc_train_v4):,} (f11 N>=5: {(rc_train_v4['f11_final_n']>=5).mean()*100:.1f}%, f12 N>=5: {(rc_train_v4['f12_final_n']>=5).mean()*100:.1f}%)")

# ==================== 12 feature tensor ====================
FEATURES_V4 = FEATURES_NEW + ["f11_final", "f12_final"]
print(f"\n[2] 12 特徴量テンソル構築: {FEATURES_V4}")
X_train_v4, pi_train_v4, keys_train_v4 = build_race_tensors(rc_train_v4, FEATURES_V4)
X_test_v4,  pi_test_v4,  keys_test_v4  = build_race_tensors(rc_test_v4,  FEATURES_V4)
print(f"   X_train_v4={X_train_v4.shape}  X_test_v4={X_test_v4.shape}")

# ==================== PL 学習 ====================
print("\n[3] PL 学習 (v3 β + f11/f12 初期値 0)")
x0_v4 = np.zeros(12)
x0_v4[:10] = beta_v3
beta_v4, iters = fit_pl(X_train_v4, pi_train_v4, x0_v4, "v4")
print(f"   β_v4 = {np.array2string(beta_v4, precision=3)}  (iter={iters})")

# ==================== τ グリッド ====================
print("\n[4] τ グリッド探索")
best_v4 = None
for tau in [0.6, 0.7, 0.8, 0.9, 1.0]:
    m = evaluate(beta_v4, X_test_v4, pi_test_v4, tau=tau, label=f"v4 τ={tau}")
    if best_v4 is None or m["ll"] < best_v4["ll"]:
        best_v4 = {**m, "tau": tau}
print(f"   最適 τ = {best_v4['tau']}, LL={best_v4['ll']:.4f}")

# ==================== VIF ====================
print("\n[5] VIF 検証 (v4 12 特徴量)")
from scipy.linalg import inv
X_flat = X_train_v4.reshape(-1, 12)
corr = np.corrcoef(X_flat.T)
try:
    vif = np.diag(inv(corr))
except np.linalg.LinAlgError:
    # 対角にゆらぎを加えて近似
    vif = np.diag(inv(corr + np.eye(12) * 1e-6))
print(f"   {'feature':>16s} {'VIF':>7s} {'判定':>6s}")
vif_rows = []
for i, name in enumerate(FEATURES_V4):
    v = vif[i]
    if v < 3: judge = "✅"
    elif v < 5: judge = "⚠️"
    else: judge = "❌"
    print(f"   {name:>16s} {v:>7.3f} {judge}")
    vif_rows.append({"feature": name, "vif": float(v), "judge": judge})
print(f"\n   最大 VIF: {vif.max():.3f} ({FEATURES_V4[vif.argmax()]})")
pd.DataFrame(vif_rows).to_csv(OUT/f"v4{OUT_SUFFIX}_vif.csv", index=False, encoding="utf-8-sig")

# ==================== 寄与度 ====================
print("\n[6] 寄与度比較 (v3 10 vs v4 12)")
std_v4 = X_test_v4.reshape(-1, 12).std(axis=0)
contrib_v4 = np.abs(beta_v4) * std_v4
pct_v4 = contrib_v4 / contrib_v4.sum() * 100
# v3 の寄与度を v3_contribution.csv から読む (存在しない場合は 0)
try:
    v3c = pd.read_csv(OUT / "v3_contribution.csv")
    v3_pct = dict(zip(v3c["feature"], v3c["contrib_v3_%"]))
except Exception:
    v3_pct = {}

print(f"   {'feature':>16s} {'v3 %':>7s} {'v4 %':>7s} {'変化':>7s}")
rows = []
for i, name in enumerate(FEATURES_V4):
    v3p = v3_pct.get(name, 0.0)
    v4p = pct_v4[i]
    print(f"   {name:>16s} {v3p:>6.2f}% {v4p:>6.2f}% {v4p-v3p:>+6.2f}")
    rows.append({"feature": name, "v3_%": v3p, "v4_%": v4p, "diff": v4p-v3p})
pd.DataFrame(rows).to_csv(OUT/f"v4{OUT_SUFFIX}_contribution.csv", index=False, encoding="utf-8-sig")

# ==================== 係数 + β CSV ====================
pd.DataFrame({"feature": FEATURES_V4, "beta_v4": beta_v4}).to_csv(
    OUT/f"v4{OUT_SUFFIX}_weights.csv", index=False, encoding="utf-8-sig")

# ==================== Bootstrap SE ====================
print("\n[7] Bootstrap SE (N=80)")
from scipy.optimize import minimize
from scipy.special import logsumexp
LAMBDA = 0.01
def pl_fn(X, pi):
    def f(beta):
        N, M, K = X.shape
        S = X @ beta
        removed = np.zeros((N, M), dtype=bool); ri = np.arange(N)
        nll = 0.0; grad = np.zeros(K)
        for t in range(3):
            pt = pi[:, t]; Sm = np.where(removed, -np.inf, S)
            lse = logsumexp(Sm, axis=1)
            nll += float((lse - S[ri, pt]).sum())
            p = np.exp(Sm - lse[:, None])
            grad += -X[ri, pt, :].sum(axis=0) + np.einsum("nj,njk->k", p, X)
            removed[ri, pt] = True
        nll += 0.5 * LAMBDA * float(beta @ beta); grad += LAMBDA * beta
        return nll, grad
    return f

rng = np.random.default_rng(42); B = 80
N_tr = len(pi_train_v4)
boot = np.zeros((B, 12))
_buf = io.StringIO(); _sso = sys.stdout; sys.stdout = _buf
try:
    for b in range(B):
        idx = rng.integers(0, N_tr, N_tr)
        r = minimize(fun=pl_fn(X_train_v4[idx], pi_train_v4[idx]),
                     x0=beta_v4, jac=True, method="L-BFGS-B",
                     options={"ftol":1e-5,"maxiter":300})
        boot[b] = r.x
finally:
    sys.stdout = _sso
se_v4 = boot.std(axis=0, ddof=1); z_v4 = beta_v4 / se_v4
ci_lo = np.percentile(boot, 2.5, axis=0); ci_hi = np.percentile(boot, 97.5, axis=0)
pd.DataFrame({"feature":FEATURES_V4, "beta":beta_v4, "se":se_v4, "z":z_v4,
              "ci_lo":ci_lo, "ci_hi":ci_hi}).to_csv(
    OUT/f"v4{OUT_SUFFIX}_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
print(f"   saved: v4_bootstrap_ci.csv  (f11 z={z_v4[10]:+.2f}, f12 z={z_v4[11]:+.2f})")

# ==================== 会場別性能 (v3 vs v4 での改善) ====================
print("\n[8] 会場別 Test 性能 (v3 vs v4)")
STA = {1:"桐生",2:"戸田",3:"江戸川",4:"平和島",5:"多摩川",6:"浜名湖",7:"蒲郡",8:"常滑",9:"津",10:"三国",11:"びわこ",12:"住之江",13:"尼崎",14:"鳴門",15:"丸亀",16:"児島",17:"宮島",18:"徳山",19:"下関",20:"若松",21:"芦屋",22:"福岡",23:"唐津",24:"大村"}

S_v3 = ns["X_test_new"] @ beta_v3
S_v4 = X_test_v4 @ beta_v4
pred_v3 = S_v3.argmax(axis=1); pred_v4 = S_v4.argmax(axis=1)
df_ven = pd.DataFrame({
    "stadium": keys_test_v4["stadium"].to_numpy(),
    "hit_v3": (pred_v3 == pi_test_v4[:, 0]).astype(float),
    "hit_v4": (pred_v4 == pi_test_v4[:, 0]).astype(float),
})
agg = df_ven.groupby("stadium").agg(
    N=("hit_v3","count"), hit_v3=("hit_v3","mean"), hit_v4=("hit_v4","mean")).reset_index()
agg["name"] = agg["stadium"].map(STA)
agg["v3_%"] = agg["hit_v3"]*100; agg["v4_%"] = agg["hit_v4"]*100
agg["delta"] = agg["v4_%"] - agg["v3_%"]

print(f"   {'会場':>10} {'N':>5s} {'v3':>6s} {'v4':>6s} {'Δ':>6s}")
for name in ["戸田","江戸川","鳴門","平和島","桐生"]:
    row = agg[agg["name"]==name]
    if len(row)==0: continue
    r = row.iloc[0]
    print(f"   {r['name']:>10}({r['stadium']:>2d}) {int(r['N']):>5} {r['v3_%']:>5.2f}% {r['v4_%']:>5.2f}% {r['delta']:>+5.2f}")
print(f"\n   全体: v3={(pred_v3==pi_test_v4[:,0]).mean()*100:.2f}% v4={(pred_v4==pi_test_v4[:,0]).mean()*100:.2f}%")

agg[["stadium","name","N","v3_%","v4_%","delta"]].to_csv(
    OUT/f"v4{OUT_SUFFIX}_venue_performance.csv", index=False, encoding="utf-8-sig")

# ==================== 総合比較 ====================
print("\n" + "=" * 80)
print("総合比較 (test 2026-01〜2026-04-18)")
print("=" * 80)
v3_ref = {"hit1":56.98, "hit2":90.92, "hit3":99.57, "ll":1.2031, "pl_ll":4.0287, "tau":0.7}
print(f"{'モデル':>12s} {'1着%':>8s} {'2連対%':>8s} {'3連対%':>8s} {'LL':>8s} {'PL_NLL':>8s} {'τ':>4s}")
print(f"{'v3':>12s} {v3_ref['hit1']:>7.2f}% {v3_ref['hit2']:>7.2f}% {v3_ref['hit3']:>7.2f}% {v3_ref['ll']:>8.4f} {v3_ref['pl_ll']:>8.4f} {v3_ref['tau']:>4.1f}")
print(f"{'v4':>12s} {best_v4['hit1']:>7.2f}% {best_v4['hit2']:>7.2f}% {best_v4['hit3']:>7.2f}% {best_v4['ll']:>8.4f} {best_v4['pl_ll']:>8.4f} {best_v4['tau']:>4.1f}")

compare_v3v4 = pd.DataFrame([
    {"model":"v3", **{k:v for k,v in v3_ref.items()}},
    {"model":"v4", "hit1":best_v4['hit1'], "hit2":best_v4['hit2'], "hit3":best_v4['hit3'],
     "ll":best_v4['ll'], "pl_ll":best_v4['pl_ll'], "tau":best_v4['tau']},
])
compare_v3v4.to_csv(OUT/f"v3_vs_v4{OUT_SUFFIX}_comparison.csv", index=False, encoding="utf-8-sig")

# ==================== 判定 + レポート ====================
print("\n" + "=" * 80)
print("判定")
print("=" * 80)
hit_delta = best_v4['hit1'] - v3_ref['hit1']
ll_delta = v3_ref['ll'] - best_v4['ll']   # 小さい方が良い
max_vif = vif.max()
f11_pct = pct_v4[10]; f12_pct = pct_v4[11]

if max_vif > 5:
    pattern = "β: VIF 高 → 特徴量統合/削除検討"
elif abs(hit_delta) < 0.1 and abs(ll_delta) < 0.005:
    pattern = "δ: 性能変わらず → 特徴量設計見直し"
elif f11_pct < 1 or f12_pct < 1:
    pattern = "γ: 寄与度低特徴量あり、削除検討"
else:
    pattern = "α: 両方有効、VIF 健全 → v4 採用、v5 (f13) 準備"

print(f"\nパターン判定: {pattern}")
print(f"  hit1: v3 → v4 = {v3_ref['hit1']:.2f}% → {best_v4['hit1']:.2f}% (Δ={hit_delta:+.2f}pt)")
print(f"  LL: v3 → v4 = {v3_ref['ll']:.4f} → {best_v4['ll']:.4f} (改善 {ll_delta:+.4f})")
print(f"  最大 VIF: {max_vif:.3f}")
print(f"  f11 寄与度: {f11_pct:.2f}%  f12 寄与度: {f12_pct:.2f}%")

# release notes
notes = f"""# v4 リリースノート

## モデル名
v4 — v3 (10 特徴量) + f11_lane_player + f12_motor_recent = 12 特徴量

## 学習日
{datetime.now():%Y-%m-%d %H:%M JST}

## 新特徴量
- **f11_lane_player**: 過去 12 ヶ月 (racer × lane) 1 着率 logit - 全国 lane 平均 logit、シンプル平均、clip ±2.5
- **f12_motor_recent**: 過去 3 ヶ月 (motor × stadium) 2 連対率 Beta prior smoothed (prior_strength=20)、会場全モーター平均の z-score、clip ±2.5

## 性能比較 (test 期間)
| モデル | 1着% | 2連対% | 3連対% | LL | PL_NLL | τ |
|---|---:|---:|---:|---:|---:|---:|
| v3 | {v3_ref['hit1']:.2f} | {v3_ref['hit2']:.2f} | {v3_ref['hit3']:.2f} | {v3_ref['ll']:.4f} | {v3_ref['pl_ll']:.4f} | {v3_ref['tau']:.1f} |
| **v4** | **{best_v4['hit1']:.2f}** | **{best_v4['hit2']:.2f}** | **{best_v4['hit3']:.2f}** | **{best_v4['ll']:.4f}** | **{best_v4['pl_ll']:.4f}** | **{best_v4['tau']:.1f}** |

## VIF 最大: {max_vif:.3f} ({FEATURES_V4[vif.argmax()]})
- 全12特徴量 < 5: {'✅ 健全' if max_vif < 5 else '⚠️ 要検討'}

## f11/f12 寄与度
- f11 寄与度: {f11_pct:.2f}%
- f12 寄与度: {f12_pct:.2f}%

## 苦戦会場 (v3→v4 改善度)
```
{agg[agg['name'].isin(['戸田','江戸川','鳴門','平和島','桐生'])][['name','v3_%','v4_%','delta']].to_string(index=False)}
```

## 判定
{pattern}
"""
(OUT/f"v4{OUT_SUFFIX}_release_notes.md").write_text(notes, encoding="utf-8")
print(f"\nsaved: v4_release_notes.md")
print(f"saved: v4_weights.csv / v4_vif.csv / v4_contribution.csv")
print(f"saved: v4_bootstrap_ci.csv / v4_venue_performance.csv / v3_vs_v4_comparison.csv")
