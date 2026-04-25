# -*- coding: utf-8 -*-
"""Phase C 最終戦略検証: calibrated バックテストからシナリオ集計."""
from __future__ import annotations
import io, sys, math, runpy, warnings
from pathlib import Path
from itertools import permutations
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "notebooks" / "output"
sys.path.insert(0, str(BASE))
sys.stdout.reconfigure(encoding="utf-8")

PERMS = [(a, b, c) for a, b, c in permutations([1,2,3,4,5,6], 3)]

def binomial_ci(hits, total):
    if total == 0: return (0.0, 0.0)
    p = hits/total
    se = math.sqrt(p*(1-p)/total); z = 1.96
    return (max(0, p-z*se), min(1, p+z*se))

def roi_ci_approx(stake, pay, hit_races, bet_races):
    """ROI CI 近似: hit_rate_race × (avg_pay_per_hit / avg_stake_per_race)."""
    if bet_races == 0 or stake == 0: return (0,0)
    avg_stake = stake / bet_races
    avg_pay_per_hit = pay / hit_races if hit_races else 0
    ci = binomial_ci(hit_races, bet_races)
    return (ci[0]*avg_pay_per_hit/avg_stake, ci[1]*avg_pay_per_hit/avg_stake)

print("=" * 80); print("Phase C 最終戦略検証"); print("=" * 80)

# ========== [0] データロード ==========
print("\n[0] データロード ...")
_b = io.StringIO(); _o = sys.stdout; sys.stdout = _b
try: ns = runpy.run_path(str(BASE / "notebooks" / "pl_v4_training.py"))
finally: sys.stdout = _o
beta_v4 = ns["beta_v4"]
X_te = ns["X_test_v4"]; keys_te = ns["keys_test_v4"].reset_index(drop=True)
N = len(keys_te)

bt = pd.read_csv(OUT/"calibrated_backtest_per_race.csv")
bt["date"] = pd.to_datetime(bt["date"]).dt.date
print(f"   backtest rows: {len(bt):,}, v4 keys: {N:,}")

# G_S, O_S per race
rows_idx = []
for i in range(N):
    k = keys_te.iloc[i]
    S = X_te[i] @ beta_v4
    std_S = max(S.std(ddof=0), 0.3)
    s_sorted = np.sort(S)[::-1]
    G_S = (s_sorted[0] - s_sorted[1]) / std_S
    O_S = (S[3:6].max() - S.mean()) / std_S
    rows_idx.append({"date": pd.Timestamp(k["date"]).date(),
                     "stadium": int(k["stadium"]),
                     "race_number": int(k["race_number"]),
                     "G_S": G_S, "O_S": O_S})
bt_m = bt.merge(pd.DataFrame(rows_idx), on=["date","stadium","race_number"], how="left")
bt_m["month"] = pd.to_datetime(bt_m["date"]).dt.to_period("M").astype(str)
print(f"   merged: {len(bt_m):,}")

# ========== [Step 1] シナリオ別 ROI 再集計 ==========
print("\n[Step 1] シナリオ別 ROI ...")
scenarios = [
    ("現状",        ["型1_逃げ本命","型2_イン残りヒモ荒れ","型3_頭荒れ"]),
    ("A 型2見送り", ["型1_逃げ本命","型3_頭荒れ"]),
    ("B 型1のみ",   ["型1_逃げ本命"]),
    ("C 型3のみ",   ["型3_頭荒れ"]),
    ("D 型3見送り", ["型1_逃げ本命","型2_イン残りヒモ荒れ"]),
]

def agg_scenario(df):
    bet = df[df["verdict"]=="bet"]
    stake = int(bet["total_stake"].sum()); pay = int(bet["total_payout"].sum())
    n_bet_races = len(bet); n_hit = int(bet["hit"].sum())
    roi = pay/stake if stake else 0
    ci = roi_ci_approx(stake, pay, n_hit, n_bet_races)
    # 月別 std
    mon_roi = []
    for m, sub in bet.groupby("month"):
        ss = int(sub["total_stake"].sum()); pp = int(sub["total_payout"].sum())
        if ss: mon_roi.append(pp/ss)
    mon_std = float(np.std(mon_roi)) if mon_roi else 0
    return {"n_bet_races": n_bet_races, "n_hit": n_hit,
            "stake": stake, "payout": pay, "profit": pay-stake,
            "roi": roi, "ci_lo": ci[0], "ci_hi": ci[1],
            "hit_rate_race": n_hit/n_bet_races if n_bet_races else 0,
            "month_std": mon_std}

scen_rows = []
for name, types in scenarios:
    sub = bt_m[bt_m["type_d"].isin(types)]
    a = agg_scenario(sub)
    a["scenario"] = name; a["types"] = "+".join(t.split("_")[0] for t in types)
    scen_rows.append(a)
scen_df = pd.DataFrame(scen_rows)[
    ["scenario","types","n_bet_races","n_hit","stake","payout","profit",
     "roi","ci_lo","ci_hi","hit_rate_race","month_std"]]
scen_df.to_csv(OUT/"scenario_comparison.csv", index=False, encoding="utf-8-sig")

print(f"\n  {'シナリオ':<14} {'bets':>6} {'hit':>5} {'profit':>12} {'ROI':>8} "
      f"{'CI下':>7} {'CI上':>7} {'月std':>7}")
for _, rw in scen_df.iterrows():
    print(f"  {rw['scenario']:<14} {rw['n_bet_races']:>6,} {rw['n_hit']:>5,} "
          f"¥{int(rw['profit']):>+11,} {rw['roi']:>8.4f} "
          f"{rw['ci_lo']:>7.4f} {rw['ci_hi']:>7.4f} {rw['month_std']:>7.4f}")

# ========== [Step 2] 型2 内部分析 ==========
print("\n[Step 2] 型2 内部分析 ...")
t2 = bt_m[bt_m["type_d"] == "型2_イン残りヒモ荒れ"].copy()
t2_bet = t2[t2["verdict"] == "bet"]

# 軸1: G_S 帯
print("\n  G_S 帯:")
gs_bands = [(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,float("inf"))]
gs_rows = []
for lo, hi in gs_bands:
    m = (t2_bet["G_S"] >= lo) & (t2_bet["G_S"] < hi)
    sub = t2_bet[m]
    st=int(sub["total_stake"].sum());pay=int(sub["total_payout"].sum())
    nh=int(sub["hit"].sum()); roi=pay/st if st else 0
    ci = roi_ci_approx(st, pay, nh, len(sub))
    gs_rows.append({"gs_lo":lo,"gs_hi":hi,"n":len(sub),"n_hit":nh,
                    "stake":st,"payout":pay,"roi":roi,
                    "ci_lo":ci[0],"ci_hi":ci[1]})
    print(f"    [{lo:.1f}, {hi:.1f}): N={len(sub):>5}  hit={nh:>3}  ROI={roi:.4f}  "
          f"CI[{ci[0]:.3f},{ci[1]:.3f}]")
pd.DataFrame(gs_rows).to_csv(OUT/"type2_gs_breakdown.csv", index=False, encoding="utf-8-sig")

# 軸2: O_S 帯
print("\n  O_S 帯:")
os_bands = [(0.2,0.3),(0.3,0.4),(0.4,0.5),(0.5,float("inf"))]
os_rows = []
for lo, hi in os_bands:
    m = (t2_bet["O_S"] >= lo) & (t2_bet["O_S"] < hi)
    sub = t2_bet[m]
    st=int(sub["total_stake"].sum());pay=int(sub["total_payout"].sum())
    nh=int(sub["hit"].sum()); roi=pay/st if st else 0
    ci = roi_ci_approx(st, pay, nh, len(sub))
    os_rows.append({"os_lo":lo,"os_hi":hi,"n":len(sub),"n_hit":nh,
                    "stake":st,"payout":pay,"roi":roi,
                    "ci_lo":ci[0],"ci_hi":ci[1]})
    print(f"    [{lo:.1f}, {hi:.1f}): N={len(sub):>5}  hit={nh:>3}  ROI={roi:.4f}  "
          f"CI[{ci[0]:.3f},{ci[1]:.3f}]")
pd.DataFrame(os_rows).to_csv(OUT/"type2_os_breakdown.csv", index=False, encoding="utf-8-sig")

# 軸3: 4x4 マトリクス
print("\n  G_S × O_S マトリクス:")
mat_rows = []
for glo, ghi in gs_bands:
    for olo, ohi in os_bands:
        m = ((t2_bet["G_S"] >= glo) & (t2_bet["G_S"] < ghi) &
             (t2_bet["O_S"] >= olo) & (t2_bet["O_S"] < ohi))
        sub = t2_bet[m]
        st=int(sub["total_stake"].sum());pay=int(sub["total_payout"].sum())
        nh=int(sub["hit"].sum()); roi=pay/st if st else 0
        mat_rows.append({"gs_lo":glo,"gs_hi":ghi,"os_lo":olo,"os_hi":ohi,
                         "n":len(sub),"n_hit":nh,"stake":st,"payout":pay,"roi":roi})
mat_df = pd.DataFrame(mat_rows)
mat_df.to_csv(OUT/"type2_matrix.csv", index=False, encoding="utf-8-sig")

# ヒートマップ
roi_mat = np.zeros((len(gs_bands), len(os_bands)))
n_mat = np.zeros((len(gs_bands), len(os_bands)), dtype=int)
for i,(glo,ghi) in enumerate(gs_bands):
    for j,(olo,ohi) in enumerate(os_bands):
        r = mat_df[(mat_df["gs_lo"]==glo)&(mat_df["os_lo"]==olo)].iloc[0]
        roi_mat[i,j] = r["roi"]; n_mat[i,j] = int(r["n"])

fig, ax = plt.subplots(figsize=(9,7))
im = ax.imshow(roi_mat, cmap="RdBu_r", vmin=0.0, vmax=2.0, aspect="auto")
for i in range(len(gs_bands)):
    for j in range(len(os_bands)):
        if n_mat[i,j] > 0:
            color = "white" if abs(roi_mat[i,j]-1.0) > 0.5 else "black"
            ax.text(j, i, f"{roi_mat[i,j]:.3f}\nN={n_mat[i,j]}",
                    ha="center", va="center", color=color, fontsize=9)
        else:
            ax.text(j, i, "—", ha="center", va="center", fontsize=9)
ax.set_xticks(range(len(os_bands)))
ax.set_xticklabels([f"[{lo:.1f},{hi:.1f})" if hi!=float("inf") else f"[{lo:.1f},∞)"
                    for lo,hi in os_bands])
ax.set_yticks(range(len(gs_bands)))
ax.set_yticklabels([f"[{lo:.1f},{hi:.1f})" if hi!=float("inf") else f"[{lo:.1f},∞)"
                    for lo,hi in gs_bands])
ax.set_xlabel("O_S"); ax.set_ylabel("G_S")
ax.set_title("型2 ROI (only bet races) — calibrated+EV>=1.2")
fig.colorbar(im, ax=ax, label="ROI")
fig.tight_layout()
fig.savefig(OUT/"type2_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"   saved heatmap")

# ========== [Step 3] 型2 閾値案 ==========
print("\n[Step 3] 型2 閾値案シミュレーション ...")
# 型1 + 型3 固定
fixed = bt_m[bt_m["type_d"].isin(["型1_逃げ本命","型3_頭荒れ"]) & (bt_m["verdict"]=="bet")]
fix_stake = int(fixed["total_stake"].sum()); fix_pay = int(fixed["total_payout"].sum())
fix_races = len(fixed); fix_hit = int(fixed["hit"].sum())

t2_all = t2[t2["verdict"] == "bet"]
cases = [
    ("現状", 0.6, 0.2),
    ("α", 0.8, 0.2),
    ("β", 0.6, 0.3),
    ("γ", 0.8, 0.3),
    ("δ", 0.9, 0.4),
]
thr_rows = []
for lbl, g_thr, o_thr in cases:
    keep = t2_all[(t2_all["G_S"] > g_thr) & (t2_all["O_S"] > o_thr)]
    k_stake = int(keep["total_stake"].sum()); k_pay = int(keep["total_payout"].sum())
    k_races = len(keep); k_hit = int(keep["hit"].sum())
    t2_roi = k_pay/k_stake if k_stake else 0
    t2_ci = roi_ci_approx(k_stake, k_pay, k_hit, k_races)
    total_stake = fix_stake + k_stake; total_pay = fix_pay + k_pay
    total_races = fix_races + k_races; total_hit = fix_hit + k_hit
    overall_roi = total_pay/total_stake if total_stake else 0
    overall_ci = roi_ci_approx(total_stake, total_pay, total_hit, total_races)
    thr_rows.append({"scenario":lbl,"g_thr":g_thr,"o_thr":o_thr,
                     "t2_n":k_races,"t2_roi":t2_roi,
                     "t2_ci_lo":t2_ci[0],"t2_ci_hi":t2_ci[1],
                     "total_n":total_races,"total_stake":total_stake,"total_payout":total_pay,
                     "profit":total_pay-total_stake,
                     "overall_roi":overall_roi,
                     "overall_ci_lo":overall_ci[0],"overall_ci_hi":overall_ci[1]})
thr_df = pd.DataFrame(thr_rows)
thr_df.to_csv(OUT/"type2_threshold_scenarios.csv", index=False, encoding="utf-8-sig")
print(f"\n  {'case':<6} {'条件':<24} {'型2 N':>6} {'型2 ROI':>8} "
      f"{'全 N':>6} {'全 ROI':>8} {'CI下':>7} {'CI上':>7} {'profit':>12}")
for _, rw in thr_df.iterrows():
    cond = f"G>{rw['g_thr']:.1f} ∧ O>{rw['o_thr']:.1f}"
    print(f"  {rw['scenario']:<6} {cond:<24} {int(rw['t2_n']):>6,} "
          f"{rw['t2_roi']:>8.4f} {int(rw['total_n']):>6,} "
          f"{rw['overall_roi']:>8.4f} {rw['overall_ci_lo']:>7.4f} "
          f"{rw['overall_ci_hi']:>7.4f} ¥{int(rw['profit']):>+11,}")

# ========== [Step 4] 型3 信頼性 ==========
print("\n[Step 4] 型3 信頼性 ...")
t3_bet = bt_m[(bt_m["type_d"]=="型3_頭荒れ") & (bt_m["verdict"]=="bet")]
print(f"  型3 bet races: {len(t3_bet):,}")

# 月別
print("\n  月別:")
t3_monthly = []
for m, sub in t3_bet.groupby("month"):
    st=int(sub["total_stake"].sum());pay=int(sub["total_payout"].sum())
    nh=int(sub["hit"].sum()); roi=pay/st if st else 0
    ci = roi_ci_approx(st, pay, nh, len(sub))
    t3_monthly.append({"month":m,"n":len(sub),"hit":nh,"stake":st,
                       "payout":pay,"roi":roi,"ci_lo":ci[0],"ci_hi":ci[1]})
    print(f"    {m}: N={len(sub):>3}  hit={nh:>2}  ROI={roi:.4f}  CI[{ci[0]:.3f},{ci[1]:.3f}]")

# 会場別
print("\n  会場別 (Top 10 by N):")
stad_rows = []
for s, sub in t3_bet.groupby("stadium"):
    st=int(sub["total_stake"].sum());pay=int(sub["total_payout"].sum())
    nh=int(sub["hit"].sum()); roi=pay/st if st else 0
    stad_rows.append({"stadium":s,"n":len(sub),"hit":nh,"stake":st,
                      "payout":pay,"roi":roi,"profit":pay-st})
stad_df = pd.DataFrame(stad_rows).sort_values("n", ascending=False)
for _, rw in stad_df.head(10).iterrows():
    print(f"    場{int(rw['stadium']):>2}: N={int(rw['n']):>3}  hit={int(rw['hit']):>2}  "
          f"ROI={rw['roi']:.4f}  profit={int(rw['profit']):+,}")

# 合併データ
reliab = {
    "total_races": len(t3_bet),
    "total_hit": int(t3_bet["hit"].sum()),
    "total_stake": int(t3_bet["total_stake"].sum()),
    "total_payout": int(t3_bet["total_payout"].sum()),
    "total_roi": int(t3_bet["total_payout"].sum())/max(int(t3_bet["total_stake"].sum()),1),
    "month_roi_std": float(np.std([r["roi"] for r in t3_monthly])) if t3_monthly else 0,
    "stadium_count": len(stad_rows),
    "n_positive_months": sum(1 for r in t3_monthly if r["roi"]>=1.0),
    "n_positive_stadiums": int((stad_df["roi"]>=1.0).sum()),
}
pd.DataFrame(t3_monthly).to_csv(OUT/"type3_monthly_reliability.csv", index=False, encoding="utf-8-sig")
stad_df.to_csv(OUT/"type3_stadium_reliability.csv", index=False, encoding="utf-8-sig")
pd.DataFrame([reliab]).to_csv(OUT/"type3_reliability.csv", index=False, encoding="utf-8-sig")

print(f"\n  月別 ROI std: {reliab['month_roi_std']:.4f}")
print(f"  ROI≥1.0 月数: {reliab['n_positive_months']}/{len(t3_monthly)}")
print(f"  ROI≥1.0 会場数: {reliab['n_positive_stadiums']}/{len(stad_rows)}")

# ========== [Step 5] 最適戦略特定 + 最終判定 ==========
print("\n[Step 5] 最終判定 ...")
# 候補: シナリオ + 型2 閾値案 の組み合わせ
# 実効的な最良は thr_df の overall_roi 最大
best_idx = thr_df["overall_roi"].idxmax()
best = thr_df.loc[best_idx]
# CI 下限を判定材料に
best_roi = float(best["overall_roi"])
best_ci_lo = float(best["overall_ci_lo"])

if best_roi >= 1.0 and best_ci_lo >= 0.95:
    verdict = "α"; desc = "運用候補確定 (ROI≥1.0 AND CI下≥0.95)"
elif best_roi >= 1.0:
    verdict = "β"; desc = "期待値プラスだが不安定 (ROI≥1.0 だが CI下<0.95)"
elif best_roi >= 0.95:
    verdict = "γ"; desc = "実質 break-even — 他券種検討 or 運用諦める"
else:
    verdict = "δ"; desc = "型フィルタでも不十分 — 他券種検証へ"

# 型2 廃止 scenario
noType2 = scen_df[scen_df["scenario"]=="A 型2見送り"].iloc[0]
# 型1のみ
onlyT1 = scen_df[scen_df["scenario"]=="B 型1のみ"].iloc[0]
# 型3のみ
onlyT3 = scen_df[scen_df["scenario"]=="C 型3のみ"].iloc[0]

# ========== [Step 6] レポート ==========
print("\n[Step 6] レポート生成 ...")
md = f"""# Phase C 最終戦略レポート

## 背景
キャリブレーション+EV≥1.2 の既存データを**再バックテストせず集計**で解析.

- 現状 (全型): ROI 0.9539
- 型1: 0.9974 / 型2: 0.8510 / 型3: 1.1119

## Step 1: シナリオ別 ROI

| シナリオ | 型 | N_bet | hit | profit | ROI | CI下 | CI上 | 月別 std |
|---|---|---|---|---|---|---|---|---|
"""
for _, rw in scen_df.iterrows():
    md += (f"| {rw['scenario']} | {rw['types']} | {rw['n_bet_races']:,} | "
           f"{rw['n_hit']:,} | ¥{int(rw['profit']):+,} | "
           f"**{rw['roi']:.4f}** | {rw['ci_lo']:.4f} | {rw['ci_hi']:.4f} | "
           f"{rw['month_std']:.4f} |\n")

md += """

## Step 2: 型2 内部分析

### 軸1: G_S 帯別 ROI

| G_S 帯 | N | hit | ROI | CI下 | CI上 |
|---|---|---|---|---|---|
"""
for r in gs_rows:
    hi = f"{r['gs_hi']:.1f}" if r['gs_hi']!=float("inf") else "∞"
    md += (f"| [{r['gs_lo']:.1f}, {hi}) | {r['n']:,} | {r['n_hit']} | "
           f"**{r['roi']:.4f}** | {r['ci_lo']:.4f} | {r['ci_hi']:.4f} |\n")

md += """

### 軸2: O_S 帯別 ROI

| O_S 帯 | N | hit | ROI | CI下 | CI上 |
|---|---|---|---|---|---|
"""
for r in os_rows:
    hi = f"{r['os_hi']:.1f}" if r['os_hi']!=float("inf") else "∞"
    md += (f"| [{r['os_lo']:.1f}, {hi}) | {r['n']:,} | {r['n_hit']} | "
           f"**{r['roi']:.4f}** | {r['ci_lo']:.4f} | {r['ci_hi']:.4f} |\n")

md += """

### 軸3: G_S × O_S マトリクス (bet only)

| G_S ＼ O_S | [0.2,0.3) | [0.3,0.4) | [0.4,0.5) | [0.5,∞) |
|---|---|---|---|---|
"""
for glo, ghi in gs_bands:
    gs_label = f"[{glo:.1f},{ghi:.1f})" if ghi!=float("inf") else f"[{glo:.1f},∞)"
    md += f"| {gs_label} "
    for olo, ohi in os_bands:
        r = mat_df[(mat_df["gs_lo"]==glo)&(mat_df["os_lo"]==olo)].iloc[0]
        if r["n"] > 0:
            md += f"| **{r['roi']:.3f}** ({int(r['n'])}) "
        else:
            md += "| — "
    md += "|\n"

md += f"""

![heatmap](type2_heatmap.png)

## Step 3: 型2 閾値案シミュレーション

| case | 型2 条件 | 型2 N | 型2 ROI | 全 N | **全 ROI** | CI下 | CI上 | profit |
|---|---|---|---|---|---|---|---|---|
"""
for _, rw in thr_df.iterrows():
    cond = f"G>{rw['g_thr']:.1f} ∧ O>{rw['o_thr']:.1f}"
    md += (f"| {rw['scenario']} | {cond} | {int(rw['t2_n']):,} | "
           f"{rw['t2_roi']:.4f} | {int(rw['total_n']):,} | "
           f"**{rw['overall_roi']:.4f}** | {rw['overall_ci_lo']:.4f} | "
           f"{rw['overall_ci_hi']:.4f} | ¥{int(rw['profit']):+,} |\n")

md += f"""

## Step 4: 型3 信頼性

### 月別
| 月 | N | hit | ROI | CI下 | CI上 |
|---|---|---|---|---|---|
"""
for r in t3_monthly:
    md += (f"| {r['month']} | {r['n']} | {r['hit']} | "
           f"{r['roi']:.4f} | {r['ci_lo']:.4f} | {r['ci_hi']:.4f} |\n")

md += """

### 会場別 (Top 10 by N)
| 会場 | N | hit | ROI | profit |
|---|---|---|---|---|
"""
for _, rw in stad_df.head(10).iterrows():
    md += (f"| {int(rw['stadium'])} | {int(rw['n'])} | {int(rw['hit'])} | "
           f"{rw['roi']:.4f} | ¥{int(rw['profit']):+,} |\n")

md += f"""

### 型3 安定性サマリ
- 月別 ROI std: {reliab['month_roi_std']:.4f}
- ROI≥1.0 月数: {reliab['n_positive_months']}/{len(t3_monthly)}
- ROI≥1.0 会場数: {reliab['n_positive_stadiums']}/{len(stad_rows)}

## Step 5: 最終判定 — パターン {verdict}

**{desc}**

- 最適戦略: **Step3 case {best['scenario']}** (型2 条件: G>{best['g_thr']:.1f} ∧ O>{best['o_thr']:.1f})
- ROI: **{best_roi:.4f}**
- 95% CI: [{best_ci_lo:.4f}, {float(best['overall_ci_hi']):.4f}]
- profit: ¥{int(best['profit']):+,}
- ベット数: {int(best['total_n']):,}

### Step1 シナリオとの比較
- 現状 (全型): ROI 0.9539
- A 型2見送り (型1+型3): ROI {float(noType2['roi']):.4f}, profit ¥{int(noType2['profit']):+,}
- B 型1のみ: ROI {float(onlyT1['roi']):.4f}
- C 型3のみ: ROI {float(onlyT3['roi']):.4f} (N={int(onlyT3['n_bet_races'])}, 小サンプル)
- **最適 (case {best['scenario']})**: ROI {best_roi:.4f}, profit ¥{int(best['profit']):+,}

### 次アクション提案
"""
if verdict == "α":
    md += f"""
1. **運用候補戦略確定**: case {best['scenario']} (型2 条件 G>{best['g_thr']:.1f} ∧ O>{best['o_thr']:.1f})
2. ペーパートレード検証
3. 型1 (ROI 0.9974) もさらに絞り込み検討
"""
elif verdict == "β":
    md += f"""
1. 期待値プラスだが CI 下限が弱い — 追加検証
2. 月別分散を見て安定性確認
3. 運用前にサンプルサイズ増強 (データ期間延長など)
"""
elif verdict == "γ":
    md += f"""
1. 3連単では **{best_roi:.4f}** が上限
2. 他券種検討 (2連単・2連複・3連複)
3. 型3 単独運用: ROI {float(onlyT3['roi']):.4f} (ただし N 少)
"""
else:
    md += """
1. 3連単では構造的に黒字化困難
2. 他券種 (2連単など) のバックテスト必須
3. モデル再設計も選択肢
"""

md += """

## 出力ファイル
- `scenario_comparison.csv`
- `type2_gs_breakdown.csv` / `type2_os_breakdown.csv` / `type2_matrix.csv`
- `type2_heatmap.png`
- `type2_threshold_scenarios.csv`
- `type3_monthly_reliability.csv` / `type3_stadium_reliability.csv` / `type3_reliability.csv`
"""

with open(OUT/"phase_c_final_strategy.md", "w", encoding="utf-8") as f:
    f.write(md)
print(f"   saved: phase_c_final_strategy.md")

print(f"\n=== 判定: パターン {verdict} (最適 case {best['scenario']}, "
      f"ROI={best_roi:.4f}, CI下={best_ci_lo:.4f}) ===")
print("完了")
