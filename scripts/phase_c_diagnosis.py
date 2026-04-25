# -*- coding: utf-8 -*-
"""
Phase C 診断 5 種 — ROI 0.80 の根本原因分析
1. キャリブレーション (モデル確率 vs 実 hit)
2. PL 構造 (2 着・3 着予想の精度)
3. 月別変動
4. 閾値感度
5. 型3 深掘り
"""
from __future__ import annotations
import io, sys, json, runpy, math
from pathlib import Path
from itertools import permutations
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "notebooks" / "output"
sys.path.insert(0, str(BASE))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

TAU = 0.8
PERMS = [(a, b, c) for a, b, c in permutations([1,2,3,4,5,6], 3)]

print("=" * 80)
print("Phase C Diagnosis — 5 analyses")
print("=" * 80)

# --- v4 state ロード ---
print("\n[0] v4 runpy ...")
_b = io.StringIO(); _o = sys.stdout; sys.stdout = _b
try: ns = runpy.run_path(str(BASE / "notebooks" / "pl_v4_training.py"))
finally: sys.stdout = _o
beta_v4 = ns["beta_v4"]
X_test = ns["X_test_v4"]; pi_test = ns["pi_test_v4"]; keys_test = ns["keys_test_v4"].reset_index(drop=True)
print(f"  β_v4={beta_v4.shape}  X_test={X_test.shape}")

# --- バックテスト結果 ---
print("\n[1] backtest_stage1/2 ロード ...")
bt1 = pd.read_csv(OUT/"backtest_stage1_per_race.csv")
bt2 = pd.read_csv(OUT/"backtest_stage2_per_race.csv")
print(f"  bt1: {len(bt1):,} / bt2: {len(bt2):,}")

# --- オッズ + 結果 ---
print("\n[2] オッズ + 結果 ロード ...")
from scripts.db import get_connection
conn = get_connection()
odds_df = pd.read_sql_query("""
    SELECT date, stadium, race_number, combination, odds_1min
    FROM trifecta_odds
    WHERE date BETWEEN '2026-01-01' AND '2026-04-18' AND odds_1min IS NOT NULL
""", conn.native)
odds_df["date"] = pd.to_datetime(odds_df["date"]).dt.date

res_df = pd.read_sql_query("""
    SELECT date, stadium, race_number, rank, boat FROM race_results
    WHERE date BETWEEN '2026-01-01' AND '2026-04-18' AND rank BETWEEN 1 AND 3
""", conn.native)
res_df["date"] = pd.to_datetime(res_df["date"]).dt.date
conn.close()

# Race → actual_combo
res_map = {}
for (d,s,r),g in res_df.groupby(["date","stadium","race_number"]):
    g2 = g.sort_values("rank")
    if len(g2) < 3: continue
    res_map[(d,s,r)] = (int(g2.iloc[0]["boat"]), int(g2.iloc[1]["boat"]), int(g2.iloc[2]["boat"]))

# Race → odds dict
odds_map = {}
for (d,s,r),g in odds_df.groupby(["date","stadium","race_number"]):
    book = {}
    for _,row in g.iterrows():
        try:
            a,b,c = map(int, row["combination"].split("-"))
            o = float(row["odds_1min"])
            if o > 0: book[(a,b,c)] = o
        except: pass
    odds_map[(d,s,r)] = book


# =========== 診断 1: キャリブレーション ===========
print("\n" + "=" * 80)
print("診断 1: モデル確率キャリブレーション (120 通り × 15k レース = 1.8M 買い目)")
print("=" * 80)

def pl_probs(p_lane):
    p = np.asarray(p_lane)
    out = np.zeros(120)
    for i, (a,b,c) in enumerate(PERMS):
        pa = p[a-1]
        pb = p[b-1] / max(1-pa, 1e-9)
        pc = p[c-1] / max(1-pa-p[b-1], 1e-9)
        out[i] = pa*pb*pc
    return out

# 全買い目の (p_model, hit) を集める
N = len(keys_test)
all_p = []; all_hit = []
for i in range(N):
    k = keys_test.iloc[i]
    d = pd.Timestamp(k["date"]).date(); s = int(k["stadium"]); r = int(k["race_number"])
    if (d,s,r) not in res_map: continue
    actual = res_map[(d,s,r)]
    S = X_test[i] @ beta_v4
    s_t = S / TAU
    s_t = s_t - s_t.max()
    p_lane = np.exp(s_t) / np.exp(s_t).sum()
    probs = pl_probs(p_lane)
    for j, combo in enumerate(PERMS):
        all_p.append(probs[j])
        all_hit.append(1 if combo == actual else 0)

all_p = np.asarray(all_p); all_hit = np.asarray(all_hit)
print(f"  総買い目数: {len(all_p):,}, 的中数: {all_hit.sum():,}")

# キャリブレーション (対数ビン、15 bin)
bins = np.concatenate([[0], np.logspace(-4, 0, 15)])
bin_idx = np.clip(np.digitize(all_p, bins) - 1, 0, len(bins)-2)
rows = []
for b in range(len(bins) - 1):
    mask = bin_idx == b
    if mask.sum() < 10: continue
    avg_p = all_p[mask].mean(); hit_rate = all_hit[mask].mean()
    rows.append({"bin_lower": bins[b], "bin_upper": bins[b+1],
                 "n": int(mask.sum()), "avg_p_model": float(avg_p),
                 "actual_hit_rate": float(hit_rate),
                 "bias": float(hit_rate - avg_p)})
cal_df = pd.DataFrame(rows)
cal_df.to_csv(OUT/"calibration_curve.csv", index=False, encoding="utf-8-sig")
print("\n  キャリブレーション bin 表 (1% 以上のビン):")
print(f"  {'p_bin':>15} {'n':>8} {'avg_p':>8} {'hit':>7} {'bias':>8}")
for r in rows:
    if r["avg_p_model"] < 0.005: continue
    print(f"  [{r['bin_lower']:.4f},{r['bin_upper']:.4f}] {r['n']:>8,} "
          f"{r['avg_p_model']:>7.4f} {r['actual_hit_rate']:>6.4f} {r['bias']:>+8.4f}")

# プロット
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
xs = cal_df["avg_p_model"].values; ys = cal_df["actual_hit_rate"].values
sizes = np.sqrt(cal_df["n"].values) * 0.5
ax.scatter(xs, ys, s=sizes, alpha=0.7, color="#3B82F6")
for i in range(len(xs)):
    ax.annotate(f"{int(cal_df.iloc[i]['n']):,}", (xs[i], ys[i]), fontsize=7)
ax.set_xlabel("Model probability (PL)"); ax.set_ylabel("Actual hit rate")
ax.set_title("v4 PL 3-trifecta calibration curve")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-4, 0.5); ax.set_ylim(1e-4, 0.5)
ax.grid(alpha=0.2); ax.legend()
fig.savefig(OUT/"calibration_curve.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"\n  saved: calibration_curve.csv, calibration_curve.png")

# 全体のバイアス
total_pred = all_p.sum(); total_actual = all_hit.sum()
print(f"\n  全体 Σp_model = {total_pred:.1f} vs 実 hit = {total_actual}")
print(f"  比率 (actual/model) = {total_actual/total_pred:.4f}")
if total_actual/total_pred < 0.95: calib_verdict = "モデル楽観 (確率を下げる必要)"
elif total_actual/total_pred > 1.05: calib_verdict = "モデル悲観 (確率を上げる必要)"
else: calib_verdict = "ほぼ整合"
print(f"  判定: {calib_verdict}")


# =========== 診断 2: PL 構造 ===========
print("\n" + "=" * 80)
print("診断 2: PL 構造 (2着・3着予想)")
print("=" * 80)

hit1 = hit2_given_1 = hit3_given_12 = 0
n_races = n_1 = n_12 = 0
for i in range(N):
    k = keys_test.iloc[i]
    d = pd.Timestamp(k["date"]).date(); s = int(k["stadium"]); r = int(k["race_number"])
    if (d,s,r) not in res_map: continue
    actual = res_map[(d,s,r)]
    S = X_test[i] @ beta_v4
    # 予想順 (降順)
    pred_order = np.argsort(-S) + 1  # 1..6
    n_races += 1
    if int(pred_order[0]) == actual[0]:
        hit1 += 1; n_1 += 1
        if int(pred_order[1]) == actual[1]:
            hit2_given_1 += 1; n_12 += 1
            if int(pred_order[2]) == actual[2]:
                hit3_given_12 += 1

p1 = hit1/n_races
p2_c = hit2_given_1/n_1 if n_1 else 0
p3_c = hit3_given_12/n_12 if n_12 else 0
print(f"\n  レース数: {n_races:,}")
print(f"  1 着 hit 率: {p1*100:.2f}% (of {n_races:,})")
print(f"  1 着的中時 2 着 hit 率: {p2_c*100:.2f}% (of {n_1:,})")
print(f"  1+2 着的中時 3 着 hit 率: {p3_c*100:.2f}% (of {n_12:,})")

pl_df = pd.DataFrame([{
    "hit1_rate": p1, "hit2_given_1": p2_c, "hit3_given_12": p3_c,
    "n_races": n_races, "n_hit1": hit1, "n_hit12": n_12,
}])
pl_df.to_csv(OUT/"pl_structure_validation.csv", index=False, encoding="utf-8-sig")

# PL 仮定の妥当性判定
# 一般に「1 着が分かっているときの 2 着予想」は「1 着予想」より高く出るはず
# (選択肢が 6 → 5 に減るため)。p2_c > p1 なら妥当
if p2_c >= p1:
    pl_v = "PL 仮定は妥当 (条件付き精度が上昇)"
elif p2_c < p1 / 2:
    pl_v = "PL 仮定に問題 (条件付き精度が大幅低下)"
else:
    pl_v = "中立"
print(f"  判定: {pl_v}")


# =========== 診断 3: 月別変動 ===========
print("\n" + "=" * 80)
print("診断 3: 月別変動")
print("=" * 80)

# bt2 から月別集計
bt2["date_dt"] = pd.to_datetime(bt2["date"])
bt2["month"] = bt2["date_dt"].dt.to_period("M").astype(str)
# 型分布 (全レース)
print(f"\n  {'月':>8} {'N':>6} {'型1':>6} {'型2':>6} {'型3':>5} {'型4':>6} "
      f"{'hit1(model)':>11} {'ベット率':>8} {'ROI':>7}")
mon_rows = []
for m in sorted(bt2["month"].unique()):
    sub = bt2[bt2["month"]==m]
    n = len(sub)
    t1 = (sub["type_d"]=="型1_逃げ本命").sum()/n*100
    t2 = (sub["type_d"]=="型2_イン残りヒモ荒れ").sum()/n*100
    t3 = (sub["type_d"]=="型3_頭荒れ").sum()/n*100
    t4 = (sub["type_d"]=="型4_ノイズ").sum()/n*100
    # モデル top1 hit 率 (1 着的中率) — v4 上で計算
    # 当該月のレース index
    mask_i = []
    for i in range(N):
        kk = keys_test.iloc[i]
        dd = pd.Timestamp(kk["date"]).date()
        mm = f"{dd.year}-{dd.month:02d}"
        if mm == m: mask_i.append(i)
    if mask_i:
        S_m = X_test[mask_i] @ beta_v4
        pred_m = S_m.argmax(axis=1)
        actual_m = pi_test[mask_i, 0]
        hit_m = (pred_m == actual_m).mean()*100
    else:
        hit_m = 0
    bet_sub = sub[sub["verdict"]=="bet"]
    stake = int(bet_sub["total_stake"].sum()); pay = int(bet_sub["total_payout"].sum())
    roi_m = pay/stake if stake else 0
    bet_rate = len(bet_sub)/n*100
    print(f"  {m:>8} {n:>6,} {t1:>5.1f}% {t2:>5.1f}% {t3:>4.1f}% {t4:>5.1f}% "
          f"{hit_m:>10.2f}% {bet_rate:>7.2f}% {roi_m:>7.4f}")
    mon_rows.append({"month":m,"n_races":n,"type1_%":t1,"type2_%":t2,
                     "type3_%":t3,"type4_%":t4,"model_hit1_%":hit_m,
                     "bet_rate_%":bet_rate,"roi":roi_m})
pd.DataFrame(mon_rows).to_csv(OUT/"monthly_diagnosis.csv", index=False, encoding="utf-8-sig")


# =========== 診断 4: 閾値感度 ===========
print("\n" + "=" * 80)
print("診断 4: 閾値感度 (Stage 2 ベース)")
print("=" * 80)

from scripts.backtest_phase_c import TYPE_RELIABILITY, ALPHA_FALLBACK, MIN_COMBOS_FOR_Z

thresholds = [(1.10, 0.01), (1.20, 0.02), (1.30, 0.03),
              (1.50, 0.05), (1.80, 0.05), (2.00, 0.05), (2.50, 0.10)]

# 各閾値で簡易バックテストシミュレーション (買い目 + 結果 だけ、均等配分抜き)
def binomial_ci(hits, total, alpha=0.05):
    if total == 0: return (0, 0)
    p = hits/total
    import math
    se = math.sqrt(p*(1-p)/total)
    z = 1.96
    return (max(0, p-z*se), min(1, p+z*se))

print(f"\n  {'EV':>5} {'Edge':>5} {'bet_n':>8} {'hit':>6} {'hit_rate':>9} "
      f"{'avg_odds':>9} {'ROI':>7} {'CI_lo':>7} {'CI_hi':>7}")
thr_rows = []
for ev_thr, edge_thr in thresholds:
    bet_n = hit_n = stake_sum = pay_sum = 0
    odds_list = []
    for i in range(N):
        k = keys_test.iloc[i]
        d = pd.Timestamp(k["date"]).date(); s=int(k["stadium"]); r=int(k["race_number"])
        if (d,s,r) not in res_map or (d,s,r) not in odds_map: continue
        actual = res_map[(d,s,r)]
        book = odds_map[(d,s,r)]
        if len(book) < MIN_COMBOS_FOR_Z:
            Z = ALPHA_FALLBACK
        else:
            Z = sum(1.0/o for o in book.values())
        S = X_test[i] @ beta_v4
        idx = {"G_S":(np.sort(-S)[0] - np.sort(-S)[1]) / max(S.std(ddof=0), 0.3),
               "O_S":(S[3:6].max() - S.mean()) / max(S.std(ddof=0), 0.3),
               "top1_lane":int(S.argmax())}
        # 型判定
        if idx["top1_lane"] == 0:
            if idx["G_S"]>1.0 and idx["O_S"]<0.3: t = "型1_逃げ本命"
            elif idx["G_S"]>0.6 and idx["O_S"]>0.2: t = "型2_イン残りヒモ荒れ"
            else: t = "型4_ノイズ"
        else:
            if idx["O_S"]>0.3 and idx["G_S"]>0.4: t = "型3_頭荒れ"
            else: t = "型4_ノイズ"
        if t == "型4_ノイズ": continue
        rel = TYPE_RELIABILITY[t]
        s_t = S/TAU; s_t = s_t - s_t.max()
        p_lane = np.exp(s_t) / np.exp(s_t).sum()
        probs = pl_probs(p_lane)
        for j, combo in enumerate(PERMS):
            odds = book.get(combo)
            if not odds: continue
            p_adj = probs[j] * rel
            ev = p_adj * odds
            edge = p_adj - 1.0/(odds*Z)
            if ev >= ev_thr and edge >= edge_thr:
                stake = 100  # 100円固定
                bet_n += 1
                stake_sum += stake
                odds_list.append(odds)
                if combo == actual:
                    pay_sum += int(stake * odds)
                    hit_n += 1
    roi = pay_sum/stake_sum if stake_sum else 0
    hit_rate = hit_n/bet_n if bet_n else 0
    # ROI の 95% CI — delta method 近似: Var(ROI) ≈ Var(payout)/stake²
    avg_odds = float(np.mean(odds_list)) if odds_list else 0
    ci = binomial_ci(hit_n, bet_n)
    # ROI ~ hit_rate * avg_odds (近似)
    roi_lo = ci[0] * avg_odds; roi_hi = ci[1] * avg_odds
    print(f"  {ev_thr:>5.2f} {edge_thr:>5.2f} {bet_n:>8,} {hit_n:>6,} "
          f"{hit_rate*100:>8.2f}% {avg_odds:>9.2f} {roi:>7.4f} "
          f"{roi_lo:>7.4f} {roi_hi:>7.4f}")
    thr_rows.append({"ev_thr":ev_thr,"edge_thr":edge_thr,"bet_n":bet_n,
                     "hit_n":hit_n,"hit_rate":hit_rate*100,"avg_odds":avg_odds,
                     "roi":roi,"roi_ci_lo":roi_lo,"roi_ci_hi":roi_hi})
pd.DataFrame(thr_rows).to_csv(OUT/"threshold_sensitivity.csv", index=False, encoding="utf-8-sig")


# =========== 診断 5: 型3 深掘り ===========
print("\n" + "=" * 80)
print("診断 5: 型3 深掘り")
print("=" * 80)

t3 = bt2[bt2["type_d"]=="型3_頭荒れ"]
t3_bet = t3[t3["verdict"]=="bet"]
n3 = len(t3); n3_bet = len(t3_bet)
stake3 = int(t3_bet["total_stake"].sum()); pay3 = int(t3_bet["total_payout"].sum())
# 的中レース
t3_hit = t3_bet[t3_bet["total_payout"]>0]
n3_hit = len(t3_hit)
roi3 = pay3/stake3 if stake3 else 0
hit_rate3 = n3_hit/n3_bet if n3_bet else 0
ci = binomial_ci(n3_hit, n3_bet)
avg_stake3 = stake3/n3_bet if n3_bet else 0
# ROI CI 近似: ROI ≈ hit_rate × avg_payout_per_hit / avg_stake
if n3_hit > 0:
    avg_pay_hit = (t3_hit["total_payout"].sum()/n3_hit) / avg_stake3
    roi_ci_lo = ci[0]*avg_pay_hit
    roi_ci_hi = ci[1]*avg_pay_hit
else:
    avg_pay_hit = 0; roi_ci_lo = roi_ci_hi = 0

print(f"\n  型3: N={n3:,}, ベット数={n3_bet:,}")
print(f"  stake=¥{stake3:,}, payout=¥{pay3:,}, profit=¥{pay3-stake3:+,}")
print(f"  ROI={roi3:.4f}  hit_rate={hit_rate3*100:.2f}%")
print(f"  95% CI (hit_rate): [{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]")
print(f"  95% CI (ROI 近似): [{roi_ci_lo:.4f}, {roi_ci_hi:.4f}]")

# 月別
print("\n  月別 型3:")
t3_bet["month"] = pd.to_datetime(t3_bet["date"]).dt.to_period("M").astype(str)
print(f"  {'月':>8} {'N_bet':>6} {'hit':>5} {'ROI':>7}")
type3_mon = []
for m in sorted(t3_bet["month"].unique()):
    sub = t3_bet[t3_bet["month"]==m]
    st = int(sub["total_stake"].sum()); pa = int(sub["total_payout"].sum())
    hh = (sub["total_payout"]>0).sum()
    ri = pa/st if st else 0
    print(f"  {m:>8} {len(sub):>6,} {hh:>5,} {ri:>7.4f}")
    type3_mon.append({"month":m,"n_bet":len(sub),"n_hit":int(hh),"roi":ri})

pd.DataFrame([{"n_total":n3,"n_bet":n3_bet,"n_hit":n3_hit,"stake":stake3,
               "payout":pay3,"roi":roi3,"hit_rate":hit_rate3,
               "ci_lo_hit":ci[0],"ci_hi_hit":ci[1],
               "ci_lo_roi":roi_ci_lo,"ci_hi_roi":roi_ci_hi}]).to_csv(
    OUT/"type3_deep_dive.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(type3_mon).to_csv(OUT/"type3_monthly.csv", index=False, encoding="utf-8-sig")


# =========== 統合判定 ===========
print("\n" + "=" * 80)
print("統合判定")
print("=" * 80)

# パターン判定
calib_ratio = total_actual / total_pred
best_thr_row = max(thr_rows, key=lambda r: r["roi"])

if calib_ratio < 0.90:
    verdict = "パターン I: モデル確率が体系的に楽観 → キャリブレーション推奨"
elif p2_c < p1 * 0.5:
    verdict = "パターン II: PL 構造に問題 → 2/3 着別モデル検討"
elif best_thr_row["roi"] >= 1.0:
    verdict = f"パターン IV: 閾値 EV={best_thr_row['ev_thr']} で ROI={best_thr_row['roi']:.4f} 達成可能"
elif max(r["roi"] for r in mon_rows) - min(r["roi"] for r in mon_rows) > 0.15:
    verdict = "パターン III: 月別変動大 (構造的不安定)"
else:
    verdict = "複合要因、詳細検討必要"

print(f"\n  計算結果:")
print(f"    Σp_model/実hit 比率: {calib_ratio:.4f} (<0.90 なら楽観)")
print(f"    p(2着|1着的中): {p2_c*100:.2f}% (1着 hit={p1*100:.2f}%)")
print(f"    最適閾値 ROI: EV={best_thr_row['ev_thr']} Edge={best_thr_row['edge_thr']} "
      f"→ ROI {best_thr_row['roi']:.4f}")
print(f"    月別 ROI 幅: {max(r['roi'] for r in mon_rows) - min(r['roi'] for r in mon_rows):.4f}")
print(f"\n  判定: {verdict}")

# レポート
rep = f"""# Phase C 診断レポート

## 診断 1: モデル確率キャリブレーション
- 全買い目数: {len(all_p):,}
- 実 hit: {int(total_actual):,}
- Σp_model: {total_pred:.1f}
- 比率 (actual/model): **{calib_ratio:.4f}**
- 判定: **{calib_verdict}**

## 診断 2: PL 構造
- 1 着 hit: {p1*100:.2f}%
- 1 着的中時 2 着 hit: {p2_c*100:.2f}%
- 1+2 着的中時 3 着 hit: {p3_c*100:.2f}%
- 判定: **{pl_v}**

## 診断 3: 月別変動
```
月      N     型1%  型2%  型3%  型4%  hit1%  ベット率  ROI
""" + "\n".join(
    f"{r['month']}  {r['n_races']:5}  {r['type1_%']:4.1f}  {r['type2_%']:4.1f}  "
    f"{r['type3_%']:4.1f}  {r['type4_%']:4.1f}  {r['model_hit1_%']:5.2f}  "
    f"{r['bet_rate_%']:6.2f}  {r['roi']:.4f}"
    for r in mon_rows
) + f"""
```

## 診断 4: 閾値感度
```
EV    Edge   bet_n   hit   hit%    avg_odds  ROI     CI_lo   CI_hi
""" + "\n".join(
    f"{r['ev_thr']:.2f}  {r['edge_thr']:.2f}  {r['bet_n']:6}  {r['hit_n']:5}  "
    f"{r['hit_rate']:5.2f}   {r['avg_odds']:6.2f}    {r['roi']:.4f}  "
    f"{r['roi_ci_lo']:.4f}  {r['roi_ci_hi']:.4f}"
    for r in thr_rows
) + f"""
```

## 診断 5: 型3 深掘り
- レース数: {n3:,}, ベット数: {n3_bet:,}, 的中: {n3_hit:,}
- ROI: **{roi3:.4f}**
- 95% CI (hit): [{ci[0]*100:.2f}%, {ci[1]*100:.2f}%]
- 95% CI (ROI): [{roi_ci_lo:.4f}, {roi_ci_hi:.4f}]

## 最終判定

**{verdict}**
"""
(OUT/"phase_c_diagnosis_report.md").write_text(rep, encoding="utf-8")
print(f"\nsaved: phase_c_diagnosis_report.md")
print("完了")
