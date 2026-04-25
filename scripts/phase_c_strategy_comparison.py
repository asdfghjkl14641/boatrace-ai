# -*- coding: utf-8 -*-
"""Phase C 再設計: 買い目選定ロジック 7 戦略比較.

固定: v4 モデル, 型分類, 型信頼度 r, 予算 3000円, min 100円, EV>=1.10 AND Edge>=0.01
変化: 候補絞り込み (A全候補 / B上位3 / C上位5 / D累積30% / E累積50% / Fハイブリッド / G1点集中)
"""
from __future__ import annotations
import io, sys, math, runpy, warnings
from pathlib import Path
from itertools import permutations
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "notebooks" / "output"
sys.path.insert(0, str(BASE))
sys.stdout.reconfigure(encoding="utf-8")

TAU = 0.8
EV_THR = 1.10; EDGE_THR = 0.01
ALPHA_FALLBACK = 1.333
MIN_COMBOS_FOR_Z = 100
BUDGET = 3000; MIN_UNIT = 100
TYPE_RELIABILITY = {
    "型1_逃げ本命": 1.0, "型2_イン残りヒモ荒れ": 0.9,
    "型3_頭荒れ": 0.8, "型4_ノイズ": 0.5,
}
PERMS = [(a, b, c) for a, b, c in permutations([1,2,3,4,5,6], 3)]

STRATEGIES = ["A_全候補", "B_EV上位3", "C_EV上位5", "D_累積30%",
              "E_累積50%", "F_ハイブリッド", "G_EV最大1点"]

def pl_probs(p_lane):
    p = np.asarray(p_lane); out = np.zeros(120)
    for i, (a,b,c) in enumerate(PERMS):
        pa = p[a-1]
        pb = p[b-1] / max(1-pa, 1e-9)
        pc = p[c-1] / max(1-pa-p[b-1], 1e-9)
        out[i] = pa*pb*pc
    return out

def classify_type(S):
    std_S = max(S.std(ddof=0), 0.3)
    s_sorted = np.sort(S)[::-1]
    G_S = (s_sorted[0] - s_sorted[1]) / std_S
    O_S = (S[3:6].max() - S.mean()) / std_S
    top1 = int(S.argmax())
    if top1 == 0:
        if G_S > 1.0 and O_S < 0.3: return "型1_逃げ本命"
        if G_S > 0.6 and O_S > 0.2: return "型2_イン残りヒモ荒れ"
        return "型4_ノイズ"
    else:
        if O_S > 0.3 and G_S > 0.4: return "型3_頭荒れ"
        return "型4_ノイズ"

def binomial_ci(hits, total):
    if total == 0: return (0.0, 0.0)
    p = hits/total
    se = math.sqrt(p*(1-p)/total); z = 1.96
    return (max(0, p-z*se), min(1, p+z*se))

def allocate(cands, budget=BUDGET, min_unit=MIN_UNIT):
    """均等配分 (端数は EV top から追加)."""
    if not cands: return []
    k = len(cands)
    base = (budget // k) // min_unit * min_unit
    alloc = [(c, base) for c in cands]
    used = base * k
    extra = budget - used
    for idx, c in enumerate(sorted(range(k), key=lambda i: -cands[i][3])):
        if extra < min_unit: break
        combo, odds, p_adj, ev, edge = alloc[c][0]
        s = alloc[c][1] + min_unit
        alloc[c] = (alloc[c][0], s); extra -= min_unit
    return [(a[0][0], a[0][1], a[1]) for a in alloc if a[1] > 0]

def select_candidates(cands, strategy):
    """候補 cands (EV 降順済) を戦略に応じて絞る."""
    if not cands: return []
    if strategy == "A_全候補":
        return cands
    if strategy == "B_EV上位3":
        return cands[:3]
    if strategy == "C_EV上位5":
        return cands[:5]
    if strategy == "D_累積30%":
        out = []; cum = 0
        for c in cands:
            out.append(c); cum += c[2]
            if cum >= 0.30: break
        return out
    if strategy == "E_累積50%":
        out = []; cum = 0
        for c in cands:
            out.append(c); cum += c[2]
            if cum >= 0.50: break
        return out
    if strategy == "F_ハイブリッド":
        out = []; cum = 0
        for c in cands:
            out.append(c); cum += c[2]
            if len(out) >= 5 or cum >= 0.30: break
        return out
    if strategy == "G_EV最大1点":
        return cands[:1]
    raise ValueError(f"unknown strategy {strategy}")


print("=" * 80)
print("Phase C 戦略比較 — 7 戦略バックテスト")
print("=" * 80)

# ========== [0] v4 state ==========
print("\n[0] v4 runpy ...")
_b = io.StringIO(); _o = sys.stdout; sys.stdout = _b
try: ns = runpy.run_path(str(BASE / "notebooks" / "pl_v4_training.py"))
finally: sys.stdout = _o
beta_v4 = ns["beta_v4"]
X_te = ns["X_test_v4"]; keys_te = ns["keys_test_v4"].reset_index(drop=True)
N = len(keys_te)
print(f"  β_v4={beta_v4.shape}  X_test={X_te.shape}  N={N}")

# ========== [1] odds + results ==========
print("\n[1] odds + results ...")
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

res_map = {}
for (d,s,r),g in res_df.groupby(["date","stadium","race_number"]):
    g2 = g.sort_values("rank")
    if len(g2) < 3: continue
    res_map[(d,s,r)] = (int(g2.iloc[0]["boat"]), int(g2.iloc[1]["boat"]), int(g2.iloc[2]["boat"]))

odds_map = {}
for (d,s,r),g in odds_df.groupby(["date","stadium","race_number"]):
    book = {}
    for _,row in g.iterrows():
        try:
            a,b,c = map(int, row["combination"].split("-"))
            o = float(row["odds_1min"])
            if o > 0: book[(a,b,c)] = o
        except Exception: pass
    if book: odds_map[(d,s,r)] = book
print(f"   res={len(res_map):,}  odds={len(odds_map):,}")

# ========== [2] 全戦略を同時に回す ==========
print("\n[2] 戦略比較バックテスト ...")
# per-strategy per-race rows and totals
per_race = {s: [] for s in STRATEGIES}
totals = {s: {"stake":0, "pay":0, "bet_n":0, "hit_n":0,
              "bet_races":0, "hit_races":0,
              "ev_sum":0.0, "odds_sum":0.0,
              "profit_list":[]} for s in STRATEGIES}

# 型別集計用
type_agg = {s: {t: {"stake":0,"pay":0,"bet_races":0,"hit_races":0,"bet_n":0,"hit_n":0}
                 for t in ["型1_逃げ本命","型2_イン残りヒモ荒れ","型3_頭荒れ"]}
            for s in STRATEGIES}
# 月別集計用
mon_agg = {s: {} for s in STRATEGIES}

import time; t0 = time.time()
for i in range(N):
    k = keys_te.iloc[i]
    d = pd.Timestamp(k["date"]).date(); s=int(k["stadium"]); r=int(k["race_number"])
    date_month = f"{d.year}-{d.month:02d}"

    if (d,s,r) not in res_map or (d,s,r) not in odds_map:
        for strat in STRATEGIES:
            per_race[strat].append({"date":str(d),"stadium":s,"race_number":r,
                "verdict":"no_data","type_d":"","n_cands":0,
                "stake":0,"payout":0,"hit":0,"race_profit":0})
        continue

    book = odds_map[(d,s,r)]; actual = res_map[(d,s,r)]
    Z = sum(1.0/o for o in book.values()) if len(book) >= MIN_COMBOS_FOR_Z else ALPHA_FALLBACK
    S = X_te[i] @ beta_v4
    t = classify_type(S)

    if t == "型4_ノイズ":
        for strat in STRATEGIES:
            per_race[strat].append({"date":str(d),"stadium":s,"race_number":r,
                "verdict":"skip_type4","type_d":t,"n_cands":0,
                "stake":0,"payout":0,"hit":0,"race_profit":0})
        continue

    rel = TYPE_RELIABILITY[t]
    s_t = S/TAU; s_t = s_t - s_t.max()
    p_lane = np.exp(s_t) / np.exp(s_t).sum()
    probs = pl_probs(p_lane)

    # 候補抽出
    cands = []
    for j, combo in enumerate(PERMS):
        odds = book.get(combo)
        if not odds: continue
        p_adj = probs[j] * rel
        ev = p_adj * odds
        edge = p_adj - 1.0/(odds*Z)
        if ev >= EV_THR and edge >= EDGE_THR:
            cands.append((combo, odds, p_adj, ev, edge))
    cands.sort(key=lambda x: -x[3])  # EV desc

    if not cands:
        for strat in STRATEGIES:
            per_race[strat].append({"date":str(d),"stadium":s,"race_number":r,
                "verdict":"no_cands","type_d":t,"n_cands":0,
                "stake":0,"payout":0,"hit":0,"race_profit":0})
        continue

    for strat in STRATEGIES:
        selected = select_candidates(cands, strat)
        if strat == "G_EV最大1点":
            # 予算全額を 1 点に
            if selected:
                combo, odds, p_adj, ev, edge = selected[0]
                alloc_final = [(combo, odds, BUDGET)]
            else:
                alloc_final = []
        else:
            alloc_final = allocate(selected)

        if not alloc_final:
            per_race[strat].append({"date":str(d),"stadium":s,"race_number":r,
                "verdict":"no_cands","type_d":t,"n_cands":0,
                "stake":0,"payout":0,"hit":0,"race_profit":0})
            continue

        race_stake = sum(a[2] for a in alloc_final)
        race_pay = 0; hit_flag = 0
        hits_in_race = 0
        for combo, odds, stake in alloc_final:
            if combo == actual:
                race_pay += int(stake * odds); hits_in_race += 1
        hit_flag = 1 if hits_in_race > 0 else 0
        race_profit = race_pay - race_stake

        # update totals
        T = totals[strat]
        T["stake"] += race_stake; T["pay"] += race_pay
        T["bet_n"] += len(alloc_final); T["hit_n"] += hits_in_race
        T["bet_races"] += 1; T["hit_races"] += hit_flag
        T["profit_list"].append(race_profit)
        for combo_, odds_, stake_ in alloc_final:
            # find EV for avg calc
            for c in selected:
                if c[0] == combo_:
                    T["ev_sum"] += c[3]; break
            T["odds_sum"] += odds_

        # type agg
        TA = type_agg[strat][t]
        TA["stake"] += race_stake; TA["pay"] += race_pay
        TA["bet_races"] += 1; TA["hit_races"] += hit_flag
        TA["bet_n"] += len(alloc_final); TA["hit_n"] += hits_in_race

        # month agg
        if date_month not in mon_agg[strat]:
            mon_agg[strat][date_month] = {"stake":0,"pay":0,"bet_races":0,
                                          "hit_races":0,"profit_list":[]}
        M = mon_agg[strat][date_month]
        M["stake"] += race_stake; M["pay"] += race_pay
        M["bet_races"] += 1; M["hit_races"] += hit_flag
        M["profit_list"].append(race_profit)

        per_race[strat].append({"date":str(d),"stadium":s,"race_number":r,
            "verdict":"bet","type_d":t,"n_cands":len(alloc_final),
            "stake":race_stake,"payout":race_pay,"hit":hit_flag,
            "race_profit":race_profit})

    if i % 3000 == 0 and i > 0:
        print(f"   {i}/{N}  elapsed {time.time()-t0:.1f}s")

print(f"   done  elapsed {time.time()-t0:.1f}s")

# ========== [3] 戦略比較テーブル ==========
print("\n[3] 戦略比較サマリ ...")
rows = []
for strat in STRATEGIES:
    T = totals[strat]
    N_bet_races = T["bet_races"]; N_bet = T["bet_n"]
    roi = T["pay"]/T["stake"] if T["stake"] else 0
    hit_rate_race = T["hit_races"]/N_bet_races if N_bet_races else 0
    avg_ev = T["ev_sum"]/N_bet if N_bet else 0
    avg_odds = T["odds_sum"]/N_bet if N_bet else 0
    ci = binomial_ci(T["hit_races"], N_bet_races)
    # ROI CI (近似): hit_rate × avg_payoff_per_hit
    # 簡易: hit_rate_race * avg_payout_per_hit_race / avg_stake
    avg_stake_per_race = T["stake"]/N_bet_races if N_bet_races else 0
    avg_pay_per_hit = T["pay"]/T["hit_races"] if T["hit_races"] else 0
    roi_lo = ci[0] * avg_pay_per_hit / avg_stake_per_race if avg_stake_per_race else 0
    roi_hi = ci[1] * avg_pay_per_hit / avg_stake_per_race if avg_stake_per_race else 0
    rows.append({"strategy":strat,
        "bet_rate_pct": N_bet_races/N*100,
        "bet_races": N_bet_races,
        "bet_n": N_bet,
        "stake": T["stake"], "payout": T["pay"],
        "profit": T["pay"]-T["stake"],
        "roi": roi,
        "roi_ci_lo": roi_lo, "roi_ci_hi": roi_hi,
        "hit_rate_race_pct": hit_rate_race*100,
        "avg_ev": avg_ev, "avg_odds": avg_odds,
    })
sc_df = pd.DataFrame(rows)
sc_df.to_csv(OUT/"strategy_comparison.csv", index=False, encoding="utf-8-sig")

print(f"\n  {'戦略':<16} {'bet%':>6} {'bet_r':>6} {'bet_n':>7} {'stake':>11} "
      f"{'payout':>11} {'profit':>11} {'ROI':>7} {'CIlo':>7} {'CIhi':>7} "
      f"{'hit%':>6} {'EV':>5} {'odds':>7}")
for _, rw in sc_df.iterrows():
    print(f"  {rw['strategy']:<16} {rw['bet_rate_pct']:>5.1f}% {rw['bet_races']:>6,} "
          f"{rw['bet_n']:>7,} {rw['stake']:>11,} {rw['payout']:>11,} "
          f"{int(rw['profit']):>+11,} {rw['roi']:>7.4f} {rw['roi_ci_lo']:>7.4f} "
          f"{rw['roi_ci_hi']:>7.4f} {rw['hit_rate_race_pct']:>5.2f}% "
          f"{rw['avg_ev']:>5.2f} {rw['avg_odds']:>7.2f}")

# ========== [4] 型別 ==========
print("\n[4] 型別 ROI ...")
type_rows = []
for strat in STRATEGIES:
    row = {"strategy": strat}
    for t in ["型1_逃げ本命","型2_イン残りヒモ荒れ","型3_頭荒れ"]:
        TA = type_agg[strat][t]
        roi_t = TA["pay"]/TA["stake"] if TA["stake"] else 0
        row[f"{t}_roi"] = roi_t
        row[f"{t}_bet_races"] = TA["bet_races"]
        row[f"{t}_profit"] = TA["pay"] - TA["stake"]
    type_rows.append(row)
type_df = pd.DataFrame(type_rows)
type_df.to_csv(OUT/"strategy_by_type.csv", index=False, encoding="utf-8-sig")
print(f"  {'戦略':<16} {'型1 ROI':>10} {'型2 ROI':>10} {'型3 ROI':>10}")
for _, rw in type_df.iterrows():
    print(f"  {rw['strategy']:<16} {rw['型1_逃げ本命_roi']:>10.4f} "
          f"{rw['型2_イン残りヒモ荒れ_roi']:>10.4f} {rw['型3_頭荒れ_roi']:>10.4f}")

# ========== [5] 月別 ==========
print("\n[5] 月別 ROI ...")
months = sorted({m for s in STRATEGIES for m in mon_agg[s]})
mon_rows = []
for strat in STRATEGIES:
    row = {"strategy": strat}
    for m in months:
        M = mon_agg[strat].get(m, {"stake":0,"pay":0})
        row[f"{m}_roi"] = M["pay"]/M["stake"] if M["stake"] else 0
    mon_rows.append(row)
mon_df = pd.DataFrame(mon_rows)
mon_df.to_csv(OUT/"strategy_by_month.csv", index=False, encoding="utf-8-sig")
header = f"  {'戦略':<16} " + " ".join(f"{m:>9}" for m in months) + f" {'std':>7}"
print(header)
for _, rw in mon_df.iterrows():
    vals = [rw[f"{m}_roi"] for m in months]
    std = float(np.std(vals))
    print(f"  {rw['strategy']:<16} " + " ".join(f"{v:>9.4f}" for v in vals) + f" {std:>7.4f}")

# ========== [6] 最適戦略の特定 + 詳細 ==========
print("\n[6] 最適戦略の特定 ...")
# ROI max (bet_races >= 500 が実用条件)
viable = sc_df[sc_df["bet_races"] >= 500].copy()
if len(viable) == 0: viable = sc_df.copy()
roi_best_idx = viable["roi"].idxmax()
roi_best = viable.loc[roi_best_idx, "strategy"]

# CI narrow & ROI ok
viable["ci_width"] = viable["roi_ci_hi"] - viable["roi_ci_lo"]
ci_best_idx = viable.sort_values(["ci_width","roi"], ascending=[True, False]).iloc[0].name
ci_best = viable.loc[ci_best_idx, "strategy"]

# 月別分散 min
mon_std_rows = []
for _, rw in mon_df.iterrows():
    if rw["strategy"] not in viable["strategy"].values: continue
    vals = [rw[f"{m}_roi"] for m in months]
    mon_std_rows.append({"strategy":rw["strategy"], "mon_std":float(np.std(vals)),
                         "mon_mean":float(np.mean(vals))})
mon_std_df = pd.DataFrame(mon_std_rows).sort_values("mon_std")
stab_best = mon_std_df.iloc[0]["strategy"]

print(f"  ROI 最大:    {roi_best}")
print(f"  CI 狭:       {ci_best}")
print(f"  月別分散小:  {stab_best}")

# 最優先: ROI max
optimal = roi_best

# 詳細分析
print(f"\n[7] 最適戦略 {optimal} 詳細分析 ...")
T = totals[optimal]
profits = T["profit_list"]
# 最大ドローダウン
cum = np.cumsum(profits)
peak = np.maximum.accumulate(cum)
drawdown = cum - peak
max_dd = float(drawdown.min())
# Sharpe 比 (daily風: per-race profit)
sharpe = float(np.mean(profits) / np.std(profits)) if np.std(profits) > 0 else 0

opt_detail = {
    "strategy": optimal,
    "bet_races": T["bet_races"],
    "bet_n": T["bet_n"],
    "stake": T["stake"], "payout": T["pay"],
    "profit": T["pay"]-T["stake"],
    "roi": T["pay"]/T["stake"] if T["stake"] else 0,
    "hit_rate_race": T["hit_races"]/T["bet_races"] if T["bet_races"] else 0,
    "max_drawdown": max_dd,
    "sharpe_per_race": sharpe,
    "avg_ev": T["ev_sum"]/T["bet_n"] if T["bet_n"] else 0,
    "avg_odds": T["odds_sum"]/T["bet_n"] if T["bet_n"] else 0,
}
pd.DataFrame([opt_detail]).to_csv(OUT/"optimal_strategy_detailed.csv", index=False, encoding="utf-8-sig")
print(f"  ROI {opt_detail['roi']:.4f}  max_DD ¥{max_dd:+,.0f}  Sharpe {sharpe:+.4f}")

# G 戦略の追加分析
print(f"\n[8] 戦略 G (1点集中) 追加分析 ...")
TG = totals["G_EV最大1点"]
g_profits = TG["profit_list"]
g_mon_std = float(np.std([mon_agg["G_EV最大1点"].get(m, {"stake":0,"pay":0})["pay"]/
                          mon_agg["G_EV最大1点"].get(m, {"stake":1})["stake"]
                          if mon_agg["G_EV最大1点"].get(m, {}).get("stake",0) else 0
                          for m in months]))
print(f"  bet数: {TG['bet_races']:,} (全 1点)")
print(f"  avg EV: {TG['ev_sum']/TG['bet_n']:.3f}")
print(f"  avg odds: {TG['odds_sum']/TG['bet_n']:.2f}")
print(f"  月別ROI std: {g_mon_std:.4f}")

# ========== [9] レポート ==========
print("\n[9] レポート生成 ...")
roi_final = sc_df.set_index("strategy").loc[optimal, "roi"]
if roi_final >= 1.0:
    verdict = "A"; desc = "絞り込みで ROI ≥ 1.0 到達 — 運用候補確定"
elif roi_final >= 0.90:
    verdict = "B"; desc = "3 連単の改善限界付近 — キャリブレーション併用 or 他券種検討"
else:
    verdict = "C"; desc = "絞ってもROI<0.90 — 他券種 (2連単・2連複・3連複) への転換"

baseline = sc_df.set_index("strategy").loc["A_全候補", "roi"]

md = f"""# Phase C 再設計 — 買い目選定戦略 7 比較

## Step 1: 戦略比較

| 戦略 | ベット率 | 賭け金 | 払戻 | profit | **ROI** | CI下 | CI上 | 的中率(race) | avg EV | avg odds |
|---|---|---|---|---|---|---|---|---|---|---|
"""
for _, rw in sc_df.iterrows():
    md += (f"| {rw['strategy']} | {rw['bet_rate_pct']:.2f}% | "
           f"¥{int(rw['stake']):,} | ¥{int(rw['payout']):,} | "
           f"¥{int(rw['profit']):+,} | **{rw['roi']:.4f}** | "
           f"{rw['roi_ci_lo']:.4f} | {rw['roi_ci_hi']:.4f} | "
           f"{rw['hit_rate_race_pct']:.2f}% | {rw['avg_ev']:.3f} | {rw['avg_odds']:.2f} |\n")

md += "\n## Step 2: 型別 ROI\n\n| 戦略 | 型1 ROI | 型2 ROI | 型3 ROI |\n|---|---|---|---|\n"
for _, rw in type_df.iterrows():
    md += (f"| {rw['strategy']} | {rw['型1_逃げ本命_roi']:.4f} | "
           f"{rw['型2_イン残りヒモ荒れ_roi']:.4f} | {rw['型3_頭荒れ_roi']:.4f} |\n")

md += "\n## Step 3: 月別 ROI\n\n| 戦略 | " + " | ".join(months) + " | std |\n"
md += "|---" * (len(months)+2) + "|\n"
for _, rw in mon_df.iterrows():
    vals = [rw[f"{m}_roi"] for m in months]
    std = float(np.std(vals))
    md += (f"| {rw['strategy']} | " + " | ".join(f"{v:.4f}" for v in vals) +
           f" | {std:.4f} |\n")

md += f"""

## Step 4: 戦略選択観点

| 観点 | 最良戦略 |
|---|---|
| ROI 最大 | **{roi_best}** |
| CI 狭 | {ci_best} |
| 月別分散小 | {stab_best} |

## Step 5: 最適戦略 ({optimal}) 詳細

- ベットレース: {opt_detail['bet_races']:,}
- ベット数: {opt_detail['bet_n']:,}
- stake: ¥{opt_detail['stake']:,}
- payout: ¥{opt_detail['payout']:,}
- profit: ¥{opt_detail['profit']:+,}
- **ROI: {opt_detail['roi']:.4f}**
- 的中率(race): {opt_detail['hit_rate_race']*100:.2f}%
- 最大ドローダウン: ¥{opt_detail['max_drawdown']:+,.0f}
- Sharpe (per-race): {opt_detail['sharpe_per_race']:+.4f}
- avg EV: {opt_detail['avg_ev']:.3f}
- avg odds: {opt_detail['avg_odds']:.2f}

## 戦略 G (1点集中) 追加

- bet数: {TG['bet_races']:,}
- avg EV: {TG['ev_sum']/TG['bet_n']:.3f}
- avg odds: {TG['odds_sum']/TG['bet_n']:.2f}
- 月別ROI std: {g_mon_std:.4f}

## 核心メッセージ

- **ベースライン (A 全候補): ROI = {baseline:.4f}**
- **最適 ({optimal}): ROI = {roi_final:.4f}** ({'+' if roi_final>baseline else ''}{(roi_final-baseline)*100:+.2f}pt)
- 「広く買う」→「絞って買う」への転換効果: {(roi_final-baseline)*100:+.2f}pt

## 最終判定: パターン {verdict}

**{desc}**

### 次アクション提案
"""
if verdict == "A":
    md += """
1. 運用候補戦略確定 → ペーパートレード
2. キャリブレーション併用で ROI をさらに押し上げる
3. 安定性 (月別分散, drawdown) の継続モニタリング
"""
elif verdict == "B":
    md += """
1. キャリブレーション + 最適戦略の組み合わせで 1.0 超え狙い
2. 型別で ROI が異なる → 型1+型3 限定運用も検討
3. 上記で 1.0 未達なら 2連単・2連複 へ展開
"""
else:
    md += """
1. 3連単からの撤退
2. 同じ v4 モデルで 2連単・2連複・3連複バックテスト
3. 券種ごとの vig 差を活用した戦略設計
"""

md += "\n## 出力ファイル\n"
md += "- `strategy_comparison.csv`\n- `strategy_by_type.csv`\n- `strategy_by_month.csv`\n"
md += "- `optimal_strategy_detailed.csv`\n"

with open(OUT/"phase_c_strategy_comparison.md", "w", encoding="utf-8") as f:
    f.write(md)
print(f"   saved: phase_c_strategy_comparison.md")

# per_race output (最適戦略のみ詳細保存)
pd.DataFrame(per_race[optimal]).to_csv(
    OUT/f"strategy_per_race_{optimal}.csv", index=False, encoding="utf-8-sig")
print(f"   saved: strategy_per_race_{optimal}.csv")

print(f"\n=== 判定: パターン {verdict} (最適={optimal}, ROI={roi_final:.4f}) ===")
print("完了")
