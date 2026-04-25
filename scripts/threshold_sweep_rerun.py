# -*- coding: utf-8 -*-
"""Rerun diagnostic 4 (threshold sweep) in isolation, without triggering
backtest_phase_c module re-execution. Constants are inlined."""
from __future__ import annotations
import io, sys, math, runpy
from pathlib import Path
from itertools import permutations
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "notebooks" / "output"
sys.path.insert(0, str(BASE))
sys.stdout.reconfigure(encoding="utf-8")

TAU = 0.8
ALPHA_FALLBACK = 1.333
MIN_COMBOS_FOR_Z = 100
TYPE_RELIABILITY = {
    "型1_逃げ本命": 1.0,
    "型2_イン残りヒモ荒れ": 0.9,
    "型3_頭荒れ": 0.8,
    "型4_ノイズ": 0.5,
}
PERMS = [(a, b, c) for a, b, c in permutations([1,2,3,4,5,6], 3)]

def pl_probs(p_lane):
    p = np.asarray(p_lane); out = np.zeros(120)
    for i, (a,b,c) in enumerate(PERMS):
        pa = p[a-1]
        pb = p[b-1] / max(1-pa, 1e-9)
        pc = p[c-1] / max(1-pa-p[b-1], 1e-9)
        out[i] = pa*pb*pc
    return out

print("=" * 80); print("Threshold sweep (standalone rerun)"); print("=" * 80)

print("\n[0] v4 runpy ...")
_b = io.StringIO(); _o = sys.stdout; sys.stdout = _b
try: ns = runpy.run_path(str(BASE / "notebooks" / "pl_v4_training.py"))
finally: sys.stdout = _o
beta_v4 = ns["beta_v4"]; X_test = ns["X_test_v4"]
pi_test = ns["pi_test_v4"]; keys_test = ns["keys_test_v4"].reset_index(drop=True)
N = len(keys_test)
print(f"  β_v4={beta_v4.shape}  X={X_test.shape}  N={N}")

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
print(f"   res_map={len(res_map):,}  odds_map={len(odds_map):,}")

# Diagnostic: check first race
print("\n[2] single-race debug ...")
for i in range(3):
    k = keys_test.iloc[i]
    d = pd.Timestamp(k["date"]).date(); s=int(k["stadium"]); r=int(k["race_number"])
    in_res = (d,s,r) in res_map; in_odds = (d,s,r) in odds_map
    print(f"  race {i}: ({d},{s},{r}) res={in_res} odds={in_odds}")
    if in_odds:
        book = odds_map[(d,s,r)]
        Z = sum(1.0/o for o in book.values()) if len(book) >= MIN_COMBOS_FOR_Z else ALPHA_FALLBACK
        S = X_test[i] @ beta_v4
        s_t = S/TAU; s_t = s_t - s_t.max()
        p_lane = np.exp(s_t) / np.exp(s_t).sum()
        probs = pl_probs(p_lane)
        # ev/edge for top-1 combination in book
        ev_edges = []
        for j, combo in enumerate(PERMS):
            odds = book.get(combo)
            if not odds: continue
            p_adj = probs[j] * 1.0
            ev = p_adj * odds; edge = p_adj - 1.0/(odds*Z)
            ev_edges.append((ev, edge, p_adj, odds))
        ev_edges.sort(reverse=True)
        print(f"    Z={Z:.4f}, book={len(book)}, top-5 by EV:")
        for e in ev_edges[:5]:
            print(f"      EV={e[0]:.3f} Edge={e[1]:+.3f} p={e[2]:.4f} odds={e[3]:.1f}")

# Full threshold sweep
print("\n[3] threshold sweep ...")
thresholds = [(1.10, 0.01), (1.20, 0.02), (1.30, 0.03),
              (1.50, 0.05), (1.80, 0.05), (2.00, 0.05), (2.50, 0.10)]

def binomial_ci(hits, total):
    if total == 0: return (0, 0)
    p = hits/total
    se = math.sqrt(p*(1-p)/total); z = 1.96
    return (max(0, p-z*se), min(1, p+z*se))

print(f"\n  {'EV':>5} {'Edge':>5} {'bet_n':>8} {'hit':>6} {'hit%':>7} "
      f"{'avg_o':>7} {'ROI':>7} {'CIlo':>7} {'CIhi':>7}")
thr_rows = []
for ev_thr, edge_thr in thresholds:
    bet_n = hit_n = stake_sum = pay_sum = 0
    odds_list = []
    skip_res = skip_odds = skip_type = skip_noodds = 0
    for i in range(N):
        k = keys_test.iloc[i]
        d = pd.Timestamp(k["date"]).date(); s=int(k["stadium"]); r=int(k["race_number"])
        if (d,s,r) not in res_map: skip_res += 1; continue
        if (d,s,r) not in odds_map: skip_odds += 1; continue
        actual = res_map[(d,s,r)]; book = odds_map[(d,s,r)]
        Z = sum(1.0/o for o in book.values()) if len(book) >= MIN_COMBOS_FOR_Z else ALPHA_FALLBACK
        S = X_test[i] @ beta_v4
        std_S = max(S.std(ddof=0), 0.3)
        G_S = (np.sort(-S)[0] - np.sort(-S)[1]) / std_S  # note: -S means max is "smallest"
        # Actually -S sorted ascending: -max, -2nd, ... → [0]-[1] = -max - (-2nd) = 2nd-max. Negative!
        # Fix below (kept as-is to match original diagnosis; computing via positive sort)
        s_sorted = np.sort(S)[::-1]
        G_S = (s_sorted[0] - s_sorted[1]) / std_S
        O_S = (S[3:6].max() - S.mean()) / std_S
        top1_lane = int(S.argmax())
        if top1_lane == 0:
            if G_S>1.0 and O_S<0.3: t = "型1_逃げ本命"
            elif G_S>0.6 and O_S>0.2: t = "型2_イン残りヒモ荒れ"
            else: t = "型4_ノイズ"
        else:
            if O_S>0.3 and G_S>0.4: t = "型3_頭荒れ"
            else: t = "型4_ノイズ"
        if t == "型4_ノイズ": skip_type += 1; continue
        rel = TYPE_RELIABILITY[t]
        s_t = S/TAU; s_t = s_t - s_t.max()
        p_lane = np.exp(s_t) / np.exp(s_t).sum()
        probs = pl_probs(p_lane)
        race_had_bet = False
        for j, combo in enumerate(PERMS):
            odds = book.get(combo)
            if not odds: continue
            p_adj = probs[j] * rel
            ev = p_adj * odds; edge = p_adj - 1.0/(odds*Z)
            if ev >= ev_thr and edge >= edge_thr:
                bet_n += 1; stake_sum += 100; odds_list.append(odds)
                race_had_bet = True
                if combo == actual:
                    pay_sum += int(100 * odds); hit_n += 1
        if not race_had_bet: skip_noodds += 1
    roi = pay_sum/stake_sum if stake_sum else 0
    hit_rate = hit_n/bet_n if bet_n else 0
    avg_odds = float(np.mean(odds_list)) if odds_list else 0
    ci = binomial_ci(hit_n, bet_n)
    roi_lo = ci[0] * avg_odds; roi_hi = ci[1] * avg_odds
    print(f"  {ev_thr:>5.2f} {edge_thr:>5.2f} {bet_n:>8,} {hit_n:>6,} "
          f"{hit_rate*100:>6.2f}% {avg_odds:>7.2f} {roi:>7.4f} {roi_lo:>7.4f} {roi_hi:>7.4f}"
          f"   (skip res={skip_res} type={skip_type} noodds={skip_noodds})")
    thr_rows.append({"ev_thr":ev_thr,"edge_thr":edge_thr,"bet_n":bet_n,
                     "hit_n":hit_n,"hit_rate":hit_rate*100,"avg_odds":avg_odds,
                     "roi":roi,"roi_ci_lo":roi_lo,"roi_ci_hi":roi_hi,
                     "stake":stake_sum,"payout":pay_sum,"profit":pay_sum-stake_sum})

pd.DataFrame(thr_rows).to_csv(OUT/"threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
print(f"\nsaved: {OUT/'threshold_sensitivity.csv'}")
print("done")
