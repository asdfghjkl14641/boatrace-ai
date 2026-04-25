# -*- coding: utf-8 -*-
"""Phase C 汎化戦略特定: 6 戦略 × train/test バックテスト."""
from __future__ import annotations
import io, sys, math, pickle, runpy, time, warnings
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
EV_THR = 1.20; EDGE_THR = 0.02
ALPHA_FALLBACK = 1.333
MIN_COMBOS_FOR_Z = 100
BUDGET = 3000; MIN_UNIT = 100
TYPE_RELIABILITY = {"型1": 1.0, "型2": 0.9, "型3": 0.8}
PERMS = [(a,b,c) for a,b,c in permutations([1,2,3,4,5,6],3)]

# 閾値セット
T1_orig = (1.0, 0.3)
T1_Q    = (1.0, 0.2)
T1_S    = (1.5, 0.2)
T2_orig = (0.6, 0.2)
T2_delta = (0.9, 0.4)
T3_orig  = (0.3, 0.4)
T3_none  = None

STRATEGIES = [
    ("1_full",          T1_orig, T2_orig,  T3_orig),
    ("2_noT3",          T1_orig, T2_orig,  T3_none),
    ("3_T2d",           T1_orig, T2_delta, T3_orig),
    ("4_T2d_noT3",      T1_orig, T2_delta, T3_none),
    ("5_T1Q_T2d_noT3",  T1_Q,    T2_delta, T3_none),
    ("6_T1S_T2d_noT3",  T1_S,    T2_delta, T3_none),
]

def pl_probs(p_lane):
    p = np.asarray(p_lane); out = np.zeros(120)
    for i, (a,b,c) in enumerate(PERMS):
        pa = p[a-1]
        pb = p[b-1] / max(1-pa, 1e-9)
        pc = p[c-1] / max(1-pa-p[b-1], 1e-9)
        out[i] = pa*pb*pc
    return out

def compute_indices(S):
    std_S = max(S.std(ddof=0), 0.3)
    s_sorted = np.sort(S)[::-1]
    return (
        (s_sorted[0] - s_sorted[1]) / std_S,  # G_S
        (S[3:6].max() - S.mean()) / std_S,    # O_S
        int(S.argmax()),                       # top1_lane
    )

def classify(idx, t1, t2, t3):
    G, O, top1 = idx
    if top1 == 0:
        if G > t1[0] and O < t1[1]: return "型1"
        if t2 is not None and G > t2[0] and O > t2[1]: return "型2"
        return None
    else:
        if t3 is not None and O > t3[0] and G > t3[1]: return "型3"
        return None

def bootstrap_roi_ci(stakes, pays, n_boot=1000, seed=42):
    stakes = np.asarray(stakes, dtype=float)
    pays = np.asarray(pays, dtype=float)
    n = len(stakes)
    if n == 0 or stakes.sum() == 0: return (0, 0)
    rng = np.random.default_rng(seed)
    rois = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = stakes[idx].sum()
        if s > 0: rois.append(pays[idx].sum()/s)
    return (float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5)))

print("=" * 80); print("Phase C 汎化戦略特定 — 6 戦略 × train/test"); print("=" * 80)

# ========== [0] データロード ==========
print("\n[0] v4 + isotonic ロード ...")
_b = io.StringIO(); _o = sys.stdout; sys.stdout = _b
try: ns = runpy.run_path(str(BASE / "notebooks" / "pl_v4_training.py"))
finally: sys.stdout = _o
beta_v4 = ns["beta_v4"]
X_tr = ns["X_train_v4"]; keys_tr = ns["keys_train_v4"].reset_index(drop=True)
X_te = ns["X_test_v4"];  keys_te = ns["keys_test_v4"].reset_index(drop=True)

with open(OUT/"calibration_isotonic_train.pkl", "rb") as f:
    iso = pickle.load(f)
print(f"   v4: train={len(keys_tr):,}, test={len(keys_te):,}")

# ========== [1] odds + results (両期間) ==========
print("\n[1] odds + results ロード ...")
from scripts.db import get_connection
conn = get_connection()

def load_period(date_from, date_to):
    odds = pd.read_sql_query(f"""
        SELECT date, stadium, race_number, combination, odds_1min
        FROM trifecta_odds
        WHERE date BETWEEN '{date_from}' AND '{date_to}' AND odds_1min IS NOT NULL
    """, conn.native)
    odds["date"] = pd.to_datetime(odds["date"]).dt.date
    res = pd.read_sql_query(f"""
        SELECT date, stadium, race_number, rank, boat FROM race_results
        WHERE date BETWEEN '{date_from}' AND '{date_to}' AND rank BETWEEN 1 AND 3
    """, conn.native)
    res["date"] = pd.to_datetime(res["date"]).dt.date

    res_map = {}
    for (d,s,r),g in res.groupby(["date","stadium","race_number"]):
        g2 = g.sort_values("rank")
        if len(g2) < 3: continue
        res_map[(d,s,r)] = (int(g2.iloc[0]["boat"]), int(g2.iloc[1]["boat"]), int(g2.iloc[2]["boat"]))
    odds_map = {}
    for (d,s,r),g in odds.groupby(["date","stadium","race_number"]):
        book = {}
        for _,row in g.iterrows():
            try:
                a,b,c = map(int, row["combination"].split("-"))
                o = float(row["odds_1min"])
                if o > 0: book[(a,b,c)] = o
            except Exception: pass
        if book: odds_map[(d,s,r)] = book
    return res_map, odds_map

tr_res, tr_odds = load_period("2023-05-01", "2025-12-31")
te_res, te_odds = load_period("2026-01-01", "2026-04-18")
conn.close()
print(f"   train: res={len(tr_res):,} odds={len(tr_odds):,}")
print(f"   test:  res={len(te_res):,} odds={len(te_odds):,}")

# ========== [2] 両期間 × 6 戦略 ==========
def run_all_strategies(X, keys, res_map, odds_map, label):
    """全戦略を同時に 1-pass で評価."""
    N = len(keys)
    # 各戦略・各型ごとに per-race stake/pay を記録
    # stat[strat][type] = {"stakes":[], "pays":[], "races":0, "hits":0}
    stat = {s[0]:{"型1":{"stakes":[],"pays":[],"races":0,"hits":0},
                  "型2":{"stakes":[],"pays":[],"races":0,"hits":0},
                  "型3":{"stakes":[],"pays":[],"races":0,"hits":0},
                  "per_race_stakes":[], "per_race_pays":[],
                  "month_stakes":{}, "month_pays":{}}
            for s in STRATEGIES}
    t0 = time.time()
    for i in range(N):
        k = keys.iloc[i]
        d = pd.Timestamp(k["date"]).date(); s=int(k["stadium"]); r=int(k["race_number"])
        if (d,s,r) not in res_map or (d,s,r) not in odds_map: continue
        book = odds_map[(d,s,r)]; actual = res_map[(d,s,r)]
        Z = sum(1.0/o for o in book.values()) if len(book) >= MIN_COMBOS_FOR_Z else ALPHA_FALLBACK
        S = X[i] @ beta_v4
        idx = compute_indices(S)
        s_t = S/TAU; s_t = s_t - s_t.max()
        p_lane = np.exp(s_t)/np.exp(s_t).sum()
        probs = pl_probs(p_lane)
        probs_cal = iso.transform(probs.astype(np.float64))
        month = f"{d.year}-{d.month:02d}"

        for strat_name, t1, t2, t3 in STRATEGIES:
            t = classify(idx, t1, t2, t3)
            if t is None: continue
            rel = TYPE_RELIABILITY[t]
            cands = []
            for j, combo in enumerate(PERMS):
                odds = book.get(combo)
                if not odds: continue
                p_adj = probs_cal[j] * rel
                ev = p_adj * odds; edge = p_adj - 1.0/(odds*Z)
                if ev >= EV_THR and edge >= EDGE_THR:
                    cands.append((combo, odds, p_adj, ev, edge))
            if not cands: continue
            ev_sum = sum(c[3] for c in cands)
            alloc = []
            for combo, odds, p_adj, ev, edge in cands:
                units = int(BUDGET * ev / ev_sum / MIN_UNIT)
                stake = max(0, units) * MIN_UNIT
                alloc.append((combo, odds, stake))
            used = sum(a[2] for a in alloc); extra = BUDGET - used
            order = sorted(range(len(alloc)), key=lambda ii: -cands[ii][3])
            for ia in order:
                if extra < MIN_UNIT: break
                c,o,ss = alloc[ia]; alloc[ia] = (c,o,ss+MIN_UNIT); extra -= MIN_UNIT
            alloc = [a for a in alloc if a[2] > 0]
            if not alloc: continue
            race_stake = sum(a[2] for a in alloc); race_pay = 0; hit_flag = 0
            for combo, odds, st in alloc:
                if combo == actual:
                    race_pay += int(st*odds); hit_flag = 1
            st_obj = stat[strat_name]
            st_obj[t]["stakes"].append(race_stake); st_obj[t]["pays"].append(race_pay)
            st_obj[t]["races"] += 1; st_obj[t]["hits"] += hit_flag
            st_obj["per_race_stakes"].append(race_stake); st_obj["per_race_pays"].append(race_pay)
            st_obj["month_stakes"].setdefault(month, []).append(race_stake)
            st_obj["month_pays"].setdefault(month, []).append(race_pay)
        if (i+1) % 10000 == 0:
            print(f"     [{label}] {i+1}/{N}  elapsed {time.time()-t0:.1f}s")
    print(f"   [{label}] done {time.time()-t0:.1f}s")
    return stat

print("\n[2] train バックテスト (6 戦略同時)...")
stat_tr = run_all_strategies(X_tr, keys_tr, tr_res, tr_odds, "train")
print("\n[2b] test バックテスト (6 戦略同時)...")
stat_te = run_all_strategies(X_te, keys_te, te_res, te_odds, "test")

# ========== [3] 集計 ==========
def roll_up(stat):
    rows = []
    for strat_name, _, _, _ in STRATEGIES:
        obj = stat[strat_name]
        stakes = obj["per_race_stakes"]; pays = obj["per_race_pays"]
        total_stake = sum(stakes); total_pay = sum(pays)
        n_bet = len(stakes)
        n_hit = sum(obj[t]["hits"] for t in ["型1","型2","型3"])
        roi = total_pay/total_stake if total_stake else 0
        ci = bootstrap_roi_ci(stakes, pays, n_boot=1000)
        # 型別 ROI
        trois = {}
        for t in ["型1","型2","型3"]:
            ts = sum(obj[t]["stakes"]); tp = sum(obj[t]["pays"])
            trois[t] = tp/ts if ts else 0
        # 月別 ROI std
        mon_rois = []
        for m, m_st in obj["month_stakes"].items():
            m_pay = obj["month_pays"][m]
            ms = sum(m_st); mp = sum(m_pay)
            if ms: mon_rois.append(mp/ms)
        mon_std = float(np.std(mon_rois)) if mon_rois else 0
        rows.append({"strategy":strat_name,"n_bet":n_bet,"n_hit":n_hit,
                     "stake":total_stake,"payout":total_pay,
                     "profit":total_pay-total_stake,"roi":roi,
                     "ci_lo":ci[0],"ci_hi":ci[1],
                     "t1_roi":trois["型1"],"t2_roi":trois["型2"],"t3_roi":trois["型3"],
                     "t1_n":obj["型1"]["races"],"t2_n":obj["型2"]["races"],"t3_n":obj["型3"]["races"],
                     "month_std":mon_std})
    return pd.DataFrame(rows)

tr_df = roll_up(stat_tr); tr_df["period"] = "train"
te_df = roll_up(stat_te); te_df["period"] = "test"
all_df = pd.concat([tr_df, te_df], ignore_index=True)
all_df.to_csv(OUT/"generalization_matrix.csv", index=False, encoding="utf-8-sig")

print(f"\n  {'戦略':<20} {'期間':<6} {'N':>6} {'ROI':>7} {'CI下':>7} {'CI上':>7} "
      f"{'T1':>7} {'T2':>7} {'T3':>7} {'profit':>12}")
for _, rw in all_df.iterrows():
    print(f"  {rw['strategy']:<20} {rw['period']:<6} {rw['n_bet']:>6,} "
          f"{rw['roi']:>7.4f} {rw['ci_lo']:>7.4f} {rw['ci_hi']:>7.4f} "
          f"{rw['t1_roi']:>7.4f} {rw['t2_roi']:>7.4f} {rw['t3_roi']:>7.4f} "
          f"¥{int(rw['profit']):>+11,}")

# ========== [Step 2] 差分分析 ==========
print("\n[Step 2] train/test 差分 ...")
gap_rows = []
for strat_name, _, _, _ in STRATEGIES:
    tr = tr_df[tr_df["strategy"]==strat_name].iloc[0]
    te = te_df[te_df["strategy"]==strat_name].iloc[0]
    diff = te["roi"] - tr["roi"]
    t1d = te["t1_roi"] - tr["t1_roi"]
    t2d = te["t2_roi"] - tr["t2_roi"]
    t3d = te["t3_roi"] - tr["t3_roi"]
    bet_rate_tr = tr["n_bet"] / len(keys_tr) * 100
    bet_rate_te = te["n_bet"] / len(keys_te) * 100
    if abs(diff) < 0.05: verdict = "汎化"
    elif abs(diff) < 0.10: verdict = "軽度過学習"
    else: verdict = "明確過学習"
    gap_rows.append({"strategy":strat_name,
                     "train_roi":tr["roi"],"test_roi":te["roi"],"roi_diff":diff,
                     "t1_diff":t1d,"t2_diff":t2d,"t3_diff":t3d,
                     "bet_rate_train":bet_rate_tr,"bet_rate_test":bet_rate_te,
                     "verdict":verdict,
                     "train_ci_lo":tr["ci_lo"],"train_ci_hi":tr["ci_hi"],
                     "test_ci_lo":te["ci_lo"],"test_ci_hi":te["ci_hi"]})
gap_df = pd.DataFrame(gap_rows)
gap_df.to_csv(OUT/"train_test_gap_analysis.csv", index=False, encoding="utf-8-sig")

print(f"\n  {'戦略':<20} {'train':>7} {'test':>7} {'差':>8} {'判定':<12}")
for _, rw in gap_df.iterrows():
    print(f"  {rw['strategy']:<20} {rw['train_roi']:>7.4f} {rw['test_roi']:>7.4f} "
          f"{rw['roi_diff']:>+8.4f} {rw['verdict']:<12}")

# ========== [Step 3] 汎化戦略の評価 ==========
print("\n[Step 3] 汎化戦略 (|差|<0.05) の評価 ...")
generalized = gap_df[abs(gap_df["roi_diff"]) < 0.05]
if len(generalized) == 0:
    print("   **汎化している戦略なし**")
    best_gen = None
else:
    # 各観点
    gen_with_train = generalized.merge(tr_df, left_on="strategy", right_on="strategy", suffixes=("","_tr"))
    # ROI 最大 (train ベース — 汎化しているので test も同等)
    roi_max_idx = generalized["train_roi"].idxmax()
    ci_max_idx = gap_df[gap_df.index.isin(generalized.index)].assign(
        min_ci=lambda d: d[["train_ci_lo","test_ci_lo"]].min(axis=1))["min_ci"].idxmax()
    # 月別分散 min
    gen_mon = []
    for _, rw in generalized.iterrows():
        tr_mon = tr_df[tr_df["strategy"]==rw["strategy"]].iloc[0]["month_std"]
        te_mon = te_df[te_df["strategy"]==rw["strategy"]].iloc[0]["month_std"]
        gen_mon.append((rw["strategy"], tr_mon, te_mon, (tr_mon+te_mon)/2))
    gen_mon_df = pd.DataFrame(gen_mon, columns=["strategy","tr_mon","te_mon","avg_mon"]).sort_values("avg_mon")
    print(f"   ROI 最大 (train): {generalized.loc[roi_max_idx, 'strategy']}")
    print(f"   CI 下最も高い:    {generalized.loc[ci_max_idx, 'strategy']}")
    print(f"   月別分散小:       {gen_mon_df.iloc[0]['strategy']}")
    best_gen = generalized.loc[roi_max_idx, "strategy"]

# ========== [Step 4] 最終判定 ==========
print("\n[Step 4] 最終判定 ...")
if best_gen:
    best_tr = tr_df[tr_df["strategy"]==best_gen].iloc[0]
    best_te = te_df[te_df["strategy"]==best_gen].iloc[0]
    best_roi = (best_tr["roi"] + best_te["roi"]) / 2
    best_ci_lo = min(best_tr["ci_lo"], best_te["ci_lo"])
    if best_roi >= 1.0 and best_ci_lo >= 0.95:
        pattern = "A"; desc = f"汎化 & ROI≥1.0 — 運用候補 ({best_gen})"
    elif best_roi >= 0.95:
        pattern = "B"; desc = f"汎化するが break-even ({best_roi:.4f})"
    else:
        pattern = "C"; desc = f"汎化するが ROI<0.95 ({best_roi:.4f}) — 3 連単撤退"
else:
    pattern = "D"; desc = "全戦略で train/test 乖離 — モデル再設計必要"
    best_roi = 0; best_ci_lo = 0

print(f"   パターン: {pattern} — {desc}")

# ========== [Step 5] レポート ==========
print("\n[Step 5] レポート生成 ...")
md = f"""# Phase C 汎化戦略特定レポート

## 検証戦略
| # | 戦略 | 型1 | 型2 | 型3 |
|---|---|---|---|---|
| 1 | full | G>1.0, O<0.3 | G>0.6, O>0.2 | O>0.3, G>0.4 |
| 2 | noT3 | G>1.0, O<0.3 | G>0.6, O>0.2 | 除外 |
| 3 | T2d | G>1.0, O<0.3 | G>0.9, O>0.4 | O>0.3, G>0.4 |
| 4 | T2d_noT3 | G>1.0, O<0.3 | G>0.9, O>0.4 | 除外 |
| 5 | T1Q_T2d_noT3 | G>1.0, O<0.2 | G>0.9, O>0.4 | 除外 |
| 6 | T1S_T2d_noT3 | G>1.5, O<0.2 | G>0.9, O>0.4 | 除外 |

共通: Isotonic キャリ + EV≥1.2 + Edge≥0.02, 予算 3000/レース, EV比例配分

## Step 1: 全体マトリクス

| 戦略 | 期間 | N_bet | ROI | CI下 | CI上 | T1 | T2 | T3 | profit |
|---|---|---|---|---|---|---|---|---|---|
"""
for _, rw in all_df.iterrows():
    md += (f"| {rw['strategy']} | {rw['period']} | {rw['n_bet']:,} | "
           f"**{rw['roi']:.4f}** | {rw['ci_lo']:.4f} | {rw['ci_hi']:.4f} | "
           f"{rw['t1_roi']:.4f} | {rw['t2_roi']:.4f} | {rw['t3_roi']:.4f} | "
           f"¥{int(rw['profit']):+,} |\n")

md += """

## Step 2: train/test 差分分析

| 戦略 | train ROI | test ROI | 差 | 判定 |
|---|---|---|---|---|
"""
for _, rw in gap_df.iterrows():
    md += (f"| {rw['strategy']} | {rw['train_roi']:.4f} | {rw['test_roi']:.4f} | "
           f"{rw['roi_diff']:+.4f} | **{rw['verdict']}** |\n")

md += "\n## Step 3: 汎化戦略の評価\n\n"
if len(generalized) == 0:
    md += "**汎化している戦略なし (|差|<0.05)**\n"
else:
    md += f"汎化戦略: {', '.join(generalized['strategy'])}\n\n"
    md += f"- ROI 最大 (train): `{generalized.loc[roi_max_idx, 'strategy']}`\n"
    md += f"- CI 下最高: `{generalized.loc[ci_max_idx, 'strategy']}`\n"
    md += f"- 月別分散小: `{gen_mon_df.iloc[0]['strategy']}`\n"

md += f"""

## Step 4: 最終判定

**パターン {pattern}**: {desc}
"""

if best_gen:
    best_tr_row = tr_df[tr_df["strategy"]==best_gen].iloc[0]
    best_te_row = te_df[te_df["strategy"]==best_gen].iloc[0]
    md += f"""
### 最良汎化戦略 ({best_gen})

- train ROI: {best_tr_row['roi']:.4f} CI [{best_tr_row['ci_lo']:.4f}, {best_tr_row['ci_hi']:.4f}]
- test ROI: {best_te_row['roi']:.4f} CI [{best_te_row['ci_lo']:.4f}, {best_te_row['ci_hi']:.4f}]
- train bet数: {int(best_tr_row['n_bet']):,}, test bet数: {int(best_te_row['n_bet']):,}
- 型別 ROI (train / test):
  - 型1: {best_tr_row['t1_roi']:.4f} / {best_te_row['t1_roi']:.4f}
  - 型2: {best_tr_row['t2_roi']:.4f} / {best_te_row['t2_roi']:.4f}
  - 型3: {best_tr_row['t3_roi']:.4f} / {best_te_row['t3_roi']:.4f}
"""

md += """

### 次アクション
"""
if pattern == "A":
    md += "1. 運用候補戦略確定\n2. ペーパートレード\n3. 仕様書反映\n"
elif pattern == "B":
    md += "1. break-even ライン — 運用は消極的\n2. 他券種 (2連単) バックテスト推奨\n3. データ延長で CI 絞り\n"
elif pattern == "C":
    md += "1. **3連単撤退**\n2. 他券種 (2連単・2連複・3連複) バックテスト必須\n3. モデル改善 (Phase D)\n"
else:
    md += "1. モデル自体に問題あり\n2. Phase D (新特徴量) or 非線形化\n3. v5 設計検討\n"

md += """

## 出力ファイル
- `generalization_matrix.csv`
- `train_test_gap_analysis.csv`
"""

with open(OUT/"phase_c_generalization_final.md", "w", encoding="utf-8") as f:
    f.write(md)
print(f"   saved: phase_c_generalization_final.md")
print(f"\n=== パターン {pattern}: {desc} ===")
print("完了")
