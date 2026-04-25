# -*- coding: utf-8 -*-
"""Phase A-ext-6: v4_ext_fixed で 6 戦略 × train/test 過学習再検証."""
from __future__ import annotations
import io, os, sys, math, pickle, runpy, time, warnings
from pathlib import Path
from itertools import permutations
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "notebooks" / "output"
sys.path.insert(0, str(BASE))
sys.stdout.reconfigure(encoding="utf-8")

# 日付 (v4_ext_fixed 想定)
TRAIN_FROM = "2020-02-01"; TRAIN_TO = "2025-06-30"
TEST_FROM  = "2025-07-01"; TEST_TO  = "2026-04-18"

TAU = 0.8
EV_THR = 1.20; EDGE_THR = 0.02
ALPHA_FALLBACK = 1.333
MIN_COMBOS_FOR_Z = 100
BUDGET = 3000; MIN_UNIT = 100
TYPE_RELIABILITY = {"型1": 1.0, "型2": 0.9, "型3": 0.8}
PERMS = [(a,b,c) for a,b,c in permutations([1,2,3,4,5,6],3)]

T1_orig = (1.0, 0.3); T1_Q = (1.0, 0.2); T1_S = (1.5, 0.2)
T2_orig = (0.6, 0.2); T2_delta = (0.9, 0.4)
T3_orig = (0.3, 0.4); T3_none = None
STRATEGIES = [
    ("1_full",         T1_orig, T2_orig,  T3_orig),
    ("2_noT3",         T1_orig, T2_orig,  T3_none),
    ("3_T2d",          T1_orig, T2_delta, T3_orig),
    ("4_T2d_noT3",     T1_orig, T2_delta, T3_none),
    ("5_T1Q_T2d_noT3", T1_Q,    T2_delta, T3_none),
    ("6_T1S_T2d_noT3", T1_S,    T2_delta, T3_none),
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
    return ((s_sorted[0] - s_sorted[1]) / std_S,
            (S[3:6].max() - S.mean()) / std_S,
            int(S.argmax()))

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
    stakes = np.asarray(stakes, dtype=float); pays = np.asarray(pays, dtype=float)
    n = len(stakes)
    if n == 0 or stakes.sum() == 0: return (0, 0)
    rng = np.random.default_rng(seed)
    rois = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = stakes[idx].sum()
        if s > 0: rois.append(pays[idx].sum()/s)
    return (float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5)))


print("=" * 80)
print("v4_ext_fixed 過学習再検証 — 6 戦略 × train/test")
print("=" * 80)

# ========== [0] v4_ext_fixed runpy ==========
print("\n[0] v4_ext_fixed runpy (β + tensor + keys)...")
os.environ["V4_TRAIN_FROM"] = TRAIN_FROM
os.environ["V4_TRAIN_TO"]   = TRAIN_TO
os.environ["V4_TEST_FROM"]  = TEST_FROM
os.environ["V4_TEST_TO"]    = TEST_TO
os.environ["V4_OUT_SUFFIX"] = "_ext_fixed"

_buf = io.StringIO(); _o = sys.stdout; sys.stdout = _buf
try:
    ns = runpy.run_path(str(BASE / "notebooks" / "pl_v4_training.py"))
finally:
    sys.stdout = _o

beta = ns["beta_v4"]
X_tr = ns["X_train_v4"]; keys_tr = ns["keys_train_v4"].reset_index(drop=True)
X_te = ns["X_test_v4"];  keys_te = ns["keys_test_v4"].reset_index(drop=True)
pi_tr = ns["pi_train_v4"]; pi_te = ns["pi_test_v4"]
print(f"  β shape={beta.shape}, X_train={X_tr.shape}, X_test={X_te.shape}")

# ========== [1] Isotonic fit on new train ==========
print("\n[1] Isotonic fit (train 2020-02〜2025-06)...")
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

print("  [1a] train odds + results...")
tr_res, tr_odds = load_period(TRAIN_FROM, TRAIN_TO)
print(f"   train: res={len(tr_res):,}  odds={len(tr_odds):,}")
print("  [1b] test odds + results...")
te_res, te_odds = load_period(TEST_FROM, TEST_TO)
print(f"   test:  res={len(te_res):,}  odds={len(te_odds):,}")
conn.close()

# train pairs (p_model, hit) for isotonic
print("\n[1c] train pairs for isotonic...")
N_tr = len(keys_tr)
tr_p = np.empty(N_tr * 120, dtype=np.float32)
tr_y = np.zeros(N_tr * 120, dtype=np.int8)
tr_valid = np.zeros(N_tr * 120, dtype=bool)
for i in range(N_tr):
    k = keys_tr.iloc[i]
    d = pd.Timestamp(k["date"]).date(); s=int(k["stadium"]); r=int(k["race_number"])
    S = X_tr[i] @ beta
    s_t = S/TAU; s_t = s_t - s_t.max()
    p_lane = np.exp(s_t) / np.exp(s_t).sum()
    probs = pl_probs(p_lane)
    base = i * 120
    tr_p[base:base+120] = probs.astype(np.float32)
    if (d,s,r) in tr_res:
        actual = tr_res[(d,s,r)]
        try:
            pos = PERMS.index(actual); tr_y[base+pos] = 1
        except ValueError: pass
        tr_valid[base:base+120] = True
tr_p_v = tr_p[tr_valid]; tr_y_v = tr_y[tr_valid]
print(f"   train pairs: {len(tr_p_v):,}, hits: {int(tr_y_v.sum()):,}")

iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
iso.fit(tr_p_v.astype(np.float64), tr_y_v.astype(np.float64))
with open(OUT/"calibration_isotonic_ext_fixed.pkl", "wb") as f:
    pickle.dump(iso, f)
print("  isotonic saved: calibration_isotonic_ext_fixed.pkl")

# ========== [2] 6 戦略 × train/test 同時評価 ==========
def run_all_strategies(X, keys, res_map, odds_map, label):
    N = len(keys)
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
        S = X[i] @ beta
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
        if (i+1) % 30000 == 0:
            print(f"     [{label}] {i+1}/{N}  elapsed {time.time()-t0:.1f}s")
    print(f"   [{label}] done {time.time()-t0:.1f}s")
    return stat

print("\n[2] train バックテスト (6 戦略同時)...")
stat_tr = run_all_strategies(X_tr, keys_tr, tr_res, tr_odds, "train")
print("\n[2b] test バックテスト (6 戦略同時)...")
stat_te = run_all_strategies(X_te, keys_te, te_res, te_odds, "test")

def roll_up(stat, period):
    rows = []
    for strat_name, _, _, _ in STRATEGIES:
        obj = stat[strat_name]
        stakes = obj["per_race_stakes"]; pays = obj["per_race_pays"]
        total_stake = sum(stakes); total_pay = sum(pays)
        n_bet = len(stakes)
        n_hit = sum(obj[t]["hits"] for t in ["型1","型2","型3"])
        roi = total_pay/total_stake if total_stake else 0
        ci = bootstrap_roi_ci(stakes, pays, n_boot=1000)
        trois = {}
        for t in ["型1","型2","型3"]:
            ts = sum(obj[t]["stakes"]); tp = sum(obj[t]["pays"])
            trois[t] = tp/ts if ts else 0
        mon_rois = []
        for m, m_st in obj["month_stakes"].items():
            m_pay = obj["month_pays"][m]
            ms = sum(m_st); mp = sum(m_pay)
            if ms: mon_rois.append(mp/ms)
        mon_std = float(np.std(mon_rois)) if mon_rois else 0
        rows.append({"strategy":strat_name,"period":period,"n_bet":n_bet,"n_hit":n_hit,
                     "stake":total_stake,"payout":total_pay,"profit":total_pay-total_stake,
                     "roi":roi,"ci_lo":ci[0],"ci_hi":ci[1],
                     "t1_roi":trois["型1"],"t2_roi":trois["型2"],"t3_roi":trois["型3"],
                     "t1_n":obj["型1"]["races"],"t2_n":obj["型2"]["races"],"t3_n":obj["型3"]["races"],
                     "month_std":mon_std})
    return pd.DataFrame(rows)

tr_df = roll_up(stat_tr, "train")
te_df = roll_up(stat_te, "test")
all_df = pd.concat([tr_df, te_df], ignore_index=True)
all_df.to_csv(OUT/"v4_ext_fixed_generalization_matrix.csv", index=False, encoding="utf-8-sig")

print(f"\n  {'戦略':<20} {'期間':<6} {'N':>6} {'ROI':>7} {'CI下':>7} {'CI上':>7} "
      f"{'T1':>7} {'T2':>7} {'T3':>7} {'profit':>12}")
for _, rw in all_df.iterrows():
    print(f"  {rw['strategy']:<20} {rw['period']:<6} {rw['n_bet']:>6,} "
          f"{rw['roi']:>7.4f} {rw['ci_lo']:>7.4f} {rw['ci_hi']:>7.4f} "
          f"{rw['t1_roi']:>7.4f} {rw['t2_roi']:>7.4f} {rw['t3_roi']:>7.4f} "
          f"¥{int(rw['profit']):>+11,}")

# ========== [3] 差分分析 + Phase C 比較 ==========
print("\n[3] train/test 差 + Phase C 比較...")
gap_rows = []
# Phase C 結果 (既知値)
phase_c = {
    "1_full": (0.867, 0.954), "2_noT3": (0.871, 0.950),
    "3_T2d": (0.858, 1.020), "4_T2d_noT3": (0.863, 1.017),
    "5_T1Q_T2d_noT3": (0.865, 1.059), "6_T1S_T2d_noT3": (0.861, 1.145),
}
for strat_name, _, _, _ in STRATEGIES:
    tr = tr_df[tr_df["strategy"]==strat_name].iloc[0]
    te = te_df[te_df["strategy"]==strat_name].iloc[0]
    diff = te["roi"] - tr["roi"]
    if abs(diff) < 0.05: verdict = "汎化"
    elif abs(diff) < 0.10: verdict = "軽度過学習"
    else: verdict = "明確過学習"
    pc_tr, pc_te = phase_c.get(strat_name, (0,0))
    pc_diff = pc_te - pc_tr
    gap_rows.append({"strategy":strat_name,
                     "phase_c_tr":pc_tr, "phase_c_te":pc_te, "phase_c_diff":pc_diff,
                     "ext_tr":tr["roi"], "ext_te":te["roi"], "ext_diff":diff,
                     "ext_ci_lo":te["ci_lo"], "ext_ci_hi":te["ci_hi"],
                     "verdict":verdict,
                     "train_n": tr["n_bet"], "test_n": te["n_bet"]})
gap_df = pd.DataFrame(gap_rows)
gap_df.to_csv(OUT/"v4_ext_fixed_vs_phase_c_comparison.csv", index=False, encoding="utf-8-sig")

print(f"\n  {'戦略':<20} {'PC差':>7} {'ext差':>7} {'縮小':>7} {'判定':<12}")
for _, rw in gap_df.iterrows():
    shrink = rw['phase_c_diff'] - rw['ext_diff']
    print(f"  {rw['strategy']:<20} {rw['phase_c_diff']:>+7.4f} {rw['ext_diff']:>+7.4f} "
          f"{shrink:>+7.4f} {rw['verdict']:<12}")

# ========== [4] 最良戦略特定 ==========
print("\n[4] 最良戦略特定...")
# 判定基準: (1)汎化 AND (2)ROI>=1.0 AND (3)CI下>=0.95
candidates = []
for _, rw in gap_df.iterrows():
    crit1 = abs(rw["ext_diff"]) < 0.05
    crit2 = rw["ext_te"] >= 1.0 and rw["ext_tr"] >= 1.0
    crit3 = rw["ext_ci_lo"] >= 0.95
    candidates.append({**rw.to_dict(),
                       "汎化": crit1, "黒字": crit2, "CI": crit3,
                       "score": sum([crit1, crit2, crit3])})
cand_df = pd.DataFrame(candidates).sort_values(["score","ext_te"], ascending=[False, False])
print(f"\n  {'戦略':<20} {'汎化':>4} {'黒字':>4} {'CI':>4} {'ext ROI':>8} {'CI下':>7}")
for _, rw in cand_df.iterrows():
    print(f"  {rw['strategy']:<20} {'✅' if rw['汎化'] else '❌':>3} "
          f"{'✅' if rw['黒字'] else '❌':>3} {'✅' if rw['CI'] else '❌':>3} "
          f"{rw['ext_te']:>7.4f} {rw['ext_ci_lo']:>7.4f}")

best = cand_df.iloc[0]
all_criteria = best["score"] == 3

# ========== [5] 最終判定 + レポート ==========
print("\n[5] 最終判定...")
# パターン判定
max_diff = gap_df["ext_diff"].abs().max()
max_roi_test = te_df["roi"].max()
phase_c_max_diff = gap_df["phase_c_diff"].abs().max()

if all_criteria:
    pattern = "α"; desc = f"運用候補確定 ({best['strategy']})"
elif max_diff < 0.05 and max_roi_test < 1.0:
    pattern = "β"; desc = "構造的限界 — 過学習解消したが ROI<1.0"
elif max_diff > 0.10:
    pattern = "γ"; desc = "Phase C の過学習は本物 — 戦略選択手法見直し必要"
elif max(te_df["roi"] - tr_df.iloc[0]["roi"] for _, row in tr_df.iterrows()) > 0.05:
    pattern = "δ"; desc = "期間拡大で改善の兆しあり"
else:
    pattern = "β"; desc = "構造的限界"

print(f"\n  パターン {pattern}: {desc}")
print(f"  最大過学習差: Phase C {phase_c_max_diff:.4f} → v4_ext_fixed {max_diff:.4f}")
print(f"  test 最高 ROI: {max_roi_test:.4f}")

md = f"""# v4_ext_fixed 最終判定レポート

## 核心メッセージ
- **Phase C 過学習は{'解消' if max_diff < 0.05 else ('部分解消' if max_diff < phase_c_max_diff*0.5 else '解消せず')}**
- Phase C 最大差 {phase_c_max_diff:.4f} → v4_ext_fixed {max_diff:.4f} ({(phase_c_max_diff-max_diff)/phase_c_max_diff*100:+.1f}% 縮小)
- test 最高 ROI: {max_roi_test:.4f}
- 最良戦略 ({best['strategy']}): ROI={best['ext_te']:.4f}, CI[{best['ext_ci_lo']:.4f}, {best['ext_ci_hi']:.4f}], 差={best['ext_diff']:+.4f}

## 6 戦略 全体マトリクス

| 戦略 | 期間 | N_bet | ROI | CI下 | CI上 | T1 | T2 | T3 | profit |
|---|---|---|---|---|---|---|---|---|---|
"""
for _, rw in all_df.iterrows():
    md += (f"| {rw['strategy']} | {rw['period']} | {rw['n_bet']:,} | "
           f"**{rw['roi']:.4f}** | {rw['ci_lo']:.4f} | {rw['ci_hi']:.4f} | "
           f"{rw['t1_roi']:.4f} | {rw['t2_roi']:.4f} | {rw['t3_roi']:.4f} | "
           f"¥{int(rw['profit']):+,} |\n")

md += "\n## Phase C との比較 (過学習縮小)\n\n"
md += "| 戦略 | Phase C 差 | v4_ext_fixed 差 | 縮小 | 判定 |\n"
md += "|---|---|---|---|---|\n"
for _, rw in gap_df.iterrows():
    shrink = rw['phase_c_diff'] - rw['ext_diff']
    md += (f"| {rw['strategy']} | {rw['phase_c_diff']:+.4f} | {rw['ext_diff']:+.4f} | "
           f"{shrink:+.4f} | {rw['verdict']} |\n")

md += f"""

## 最良戦略 ({best['strategy']})

| 項目 | 値 |
|---|---|
| train ROI | {best['ext_tr']:.4f} |
| test ROI | {best['ext_te']:.4f} |
| 差 | {best['ext_diff']:+.4f} |
| 95% CI (test) | [{best['ext_ci_lo']:.4f}, {best['ext_ci_hi']:.4f}] |
| 汎化 (|差|<0.05) | {'✅' if best['汎化'] else '❌'} |
| 黒字 (ROI≥1.0) | {'✅' if best['黒字'] else '❌'} |
| CI (下限≥0.95) | {'✅' if best['CI'] else '❌'} |

## 最終判定: パターン {pattern}

**{desc}**

### 次アクション
"""
if pattern == "α":
    md += f"""
1. 運用候補戦略 {best['strategy']} で仕様書更新
2. 配分最適化 (ハーフケリー等) 検討
3. ペーパートレード開始
"""
elif pattern == "β":
    md += """
1. 3連単での黒字化不可能を確定
2. 他券種 (2連単・2連複・3連複) バックテストへ
3. または構想メモ段階の戦略に戻る
"""
elif pattern == "γ":
    md += """
1. 戦略選択手法の見直し (閾値を train のみで決定する設計)
2. Nested cross-validation 検討
3. または v5 (新特徴量) 設計
"""
else:
    md += """
1. 期間拡大で改善の兆し — 追加期間や特徴量追加で更に改善狙う
2. Phase A-ext-6 の詳細分析
"""

md += """

## 出力ファイル
- v4_ext_fixed_generalization_matrix.csv
- v4_ext_fixed_vs_phase_c_comparison.csv
- calibration_isotonic_ext_fixed.pkl
"""
(OUT/"v4_ext_final_verdict.md").write_text(md, encoding="utf-8")
print(f"\nsaved: v4_ext_final_verdict.md")
print(f"\n=== パターン {pattern}: {desc} ===")
