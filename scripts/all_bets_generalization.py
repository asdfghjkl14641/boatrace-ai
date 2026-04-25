# -*- coding: utf-8 -*-
"""全券種 (3連単・2連単・2連複・3連複) 汎化検証 — 6 戦略 × 4 券種 × train/test."""
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

TRAIN_FROM = "2020-02-01"; TRAIN_TO = "2025-06-30"
TEST_FROM  = "2025-07-01"; TEST_TO  = "2026-04-18"

TAU = 0.8
EV_THR = 1.20; EDGE_THR = 0.02
ALPHA_FALLBACK = 1.333
MIN_COMBOS_FOR_Z = {"trifecta": 100, "exacta": 25, "quinella": 12, "trio": 16}
BUDGET = 3000; MIN_UNIT = 100
TYPE_RELIABILITY = {"型1": 1.0, "型2": 0.9, "型3": 0.8}

# 組み合わせリスト (各券種)
PERMS_TRI = [(a,b,c) for a,b,c in permutations([1,2,3,4,5,6], 3)]  # 120
PERMS_EX  = [(a,b) for a,b in permutations([1,2,3,4,5,6], 2)]       # 30
COMBS_QU  = sorted([tuple(sorted([a,b])) for a,b in permutations([1,2,3,4,5,6],2) if a<b])  # 15
COMBS_TR  = sorted([tuple(sorted([a,b,c])) for a,b,c in permutations([1,2,3,4,5,6],3) if a<b<c])  # 20

T1_orig=(1.0,0.3); T1_Q=(1.0,0.2); T1_S=(1.5,0.2)
T2_orig=(0.6,0.2); T2_delta=(0.9,0.4)
T3_orig=(0.3,0.4); T3_none=None
STRATEGIES = [
    ("1_full",         T1_orig, T2_orig,  T3_orig),
    ("2_noT3",         T1_orig, T2_orig,  T3_none),
    ("3_T2d",          T1_orig, T2_delta, T3_orig),
    ("4_T2d_noT3",     T1_orig, T2_delta, T3_none),
    ("5_T1Q_T2d_noT3", T1_Q,    T2_delta, T3_none),
    ("6_T1S_T2d_noT3", T1_S,    T2_delta, T3_none),
]

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

def compute_all_probs(p_lane):
    """PL から 4 券種の確率を計算. 返り値: dict {bet_type: ndarray}"""
    out = {}
    p = np.asarray(p_lane)
    # trifecta (120)
    arr = np.zeros(120)
    for i, (a,b,c) in enumerate(PERMS_TRI):
        pa = p[a-1]; pb = p[b-1]/max(1-pa, 1e-9); pc = p[c-1]/max(1-pa-p[b-1], 1e-9)
        arr[i] = pa*pb*pc
    out["trifecta"] = arr
    # exacta (30): P(1=a, 2=b)
    arr = np.zeros(30)
    for i, (a,b) in enumerate(PERMS_EX):
        pa = p[a-1]; pb = p[b-1]/max(1-pa, 1e-9)
        arr[i] = pa*pb
    out["exacta"] = arr
    # quinella (15): P(1=a,2=b) + P(1=b,2=a)
    arr = np.zeros(15)
    for i, (a,b) in enumerate(COMBS_QU):
        pa = p[a-1]; pb = p[b-1]
        arr[i] = pa*(p[b-1]/max(1-pa,1e-9)) + pb*(p[a-1]/max(1-pb,1e-9))
    out["quinella"] = arr
    # trio (20): 6 permutations sum
    arr = np.zeros(20)
    for i, (a,b,c) in enumerate(COMBS_TR):
        s = 0.0
        for (x,y,z) in permutations([a,b,c], 3):
            px = p[x-1]; py = p[y-1]/max(1-px,1e-9); pz = p[z-1]/max(1-px-p[y-1],1e-9)
            s += px*py*pz
        arr[i] = s
    out["trio"] = arr
    return out

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
print("全券種汎化検証 — 4 券種 × 6 戦略 × train/test")
print("=" * 80)

# ========== [0] v4_ext_fixed runpy ==========
print("\n[0] v4_ext_fixed runpy ...")
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
print(f"  β={beta.shape}, X_tr={X_tr.shape}, X_te={X_te.shape}")

# ========== [1] odds + results (4 券種) ==========
print("\n[1] オッズ + 結果 ロード ...")
from scripts.db import get_connection
conn = get_connection()

def load_results(date_from, date_to):
    df = pd.read_sql_query(f"""
        SELECT date, stadium, race_number, rank, boat FROM race_results
        WHERE date BETWEEN '{date_from}' AND '{date_to}' AND rank BETWEEN 1 AND 3
    """, conn.native)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    res_map = {}
    for (d,s,r),g in df.groupby(["date","stadium","race_number"]):
        g2 = g.sort_values("rank")
        if len(g2) < 3: continue
        res_map[(d,s,r)] = (int(g2.iloc[0]["boat"]), int(g2.iloc[1]["boat"]), int(g2.iloc[2]["boat"]))
    return res_map

def load_odds(tbl, date_from, date_to):
    df = pd.read_sql_query(f"""
        SELECT date, stadium, race_number, combination, odds_1min
        FROM {tbl}
        WHERE date BETWEEN '{date_from}' AND '{date_to}' AND odds_1min IS NOT NULL AND odds_1min > 0
    """, conn.native)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    m = {}
    for (d,s,r),g in df.groupby(["date","stadium","race_number"]):
        book = {}
        for _, row in g.iterrows():
            combo_str = row["combination"]
            try:
                if "-" in combo_str:  # trifecta/exacta (ordered)
                    parts = tuple(int(x) for x in combo_str.split("-"))
                elif "=" in combo_str:  # quinella/trio (unordered)
                    parts = tuple(sorted(int(x) for x in combo_str.split("=")))
                else: continue
                book[parts] = float(row["odds_1min"])
            except Exception: continue
        if book: m[(d,s,r)] = book
    return m

# Load all odds (heavy but one-time)
print("  train...")
t0 = time.time()
tr_res = load_results(TRAIN_FROM, TRAIN_TO)
print(f"    results: {len(tr_res):,}")
tr_odds = {}
for bt, tbl in [("trifecta","trifecta_odds"),("exacta","odds_exacta"),("quinella","odds_quinella"),("trio","odds_trio")]:
    tr_odds[bt] = load_odds(tbl, TRAIN_FROM, TRAIN_TO)
    print(f"    {bt}: {len(tr_odds[bt]):,}  ({time.time()-t0:.1f}s)")

print("  test...")
te_res = load_results(TEST_FROM, TEST_TO)
print(f"    results: {len(te_res):,}")
te_odds = {}
for bt, tbl in [("trifecta","trifecta_odds"),("exacta","odds_exacta"),("quinella","odds_quinella"),("trio","odds_trio")]:
    te_odds[bt] = load_odds(tbl, TEST_FROM, TEST_TO)
    print(f"    {bt}: {len(te_odds[bt]):,}  ({time.time()-t0:.1f}s)")
conn.close()

# ========== [2] Isotonic fit per bet type (train) ==========
print("\n[2] Isotonic fit per bet type (train)...")
# 事前準備: train の probs と hits をベクトル化
N_tr = len(keys_tr)

# 各券種の actual combination を決める関数
def actual_trifecta(t): return t  # (a, b, c)
def actual_exacta(t):   return (t[0], t[1])
def actual_quinella(t): return tuple(sorted([t[0], t[1]]))
def actual_trio(t):     return tuple(sorted([t[0], t[1], t[2]]))

ACT_FN = {"trifecta": actual_trifecta, "exacta": actual_exacta,
          "quinella": actual_quinella, "trio": actual_trio}
COMBS  = {"trifecta": PERMS_TRI, "exacta": PERMS_EX,
          "quinella": COMBS_QU, "trio": COMBS_TR}

isotonics = {}
for bt in ["trifecta","exacta","quinella","trio"]:
    print(f"  [{bt}] collecting pairs ...")
    t0 = time.time()
    n_combs = len(COMBS[bt])
    # allocate
    pairs_p = np.empty(N_tr * n_combs, dtype=np.float32)
    pairs_y = np.zeros(N_tr * n_combs, dtype=np.int8)
    pairs_valid = np.zeros(N_tr * n_combs, dtype=bool)
    for i in range(N_tr):
        k = keys_tr.iloc[i]
        d = pd.Timestamp(k["date"]).date(); s=int(k["stadium"]); r=int(k["race_number"])
        if (d,s,r) not in tr_res: continue
        actual = tr_res[(d,s,r)]
        S = X_tr[i] @ beta
        s_t = S/TAU; s_t = s_t - s_t.max()
        p_lane = np.exp(s_t) / np.exp(s_t).sum()
        probs = compute_all_probs(p_lane)[bt]
        base = i * n_combs
        pairs_p[base:base+n_combs] = probs.astype(np.float32)
        act_key = ACT_FN[bt](actual)
        try:
            pos = COMBS[bt].index(act_key)
            pairs_y[base+pos] = 1
        except ValueError: pass
        pairs_valid[base:base+n_combs] = True
    p_v = pairs_p[pairs_valid]; y_v = pairs_y[pairs_valid]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_v.astype(np.float64), y_v.astype(np.float64))
    isotonics[bt] = iso
    with open(OUT/f"calibration_isotonic_{bt}.pkl", "wb") as f:
        pickle.dump(iso, f)
    print(f"    {bt}: pairs={len(p_v):,}, hits={int(y_v.sum()):,}, elapsed {time.time()-t0:.1f}s")

# ========== [3] バックテスト (4 券種 × 6 戦略 × train/test) ==========
print("\n[3] バックテスト実行 ...")
ALL_BET_TYPES = ["trifecta","exacta","quinella","trio"]

def run_all(X, keys, res_map, odds_maps, label):
    """全 (券種 × 戦略) を 1 pass で評価."""
    N = len(keys)
    # stat[bet_type][strategy] = {"stakes":[],"pays":[],"types":{型1:{races,hits},...}, "month":{ym:{stakes,pays}}}
    stat = {}
    for bt in ALL_BET_TYPES:
        stat[bt] = {}
        for s in STRATEGIES:
            stat[bt][s[0]] = {"stakes":[],"pays":[],
                               "types":{"型1":[0,0],"型2":[0,0],"型3":[0,0]},
                               "month":{}}
    t0 = time.time()
    for i in range(N):
        k = keys.iloc[i]
        d = pd.Timestamp(k["date"]).date(); s=int(k["stadium"]); r=int(k["race_number"])
        if (d,s,r) not in res_map: continue
        actual = res_map[(d,s,r)]
        month = f"{d.year}-{d.month:02d}"
        S = X[i] @ beta
        idx = compute_indices(S)
        s_t = S/TAU; s_t = s_t - s_t.max()
        p_lane = np.exp(s_t) / np.exp(s_t).sum()
        all_probs = compute_all_probs(p_lane)

        for bt in ALL_BET_TYPES:
            if (d,s,r) not in odds_maps[bt]: continue
            book = odds_maps[bt][(d,s,r)]
            n_min = MIN_COMBOS_FOR_Z[bt]
            Z = sum(1.0/o for o in book.values()) if len(book) >= n_min else ALPHA_FALLBACK
            probs_cal = isotonics[bt].transform(all_probs[bt].astype(np.float64))
            act_key = ACT_FN[bt](actual)

            for strat_name, t1, t2, t3 in STRATEGIES:
                t = classify(idx, t1, t2, t3)
                if t is None: continue
                rel = TYPE_RELIABILITY[t]
                cands = []
                for j, combo in enumerate(COMBS[bt]):
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
                    if combo == act_key:
                        race_pay += int(st*odds); hit_flag = 1
                S_obj = stat[bt][strat_name]
                S_obj["stakes"].append(race_stake); S_obj["pays"].append(race_pay)
                S_obj["types"][t][0] += race_stake; S_obj["types"][t][1] += race_pay
                S_obj["month"].setdefault(month, [0, 0])
                S_obj["month"][month][0] += race_stake; S_obj["month"][month][1] += race_pay
        if (i+1) % 30000 == 0:
            print(f"    [{label}] {i+1}/{N}  elapsed {time.time()-t0:.1f}s")
    print(f"   [{label}] done {time.time()-t0:.1f}s")
    return stat

print("\n  train ...")
stat_tr = run_all(X_tr, keys_tr, tr_res, tr_odds, "train")
print("\n  test ...")
stat_te = run_all(X_te, keys_te, te_res, te_odds, "test")

# ========== [4] 集計 ==========
def roll_up(stat, period):
    rows = []
    for bt in ALL_BET_TYPES:
        for strat_name, _, _, _ in STRATEGIES:
            o = stat[bt][strat_name]
            stake = sum(o["stakes"]); pay = sum(o["pays"])
            n = len(o["stakes"])
            roi = pay/stake if stake else 0
            ci = bootstrap_roi_ci(o["stakes"], o["pays"], n_boot=1000)
            # 型別
            t1 = o["types"]["型1"][1]/o["types"]["型1"][0] if o["types"]["型1"][0] else 0
            t2 = o["types"]["型2"][1]/o["types"]["型2"][0] if o["types"]["型2"][0] else 0
            t3 = o["types"]["型3"][1]/o["types"]["型3"][0] if o["types"]["型3"][0] else 0
            mon_rois = [v[1]/v[0] for v in o["month"].values() if v[0] > 0]
            mon_std = float(np.std(mon_rois)) if mon_rois else 0
            rows.append({
                "bet_type": bt, "strategy": strat_name, "period": period,
                "n_bet": n, "stake": stake, "payout": pay, "profit": pay-stake,
                "roi": roi, "ci_lo": ci[0], "ci_hi": ci[1],
                "t1_roi": t1, "t2_roi": t2, "t3_roi": t3,
                "month_std": mon_std,
            })
    return pd.DataFrame(rows)

tr_df = roll_up(stat_tr, "train")
te_df = roll_up(stat_te, "test")
all_df = pd.concat([tr_df, te_df], ignore_index=True)
all_df.to_csv(OUT/"all_bets_generalization.csv", index=False, encoding="utf-8-sig")
print(f"\nsaved: all_bets_generalization.csv")

# ========== [5] 差分分析 + ランキング ==========
print("\n[5] 差分 + ランキング ...")
gap = []
for bt in ALL_BET_TYPES:
    for strat_name, _, _, _ in STRATEGIES:
        tr = tr_df[(tr_df["bet_type"]==bt)&(tr_df["strategy"]==strat_name)].iloc[0]
        te = te_df[(te_df["bet_type"]==bt)&(te_df["strategy"]==strat_name)].iloc[0]
        diff = te["roi"] - tr["roi"]
        if abs(diff) < 0.05: verdict = "汎化"
        elif abs(diff) < 0.10: verdict = "軽度過学習"
        else: verdict = "明確過学習"
        gap.append({"bet_type":bt,"strategy":strat_name,
                    "train_roi":tr["roi"],"test_roi":te["roi"],"diff":diff,
                    "test_n":te["n_bet"],"train_n":tr["n_bet"],
                    "test_ci_lo":te["ci_lo"],"test_ci_hi":te["ci_hi"],
                    "verdict":verdict,
                    "t1_te_roi":te["t1_roi"],"t2_te_roi":te["t2_roi"],"t3_te_roi":te["t3_roi"]})
gap_df = pd.DataFrame(gap)
gap_df.to_csv(OUT/"all_bets_gap_analysis.csv", index=False, encoding="utf-8-sig")

# 券種別最良戦略 (test ROI max, bet_n >= 500 優先)
print(f"\n  {'bet_type':<10} {'best_strat':<20} {'train':>7} {'test':>7} {'diff':>7} {'CI 下':>7} {'CI 上':>7} {'n_bet':>7}")
best_per_bt = []
for bt in ALL_BET_TYPES:
    sub = gap_df[(gap_df["bet_type"]==bt) & (gap_df["test_n"]>=500)].copy()
    if len(sub) == 0:
        sub = gap_df[gap_df["bet_type"]==bt]
    best = sub.sort_values("test_roi", ascending=False).iloc[0]
    print(f"  {bt:<10} {best['strategy']:<20} {best['train_roi']:>7.4f} "
          f"{best['test_roi']:>7.4f} {best['diff']:>+7.4f} "
          f"{best['test_ci_lo']:>7.4f} {best['test_ci_hi']:>7.4f} {best['test_n']:>7,}")
    best_per_bt.append(best.to_dict())
best_df = pd.DataFrame(best_per_bt)
best_df.to_csv(OUT/"bet_type_ranking.csv", index=False, encoding="utf-8-sig")

# 全体 top 10
print(f"\n  Top 10 (全券種 × 戦略) by test ROI:")
top10 = gap_df[gap_df["test_n"]>=500].sort_values("test_roi", ascending=False).head(10)
for _, rw in top10.iterrows():
    print(f"    {rw['bet_type']:<9} {rw['strategy']:<18} "
          f"train={rw['train_roi']:.4f} test={rw['test_roi']:.4f} "
          f"diff={rw['diff']:+.4f} CI[{rw['test_ci_lo']:.3f},{rw['test_ci_hi']:.3f}] n={rw['test_n']:,}")

# ========== [6] 判定 ==========
# パターン判定
runnable = gap_df[(gap_df["diff"].abs() < 0.05) &
                   (gap_df["test_roi"] >= 1.0) &
                   (gap_df["test_ci_lo"] >= 0.95) &
                   (gap_df["test_n"] >= 500)]
nearly_ok = gap_df[(gap_df["diff"].abs() < 0.05) &
                    (gap_df["test_roi"] >= 1.0) &
                    (gap_df["test_n"] >= 500)]

if len(runnable) > 0:
    pattern = "α"; desc = f"運用候補 {len(runnable)} 戦略 (ROI≥1.0 AND CI下≥0.95 AND 汎化)"
elif len(nearly_ok) > 0:
    pattern = "α'"; desc = f"ROI≥1.0 AND 汎化の戦略 {len(nearly_ok)} 件 (CI 下は 0.95 未達)"
else:
    # 3連単以外で黒字があるか
    non_tri_max = gap_df[gap_df["bet_type"]!="trifecta"]["test_roi"].max()
    tri_max = gap_df[gap_df["bet_type"]=="trifecta"]["test_roi"].max()
    if non_tri_max > tri_max + 0.03:
        pattern = "δ"; desc = f"3連単以外で改善 ({non_tri_max:.4f} vs trifecta {tri_max:.4f})"
    else:
        pattern = "γ"; desc = "どの券種でも運用不可"

print(f"\n  判定: パターン {pattern}")
print(f"  {desc}")

# レポート
md = f"""# 全券種バックテスト最終判定

## 核心メッセージ
- **パターン {pattern}**: {desc}
- 検証: 4 券種 × 6 戦略 × train/test = 48 バックテスト

## 券種別 最良戦略 (test ROI max, bet_n>=500)

| 券種 | 最良戦略 | train ROI | test ROI | 差 | CI下 | CI上 | n_bet |
|---|---|---|---|---|---|---|---|
"""
for _, rw in best_df.iterrows():
    md += (f"| {rw['bet_type']} | {rw['strategy']} | "
           f"{rw['train_roi']:.4f} | {rw['test_roi']:.4f} | "
           f"{rw['diff']:+.4f} | {rw['test_ci_lo']:.4f} | "
           f"{rw['test_ci_hi']:.4f} | {int(rw['test_n']):,} |\n")

md += "\n## Top 10 (全組合せ) by test ROI\n\n"
md += "| 券種 | 戦略 | train | test | diff | CI下 | CI上 | n_bet |\n"
md += "|---|---|---|---|---|---|---|---|\n"
for _, rw in top10.iterrows():
    md += (f"| {rw['bet_type']} | {rw['strategy']} | {rw['train_roi']:.4f} | "
           f"**{rw['test_roi']:.4f}** | {rw['diff']:+.4f} | {rw['test_ci_lo']:.4f} | "
           f"{rw['test_ci_hi']:.4f} | {int(rw['test_n']):,} |\n")

md += f"""

## 判定: パターン {pattern}

**{desc}**

### 運用判定基準
- 汎化 (|diff|<0.05): {'✅' if pattern in ('α',"α'") else '△'}
- 黒字 (ROI≥1.0): {'✅' if pattern in ('α',"α'") else '❌'}
- CI 下限 ≥ 0.95: {'✅' if pattern == 'α' else '❌'}

## 出力ファイル
- all_bets_generalization.csv
- all_bets_gap_analysis.csv
- bet_type_ranking.csv
- calibration_isotonic_trifecta/exacta/quinella/trio.pkl
"""
(OUT/"all_bets_final_verdict.md").write_text(md, encoding="utf-8")
print(f"\nsaved: all_bets_final_verdict.md")
print(f"\n=== パターン {pattern}: {desc} ===")
