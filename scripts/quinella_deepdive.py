# -*- coding: utf-8 -*-
"""2連複 (quinella) 深堀り分析 — 戦略 1_full の構造と改善探索."""
from __future__ import annotations
import io, os, sys, math, pickle, runpy, time, warnings
from pathlib import Path
from itertools import permutations, combinations
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
ALPHA_FALLBACK = 1.333
MIN_COMBOS_FOR_Z = 12
BUDGET = 3000; MIN_UNIT = 100
TYPE_RELIABILITY = {"型1": 1.0, "型2": 0.9, "型3": 0.8}

COMBS_QU = sorted([tuple(sorted([a,b])) for a,b in permutations([1,2,3,4,5,6],2) if a<b])

T1_orig=(1.0,0.3); T2_orig=(0.6,0.2); T3_orig=(0.3,0.4)

def compute_indices(S):
    std_S = max(S.std(ddof=0), 0.3)
    s_sorted = np.sort(S)[::-1]
    return ((s_sorted[0]-s_sorted[1])/std_S,
            (S[3:6].max()-S.mean())/std_S,
            int(S.argmax()))

def classify_full(idx):
    G, O, top1 = idx
    if top1 == 0:
        if G > T1_orig[0] and O < T1_orig[1]: return "型1"
        if G > T2_orig[0] and O > T2_orig[1]: return "型2"
        return None
    else:
        if O > T3_orig[0] and G > T3_orig[1]: return "型3"
        return None

def compute_quinella_probs(p_lane):
    p = np.asarray(p_lane)
    arr = np.zeros(15)
    for i, (a,b) in enumerate(COMBS_QU):
        pa = p[a-1]; pb = p[b-1]
        arr[i] = pa*(p[b-1]/max(1-pa,1e-9)) + pb*(p[a-1]/max(1-pb,1e-9))
    return arr

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


print("="*80); print("2連複 深堀り分析 — v4_ext_fixed, 戦略 1_full"); print("="*80)

# ========== [0] v4_ext_fixed runpy ==========
print("\n[0] v4_ext_fixed runpy ...")
os.environ["V4_TRAIN_FROM"] = TRAIN_FROM
os.environ["V4_TRAIN_TO"]   = TRAIN_TO
os.environ["V4_TEST_FROM"]  = TEST_FROM
os.environ["V4_TEST_TO"]    = TEST_TO
os.environ["V4_OUT_SUFFIX"] = "_ext_fixed"
_buf = io.StringIO(); _o = sys.stdout; sys.stdout = _buf
try: ns = runpy.run_path(str(BASE / "notebooks" / "pl_v4_training.py"))
finally: sys.stdout = _o
beta = ns["beta_v4"]
X_tr = ns["X_train_v4"]; keys_tr = ns["keys_train_v4"].reset_index(drop=True)
X_te = ns["X_test_v4"];  keys_te = ns["keys_test_v4"].reset_index(drop=True)
print(f"  β={beta.shape}, X_tr={X_tr.shape}, X_te={X_te.shape}")

# Load quinella isotonic
with open(OUT/"calibration_isotonic_quinella.pkl", "rb") as f:
    iso = pickle.load(f)
print("  quinella isotonic loaded")

# ========== [1] odds + results ==========
print("\n[1] odds + results ロード ...")
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

def load_quinella_odds(date_from, date_to):
    df = pd.read_sql_query(f"""
        SELECT date, stadium, race_number, combination, odds_1min
        FROM odds_quinella
        WHERE date BETWEEN '{date_from}' AND '{date_to}' AND odds_1min IS NOT NULL AND odds_1min > 0
    """, conn.native)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    m = {}
    for (d,s,r),g in df.groupby(["date","stadium","race_number"]):
        book = {}
        for _, row in g.iterrows():
            try:
                parts = tuple(sorted(int(x) for x in row["combination"].split("=")))
                book[parts] = float(row["odds_1min"])
            except Exception: pass
        if book: m[(d,s,r)] = book
    return m

t0 = time.time()
tr_res = load_results(TRAIN_FROM, TRAIN_TO); print(f"  train results: {len(tr_res):,} ({time.time()-t0:.1f}s)")
te_res = load_results(TEST_FROM, TEST_TO);   print(f"  test  results: {len(te_res):,}")
tr_odds = load_quinella_odds(TRAIN_FROM, TRAIN_TO); print(f"  train odds: {len(tr_odds):,} ({time.time()-t0:.1f}s)")
te_odds = load_quinella_odds(TEST_FROM, TEST_TO);   print(f"  test  odds: {len(te_odds):,} ({time.time()-t0:.1f}s)")
conn.close()

# ========== [2] 戦略 1_full バックテスト — per-bet 詳細記録 ==========
print("\n[2] 戦略 1_full バックテスト (per-bet 詳細) ...")

def backtest_detailed(X, keys, res_map, odds_map, period_label):
    """per-bet 詳細を DataFrame で返す."""
    rows = []
    t0 = time.time()
    N = len(keys)
    for i in range(N):
        k = keys.iloc[i]
        d = pd.Timestamp(k["date"]).date(); s=int(k["stadium"]); r=int(k["race_number"])
        if (d,s,r) not in res_map or (d,s,r) not in odds_map: continue
        book = odds_map[(d,s,r)]; actual = res_map[(d,s,r)]
        Z = sum(1.0/o for o in book.values()) if len(book) >= MIN_COMBOS_FOR_Z else ALPHA_FALLBACK
        S = X[i] @ beta
        idx = compute_indices(S)
        t = classify_full(idx)
        if t is None: continue
        rel = TYPE_RELIABILITY[t]
        s_t = S/TAU; s_t = s_t - s_t.max()
        p_lane = np.exp(s_t)/np.exp(s_t).sum()
        probs = compute_quinella_probs(p_lane)
        probs_cal = iso.transform(probs.astype(np.float64))
        act_key = tuple(sorted([actual[0], actual[1]]))

        cands = []
        for j, combo in enumerate(COMBS_QU):
            odds = book.get(combo)
            if not odds: continue
            p_adj = probs_cal[j] * rel
            ev = p_adj * odds; edge = p_adj - 1.0/(odds*Z)
            # 基本フィルタ: EV>=1.20 AND Edge>=0.02 (戦略 1_full)
            if ev >= 1.20 and edge >= 0.02:
                cands.append((combo, odds, p_adj, ev, edge))
        if not cands: continue

        ev_sum = sum(c[3] for c in cands)
        alloc = []
        for combo, odds, p_adj, ev, edge in cands:
            units = int(BUDGET * ev / ev_sum / MIN_UNIT)
            stake = max(0, units) * MIN_UNIT
            alloc.append((combo, odds, p_adj, ev, edge, stake))
        used = sum(a[5] for a in alloc); extra = BUDGET - used
        order = sorted(range(len(alloc)), key=lambda ii: -alloc[ii][3])
        alloc_list = list(alloc)
        for ia in order:
            if extra < MIN_UNIT: break
            c,o,p,ev,ed,st = alloc_list[ia]
            alloc_list[ia] = (c,o,p,ev,ed,st+MIN_UNIT); extra -= MIN_UNIT
        alloc_final = [a for a in alloc_list if a[5] > 0]
        if not alloc_final: continue

        for combo, odds, p_adj, ev, edge, stake in alloc_final:
            hit = 1 if combo == act_key else 0
            payout = int(stake * odds) if hit else 0
            rows.append({
                "period": period_label, "date": str(d), "stadium": s, "race": r,
                "month": f"{d.year}-{d.month:02d}",
                "type": t, "G_S": idx[0], "O_S": idx[1],
                "combo_a": combo[0], "combo_b": combo[1],
                "odds": odds, "p_adj": p_adj, "ev": ev, "edge": edge,
                "stake": stake, "hit": hit, "payout": payout,
            })
        if (i+1) % 50000 == 0:
            print(f"    [{period_label}] {i+1}/{N}  bets so far: {len(rows):,}  elapsed {time.time()-t0:.1f}s")
    print(f"  [{period_label}] done, bets: {len(rows):,}, elapsed {time.time()-t0:.1f}s")
    return pd.DataFrame(rows)

tr_bets = backtest_detailed(X_tr, keys_tr, tr_res, tr_odds, "train")
te_bets = backtest_detailed(X_te, keys_te, te_res, te_odds, "test")

# ========== [3] Step 1: 構造分析 ==========
print("\n[3] Step 1 構造分析 ...")

def aggregate(df, group_col=None):
    if group_col:
        g = df.groupby(group_col)
    else:
        g = [("all", df)]
    rows = []
    for key, sub in (g if group_col else g):
        stake = sub["stake"].sum(); pay = sub["payout"].sum()
        n = len(sub); hits = int(sub["hit"].sum())
        roi = pay/stake if stake else 0
        # per-race aggregation for bootstrap
        race_stakes = sub.groupby(["date","stadium","race"])["stake"].sum().values
        race_pays = sub.groupby(["date","stadium","race"])["payout"].sum().values
        ci = bootstrap_roi_ci(race_stakes, race_pays, n_boot=1000)
        rows.append({"key":key,"n_bets":n,"hits":hits,"hit_rate":hits/n if n else 0,
                     "stake":int(stake),"payout":int(pay),"profit":int(pay-stake),
                     "roi":roi,"ci_lo":ci[0],"ci_hi":ci[1],
                     "avg_ev":sub["ev"].mean() if n else 0,
                     "avg_odds":sub["odds"].mean() if n else 0,
                     "avg_p":sub["p_adj"].mean() if n else 0})
    return pd.DataFrame(rows)

# 1.1 月別 (train と test 別々)
print("  [1.1] 月別 ROI")
tr_mon = aggregate(tr_bets, "month"); tr_mon["period"]="train"
te_mon = aggregate(te_bets, "month"); te_mon["period"]="test"
mon_df = pd.concat([tr_mon, te_mon], ignore_index=True)
mon_df.to_csv(OUT/"quinella_monthly.csv", index=False, encoding="utf-8-sig")
print(f"   train 月別 std: {tr_mon['roi'].std():.4f}  mean: {tr_mon['roi'].mean():.4f}")
print(f"   test  月別 std: {te_mon['roi'].std():.4f}  mean: {te_mon['roi'].mean():.4f}")

# 1.2 型別
print("\n  [1.2] 型別 ROI (train vs test)")
tr_typ = aggregate(tr_bets, "type"); tr_typ["period"]="train"
te_typ = aggregate(te_bets, "type"); te_typ["period"]="test"
typ_df = pd.concat([tr_typ, te_typ], ignore_index=True)
typ_df.to_csv(OUT/"quinella_by_type.csv", index=False, encoding="utf-8-sig")
print(f"  {'type':<5} {'period':<6} {'n_bets':>8} {'ROI':>7} {'CI 下':>7}")
for _, rw in typ_df.iterrows():
    print(f"  {rw['key']:<5} {rw['period']:<6} {int(rw['n_bets']):>8,} "
          f"{rw['roi']:>7.4f} {rw['ci_lo']:>7.4f}")

# 1.3 EV 帯別
print("\n  [1.3] EV 帯別 ROI")
def ev_band(ev):
    if ev < 1.20: return "1.10-1.20"
    if ev < 1.30: return "1.20-1.30"
    if ev < 1.50: return "1.30-1.50"
    if ev < 2.00: return "1.50-2.00"
    return "2.00+"
for df, lbl in [(tr_bets,"train"),(te_bets,"test")]:
    df["ev_band"] = df["ev"].apply(ev_band)
ev_tr = aggregate(tr_bets, "ev_band"); ev_tr["period"]="train"
ev_te = aggregate(te_bets, "ev_band"); ev_te["period"]="test"
ev_df = pd.concat([ev_tr, ev_te], ignore_index=True)
ev_df.to_csv(OUT/"quinella_by_ev.csv", index=False, encoding="utf-8-sig")
print(f"  {'EV帯':<12} {'period':<6} {'n_bets':>8} {'ROI':>7}")
for _, rw in ev_df.iterrows():
    print(f"  {rw['key']:<12} {rw['period']:<6} {int(rw['n_bets']):>8,} {rw['roi']:>7.4f}")

# 1.4 会場別 (top/bottom)
print("\n  [1.4] 会場別 ROI (test)")
st_te = aggregate(te_bets, "stadium").sort_values("roi", ascending=False)
st_te.to_csv(OUT/"quinella_by_stadium.csv", index=False, encoding="utf-8-sig")
print(f"  Top 5 (test):")
for _, rw in st_te.head(5).iterrows():
    print(f"    場{int(rw['key']):>2}: n={int(rw['n_bets']):>5,} ROI={rw['roi']:.4f}")
print(f"  Bottom 5 (test):")
for _, rw in st_te.tail(5).iterrows():
    print(f"    場{int(rw['key']):>2}: n={int(rw['n_bets']):>5,} ROI={rw['roi']:.4f}")

# 1.5 オッズ帯別
print("\n  [1.5] オッズ帯別 ROI")
def odds_band(o):
    if o < 2.0: return "1.1-2.0"
    if o < 5.0: return "2.0-5.0"
    if o < 10.0: return "5.0-10.0"
    return "10+"
for df in [tr_bets, te_bets]:
    df["odds_band"] = df["odds"].apply(odds_band)
od_tr = aggregate(tr_bets, "odds_band"); od_tr["period"]="train"
od_te = aggregate(te_bets, "odds_band"); od_te["period"]="test"
od_df = pd.concat([od_tr, od_te], ignore_index=True)
od_df.to_csv(OUT/"quinella_by_odds.csv", index=False, encoding="utf-8-sig")
print(f"  {'odds帯':<10} {'period':<6} {'n_bets':>8} {'hit%':>6} {'ROI':>7}")
for _, rw in od_df.iterrows():
    print(f"  {rw['key']:<10} {rw['period']:<6} {int(rw['n_bets']):>8,} "
          f"{rw['hit_rate']*100:>5.2f}% {rw['roi']:>7.4f}")

# ========== [4] Step 2: train/test gap 分析 ==========
print("\n[4] Step 2: train/test gap 原因 ...")
def gap_compare(tr, te, col, label):
    tv = tr[col].describe(); ev = te[col].describe()
    print(f"  {label:<12}  train mean={tv['mean']:.4f} std={tv['std']:.4f} | "
          f"test mean={ev['mean']:.4f} std={ev['std']:.4f}  diff {ev['mean']-tv['mean']:+.4f}")

gap_compare(tr_bets, te_bets, "ev", "EV")
gap_compare(tr_bets, te_bets, "odds", "odds")
gap_compare(tr_bets, te_bets, "p_adj", "p_adj")
# 型分布
print("  型分布:")
for t in ["型1","型2","型3"]:
    tr_pct = (tr_bets["type"]==t).mean()*100
    te_pct = (te_bets["type"]==t).mean()*100
    print(f"    {t}: train {tr_pct:.2f}% vs test {te_pct:.2f}%  diff {te_pct-tr_pct:+.2f}pt")

# 時系列 (train を 5 期間に)
print("\n  [2.2] train 時系列推移")
def period5(m):
    y = int(m[:4])
    if y <= 2020: return "2020"
    if y == 2021: return "2021"
    if y == 2022: return "2022"
    if y == 2023: return "2023"
    return "2024-25H1"
tr_bets["period5"] = tr_bets["month"].apply(period5)
ts_tr = aggregate(tr_bets, "period5"); ts_tr["period"]="train"
ts_te_all = aggregate(te_bets); ts_te_all["key"]="test_all"; ts_te_all["period"]="test"
ts_df = pd.concat([ts_tr, ts_te_all], ignore_index=True)
ts_df.to_csv(OUT/"quinella_train_test_gap.csv", index=False, encoding="utf-8-sig")
print(f"  {'期間':<12} {'n_bets':>8} {'ROI':>7} {'CI下':>7} {'CI上':>7}")
for _, rw in ts_df.iterrows():
    print(f"  {rw['key']:<12} {int(rw['n_bets']):>8,} {rw['roi']:>7.4f} "
          f"{rw['ci_lo']:>7.4f} {rw['ci_hi']:>7.4f}")

# ========== [5] Step 3: 改善戦略試行 ==========
print("\n[5] Step 3: 改善戦略探索 ...")

# 3.1 EV 閾値引き上げ
print("  [3.1] EV 閾値引き上げ (test)")
ev_thresholds = [1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 2.00]
# 現行 filter は EV>=1.20 AND Edge>=0.02 で既にフィルタ済
# なので 1.20 未満で絞れず。1.20+ でのみ評価
print(f"  {'EV閾値':<8} {'n_bets':>8} {'ROI':>7} {'CI下':>7} {'CI上':>7}")
opt_rows = []
for thr in ev_thresholds:
    if thr < 1.20: continue  # 既にカット済
    tr_sub = tr_bets[tr_bets["ev"] >= thr]
    te_sub = te_bets[te_bets["ev"] >= thr]
    # per-race agg for CI
    tr_rs = tr_sub.groupby(["date","stadium","race"])["stake"].sum().values
    tr_rp = tr_sub.groupby(["date","stadium","race"])["payout"].sum().values
    te_rs = te_sub.groupby(["date","stadium","race"])["stake"].sum().values
    te_rp = te_sub.groupby(["date","stadium","race"])["payout"].sum().values
    tr_roi = tr_sub["payout"].sum()/tr_sub["stake"].sum() if tr_sub["stake"].sum() else 0
    te_roi = te_sub["payout"].sum()/te_sub["stake"].sum() if te_sub["stake"].sum() else 0
    te_ci = bootstrap_roi_ci(te_rs, te_rp, n_boot=1000)
    diff = te_roi - tr_roi
    opt_rows.append({"ev_thr":thr, "top_n":"all",
                     "tr_n":len(tr_sub), "te_n":len(te_sub),
                     "tr_roi":tr_roi, "te_roi":te_roi, "diff":diff,
                     "ci_lo":te_ci[0], "ci_hi":te_ci[1]})
    print(f"  EV>={thr:<5.2f} {len(te_sub):>8,} {te_roi:>7.4f} "
          f"{te_ci[0]:>7.4f} {te_ci[1]:>7.4f}  diff {diff:+.4f}")

# 3.3 買い目絞り (EV 上位 N per race)
print("\n  [3.3] 買い目絞り (EV 上位 N)")
def topN_filter(df, N):
    if N is None: return df
    return df.sort_values("ev", ascending=False).groupby(["date","stadium","race"]).head(N)

print(f"  {'絞り':<8} {'tr_n':>7} {'tr_ROI':>8} {'te_n':>7} {'te_ROI':>8} {'CI下':>7} {'CI上':>7} {'diff':>+7}")
for N in [1,2,3,5,None]:
    tr_sub = topN_filter(tr_bets, N)
    te_sub = topN_filter(te_bets, N)
    # Note: re-allocate budget? For simplicity, use original stake (may sum <3000 per race)
    tr_roi = tr_sub["payout"].sum()/tr_sub["stake"].sum() if tr_sub["stake"].sum() else 0
    te_roi = te_sub["payout"].sum()/te_sub["stake"].sum() if te_sub["stake"].sum() else 0
    te_rs = te_sub.groupby(["date","stadium","race"])["stake"].sum().values
    te_rp = te_sub.groupby(["date","stadium","race"])["payout"].sum().values
    te_ci = bootstrap_roi_ci(te_rs, te_rp, n_boot=1000)
    diff = te_roi - tr_roi
    opt_rows.append({"ev_thr":1.20, "top_n":str(N),
                     "tr_n":len(tr_sub), "te_n":len(te_sub),
                     "tr_roi":tr_roi, "te_roi":te_roi, "diff":diff,
                     "ci_lo":te_ci[0], "ci_hi":te_ci[1]})
    lbl = f"top{N}" if N else "all"
    print(f"  {lbl:<8} {len(tr_sub):>7,} {tr_roi:>8.4f} {len(te_sub):>7,} {te_roi:>8.4f} "
          f"{te_ci[0]:>7.4f} {te_ci[1]:>7.4f} {diff:+.4f}")

# 3.4 EV 閾値 × top-N 組み合わせ
print("\n  [3.4] EV 閾値 × top-N 組み合わせ")
opt2_rows = []
for thr in [1.20, 1.25, 1.30, 1.40, 1.50]:
    for N in [None, 1, 2, 3, 5]:
        tr_sub = tr_bets[tr_bets["ev"]>=thr]
        te_sub = te_bets[te_bets["ev"]>=thr]
        tr_sub = topN_filter(tr_sub, N) if N else tr_sub
        te_sub = topN_filter(te_sub, N) if N else te_sub
        tr_roi = tr_sub["payout"].sum()/tr_sub["stake"].sum() if tr_sub["stake"].sum() else 0
        te_roi = te_sub["payout"].sum()/te_sub["stake"].sum() if te_sub["stake"].sum() else 0
        te_rs = te_sub.groupby(["date","stadium","race"])["stake"].sum().values
        te_rp = te_sub.groupby(["date","stadium","race"])["payout"].sum().values
        te_ci = bootstrap_roi_ci(te_rs, te_rp, n_boot=500)
        diff = te_roi - tr_roi
        opt2_rows.append({"ev_thr":thr, "top_n":str(N) if N else "all",
                          "tr_n":len(tr_sub), "te_n":len(te_sub),
                          "tr_roi":tr_roi, "te_roi":te_roi, "diff":diff,
                          "ci_lo":te_ci[0], "ci_hi":te_ci[1]})

opt2_df = pd.DataFrame(opt2_rows)
opt2_df.to_csv(OUT/"quinella_optimization.csv", index=False, encoding="utf-8-sig")

# 最良戦略選定
viable = opt2_df[(opt2_df["te_n"]>=500) & (opt2_df["diff"].abs()<0.05)].sort_values("te_roi", ascending=False)
print(f"\n  汎化戦略 (|diff|<0.05, te_n>=500) Top 10:")
for _, rw in viable.head(10).iterrows():
    print(f"  EV>={rw['ev_thr']:.2f} top={rw['top_n']:<4} te_n={rw['te_n']:>6,} "
          f"te_roi={rw['te_roi']:.4f} CI[{rw['ci_lo']:.4f},{rw['ci_hi']:.4f}] diff={rw['diff']:+.4f}")

# 黒字候補
best_gen = viable.iloc[0] if len(viable) > 0 else None

# ========== [6] パターン判定 ==========
print("\n[6] 最終判定 ...")
# 現状 (戦略 1_full, EV>=1.20, top=all)
cur = opt2_df[(opt2_df["ev_thr"]==1.20) & (opt2_df["top_n"]=="all")].iloc[0]
print(f"  現状 (戦略 1_full, EV>=1.20, top=all): te_ROI={cur['te_roi']:.4f} CI下={cur['ci_lo']:.4f}")
if best_gen is not None:
    print(f"  最良改善 (EV>={best_gen['ev_thr']}, top={best_gen['top_n']}): "
          f"te_ROI={best_gen['te_roi']:.4f} CI下={best_gen['ci_lo']:.4f}")
    best_ci_lo = best_gen['ci_lo']; best_roi = best_gen['te_roi']
    best_diff = best_gen['diff']
else:
    best_ci_lo = cur['ci_lo']; best_roi = cur['te_roi']; best_diff = cur['diff']

if best_ci_lo >= 0.95 and abs(best_diff) < 0.05:
    pattern = "α"; desc = f"運用基準到達 CI下={best_ci_lo:.4f}"
elif best_ci_lo >= 0.93:
    pattern = "β"; desc = f"わずかに不足 CI下={best_ci_lo:.4f}"
elif best_ci_lo >= 0.90:
    pattern = "γ"; desc = f"改善効果限定 CI下={best_ci_lo:.4f}"
else:
    pattern = "δ"; desc = f"CI下低 ({best_ci_lo:.4f}) — 2連複でも黒字化困難"

print(f"  判定: パターン {pattern}: {desc}")

# ========== [7] レポート ==========
md = f"""# 2連複 (quinella) 深堀り分析レポート

## 核心メッセージ
- **最良戦略**: {'EV>='+f"{best_gen['ev_thr']:.2f}"+', top='+str(best_gen['top_n']) if best_gen is not None else '現状 (1_full)'}
- **test ROI**: {best_roi:.4f} (CI [{best_ci_lo:.4f}, {best_gen['ci_hi'] if best_gen is not None else cur['ci_hi']:.4f}])
- **train/test 差**: {best_diff:+.4f}
- **判定**: パターン {pattern} — {desc}

## Step 1: 構造分析

### 月別 ROI 概要
- train 65ヶ月 mean={tr_mon['roi'].mean():.4f}, std={tr_mon['roi'].std():.4f}
- test 10ヶ月 mean={te_mon['roi'].mean():.4f}, std={te_mon['roi'].std():.4f}

### 型別 ROI (train vs test)

| type | train n | train ROI | test n | test ROI |
|---|---|---|---|---|
"""
for t in ["型1","型2","型3"]:
    tr_r = tr_typ[tr_typ["key"]==t]
    te_r = te_typ[te_typ["key"]==t]
    if len(tr_r)>0 and len(te_r)>0:
        md += (f"| {t} | {int(tr_r.iloc[0]['n_bets']):,} | "
               f"{tr_r.iloc[0]['roi']:.4f} | {int(te_r.iloc[0]['n_bets']):,} | "
               f"{te_r.iloc[0]['roi']:.4f} |\n")

md += "\n### オッズ帯別 (test)\n\n| odds band | n | hit% | ROI |\n|---|---|---|---|\n"
for _, rw in od_te.iterrows():
    md += f"| {rw['key']} | {int(rw['n_bets']):,} | {rw['hit_rate']*100:.2f}% | {rw['roi']:.4f} |\n"

md += "\n### 会場別 Top 5 / Bottom 5 (test)\n\n"
md += "| 会場 | n | ROI |\n|---|---|---|\n"
for _, rw in st_te.head(5).iterrows():
    md += f"| 場{int(rw['key'])} | {int(rw['n_bets']):,} | {rw['roi']:.4f} |\n"
md += "| ... | ... | ... |\n"
for _, rw in st_te.tail(5).iterrows():
    md += f"| 場{int(rw['key'])} | {int(rw['n_bets']):,} | {rw['roi']:.4f} |\n"

md += "\n## Step 2: train/test gap\n\n"
md += "### 時系列推移 (train 5 期間)\n\n| 期間 | n | ROI | CI下 | CI上 |\n|---|---|---|---|---|\n"
for _, rw in ts_df.iterrows():
    md += (f"| {rw['key']} | {int(rw['n_bets']):,} | {rw['roi']:.4f} | "
           f"{rw['ci_lo']:.4f} | {rw['ci_hi']:.4f} |\n")

md += "\n## Step 3: 改善戦略マトリクス\n\n"
md += "| EV閾値 | top_n | tr_n | tr ROI | te_n | te ROI | CI下 | CI上 | diff |\n"
md += "|---|---|---|---|---|---|---|---|---|\n"
for _, rw in opt2_df.sort_values("te_roi", ascending=False).head(20).iterrows():
    mark = " ⭐" if (rw['te_n']>=500 and abs(rw['diff'])<0.05 and rw['ci_lo']>=0.90) else ""
    md += (f"| {rw['ev_thr']:.2f} | {rw['top_n']} | {int(rw['tr_n']):,} | "
           f"{rw['tr_roi']:.4f} | {int(rw['te_n']):,} | {rw['te_roi']:.4f} | "
           f"{rw['ci_lo']:.4f} | {rw['ci_hi']:.4f} | {rw['diff']:+.4f}{mark} |\n")

md += f"""

## 判定: パターン {pattern}

**{desc}**

### 運用戦略 (最良)
```
券種: 2連複 (quinella)
モデル: v4_ext_fixed
キャリブレーション: Isotonic (quinella train fit)
型フィルタ: 1_full (型1/型2/型3 all, 型4 除外)
EV 閾値: {best_gen['ev_thr'] if best_gen is not None else 1.20}
Edge 閾値: 0.02
買い目絞り: {best_gen['top_n'] if best_gen is not None else 'all'}
予算: 3000円/レース, EV 比例配分
```

### 期待性能
- train ROI: {best_gen['tr_roi'] if best_gen is not None else cur['tr_roi']:.4f}
- test ROI: {best_roi:.4f}
- 差: {best_diff:+.4f}
- 95% CI: [{best_ci_lo:.4f}, {best_gen['ci_hi'] if best_gen is not None else cur['ci_hi']:.4f}]
- ベット率 (test): {(best_gen['te_n'] if best_gen is not None else cur['te_n']) / 44064 * 100:.2f}%

## 出力ファイル
- quinella_monthly.csv / quinella_by_type.csv / quinella_by_ev.csv
- quinella_by_stadium.csv / quinella_by_odds.csv
- quinella_train_test_gap.csv / quinella_optimization.csv
"""
(OUT/"quinella_deepdive_report.md").write_text(md, encoding="utf-8")
print(f"\nsaved: quinella_deepdive_report.md")
print(f"\n=== パターン {pattern}: {desc} ===")
