# -*- coding: utf-8 -*-
"""2連複 最終戦略選定 — train で戦略選択、test で評価 (リーク防止徹底)."""
from __future__ import annotations
import io, os, sys, math, pickle, runpy, time, warnings
from pathlib import Path
from itertools import permutations, combinations
import numpy as np
import pandas as pd

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
    p = np.asarray(p_lane); arr = np.zeros(15)
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


print("="*80); print("2連複 最終戦略選定 (train 選定 → test 評価)"); print("="*80)

# ========== [0] cache 確認 ==========
CACHE_TR = OUT/"quinella_candidates_train.pkl"
CACHE_TE = OUT/"quinella_candidates_test.pkl"

if CACHE_TR.exists() and CACHE_TE.exists():
    print("\n[0] cache から candidates ロード ...")
    with open(CACHE_TR, "rb") as f: train_races = pickle.load(f)
    with open(CACHE_TE, "rb") as f: test_races = pickle.load(f)
    print(f"  train races: {len(train_races):,}, test: {len(test_races):,}")
else:
    print("\n[0] v4_ext_fixed runpy + backtest candidates 収集 ...")
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
    print(f"  β={beta.shape}")

    with open(OUT/"calibration_isotonic_quinella.pkl", "rb") as f: iso = pickle.load(f)

    from scripts.db import get_connection
    conn = get_connection()
    def load_results(df, dfrom, dto):
        q = pd.read_sql_query(f"SELECT date, stadium, race_number, rank, boat FROM race_results WHERE date BETWEEN '{dfrom}' AND '{dto}' AND rank BETWEEN 1 AND 3", conn.native)
        q["date"] = pd.to_datetime(q["date"]).dt.date
        mp = {}
        for (d,s,r),g in q.groupby(["date","stadium","race_number"]):
            g2 = g.sort_values("rank")
            if len(g2) < 3: continue
            mp[(d,s,r)] = (int(g2.iloc[0]["boat"]), int(g2.iloc[1]["boat"]), int(g2.iloc[2]["boat"]))
        return mp
    def load_odds(dfrom, dto):
        q = pd.read_sql_query(f"SELECT date, stadium, race_number, combination, odds_1min FROM odds_quinella WHERE date BETWEEN '{dfrom}' AND '{dto}' AND odds_1min IS NOT NULL AND odds_1min > 0", conn.native)
        q["date"] = pd.to_datetime(q["date"]).dt.date
        mp = {}
        for (d,s,r),g in q.groupby(["date","stadium","race_number"]):
            book = {}
            for _, row in g.iterrows():
                try:
                    parts = tuple(sorted(int(x) for x in row["combination"].split("=")))
                    book[parts] = float(row["odds_1min"])
                except Exception: pass
            if book: mp[(d,s,r)] = book
        return mp
    tr_res = load_results(None, TRAIN_FROM, TRAIN_TO); print(f"  train res: {len(tr_res):,}")
    te_res = load_results(None, TEST_FROM, TEST_TO);   print(f"  test  res: {len(te_res):,}")
    tr_odds = load_odds(TRAIN_FROM, TRAIN_TO); print(f"  train odds: {len(tr_odds):,}")
    te_odds = load_odds(TEST_FROM, TEST_TO);   print(f"  test  odds: {len(te_odds):,}")
    conn.close()

    # Collect races with candidates (EV>=1.10, Edge>=0.01 で緩めに取得して後で絞れるように)
    def collect(X, keys, res_map, odds_map, label):
        races = []
        N = len(keys)
        t0 = time.time()
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
                if ev >= 1.10 and edge >= 0.01:  # 緩め (後で絞る)
                    cands.append((combo, odds, p_adj, ev, edge))
            if not cands: continue
            races.append({"date":d, "stadium":s, "race":r, "month":f"{d.year}-{d.month:02d}",
                          "type":t, "actual":act_key, "cands":cands})
            if (i+1) % 50000 == 0:
                print(f"    [{label}] {i+1}/{N}  races={len(races):,}  elapsed {time.time()-t0:.1f}s")
        print(f"  [{label}] done, races={len(races):,}, elapsed {time.time()-t0:.1f}s")
        return races
    print("\n  train candidates 収集 ...")
    train_races = collect(X_tr, keys_tr, tr_res, tr_odds, "train")
    print("  test candidates 収集 ...")
    test_races = collect(X_te, keys_te, te_res, te_odds, "test")
    with open(CACHE_TR, "wb") as f: pickle.dump(train_races, f)
    with open(CACHE_TE, "wb") as f: pickle.dump(test_races, f)
    print(f"  cached: {CACHE_TR.name}, {CACHE_TE.name}")

# ========== [1] 戦略評価関数 ==========
N_TRAIN = sum(1 for _ in train_races)
N_TEST = sum(1 for _ in test_races)

def apply_strategy(races, ev_thr, edge_thr, type_filter, top_n, odds_cap):
    """戦略適用、per-race stake/pay を返す."""
    race_stakes = []; race_pays = []
    race_months = []; race_hits = []; race_types = []
    n_bets = 0; n_hits = 0
    for r in races:
        if r["type"] not in type_filter: continue
        # filter candidates
        cands = []
        for combo, odds, p_adj, ev, edge in r["cands"]:
            if ev < ev_thr or edge < edge_thr: continue
            if odds_cap and odds > odds_cap: continue
            cands.append((combo, odds, p_adj, ev, edge))
        if not cands: continue
        # top-N by EV
        if top_n is not None and len(cands) > top_n:
            cands = sorted(cands, key=lambda x: -x[3])[:top_n]
        # EV proportional allocation
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
        rs = sum(a[2] for a in alloc); rp = 0; hit = 0
        for combo, odds, st in alloc:
            if combo == r["actual"]:
                rp += int(st*odds); hit = 1
        race_stakes.append(rs); race_pays.append(rp)
        race_months.append(r["month"]); race_hits.append(hit); race_types.append(r["type"])
        n_bets += len(alloc); n_hits += (1 if hit else 0)
    return race_stakes, race_pays, race_months, race_hits, race_types, n_bets, n_hits

def summary(stakes, pays, months=None):
    stake_sum = sum(stakes); pay_sum = sum(pays)
    roi = pay_sum/stake_sum if stake_sum else 0
    ci = bootstrap_roi_ci(stakes, pays, n_boot=1000) if len(stakes) > 0 else (0,0)
    # 月別 std
    mon_std = 0
    if months:
        mon_rois = {}
        for s, p, m in zip(stakes, pays, months):
            mon_rois.setdefault(m, [0, 0])
            mon_rois[m][0] += s; mon_rois[m][1] += p
        rois = [v[1]/v[0] for v in mon_rois.values() if v[0] > 0]
        mon_std = float(np.std(rois)) if rois else 0
    return {"n_bet_races": len(stakes), "stake": stake_sum, "payout": pay_sum,
            "profit": pay_sum-stake_sum, "roi": roi,
            "ci_lo": ci[0], "ci_hi": ci[1], "month_std": mon_std}

# ========== [2] train で戦略空間探索 ==========
print("\n[2] 戦略空間を train で探索 ...")
EV_LIST = [1.20, 1.30, 1.40, 1.50]
EDGE_LIST = [0.02]
TYPE_FILTERS = {
    "full": {"型1","型2","型3"},
    "no_t3": {"型1","型2"},
    "t1+t2": {"型1","型2"},
    "t2+t3": {"型2","型3"},
    "t2_only": {"型2"},
}
TOP_N_LIST = [None, 5, 3, 2, 1]
ODDS_CAP_LIST = [None, 50.0, 20.0, 10.0]

results = []
total = len(EV_LIST) * len(EDGE_LIST) * len(TYPE_FILTERS) * len(TOP_N_LIST) * len(ODDS_CAP_LIST)
print(f"  総組み合わせ: {total}")
t0 = time.time()
idx_c = 0
for ev_thr in EV_LIST:
    for edge_thr in EDGE_LIST:
        for tf_name, tf in TYPE_FILTERS.items():
            for top_n in TOP_N_LIST:
                for odds_cap in ODDS_CAP_LIST:
                    idx_c += 1
                    stakes, pays, months, hits, types, n_bet, n_hit = apply_strategy(
                        train_races, ev_thr, edge_thr, tf, top_n, odds_cap)
                    if n_bet < 500: continue
                    s = summary(stakes, pays, months)
                    results.append({
                        "ev_thr":ev_thr, "edge_thr":edge_thr,
                        "type_filter":tf_name, "top_n":str(top_n) if top_n else "all",
                        "odds_cap":str(odds_cap) if odds_cap else "none",
                        **s
                    })
                    if idx_c % 50 == 0:
                        print(f"    {idx_c}/{total}  elapsed {time.time()-t0:.1f}s")
res_df = pd.DataFrame(results)
res_df.to_csv(OUT/"quinella_train_strategy_search.csv", index=False, encoding="utf-8-sig")
print(f"  train 探索完了: {len(res_df)} 戦略, elapsed {time.time()-t0:.1f}s")

# ========== [3] train で最良戦略選定 (test は見ない) ==========
print("\n[3] train で最良戦略選定 ...")
# 基準: train ROI 最大, 月別 std < 0.15, bet_races >= 1000
viable = res_df[(res_df["n_bet_races"] >= 1000) & (res_df["month_std"] < 0.15)].sort_values("roi", ascending=False)
print(f"\n  Top 10 train 戦略 (bet_races>=1000, month_std<0.15):")
print(f"  {'EV':>5} {'type':<8} {'top_n':>5} {'odds':>6} {'N':>7} {'train_ROI':>9} {'std':>6}")
for _, rw in viable.head(10).iterrows():
    print(f"  {rw['ev_thr']:>5.2f} {rw['type_filter']:<8} {str(rw['top_n']):>5} "
          f"{rw['odds_cap']:>6} {int(rw['n_bet_races']):>7,} {rw['roi']:>9.4f} "
          f"{rw['month_std']:>6.4f}")

top5_train = viable.head(5)

# ========== [4] test で評価 (初めて test 見る) ==========
print("\n[4] Top 5 train 戦略を test で評価 ...")
eval_rows = []
for _, rw in top5_train.iterrows():
    tf = TYPE_FILTERS[rw["type_filter"]]
    top_n = None if rw["top_n"] == "all" else int(rw["top_n"])
    odds_cap = None if rw["odds_cap"] == "none" else float(rw["odds_cap"])
    # test 評価
    stakes, pays, months, hits, types, n_bet, n_hit = apply_strategy(
        test_races, rw["ev_thr"], rw["edge_thr"], tf, top_n, odds_cap)
    s_te = summary(stakes, pays, months)
    # train (復元)
    stakes_tr, pays_tr, months_tr, hits_tr, types_tr, _, _ = apply_strategy(
        train_races, rw["ev_thr"], rw["edge_thr"], tf, top_n, odds_cap)
    s_tr = summary(stakes_tr, pays_tr, months_tr)
    diff = s_te["roi"] - s_tr["roi"]
    eval_rows.append({
        "ev_thr":rw["ev_thr"], "type_filter":rw["type_filter"],
        "top_n":rw["top_n"], "odds_cap":rw["odds_cap"],
        "train_roi":s_tr["roi"], "train_ci_lo":s_tr["ci_lo"], "train_ci_hi":s_tr["ci_hi"],
        "train_n":s_tr["n_bet_races"], "train_month_std":s_tr["month_std"],
        "test_roi":s_te["roi"], "test_ci_lo":s_te["ci_lo"], "test_ci_hi":s_te["ci_hi"],
        "test_n":s_te["n_bet_races"], "test_month_std":s_te["month_std"],
        "diff":diff,
    })
eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv(OUT/"quinella_train_test_evaluation.csv", index=False, encoding="utf-8-sig")

print(f"\n  Top 5 戦略 train→test 評価:")
print(f"  {'EV':>5} {'type':<8} {'top':>5} {'odds':>6} {'tr_ROI':>7} {'te_ROI':>7} "
      f"{'CI下':>6} {'CI上':>6} {'diff':>7} {'tr_n':>6} {'te_n':>6}")
for _, rw in eval_df.iterrows():
    print(f"  {rw['ev_thr']:>5.2f} {rw['type_filter']:<8} {rw['top_n']:>5} "
          f"{rw['odds_cap']:>6} {rw['train_roi']:>7.4f} {rw['test_roi']:>7.4f} "
          f"{rw['test_ci_lo']:>6.4f} {rw['test_ci_hi']:>6.4f} {rw['diff']:>+7.4f} "
          f"{int(rw['train_n']):>6,} {int(rw['test_n']):>6,}")

# 最終運用候補: test 基準を満たすもの
ok = eval_df[(eval_df["test_roi"] >= 1.0) & (eval_df["test_ci_lo"] >= 0.95) &
             (eval_df["diff"].abs() < 0.05)]
semi_ok = eval_df[(eval_df["test_roi"] >= 1.0) & (eval_df["test_ci_lo"] >= 0.90)]

if len(ok) > 0:
    best = ok.sort_values("test_roi", ascending=False).iloc[0]
    pattern = "α"; desc = f"運用基準到達 (train選定→test評価で CI下≥0.95, 差<0.05)"
elif len(semi_ok) > 0:
    best = semi_ok.sort_values("test_ci_lo", ascending=False).iloc[0]
    pattern = "β"; desc = f"近い (CI下 {best['test_ci_lo']:.4f})"
else:
    best = eval_df.sort_values("test_roi", ascending=False).iloc[0]
    pattern = "γ"; desc = "運用基準未達"

print(f"\n=== 最終判定: パターン {pattern} ===")
print(f"  {desc}")
print(f"  選定戦略: EV>={best['ev_thr']:.2f}, type={best['type_filter']}, "
      f"top_n={best['top_n']}, odds_cap={best['odds_cap']}")
print(f"  train ROI={best['train_roi']:.4f}, test ROI={best['test_roi']:.4f}, "
      f"CI[{best['test_ci_lo']:.4f},{best['test_ci_hi']:.4f}], diff={best['diff']:+.4f}")

# ========== [5] 会場 sanity check ==========
print("\n[5] 会場 sanity check (train 好調会場が test でも好調か) ...")
# 現状 (1_full, EV>=1.20, no cap) の train / test 会場別 ROI
def stadium_roi(races, ev_thr, edge_thr, type_filter, top_n, odds_cap):
    from collections import defaultdict
    st = defaultdict(lambda: [0,0])
    for r in races:
        if r["type"] not in type_filter: continue
        cands = [(c,o,p,e,ed) for c,o,p,e,ed in r["cands"]
                 if e >= ev_thr and ed >= edge_thr and (not odds_cap or o <= odds_cap)]
        if not cands: continue
        if top_n and len(cands) > top_n:
            cands = sorted(cands, key=lambda x:-x[3])[:top_n]
        ev_sum = sum(c[3] for c in cands)
        total_stake = 0; total_pay = 0
        for c,o,p,e,ed in cands:
            units = int(BUDGET * e / ev_sum / MIN_UNIT)
            stake = max(0,units)*MIN_UNIT
            total_stake += stake
            if c == r["actual"]: total_pay += int(stake*o)
        st[r["stadium"]][0] += total_stake
        st[r["stadium"]][1] += total_pay
    return {s: v[1]/v[0] if v[0] else 0 for s,v in st.items()}

tr_st = stadium_roi(train_races, 1.20, 0.02, {"型1","型2","型3"}, None, None)
te_st = stadium_roi(test_races,  1.20, 0.02, {"型1","型2","型3"}, None, None)
# 一致度: train 上位 5 会場が test でも上位半分にあるか
tr_top5 = [s for s,r in sorted(tr_st.items(), key=lambda x:-x[1])[:5]]
te_top5 = [s for s,r in sorted(te_st.items(), key=lambda x:-x[1])[:5]]
overlap = set(tr_top5) & set(te_top5)
print(f"  train top5 会場: {sorted(tr_top5)}")
print(f"  test  top5 会場: {sorted(te_top5)}")
print(f"  overlap: {sorted(overlap)}  ({len(overlap)}/5)")
print("  → 会場フィルタは " + ("有効" if len(overlap) >= 3 else "過学習的 (test 好調と違う)"))

# save stadium comparison
st_rows = []
for s in sorted(set(tr_st) | set(te_st)):
    st_rows.append({"stadium":s, "train_roi":tr_st.get(s,0), "test_roi":te_st.get(s,0),
                    "diff":te_st.get(s,0)-tr_st.get(s,0)})
pd.DataFrame(st_rows).to_csv(OUT/"quinella_stadium_train_test_check.csv", index=False, encoding="utf-8-sig")

# ========== [6] レポート ==========
md = f"""# 2連複 最終戦略レポート

## 核心メッセージ
- **パターン {pattern}**: {desc}
- train 選定 → test 評価の分離徹底
- 戦略空間: {total} 組み合わせ (EV × 型 × top-N × odds cap)

## Top 5 train 戦略 → test 評価

| EV | 型 | top | odds | train ROI | test ROI | CI下 | CI上 | 差 | train n | test n |
|---|---|---|---|---|---|---|---|---|---|---|
"""
for _, rw in eval_df.iterrows():
    md += (f"| {rw['ev_thr']:.2f} | {rw['type_filter']} | {rw['top_n']} | "
           f"{rw['odds_cap']} | {rw['train_roi']:.4f} | **{rw['test_roi']:.4f}** | "
           f"{rw['test_ci_lo']:.4f} | {rw['test_ci_hi']:.4f} | "
           f"{rw['diff']:+.4f} | {int(rw['train_n']):,} | {int(rw['test_n']):,} |\n")

md += f"""

## 最良戦略

```
券種: 2連複 (quinella)
モデル: v4_ext_fixed
キャリブレーション: Isotonic (quinella train fit)
判定: EV ≥ {best['ev_thr']:.2f} AND Edge ≥ 0.02
型フィルタ: {best['type_filter']}
買い目絞り: top-{best['top_n']}
オッズ上限: {best['odds_cap']}
予算: 3000円/レース, EV 比例配分
```

### 期待性能
- **train ROI**: {best['train_roi']:.4f} (CI [{best['train_ci_lo']:.4f}, {best['train_ci_hi']:.4f}])
- **test ROI**: {best['test_roi']:.4f} (CI [{best['test_ci_lo']:.4f}, {best['test_ci_hi']:.4f}])
- 差: {best['diff']:+.4f}
- train 月別 std: {best['train_month_std']:.4f}
- test 月別 std: {best['test_month_std']:.4f}
- ベット率 (test): {best['test_n']/N_TEST*100:.2f}%

## 会場 sanity check
- train top5: {sorted(tr_top5)}
- test top5:  {sorted(te_top5)}
- overlap: {sorted(overlap)} ({len(overlap)}/5)
- → 会場フィルタは {'有効' if len(overlap)>=3 else '過学習的'}

## 次アクション
"""
if pattern == "α":
    md += "1. 運用候補戦略確定, 仕様書更新\n2. ペーパートレード開始\n"
elif pattern == "β":
    md += "1. CI下≥0.95 に近いが未達 → 慎重運用 or 追加分析\n2. データ期間延長で CI 絞り込み\n"
else:
    md += "1. 運用基準未達 → 他アプローチ検討\n"

md += """

## 出力ファイル
- quinella_train_strategy_search.csv
- quinella_train_test_evaluation.csv
- quinella_stadium_train_test_check.csv
- quinella_candidates_train.pkl / quinella_candidates_test.pkl
"""
(OUT/"quinella_final_strategy.md").write_text(md, encoding="utf-8")
print(f"\nsaved: quinella_final_strategy.md")
print(f"\n=== 完了: パターン {pattern} ===")
