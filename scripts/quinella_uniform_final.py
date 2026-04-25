# -*- coding: utf-8 -*-
"""均等配分で 2連複 最終バックテスト + ベット率確定."""
from __future__ import annotations
import sys, pickle, math, sqlite3, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "notebooks" / "output"
sys.path.insert(0, str(BASE))
sys.stdout.reconfigure(encoding="utf-8")

BUDGET = 3000; MIN_UNIT = 100
EV_THR = 1.50; EDGE_THR = 0.02; ODDS_CAP = 10.0
TYPE_FILTER = {"型1", "型2"}

TRAIN_FROM = "2020-02-01"; TRAIN_TO = "2025-06-30"
TEST_FROM  = "2025-07-01"; TEST_TO  = "2026-04-18"

print("="*80); print("2連複 均等配分 最終 + ベット率確定"); print("="*80)

# ========== [0] ベット率の分母を正確に ==========
print("\n[0] ベット率の分母確認 (test 期間)")
DB = BASE / "boatrace.db"
conn = sqlite3.connect(str(DB))

# a) test 期間の race_cards レース総数 (モデル対象になり得る)
n_rc = conn.execute(f"SELECT COUNT(DISTINCT date||'|'||stadium||'|'||race_number) FROM race_cards WHERE date BETWEEN '{TEST_FROM}' AND '{TEST_TO}'").fetchone()[0]
# b) race_results JOIN 可能
n_rr_join = conn.execute(f"""
SELECT COUNT(DISTINCT rc.date||'|'||rc.stadium||'|'||rc.race_number)
FROM race_cards rc
JOIN race_results rr ON rc.date=rr.date AND rc.stadium=rr.stadium AND rc.race_number=rr.race_number
WHERE rc.date BETWEEN '{TEST_FROM}' AND '{TEST_TO}'""").fetchone()[0]
# c) quinella odds 1min JOIN
n_qu_join = conn.execute(f"""
SELECT COUNT(DISTINCT rc.date||'|'||rc.stadium||'|'||rc.race_number)
FROM race_cards rc
JOIN race_results rr ON rc.date=rr.date AND rc.stadium=rr.stadium AND rc.race_number=rr.race_number
JOIN odds_quinella oq ON rc.date=oq.date AND rc.stadium=oq.stadium AND rc.race_number=oq.race_number
WHERE rc.date BETWEEN '{TEST_FROM}' AND '{TEST_TO}' AND oq.odds_1min IS NOT NULL""").fetchone()[0]
conn.close()

print(f"  (a) race_cards test races:            {n_rc:,}")
print(f"  (b) (a) AND race_results JOIN:        {n_rr_join:,}")
print(f"  (c) (b) AND quinella odds 1min:       {n_qu_join:,}")

# ========== [1] 候補キャッシュ + 戦略フィルタ適用 ==========
print("\n[1] 候補キャッシュ読み込み ...")
with open(OUT/"quinella_candidates_train.pkl", "rb") as f: train_races = pickle.load(f)
with open(OUT/"quinella_candidates_test.pkl", "rb") as f: test_races = pickle.load(f)
# 候補キャッシュ: 型×候補>=1 を持つレース (EV>=1.10, Edge>=0.01 で緩く取得)
print(f"  train races w/ candidates: {len(train_races):,}")
print(f"  test  races w/ candidates: {len(test_races):,}")

def filter_cands(race):
    if race["type"] not in TYPE_FILTER: return None
    cands = [(c,o,p,e,ed) for c,o,p,e,ed in race["cands"]
             if e >= EV_THR and ed >= EDGE_THR and o <= ODDS_CAP]
    return cands if cands else None

# ========== [2] 均等配分 ==========
def alloc_uniform(cands, budget=BUDGET, min_unit=MIN_UNIT):
    k = len(cands)
    if k == 0: return []
    base = (budget // k) // min_unit * min_unit
    alloc = [(c[0], c[1], base) for c in cands]
    used = base * k
    extra = budget - used
    # 残は EV 上位から
    order = sorted(range(k), key=lambda i: -cands[i][3])
    for ia in order:
        if extra < min_unit: break
        c, o, s = alloc[ia]; alloc[ia] = (c, o, s + min_unit); extra -= min_unit
    return [a for a in alloc if a[2] > 0]

def alloc_ev_proportional(cands, budget=BUDGET, min_unit=MIN_UNIT):
    ev_sum = sum(c[3] for c in cands)
    alloc = []
    for combo, odds, p_adj, ev, edge in cands:
        units = int(budget * ev / ev_sum / min_unit)
        alloc.append((combo, odds, max(0, units) * min_unit))
    used = sum(a[2] for a in alloc); extra = budget - used
    order = sorted(range(len(alloc)), key=lambda ii: -cands[ii][3])
    for ia in order:
        if extra < min_unit: break
        c,o,s = alloc[ia]; alloc[ia] = (c,o,s+min_unit); extra -= min_unit
    return [a for a in alloc if a[2] > 0]

def alloc_half_kelly(cands, budget=BUDGET, min_unit=MIN_UNIT):
    kv = []
    for combo, odds, p_adj, ev, edge in cands:
        if odds > 1 and p_adj * odds > 1:
            kv.append(max(0, (p_adj*odds - 1)/(odds - 1)))
        else: kv.append(0)
    ks = sum(kv)
    if ks <= 0:
        return alloc_ev_proportional(cands, budget, min_unit)
    alloc = []
    for (c,o,p,e,ed), v in zip(cands, kv):
        units = int(budget * v / ks / min_unit)
        alloc.append((c, o, max(0,units)*min_unit))
    used = sum(a[2] for a in alloc); extra = budget - used
    order = sorted(range(len(alloc)), key=lambda ii: -kv[ii])
    for ia in order:
        if extra < min_unit: break
        c,o,s = alloc[ia]; alloc[ia] = (c,o,s+min_unit); extra -= min_unit
    return [a for a in alloc if a[2] > 0]

def run(races, alloc_fn):
    rows = []; n_bets = 0; n_hits = 0; combo_stakes = []
    for r in races:
        cands = filter_cands(r)
        if not cands: continue
        alloc = alloc_fn(cands)
        if not alloc: continue
        rs = sum(a[2] for a in alloc)
        rp = 0; hit = 0
        for combo, odds, st in alloc:
            combo_stakes.append(st)
            if combo == r["actual"]:
                rp += int(st*odds); hit = 1
        rows.append({"date": r["date"], "stadium": r["stadium"], "race": r["race"],
                     "month": r["month"], "type": r["type"],
                     "stake": rs, "payout": rp, "hit": hit, "n_combos": len(alloc)})
        n_bets += len(alloc); n_hits += (1 if hit else 0)
    return pd.DataFrame(rows), n_bets, n_hits, combo_stakes

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

def summarize(df):
    if df.empty: return {}
    s = df["stake"].sum(); p = df["payout"].sum()
    n = len(df); h = int(df["hit"].sum())
    roi = p/s if s else 0
    ci = bootstrap_roi_ci(df["stake"].values, df["payout"].values)
    profits = (df["payout"] - df["stake"]).values
    mon = df.groupby("month")[["stake","payout"]].sum()
    mon["roi"] = mon["payout"]/mon["stake"]
    mon_std = float(mon["roi"].std()) if len(mon) > 1 else 0
    sharpe = float(profits.mean()/profits.std()) if profits.std() > 0 else 0
    df_s = df.sort_values(["date","stadium","race"])
    p_ts = (df_s["payout"] - df_s["stake"]).values
    cum = np.cumsum(p_ts); peak = np.maximum.accumulate(cum)
    max_dd = int((cum - peak).min())
    return {"n_bet_races":n, "n_hits":h, "hit_rate":h/n if n else 0,
            "stake":int(s), "payout":int(p), "profit":int(p-s),
            "roi":roi, "ci_lo":ci[0], "ci_hi":ci[1], "ci_width":ci[1]-ci[0],
            "mon_std":mon_std, "sharpe":sharpe, "max_dd":max_dd}

# ========== [3] 3 方式 × 2 期間 ==========
print("\n[3] 3 配分方式 × 2 期間")
methods = [("EV比例", alloc_ev_proportional),
           ("ハーフケリー", alloc_half_kelly),
           ("均等", alloc_uniform)]
rows = []; combo_info = {}
all_results = {}
for m_name, fn in methods:
    for period, races in [("train", train_races), ("test", test_races)]:
        df, n_bets, n_hits, combo_stakes = run(races, fn)
        s = summarize(df)
        combo_info[(m_name, period)] = combo_stakes
        all_results[(m_name, period)] = {"df": df, **s}
        rows.append({"method": m_name, "period": period, "n_bets": n_bets,
                     "n_hits_races": n_hits, **s})
        print(f"  {m_name:<14} {period:<6}  races={s['n_bet_races']:>6,}  bets={n_bets:>6,}  "
              f"ROI={s['roi']:.4f}  CI[{s['ci_lo']:.4f},{s['ci_hi']:.4f}]  "
              f"w={s['ci_width']:.4f}  mon_std={s['mon_std']:.4f}  DD=¥{s['max_dd']:+,}")
cmp_df = pd.DataFrame(rows)
cmp_df.to_csv(OUT/"quinella_final_methods_comparison.csv", index=False, encoding="utf-8-sig")

# ========== [4] ベット率 (複数定義) ==========
print("\n[4] ベット率 (複数定義)")
# 均等配分の結果で計算
uni_te = all_results[("均等","test")]
bet_races = uni_te["n_bet_races"]
n_bets_cnt = cmp_df[(cmp_df["method"]=="均等") & (cmp_df["period"]=="test")]["n_bets"].iloc[0]

print(f"  test 期間で 1点以上ベットしたレース数: {bet_races:,}")
print(f"  test 期間の総ベット数 (買い目単位):    {n_bets_cnt:,}")
print()
print(f"  ベット率 (レースベース):")
print(f"    分母=(a) race_cards 全レース {n_rc:,}: {bet_races/n_rc*100:.2f}%")
print(f"    分母=(b) results JOIN 可    {n_rr_join:,}: {bet_races/n_rr_join*100:.2f}%")
print(f"    分母=(c) quinella odds 有   {n_qu_join:,}: {bet_races/n_qu_join*100:.2f}%")
print(f"    分母=候補ありレース        {len(test_races):,}: {bet_races/len(test_races)*100:.2f}%")

# 運用時の「有効ベット率」は (c) モデル対象レース基準が一番実用的
bet_rate_op = bet_races / n_qu_join * 100

# ========== [5] ベット額分布 (均等) ==========
print("\n[5] ベット額分布 (均等配分 test, per combo)")
arr = np.array(combo_info[("均等","test")])
print(f"  n={len(arr):,}  mean=¥{arr.mean():.0f}  median=¥{np.median(arr):.0f}  "
      f"min=¥{arr.min():.0f}  max=¥{arr.max():.0f}  std=¥{arr.std():.0f}")
# 組数別分布
df_u = all_results[("均等","test")]["df"]
ncomb = df_u["n_combos"].value_counts().sort_index()
print(f"  レースごとの買い目数分布:")
for nc, cnt in ncomb.items():
    print(f"    {nc}点買い: {cnt:,} races ({cnt/len(df_u)*100:.2f}%)")

# ========== [6] 月別 ROI (均等 test) ==========
print("\n[6] 均等配分 test 月別 ROI")
mon = df_u.groupby("month").agg(
    n_bet=("hit","count"),
    hits=("hit","sum"),
    stake=("stake","sum"),
    payout=("payout","sum"),
).reset_index()
mon["roi"] = mon["payout"]/mon["stake"]
mon["profit"] = mon["payout"] - mon["stake"]
print(f"  {'月':<8} {'n_bet':>6} {'stake':>10} {'payout':>10} {'profit':>+10} {'ROI':>7}")
for _, rw in mon.iterrows():
    print(f"  {rw['month']:<8} {int(rw['n_bet']):>6} ¥{int(rw['stake']):>9,} "
          f"¥{int(rw['payout']):>9,} ¥{int(rw['profit']):>+9,} {rw['roi']:>7.4f}")

# 月平均 stake/profit
avg_stake_mon = mon["stake"].mean()
avg_profit_mon = mon["profit"].mean()
print(f"\n  月平均 stake: ¥{avg_stake_mon:,.0f}")
print(f"  月平均 profit: ¥{avg_profit_mon:+,.0f}")

# ========== [7] 運用資金シミュレーション ==========
print("\n[7] 運用資金シミュレーション (均等 test ベース)")
max_dd = abs(uni_te["max_dd"])
total_profit = uni_te["profit"]
total_stake = uni_te["stake"]
roi = uni_te["roi"]

fund_rows = []
for f in [300_000, 500_000, 1_000_000, 3_000_000]:
    dd_ratio = max_dd / f * 100
    # 月 profit 比率
    prof_ratio = avg_profit_mon / f * 100
    if dd_ratio > 100: judge = "**破綻不可避**"
    elif dd_ratio > 50: judge = "**高リスク**"
    elif dd_ratio > 25: judge = "許容上限"
    elif dd_ratio > 10: judge = "**推奨**"
    else: judge = "保守的"
    fund_rows.append({"fund_yen":f, "dd_ratio_pct":dd_ratio, "monthly_stake":avg_stake_mon,
                      "monthly_profit":avg_profit_mon, "monthly_profit_ratio_pct":prof_ratio,
                      "judge":judge})
print(f"  {'資金':>10} {'DD比率':>7} {'月stake':>11} {'月profit':>+11} {'月利益率':>8} {'判定':<20}")
for r in fund_rows:
    print(f"  ¥{int(r['fund_yen']):>9,} {r['dd_ratio_pct']:>6.2f}% "
          f"¥{int(r['monthly_stake']):>10,} ¥{int(r['monthly_profit']):>+10,} "
          f"{r['monthly_profit_ratio_pct']:>7.3f}% {r['judge']:<20}")
pd.DataFrame(fund_rows).to_csv(OUT/"quinella_fund_sim.csv", index=False, encoding="utf-8-sig")

# ========== [8] 最終レポート ==========
print("\n[8] レポート ...")
uni_tr = all_results[("均等","train")]
md = f"""# 2連複 最終運用仕様 (均等配分版)

## 最終運用仕様

```
券種: 2連複 (quinella)
モデル: v4_ext_fixed
キャリブレーション: Isotonic (quinella train fit)
判定: EV ≥ 1.50 AND Edge ≥ 0.02
型フィルタ: no_t3 (型1 + 型2, 型3 除外)
オッズ上限: 10.0
予算: 3000円/レース
配分: **均等** (候補数で割り 100 円単位切り捨て、残は EV 上位補填)
```

## 期待性能 (test 2025-07〜2026-04)

| 指標 | 値 |
|---|---|
| **ROI** | **{uni_te['roi']:.4f}** |
| 95% CI | [{uni_te['ci_lo']:.4f}, {uni_te['ci_hi']:.4f}] |
| CI 幅 | {uni_te['ci_width']:.4f} |
| 月別 std | {uni_te['mon_std']:.4f} |
| Sharpe (per-race) | {uni_te['sharpe']:+.4f} |
| 最大 DD (9.5ヶ月) | ¥{uni_te['max_dd']:+,} |

## ベット率 (複数定義)

| 定義 | 分母 | ベットレース数 | ベット率 |
|---|---|---|---|
| (a) race_cards 全レース | {n_rc:,} | {bet_races:,} | {bet_races/n_rc*100:.2f}% |
| (b) results JOIN 可 | {n_rr_join:,} | {bet_races:,} | {bet_races/n_rr_join*100:.2f}% |
| **(c) モデル対象 (odds 有)** | **{n_qu_join:,}** | **{bet_races:,}** | **{bet_rate_op:.2f}%** |
| (d) 候補ありレース内 | {len(test_races):,} | {bet_races:,} | {bet_races/len(test_races)*100:.2f}% |

### 運用指標
- **ベット率**: **{bet_rate_op:.2f}%** (実用定義 = c)
- 総ベット数 (買い目単位): {n_bets_cnt:,}
- 月平均 ベット数 (レース): {bet_races / 10:.0f} (10 ヶ月で {bet_races:,})
- 月平均 stake: ¥{avg_stake_mon:,.0f}
- 月平均 profit: ¥{avg_profit_mon:+,.0f}
- 平均ベット額/combo: ¥{arr.mean():.0f}
- 平均買い目数/参加レース: {arr.size / bet_races:.2f}

## 買い目数分布 (test 均等配分)

| n点買い | レース数 | 割合 |
|---|---|---|
"""
for nc, cnt in ncomb.items():
    md += f"| {nc}点 | {cnt:,} | {cnt/len(df_u)*100:.2f}% |\n"

md += f"""

## 3 配分方式比較 (test)

| 方式 | ROI | CI下 | CI上 | CI幅 | mon_std | Sharpe | maxDD |
|---|---|---|---|---|---|---|---|
"""
for _, rw in cmp_df[cmp_df["period"]=="test"].iterrows():
    md += (f"| {rw['method']} | {rw['roi']:.4f} | {rw['ci_lo']:.4f} | "
           f"{rw['ci_hi']:.4f} | {rw['ci_width']:.4f} | {rw['mon_std']:.4f} | "
           f"{rw['sharpe']:+.4f} | ¥{int(rw['max_dd']):+,} |\n")

md += f"""

## 月別 ROI (均等 test)

| 月 | n_bet | stake | payout | profit | ROI |
|---|---|---|---|---|---|
"""
for _, rw in mon.iterrows():
    md += (f"| {rw['month']} | {int(rw['n_bet'])} | ¥{int(rw['stake']):,} | "
           f"¥{int(rw['payout']):,} | ¥{int(rw['profit']):+,} | {rw['roi']:.4f} |\n")

md += f"""

## 運用資金シミュレーション

| 運用資金 | DD比率 | 月平均stake | 月平均profit | 月利益率 | 判定 |
|---|---|---|---|---|---|
"""
for r in fund_rows:
    md += (f"| ¥{int(r['fund_yen']):,} | {r['dd_ratio_pct']:.2f}% | "
           f"¥{int(r['monthly_stake']):,} | ¥{int(r['monthly_profit']):+,} | "
           f"{r['monthly_profit_ratio_pct']:.3f}% | {r['judge']} |\n")

md += """

## 実運用の注意点
- オッズは 1分前オッズ基準。実購入時に ±5% の乖離リスク。
- 100円単位切り捨て: EV 上位補填で予算フル活用。
- 流動性: odds ≤ 10 中心で高い。
- ベット頻度: 月 ~340 レース前後 (test 基準)。
- 月別 ROI は 0.12 の std を持つ → 個別月では赤字もあり得る。

## 出力ファイル
- quinella_final_methods_comparison.csv
- quinella_fund_sim.csv
"""
(OUT/"quinella_uniform_final_report.md").write_text(md, encoding="utf-8")
print(f"\nsaved: quinella_uniform_final_report.md")
print(f"\n=== 均等配分 test ROI {uni_te['roi']:.4f}, CI[{uni_te['ci_lo']:.4f},{uni_te['ci_hi']:.4f}], ベット率 {bet_rate_op:.2f}% ===")
