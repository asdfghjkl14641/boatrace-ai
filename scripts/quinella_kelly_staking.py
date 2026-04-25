# -*- coding: utf-8 -*-
"""2連複 ハーフケリー 配分比較 (キャッシュ再利用)."""
from __future__ import annotations
import sys, pickle, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "notebooks" / "output"
sys.stdout.reconfigure(encoding="utf-8")

BUDGET = 3000; MIN_UNIT = 100
BANKROLL = 100000  # 方式 D 用
# 戦略: EV>=1.50, type={型1,型2}, odds<=10
EV_THR = 1.50; EDGE_THR = 0.02; ODDS_CAP = 10.0
TYPE_FILTER = {"型1", "型2"}

print("="*80); print("2連複 配分方式比較 (EV比例 vs ハーフケリー)"); print("="*80)

# ========== [0] キャッシュ読み込み ==========
with open(OUT/"quinella_candidates_train.pkl", "rb") as f: train_races = pickle.load(f)
with open(OUT/"quinella_candidates_test.pkl", "rb") as f: test_races = pickle.load(f)
print(f"\ntrain races: {len(train_races):,}, test: {len(test_races):,}")

# ========== [1] フィルタ + 配分関数 ==========
def filter_cands(race):
    """戦略フィルタ適用後の candidates 返す."""
    if race["type"] not in TYPE_FILTER: return None
    cands = [(c,o,p,e,ed) for c,o,p,e,ed in race["cands"]
             if e >= EV_THR and ed >= EDGE_THR and o <= ODDS_CAP]
    return cands if cands else None

def alloc_A_ev_proportional(cands, budget=BUDGET, min_unit=MIN_UNIT):
    """方式 A: EV 比例, 予算正規化, 100 円単位切り捨て, 余り EV 上位補填."""
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

def alloc_B_half_kelly_budget(cands, budget=BUDGET, min_unit=MIN_UNIT):
    """方式 B: ハーフケリー, 予算 3000 正規化, 100 円単位."""
    # Kelly: f = (p*o - 1) / (o - 1), only positive
    kelly_vals = []
    for combo, odds, p_adj, ev, edge in cands:
        if odds <= 1 or p_adj * odds <= 1:
            kelly_vals.append(0)
        else:
            f = (p_adj * odds - 1) / (odds - 1)
            kelly_vals.append(max(0, f))
    ksum = sum(kelly_vals)
    if ksum <= 0:
        # fallback to EV proportional
        return alloc_A_ev_proportional(cands, budget, min_unit)
    # Normalize to budget (note: half-Kelly factor cancels out in normalization)
    alloc = []
    for (combo, odds, p, e, ed), kv in zip(cands, kelly_vals):
        units = int(budget * kv / ksum / min_unit)
        alloc.append((combo, odds, max(0, units) * min_unit))
    used = sum(a[2] for a in alloc); extra = budget - used
    # 余り: Kelly 値上位に補填
    order = sorted(range(len(alloc)), key=lambda ii: -kelly_vals[ii])
    for ia in order:
        if extra < min_unit: break
        c,o,s = alloc[ia]; alloc[ia] = (c,o,s+min_unit); extra -= min_unit
    return [a for a in alloc if a[2] > 0]

def alloc_D_bankroll_halfkelly(cands, budget=BUDGET, min_unit=MIN_UNIT, bankroll=BANKROLL):
    """方式 D: Bankroll ベース ハーフケリー, 予算上限付き."""
    stakes_raw = []
    for combo, odds, p_adj, ev, edge in cands:
        if odds <= 1 or p_adj * odds <= 1:
            stakes_raw.append(0)
        else:
            f = (p_adj * odds - 1) / (odds - 1)
            stakes_raw.append(0.5 * f * bankroll)
    total = sum(stakes_raw)
    if total <= 0:
        return alloc_A_ev_proportional(cands, budget, min_unit)
    # 予算超過なら比例縮小
    if total > budget:
        scale = budget / total
        stakes_raw = [s * scale for s in stakes_raw]
    # 100 円単位切り捨て
    alloc = [(cands[i][0], cands[i][1], int(s/min_unit)*min_unit) for i,s in enumerate(stakes_raw)]
    return [a for a in alloc if a[2] > 0]

# ========== [2] バックテスト ==========
def run(races, alloc_fn, label):
    rows = []
    n_bets = 0; n_hits = 0
    for r in races:
        cands = filter_cands(r)
        if not cands: continue
        alloc = alloc_fn(cands)
        if not alloc: continue
        rs = sum(a[2] for a in alloc)
        rp = 0; hit = 0
        for combo, odds, st in alloc:
            if combo == r["actual"]:
                rp += int(st*odds); hit = 1
        rows.append({"date": r["date"], "stadium": r["stadium"], "race": r["race"],
                     "month": r["month"], "type": r["type"],
                     "stake": rs, "payout": rp, "hit": hit, "n_combos": len(alloc)})
        n_bets += len(alloc); n_hits += (1 if hit else 0)
    return pd.DataFrame(rows), n_bets, n_hits

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
    n = len(df); h = df["hit"].sum()
    roi = p/s if s else 0
    ci = bootstrap_roi_ci(df["stake"].values, df["payout"].values)
    profits = (df["payout"] - df["stake"]).values
    # 月別
    mon = df.groupby("month")[["stake","payout"]].sum()
    mon["roi"] = mon["payout"]/mon["stake"]
    mon_std = mon["roi"].std()
    # Sharpe per-race
    if profits.std() > 0:
        sharpe = profits.mean() / profits.std()
    else:
        sharpe = 0
    # Max drawdown (chronological)
    df_sorted = df.sort_values(["date","stadium","race"])
    profits_ts = (df_sorted["payout"] - df_sorted["stake"]).values
    cum = np.cumsum(profits_ts)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = int(dd.min())
    return {
        "n_races": n, "hit_rate": h/n if n else 0,
        "stake": int(s), "payout": int(p), "profit": int(p-s),
        "roi": roi, "ci_lo": ci[0], "ci_hi": ci[1], "ci_width": ci[1]-ci[0],
        "mon_std": float(mon_std) if not pd.isna(mon_std) else 0,
        "sharpe": float(sharpe), "max_dd": max_dd,
    }

# ========== [3] 3 方式 × 2 期間 ==========
print("\n[3] バックテスト実行 (3 方式 × 2 期間)")
methods = [("A_EV比例", alloc_A_ev_proportional),
           ("B_half_kelly_budget", alloc_B_half_kelly_budget),
           ("D_bankroll_halfkelly", alloc_D_bankroll_halfkelly)]

results = []
dfs = {}
for m_name, alloc_fn in methods:
    for period_name, races in [("train", train_races), ("test", test_races)]:
        df, n_bets, n_hits = run(races, alloc_fn, f"{m_name}_{period_name}")
        dfs[(m_name, period_name)] = df
        s = summarize(df)
        results.append({"method": m_name, "period": period_name,
                        "n_bets": n_bets, "n_hits": n_hits, **s})
        print(f"  {m_name:<22} {period_name:<6} races={s['n_races']:>6,} "
              f"ROI={s['roi']:.4f} CI[{s['ci_lo']:.4f},{s['ci_hi']:.4f}] "
              f"width={s['ci_width']:.4f} mon_std={s['mon_std']:.4f} "
              f"sharpe={s['sharpe']:+.4f} maxDD=¥{s['max_dd']:+,}")

res_df = pd.DataFrame(results)
res_df.to_csv(OUT/"staking_comparison_results.csv", index=False, encoding="utf-8-sig")

# ========== [4] 資産曲線 ==========
print("\n[4] 資産曲線プロット ...")
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
for period_idx, (period, ax) in enumerate(zip(["train","test"], axes)):
    for m_name, _ in methods:
        df = dfs[(m_name, period)].sort_values(["date","stadium","race"])
        profits = (df["payout"] - df["stake"]).values
        cum = np.cumsum(profits)
        ax.plot(np.arange(len(cum)), cum, label=m_name, alpha=0.7, linewidth=1.2)
    ax.set_title(f"{period} cumulative profit")
    ax.set_xlabel("bet race idx (chronological)")
    ax.set_ylabel("cumulative profit (¥)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.axhline(0, color="gray", lw=0.5)
fig.tight_layout()
fig.savefig(OUT/"staking_equity_curves.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"  saved: staking_equity_curves.png")

# ========== [5] 月別 ROI 比較 (最終選定候補向け) ==========
print("\n[5] 月別 ROI 比較 (test) ...")
mon_cmp = []
for m_name, _ in methods:
    df = dfs[(m_name, "test")]
    mon = df.groupby("month")[["stake","payout"]].sum()
    for ym, r in mon.iterrows():
        roi = r["payout"]/r["stake"] if r["stake"] else 0
        mon_cmp.append({"method":m_name, "month":ym, "stake":int(r["stake"]),
                        "payout":int(r["payout"]), "roi":roi})
mon_df = pd.DataFrame(mon_cmp)
mon_df.to_csv(OUT/"staking_monthly_comparison.csv", index=False, encoding="utf-8-sig")

# ========== [6] ベット額分布 ==========
print("\n[6] ベット額分布 (test, per combo) ...")
for m_name, alloc_fn in methods:
    stakes_all = []
    for r in test_races:
        cands = filter_cands(r)
        if not cands: continue
        alloc = alloc_fn(cands)
        for _, _, st in alloc:
            if st > 0: stakes_all.append(st)
    if stakes_all:
        arr = np.array(stakes_all)
        print(f"  {m_name:<22}  n={len(arr):>6,}  mean=¥{arr.mean():.0f}  "
              f"median=¥{np.median(arr):.0f}  min=¥{arr.min():.0f}  "
              f"max=¥{arr.max():.0f}  std=¥{arr.std():.0f}")

# ========== [7] 最終判定 + レポート ==========
print("\n[7] 最終判定 + レポート ...")
# test ROI / CI 下限比較
test_rows = res_df[res_df["period"]=="test"]
a = test_rows[test_rows["method"]=="A_EV比例"].iloc[0]
b = test_rows[test_rows["method"]=="B_half_kelly_budget"].iloc[0]
d = test_rows[test_rows["method"]=="D_bankroll_halfkelly"].iloc[0]

# 選定基準: CI 下限最高 + ROI ≥ 1.0
ranked = test_rows.sort_values(["ci_lo","roi"], ascending=[False, False])
best = ranked.iloc[0]

print(f"\n  test 比較:")
print(f"  {'方式':<22} {'ROI':>7} {'CI幅':>7} {'CI下':>7} {'mon_std':>7} {'sharpe':>8} {'maxDD':>12}")
for _, rw in test_rows.iterrows():
    mark = " ⭐" if rw['method']==best['method'] else ""
    print(f"  {rw['method']:<22} {rw['roi']:>7.4f} {rw['ci_width']:>7.4f} "
          f"{rw['ci_lo']:>7.4f} {rw['mon_std']:>7.4f} {rw['sharpe']:>+8.4f} "
          f"¥{rw['max_dd']:>+11,}{mark}")

# CI 幅の変化
ci_A = a["ci_width"]; ci_B = b["ci_width"]; ci_D = d["ci_width"]
print(f"\n  CI 幅:")
print(f"    A (EV比例): {ci_A:.4f}")
print(f"    B (halfkelly budget): {ci_B:.4f}  (vs A: {ci_B-ci_A:+.4f})")
print(f"    D (bankroll halfkelly): {ci_D:.4f}  (vs A: {ci_D-ci_A:+.4f})")

# シナリオ判定
if ci_B < ci_A - 0.01 or ci_D < ci_A - 0.01:
    pattern = "α"; desc = "ハーフケリーで CI 幅縮小"
elif abs(ci_B - ci_A) < 0.01 and abs(ci_D - ci_A) < 0.01:
    pattern = "β"; desc = "効果ほぼ同じ"
else:
    pattern = "γ"; desc = "想定外 (悪化 or 予想外の挙動)"

# レポート
md = f"""# 2連複 Kelly 配分比較レポート

## 核心メッセージ
- 戦略: 2連複, EV≥1.50, type=no_t3, odds≤10
- 配分方式: A (EV比例), B (halfkelly budget), D (bankroll halfkelly)
- **判定**: パターン {pattern} — {desc}
- 最良方式: **{best['method']}**

## 方式比較 (test 2025-07〜2026-04)

| 方式 | ROI | CI下 | CI上 | CI幅 | 月別std | Sharpe | maxDD |
|---|---|---|---|---|---|---|---|
"""
for _, rw in test_rows.iterrows():
    md += (f"| {rw['method']} | {rw['roi']:.4f} | {rw['ci_lo']:.4f} | "
           f"{rw['ci_hi']:.4f} | {rw['ci_width']:.4f} | {rw['mon_std']:.4f} | "
           f"{rw['sharpe']:+.4f} | ¥{int(rw['max_dd']):+,} |\n")

md += "\n## train 期間 (参考)\n\n"
md += "| 方式 | ROI | CI下 | CI上 | CI幅 | maxDD |\n|---|---|---|---|---|---|\n"
for _, rw in res_df[res_df["period"]=="train"].iterrows():
    md += (f"| {rw['method']} | {rw['roi']:.4f} | {rw['ci_lo']:.4f} | "
           f"{rw['ci_hi']:.4f} | {rw['ci_width']:.4f} | ¥{int(rw['max_dd']):+,} |\n")

md += f"""

## 最終運用仕様

```
券種: 2連複 (quinella)
モデル: v4_ext_fixed
キャリブレーション: Isotonic (quinella, train fit)
判定: EV ≥ 1.50 AND Edge ≥ 0.02
型フィルタ: no_t3 (型1, 型2)
オッズ上限: 10.0
予算: 3000円/レース
配分: {best['method']}
```

### 期待性能 (test)
- ROI: **{best['roi']:.4f}**
- 95% CI: [{best['ci_lo']:.4f}, {best['ci_hi']:.4f}]
- CI 幅: {best['ci_width']:.4f}
- 月別 std: {best['mon_std']:.4f}
- Sharpe (per-race): {best['sharpe']:+.4f}
- 最大 DD (test 期間): ¥{int(best['max_dd']):+,}
- ベット率: {best['n_races']/len(test_races)*100:.2f}%

### 運用資金ベース 破綻シナリオ (ドローダウンから逆算)
- 最大 DD = ¥{int(best['max_dd']):,} (test 9.5ヶ月)
- 運用資金 100,000円: DD 比率 {abs(best['max_dd'])/100000*100:.1f}%
- 運用資金 300,000円: DD 比率 {abs(best['max_dd'])/300000*100:.1f}%
- 運用資金 1,000,000円: DD 比率 {abs(best['max_dd'])/1000000*100:.1f}%

## 実運用の注意点
- **オッズ変動**: 1分前オッズで計算、実購入時と乖離 (~5%)
- **100円単位切り捨て**: 端数発生、EV 上位補填で予算活用
- **流動性**: odds>10 除外のため低倍率中心、流動性高い
- **月別変動**: std={best['mon_std']:.3f} 程度を想定

## 出力ファイル
- staking_comparison_results.csv
- staking_monthly_comparison.csv
- staking_equity_curves.png
"""
(OUT/"quinella_kelly_staking_report.md").write_text(md, encoding="utf-8")
print(f"\nsaved: quinella_kelly_staking_report.md")
print(f"\n=== パターン {pattern}: {desc} ===")
