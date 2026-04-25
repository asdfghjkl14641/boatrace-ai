# -*- coding: utf-8 -*-
"""型2 内部分解 + 閾値案シミュレーション."""
from __future__ import annotations
import io, sys, math, runpy, warnings
from pathlib import Path
from itertools import permutations
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "notebooks" / "output"
sys.path.insert(0, str(BASE))
sys.stdout.reconfigure(encoding="utf-8")

PERMS = [(a, b, c) for a, b, c in permutations([1,2,3,4,5,6], 3)]

def binomial_ci(hits, total):
    if total == 0: return (0.0, 0.0)
    p = hits/total
    se = math.sqrt(p*(1-p)/total); z = 1.96
    return (max(0, p-z*se), min(1, p+z*se))

print("=" * 80); print("型2 内部分解 + 閾値再設計"); print("=" * 80)

# ========== [0] データロード ==========
print("\n[0] データロード ...")
_b = io.StringIO(); _o = sys.stdout; sys.stdout = _b
try: ns = runpy.run_path(str(BASE / "notebooks" / "pl_v4_training.py"))
finally: sys.stdout = _o
beta_v4 = ns["beta_v4"]
X_te = ns["X_test_v4"]; keys_te = ns["keys_test_v4"].reset_index(drop=True)
N = len(keys_te)

# 既存 calibrated backtest per-race
bt = pd.read_csv(OUT/"calibrated_backtest_per_race.csv")
bt["date"] = pd.to_datetime(bt["date"]).dt.date
print(f"   backtest rows: {len(bt):,}, v4 keys: {N:,}")

# race_results for top1 hit
from scripts.db import get_connection
conn = get_connection()
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

# ========== [1] G_S, O_S, top1_lane 計算 ==========
print("\n[1] 指数計算 ...")
rows = []
for i in range(N):
    k = keys_te.iloc[i]
    d = pd.Timestamp(k["date"]).date(); s=int(k["stadium"]); r=int(k["race_number"])
    S = X_te[i] @ beta_v4
    std_S = max(S.std(ddof=0), 0.3)
    s_sorted = np.sort(S)[::-1]
    G_S = (s_sorted[0] - s_sorted[1]) / std_S
    O_S = (S[3:6].max() - S.mean()) / std_S
    top1 = int(S.argmax())
    actual = res_map.get((d,s,r))
    top1_hit = int(actual is not None and actual[0] == top1 + 1)
    rows.append({"date":d,"stadium":s,"race_number":r,
                 "G_S":G_S,"O_S":O_S,"top1_lane":top1,"top1_hit":top1_hit,
                 "has_result": actual is not None})
idx_df = pd.DataFrame(rows)

# merge with backtest
bt_m = bt.merge(idx_df, on=["date","stadium","race_number"], how="left")
print(f"   merged: {len(bt_m):,}")

# ========== [Step 1] 型2 セル分解 ==========
print("\n[Step 1] 型2 内部 G_S × O_S マトリクス ...")
t2 = bt_m[bt_m["type_d"] == "型2_イン残りヒモ荒れ"].copy()
print(f"   型2 レース数 (全): {len(t2):,}  (bet: {(t2['verdict']=='bet').sum():,})")

gs_bins = [(0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, float("inf"))]
os_bins = [(0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, float("inf"))]

cell_rows = []
for gs_lo, gs_hi in gs_bins:
    for os_lo, os_hi in os_bins:
        mask = ((t2["G_S"] >= gs_lo) & (t2["G_S"] < gs_hi) &
                (t2["O_S"] >= os_lo) & (t2["O_S"] < os_hi))
        cell = t2[mask]
        bet_cell = cell[cell["verdict"] == "bet"]
        n_race = len(cell); n_bet_race = len(bet_cell)
        n_bet = int(bet_cell["n_cands"].sum())
        n_hit = int(bet_cell["hit"].sum())
        stake = int(bet_cell["total_stake"].sum())
        pay = int(bet_cell["total_payout"].sum())
        roi = pay/stake if stake else 0
        top1_h = int(cell["top1_hit"].sum()); top1_n = int(cell["has_result"].sum())
        top1_r = top1_h/top1_n if top1_n else 0
        ci = binomial_ci(n_hit, n_bet_race)
        cell_rows.append({
            "gs_lo": gs_lo, "gs_hi": gs_hi, "os_lo": os_lo, "os_hi": os_hi,
            "n_race": n_race, "n_bet_race": n_bet_race, "n_bet": n_bet,
            "n_hit": n_hit, "stake": stake, "payout": pay,
            "profit": pay-stake, "roi": roi,
            "hit_rate_ci_lo": ci[0], "hit_rate_ci_hi": ci[1],
            "top1_hit": top1_h, "top1_n": top1_n, "top1_rate": top1_r,
        })
cell_df = pd.DataFrame(cell_rows)
cell_df.to_csv(OUT/"type2_cell_breakdown.csv", index=False, encoding="utf-8-sig")
print(f"   saved: type2_cell_breakdown.csv")

# ヒートマップ
print("\n   ROI matrix (cell ROI / n_race):")
print(f"   {'G_S ＼ O_S':<18}", end="")
for os_lo, os_hi in os_bins:
    label = f"[{os_lo:.1f},{os_hi:.1f})" if os_hi != float("inf") else f"[{os_lo:.1f},∞)"
    print(f"{label:>14}", end="")
print()
for gs_lo, gs_hi in gs_bins:
    gs_label = f"[{gs_lo:.1f},{gs_hi:.1f})" if gs_hi != float("inf") else f"[{gs_lo:.1f},∞)"
    print(f"   {gs_label:<18}", end="")
    for os_lo, os_hi in os_bins:
        c = cell_df[(cell_df["gs_lo"]==gs_lo)&(cell_df["os_lo"]==os_lo)].iloc[0]
        print(f"   {c['roi']:>5.3f}/{int(c['n_race']):>5}", end="")
    print()

# heatmap plot
roi_matrix = np.zeros((len(gs_bins), len(os_bins)))
n_matrix = np.zeros((len(gs_bins), len(os_bins)), dtype=int)
for i, (gs_lo, gs_hi) in enumerate(gs_bins):
    for j, (os_lo, os_hi) in enumerate(os_bins):
        c = cell_df[(cell_df["gs_lo"]==gs_lo)&(cell_df["os_lo"]==os_lo)].iloc[0]
        roi_matrix[i,j] = c["roi"]; n_matrix[i,j] = int(c["n_race"])

fig, ax = plt.subplots(figsize=(9,7))
# center at 1.0
vmin = max(0, roi_matrix[n_matrix>0].min() - 0.1) if (n_matrix>0).any() else 0
vmax = roi_matrix[n_matrix>0].max() + 0.1 if (n_matrix>0).any() else 1
im = ax.imshow(roi_matrix, cmap="RdBu_r", vmin=0.5, vmax=1.5, aspect="auto")
for i in range(len(gs_bins)):
    for j in range(len(os_bins)):
        if n_matrix[i,j] > 0:
            txt = f"{roi_matrix[i,j]:.3f}\nN={n_matrix[i,j]}"
            color = "white" if abs(roi_matrix[i,j]-1.0) > 0.3 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)
        else:
            ax.text(j, i, "—", ha="center", va="center", fontsize=9)
ax.set_xticks(range(len(os_bins)))
ax.set_xticklabels([f"[{lo:.1f},{hi:.1f})" if hi != float("inf") else f"[{lo:.1f},∞)" for lo,hi in os_bins])
ax.set_yticks(range(len(gs_bins)))
ax.set_yticklabels([f"[{lo:.1f},{hi:.1f})" if hi != float("inf") else f"[{lo:.1f},∞)" for lo,hi in gs_bins])
ax.set_xlabel("O_S"); ax.set_ylabel("G_S")
ax.set_title("型2 ROI heatmap (cell value = ROI, N=race count)")
fig.colorbar(im, ax=ax, label="ROI")
fig.tight_layout()
fig.savefig(OUT/"type2_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"   saved: type2_heatmap.png")

# ========== [Step 2] 型2 vs 型4 (top1=0) 境界分析 ==========
print("\n[Step 2] 型2 vs 型4 (top1=0) 境界 ...")
t4_top0 = bt_m[(bt_m["type_d"] == "型4_ノイズ") & (bt_m["top1_lane"] == 0)].copy()
print(f"   型4 で 1号艇トップ レース数: {len(t4_top0):,}")

# 型2 中核: G_S >= 0.8 AND O_S >= 0.3
t2_core = t2[(t2["G_S"] >= 0.8) & (t2["O_S"] >= 0.3)]
# 型2 グレー: 中核以外 (条件: [0.6-0.8) AND O_S any in 型2) or (G_S any in 型2 AND O_S [0.2-0.3))
t2_gray = t2[~((t2["G_S"] >= 0.8) & (t2["O_S"] >= 0.3))]

print(f"\n   境界比較:")
print(f"   {'領域':<40} {'N_race':>8} {'top1 hit':>12} {'top1 rate':>10}")
for label, sub in [("型2 中核 (G_S>=0.8 AND O_S>=0.3)", t2_core),
                    ("型2 グレー (それ以外の型2)", t2_gray),
                    ("型4 で 1号艇トップ", t4_top0)]:
    n = int(sub["has_result"].sum())
    h = int(sub["top1_hit"].sum())
    r = h/n if n else 0
    print(f"   {label:<40} {len(sub):>8,} {h:>7,}/{n:<5,} {r*100:>9.2f}%")

# ========== [Step 3-4] 閾値案シミュレーション ==========
print("\n[Step 3-4] 閾値案シミュレーション ...")
# 型1 + 型3 bets 固定
fixed = bt_m[bt_m["type_d"].isin(["型1_逃げ本命","型3_頭荒れ"])]
fixed_bet = fixed[fixed["verdict"] == "bet"]
fix_stake = int(fixed_bet["total_stake"].sum()); fix_pay = int(fixed_bet["total_payout"].sum())
fix_bet_n = int(fixed_bet["n_cands"].sum()); fix_hit = int(fixed_bet["hit"].sum())
fix_bet_races = len(fixed_bet); fix_hit_races = int(fixed_bet["hit"].sum())

# bt_m全体の中で backtest 対象外 (型1, 型3, 型4 全部, 型2 の bet 以外) の stake は 0
# 型2 だけを条件で絞る
strategies = [
    ("現状", lambda r: (r["G_S"]>0.6) & (r["O_S"]>0.2)),
    ("案 A", lambda r: (r["G_S"]>0.8) & (r["O_S"]>0.2)),
    ("案 B", lambda r: (r["G_S"]>0.6) & (r["O_S"]>0.3)),
    ("案 C", lambda r: (r["G_S"]>0.8) & (r["O_S"]>0.3)),
    ("案 D", lambda r: (r["G_S"]>0.9) & (r["O_S"]>0.4)),
]

sim_rows = []
for label, cond in strategies:
    t2_keep = t2[cond(t2)]
    t2_bet = t2_keep[t2_keep["verdict"] == "bet"]
    t2_stake = int(t2_bet["total_stake"].sum())
    t2_pay = int(t2_bet["total_payout"].sum())
    t2_bet_n = int(t2_bet["n_cands"].sum())
    t2_hit = int(t2_bet["hit"].sum())
    t2_bet_races = len(t2_bet); t2_hit_races = int(t2_bet["hit"].sum())
    t2_roi = t2_pay/t2_stake if t2_stake else 0
    t2_ci = binomial_ci(t2_hit_races, t2_bet_races)

    total_stake = fix_stake + t2_stake
    total_pay = fix_pay + t2_pay
    total_bet_races = fix_bet_races + t2_bet_races
    total_hit_races = fix_hit_races + t2_hit_races
    total_bet_n = fix_bet_n + t2_bet_n
    roi = total_pay/total_stake if total_stake else 0
    ci_overall = binomial_ci(total_hit_races, total_bet_races)
    profit = total_pay - total_stake

    # 型1・型3 は同じなので個別 ROI
    t1_sub = fixed_bet[fixed_bet["type_d"] == "型1_逃げ本命"]
    t3_sub = fixed_bet[fixed_bet["type_d"] == "型3_頭荒れ"]
    t1_roi = (t1_sub["total_payout"].sum()/t1_sub["total_stake"].sum()) if t1_sub["total_stake"].sum() else 0
    t3_roi = (t3_sub["total_payout"].sum()/t3_sub["total_stake"].sum()) if t3_sub["total_stake"].sum() else 0

    sim_rows.append({
        "strategy": label,
        "t2_bet_races": t2_bet_races, "t2_stake": t2_stake,
        "t2_payout": t2_pay, "t2_roi": t2_roi,
        "t2_ci_lo": t2_ci[0], "t2_ci_hi": t2_ci[1],
        "t1_roi": t1_roi, "t3_roi": t3_roi,
        "total_bet_races": total_bet_races, "total_bet_n": total_bet_n,
        "total_stake": total_stake, "total_payout": total_pay,
        "profit": profit, "roi": roi,
        "hit_rate_ci_lo": ci_overall[0], "hit_rate_ci_hi": ci_overall[1],
    })

sim_df = pd.DataFrame(sim_rows)
sim_df.to_csv(OUT/"type2_threshold_comparison.csv", index=False, encoding="utf-8-sig")

print(f"\n  {'戦略':<8} {'型2 bets':>10} {'型2 ROI':>8} {'型1 ROI':>8} "
      f"{'型3 ROI':>8} {'全 bets':>8} {'全 ROI':>8} {'profit':>12}")
for _, rw in sim_df.iterrows():
    print(f"  {rw['strategy']:<8} {rw['t2_bet_races']:>10,} {rw['t2_roi']:>8.4f} "
          f"{rw['t1_roi']:>8.4f} {rw['t3_roi']:>8.4f} {rw['total_bet_races']:>8,} "
          f"{rw['roi']:>8.4f} {int(rw['profit']):>+12,}")

# ========== [Step 5] 判定 ==========
print("\n[Step 5] 最終判定 ...")
curr_roi = float(sim_df[sim_df["strategy"]=="現状"]["roi"].iloc[0])
best_idx = sim_df["roi"].idxmax()
best_label = sim_df.loc[best_idx, "strategy"]
best_roi = sim_df.loc[best_idx, "roi"]
improvement = (best_roi - curr_roi) * 100

if best_roi >= 1.0:
    verdict = "α"; desc = "閾値引き上げで ROI ≥ 1.0 達成 — 運用候補"
elif improvement >= 3.0 and best_roi >= 0.97:
    verdict = "β"; desc = "改善するが ROI < 1.0 — 型1/型3 側の調整も検討"
elif abs(improvement) < 2.0:
    verdict = "γ"; desc = "閾値調整では大差なし — 型2 廃止 (型1+型3 のみ) を検討"
elif improvement < 0:
    verdict = "δ"; desc = "現閾値が最適 — 他方向性検討"
else:
    verdict = "β"; desc = "改善幅小 — 次の設計へ"

# 型2 廃止の場合の ROI
t2_only_roi = fix_pay/fix_stake if fix_stake else 0
t2_only_profit = fix_pay - fix_stake
t2_only_races = fix_bet_races

# ========== [Step 6] レポート ==========
md = f"""# 型2 内部分解 + 閾値再設計レポート

## 背景
キャリブレーション+EV≥1.2 で:
- 型1 ROI: 0.9974
- 型2 ROI: 0.8510 (赤字主因)
- 型3 ROI: 1.1119

型2 の現行定義「G_S > 0.6 AND O_S > 0.2」を内部分解して再設計可能性を検証.

## Step 1: 型2 ROI ヒートマップ (G_S × O_S 4×4)

セル値 = ROI / N_race

| G_S ＼ O_S | [0.2,0.3) | [0.3,0.4) | [0.4,0.5) | [0.5,∞) |
|---|---|---|---|---|
"""
for gs_lo, gs_hi in gs_bins:
    gs_label = f"[{gs_lo:.1f},{gs_hi:.1f})" if gs_hi != float("inf") else f"[{gs_lo:.1f},∞)"
    md += f"| {gs_label} "
    for os_lo, os_hi in os_bins:
        c = cell_df[(cell_df["gs_lo"]==gs_lo)&(cell_df["os_lo"]==os_lo)].iloc[0]
        if c["n_race"] > 0:
            md += f"| **{c['roi']:.3f}** ({int(c['n_race'])}) "
        else:
            md += "| — "
    md += "|\n"

md += f"""

![heatmap](type2_heatmap.png)

## Step 2: 型2 vs 型4 (top1=0) 境界

| 領域 | N_race | top1 hit | top1 rate |
|---|---|---|---|
"""
for label, sub in [("型2 中核 (G_S≥0.8 AND O_S≥0.3)", t2_core),
                    ("型2 グレー (それ以外の型2)", t2_gray),
                    ("型4 で 1号艇トップ", t4_top0)]:
    n = int(sub["has_result"].sum()); h = int(sub["top1_hit"].sum())
    r = h/n if n else 0
    md += f"| {label} | {len(sub):,} | {h:,}/{n:,} | {r*100:.2f}% |\n"

md += f"""

## Step 3-4: 閾値案シミュレーション

| 戦略 | 型2 条件 | 型2 bets | 型2 ROI | 型1 ROI | 型3 ROI | 全 bets | **全 ROI** | profit |
|---|---|---|---|---|---|---|---|---|
"""
conditions_display = {
    "現状": "G_S>0.6 AND O_S>0.2",
    "案 A": "G_S>0.8 AND O_S>0.2",
    "案 B": "G_S>0.6 AND O_S>0.3",
    "案 C": "G_S>0.8 AND O_S>0.3",
    "案 D": "G_S>0.9 AND O_S>0.4",
}
for _, rw in sim_df.iterrows():
    md += (f"| {rw['strategy']} | {conditions_display[rw['strategy']]} | "
           f"{int(rw['t2_bet_races']):,} | {rw['t2_roi']:.4f} | "
           f"{rw['t1_roi']:.4f} | {rw['t3_roi']:.4f} | "
           f"{int(rw['total_bet_races']):,} | **{rw['roi']:.4f}** | "
           f"{int(rw['profit']):+,} |\n")

md += f"""

### 参考: 型2 廃止 (型1 + 型3 のみ)
- N_bet races: {t2_only_races:,}
- stake: ¥{fix_stake:,}
- payout: ¥{fix_pay:,}
- profit: ¥{t2_only_profit:+,}
- **ROI: {t2_only_roi:.4f}**

## 最終判定: パターン {verdict}

**{desc}**

- 現状 ROI: {curr_roi:.4f}
- 最良 ({best_label}) ROI: {best_roi:.4f} ({improvement:+.2f}pt)

### 次アクション提案
"""
if verdict == "α":
    md += f"""
1. 新閾値 ({best_label}: {conditions_display[best_label]}) を採用
2. 運用候補戦略確定
3. 型1 もさらに絞り込めるか分析
"""
elif verdict == "β":
    md += f"""
1. {best_label} 採用で +{improvement:.2f}pt 改善
2. 型1 側の絞り込み検討 (型1 ROI=0.9974 → 更に上を狙えるか)
3. 型3 の CI 再確認
4. それでも ROI<1.0 なら他券種へ
"""
elif verdict == "γ":
    md += f"""
1. **型2 廃止** (型1+型3 のみ) で ROI={t2_only_roi:.4f}
2. 型2 は構造的に識別困難 — 定義変えても効果薄い
3. 次: 型1+型3 限定運用、または他券種 (2連単・2連複) へ
"""
else:
    md += """
1. 現閾値維持
2. 他の方向性 (特徴量追加、モデル再学習) 検討
"""

md += """

## 出力ファイル
- `type2_cell_breakdown.csv`
- `type2_heatmap.png`
- `type2_threshold_comparison.csv`
"""

with open(OUT/"type2_redesign_report.md", "w", encoding="utf-8") as f:
    f.write(md)

print(f"\n   saved: type2_redesign_report.md")
print(f"\n=== 判定: パターン {verdict} (最良={best_label}, ROI={best_roi:.4f}, 改善{improvement:+.2f}pt) ===")
print(f"参考: 型2 廃止 → ROI {t2_only_roi:.4f} (profit {t2_only_profit:+,})")
print("完了")
