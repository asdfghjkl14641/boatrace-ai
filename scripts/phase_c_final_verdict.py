# -*- coding: utf-8 -*-
"""Phase C 最終確定: 全型 閾値再設計 + train 汎化検証.

Step 1-4: test データ (calibrated_backtest_per_race.csv) から集計.
Step 5: train 期間で最適戦略を新規バックテスト (汎化検証).
Step 6-7: 最終戦略仕様 + レポート.
"""
from __future__ import annotations
import io, sys, math, pickle, runpy, time, warnings
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

TAU = 0.8
EV_THR = 1.20; EDGE_THR = 0.02
ALPHA_FALLBACK = 1.333
MIN_COMBOS_FOR_Z = 100
BUDGET = 3000; MIN_UNIT = 100
TYPE_RELIABILITY = {
    "型1": 1.0, "型2": 0.9, "型3": 0.8, "型4": 0.5,
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

def compute_indices(S):
    std_S = max(S.std(ddof=0), 0.3)
    s_sorted = np.sort(S)[::-1]
    return {
        "G_S": (s_sorted[0] - s_sorted[1]) / std_S,
        "O_S": (S[3:6].max() - S.mean()) / std_S,
        "top1_lane": int(S.argmax()),
    }

def make_classifier(t1_gs, t1_os, t2_gs, t2_os, t3_gs, t3_os):
    """閾値セットから classifier を返す."""
    def cls(idx):
        if idx["top1_lane"] == 0:
            if idx["G_S"] > t1_gs and idx["O_S"] < t1_os: return "型1"
            if idx["G_S"] > t2_gs and idx["O_S"] > t2_os: return "型2"
            return "型4"
        else:
            if idx["G_S"] > t3_gs and idx["O_S"] > t3_os: return "型3"
            return "型4"
    return cls

def binomial_ci(hits, total):
    if total == 0: return (0.0, 0.0)
    p = hits/total; se = math.sqrt(p*(1-p)/total); z = 1.96
    return (max(0, p-z*se), min(1, p+z*se))

def bootstrap_roi_ci(stake_arr, pay_arr, n_boot=1000, seed=42):
    """per-race stake, pay から bootstrap で ROI CI."""
    stake_arr = np.asarray(stake_arr, dtype=float)
    pay_arr = np.asarray(pay_arr, dtype=float)
    n = len(stake_arr)
    if n == 0 or stake_arr.sum() == 0: return (0, 0, 0)
    rng = np.random.default_rng(seed)
    rois = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = stake_arr[idx].sum()
        if s > 0:
            rois.append(pay_arr[idx].sum() / s)
    rois = np.array(rois)
    return (float(np.percentile(rois, 2.5)), float(np.percentile(rois, 97.5)),
            float(rois.std()))

print("=" * 80); print("Phase C 最終確定 — 全型 閾値再設計 + train 汎化検証"); print("=" * 80)

# ========== [0] データロード ==========
print("\n[0] データロード ...")
_b = io.StringIO(); _o = sys.stdout; sys.stdout = _b
try: ns = runpy.run_path(str(BASE / "notebooks" / "pl_v4_training.py"))
finally: sys.stdout = _o
beta_v4 = ns["beta_v4"]
X_tr = ns["X_train_v4"]; keys_tr = ns["keys_train_v4"].reset_index(drop=True)
X_te = ns["X_test_v4"];  keys_te = ns["keys_test_v4"].reset_index(drop=True)
N_tr = len(keys_tr); N_te = len(keys_te)

bt = pd.read_csv(OUT/"calibrated_backtest_per_race.csv")
bt["date"] = pd.to_datetime(bt["date"]).dt.date

with open(OUT/"calibration_isotonic_train.pkl", "rb") as f:
    iso = pickle.load(f)

# test 指数
rows_idx = []
for i in range(N_te):
    k = keys_te.iloc[i]
    S = X_te[i] @ beta_v4
    idx = compute_indices(S)
    rows_idx.append({"date": pd.Timestamp(k["date"]).date(),
                     "stadium": int(k["stadium"]),
                     "race_number": int(k["race_number"]),
                     **idx})
bt_m = bt.merge(pd.DataFrame(rows_idx), on=["date","stadium","race_number"], how="left")
bt_m["month"] = pd.to_datetime(bt_m["date"]).dt.to_period("M").astype(str)
print(f"   test merged: {len(bt_m):,}, train: {N_tr:,}")

# ========== [Step 1] case δ (G>0.9 AND O>0.4) 詳細 ==========
print("\n[Step 1] case δ 全体 CI + 月別 + 型別 ...")
t1_data = bt_m[(bt_m["type_d"]=="型1_逃げ本命") & (bt_m["verdict"]=="bet")]
t3_data = bt_m[(bt_m["type_d"]=="型3_頭荒れ") & (bt_m["verdict"]=="bet")]
t2_data = bt_m[(bt_m["type_d"]=="型2_イン残りヒモ荒れ") & (bt_m["verdict"]=="bet")]
t2_delta = t2_data[(t2_data["G_S"]>0.9) & (t2_data["O_S"]>0.4)]
case_d = pd.concat([t1_data, t2_delta, t3_data]).sort_values(["date","stadium","race_number"])
print(f"   case δ bet races: {len(case_d)} (型1={len(t1_data)}, 型2δ={len(t2_delta)}, 型3={len(t3_data)})")

# 全体 bootstrap
stake_arr = case_d["total_stake"].values
pay_arr = case_d["total_payout"].values
ci_lo, ci_hi, std = bootstrap_roi_ci(stake_arr, pay_arr)
roi_d = pay_arr.sum()/stake_arr.sum()
print(f"   ROI={roi_d:.4f}  bootstrap 95% CI [{ci_lo:.4f}, {ci_hi:.4f}]  std={std:.4f}")

# 月別
print("\n   月別:")
mon_rows = []
for m, sub in case_d.groupby("month"):
    st=sub["total_stake"].sum(); pay=sub["total_payout"].sum()
    roi = pay/st if st else 0
    ci = bootstrap_roi_ci(sub["total_stake"].values, sub["total_payout"].values, n_boot=500)
    mon_rows.append({"month":m,"n_bet":len(sub),"n_hit":int(sub["hit"].sum()),
                     "stake":int(st),"payout":int(pay),"roi":roi,
                     "ci_lo":ci[0],"ci_hi":ci[1]})
    print(f"    {m}: N={len(sub):>4}  hit={int(sub['hit'].sum()):>3}  "
          f"ROI={roi:.4f}  CI[{ci[0]:.3f},{ci[1]:.3f}]")

# 型別 bootstrap
print("\n   型別:")
type_rows = []
for name, sub in [("型1", t1_data), ("型2 (δ)", t2_delta), ("型3", t3_data)]:
    st=sub["total_stake"].sum(); pay=sub["total_payout"].sum()
    roi = pay/st if st else 0
    ci = bootstrap_roi_ci(sub["total_stake"].values, sub["total_payout"].values)
    type_rows.append({"type":name,"n_bet":len(sub),"n_hit":int(sub["hit"].sum()),
                      "stake":int(st),"payout":int(pay),"roi":roi,
                      "ci_lo":ci[0],"ci_hi":ci[1]})
    print(f"    {name}: N={len(sub):>5}  ROI={roi:.4f}  CI[{ci[0]:.3f},{ci[1]:.3f}]")

conf_rows = mon_rows + type_rows
pd.DataFrame(conf_rows).to_csv(OUT/"case_d_confidence_analysis.csv", index=False, encoding="utf-8-sig")

# ROI 分布プロット
rng = np.random.default_rng(42)
boot_rois = []
for _ in range(2000):
    idx = rng.integers(0, len(stake_arr), len(stake_arr))
    s = stake_arr[idx].sum()
    if s > 0: boot_rois.append(pay_arr[idx].sum()/s)
fig, ax = plt.subplots(figsize=(9,5))
ax.hist(boot_rois, bins=50, alpha=0.7, color="#3B82F6")
ax.axvline(roi_d, color="black", lw=2, label=f"ROI={roi_d:.4f}")
ax.axvline(ci_lo, color="red", lw=1, ls="--", label=f"CI[{ci_lo:.4f}, {ci_hi:.4f}]")
ax.axvline(ci_hi, color="red", lw=1, ls="--")
ax.axvline(1.0, color="green", lw=1, label="break-even")
ax.set_xlabel("ROI"); ax.set_ylabel("freq"); ax.set_title("case δ bootstrap ROI (2000 samples)")
ax.legend(); ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT/"case_d_roi_distribution.png", dpi=120, bbox_inches="tight")
plt.close()

# ========== [Step 2] 型1 内部分解 ==========
print("\n[Step 2] 型1 内部分解 ...")
t1_bet = t1_data.copy()
t1_gs_bands = [(1.0,1.2),(1.2,1.5),(1.5,2.0),(2.0,float("inf"))]
t1_os_bands = [(-float("inf"),0.0),(0.0,0.1),(0.1,0.2),(0.2,0.3)]

t1_mat_rows = []
for glo, ghi in t1_gs_bands:
    for olo, ohi in t1_os_bands:
        m = ((t1_bet["G_S"]>=glo) & (t1_bet["G_S"]<ghi) &
             (t1_bet["O_S"]>=olo) & (t1_bet["O_S"]<ohi))
        sub = t1_bet[m]
        st=int(sub["total_stake"].sum());pay=int(sub["total_payout"].sum())
        nh=int(sub["hit"].sum()); roi=pay/st if st else 0
        t1_mat_rows.append({"gs_lo":glo,"gs_hi":ghi,"os_lo":olo,"os_hi":ohi,
                            "n":len(sub),"n_hit":nh,"stake":st,"payout":pay,"roi":roi})
t1_mat = pd.DataFrame(t1_mat_rows)

roi_m = np.zeros((len(t1_gs_bands),len(t1_os_bands)))
n_m = np.zeros((len(t1_gs_bands),len(t1_os_bands)),dtype=int)
for i,(glo,ghi) in enumerate(t1_gs_bands):
    for j,(olo,ohi) in enumerate(t1_os_bands):
        r = t1_mat[(t1_mat["gs_lo"]==glo)&(t1_mat["os_lo"]==olo)].iloc[0]
        roi_m[i,j] = r["roi"]; n_m[i,j] = int(r["n"])

fig, ax = plt.subplots(figsize=(10,7))
im = ax.imshow(roi_m, cmap="RdBu_r", vmin=0.5, vmax=1.5, aspect="auto")
for i in range(len(t1_gs_bands)):
    for j in range(len(t1_os_bands)):
        if n_m[i,j] > 0:
            color = "white" if abs(roi_m[i,j]-1.0) > 0.3 else "black"
            ax.text(j, i, f"{roi_m[i,j]:.3f}\nN={n_m[i,j]}",
                    ha="center", va="center", color=color, fontsize=9)
        else:
            ax.text(j, i, "—", ha="center", va="center", fontsize=9)
ax.set_xticks(range(len(t1_os_bands)))
ax.set_xticklabels([f"[{lo:.1f},{hi:.1f})" if lo!=-float("inf") else f"(-∞,{hi:.1f})"
                    for lo,hi in t1_os_bands])
ax.set_yticks(range(len(t1_gs_bands)))
ax.set_yticklabels([f"[{lo:.1f},{hi:.1f})" if hi!=float("inf") else f"[{lo:.1f},∞)"
                    for lo,hi in t1_gs_bands])
ax.set_xlabel("O_S"); ax.set_ylabel("G_S")
ax.set_title("型1 ROI heatmap (calibrated+EV>=1.2 bet races)")
fig.colorbar(im, ax=ax, label="ROI")
fig.tight_layout(); fig.savefig(OUT/"type1_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()
t1_mat.to_csv(OUT/"type1_matrix.csv", index=False, encoding="utf-8-sig")

# 型1 閾値案 (全案とも top1_lane=0 前提)
t1_cases = [
    ("現状", 1.0, 0.3),
    ("P", 1.2, 0.3),
    ("Q", 1.0, 0.2),
    ("R", 1.2, 0.2),
    ("S", 1.5, 0.2),
]
t1_sim = []
for lbl, g, o in t1_cases:
    sub = t1_bet[(t1_bet["G_S"]>g) & (t1_bet["O_S"]<o)]
    st=int(sub["total_stake"].sum());pay=int(sub["total_payout"].sum())
    nh=int(sub["hit"].sum()); roi=pay/st if st else 0
    ci = bootstrap_roi_ci(sub["total_stake"].values, sub["total_payout"].values)
    t1_sim.append({"case":lbl,"g_thr":g,"o_thr":o,"n":len(sub),"n_hit":nh,
                   "stake":st,"payout":pay,"roi":roi,
                   "ci_lo":ci[0],"ci_hi":ci[1]})
pd.DataFrame(t1_sim).to_csv(OUT/"type1_threshold_scenarios.csv", index=False, encoding="utf-8-sig")
print(f"  {'案':<6} {'条件':<22} {'N':>5} {'ROI':>8} {'CI下':>7} {'CI上':>7}")
for r in t1_sim:
    print(f"  {r['case']:<6} G>{r['g_thr']:.1f} AND O<{r['o_thr']:.1f} {'':<6} {r['n']:>5,} "
          f"{r['roi']:>8.4f} {r['ci_lo']:>7.4f} {r['ci_hi']:>7.4f}")

# ========== [Step 3] 型3 内部分解 ==========
print("\n[Step 3] 型3 内部分解 ...")
t3_bet = t3_data.copy()
t3_os_bands = [(0.3,0.5),(0.5,0.7),(0.7,1.0),(1.0,float("inf"))]
t3_gs_bands = [(0.4,0.6),(0.6,0.8),(0.8,1.0),(1.0,float("inf"))]

t3_mat_rows = []
for olo,ohi in t3_os_bands:
    for glo,ghi in t3_gs_bands:
        m = ((t3_bet["O_S"]>=olo) & (t3_bet["O_S"]<ohi) &
             (t3_bet["G_S"]>=glo) & (t3_bet["G_S"]<ghi))
        sub = t3_bet[m]
        st=int(sub["total_stake"].sum());pay=int(sub["total_payout"].sum())
        nh=int(sub["hit"].sum()); roi=pay/st if st else 0
        t3_mat_rows.append({"os_lo":olo,"os_hi":ohi,"gs_lo":glo,"gs_hi":ghi,
                            "n":len(sub),"n_hit":nh,"stake":st,"payout":pay,"roi":roi})
t3_mat = pd.DataFrame(t3_mat_rows)
t3_mat.to_csv(OUT/"type3_matrix.csv", index=False, encoding="utf-8-sig")

roi_m3 = np.zeros((len(t3_os_bands),len(t3_gs_bands)))
n_m3 = np.zeros((len(t3_os_bands),len(t3_gs_bands)),dtype=int)
for i,(olo,ohi) in enumerate(t3_os_bands):
    for j,(glo,ghi) in enumerate(t3_gs_bands):
        r = t3_mat[(t3_mat["os_lo"]==olo)&(t3_mat["gs_lo"]==glo)].iloc[0]
        roi_m3[i,j] = r["roi"]; n_m3[i,j] = int(r["n"])

fig, ax = plt.subplots(figsize=(10,7))
im = ax.imshow(roi_m3, cmap="RdBu_r", vmin=0.0, vmax=3.0, aspect="auto")
for i in range(len(t3_os_bands)):
    for j in range(len(t3_gs_bands)):
        if n_m3[i,j] > 0:
            color = "white" if abs(roi_m3[i,j]-1.0) > 0.8 else "black"
            ax.text(j, i, f"{roi_m3[i,j]:.3f}\nN={n_m3[i,j]}",
                    ha="center", va="center", color=color, fontsize=9)
        else:
            ax.text(j, i, "—", ha="center", va="center", fontsize=9)
ax.set_xticks(range(len(t3_gs_bands)))
ax.set_xticklabels([f"[{lo:.1f},{hi:.1f})" if hi!=float("inf") else f"[{lo:.1f},∞)"
                    for lo,hi in t3_gs_bands])
ax.set_yticks(range(len(t3_os_bands)))
ax.set_yticklabels([f"[{lo:.1f},{hi:.1f})" if hi!=float("inf") else f"[{lo:.1f},∞)"
                    for lo,hi in t3_os_bands])
ax.set_xlabel("G_S"); ax.set_ylabel("O_S")
ax.set_title("型3 ROI heatmap (N=128 total, sparse)")
fig.colorbar(im, ax=ax, label="ROI")
fig.tight_layout(); fig.savefig(OUT/"type3_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()

t3_cases = [
    ("現状", 0.3, 0.4),
    ("P", 0.5, 0.4),
    ("Q", 0.3, 0.7),
    ("R", 0.5, 0.7),
]
t3_sim = []
for lbl, o, g in t3_cases:
    sub = t3_bet[(t3_bet["O_S"]>o) & (t3_bet["G_S"]>g)]
    st=int(sub["total_stake"].sum());pay=int(sub["total_payout"].sum())
    nh=int(sub["hit"].sum()); roi=pay/st if st else 0
    ci = bootstrap_roi_ci(sub["total_stake"].values, sub["total_payout"].values) if len(sub)>5 else (0,0,0)
    t3_sim.append({"case":lbl,"o_thr":o,"g_thr":g,"n":len(sub),"n_hit":nh,
                   "stake":st,"payout":pay,"roi":roi,
                   "ci_lo":ci[0],"ci_hi":ci[1]})
pd.DataFrame(t3_sim).to_csv(OUT/"type3_threshold_scenarios.csv", index=False, encoding="utf-8-sig")
print(f"  {'案':<6} {'条件':<22} {'N':>5} {'ROI':>8} {'CI下':>7} {'CI上':>7}")
for r in t3_sim:
    note = " (<50, 参考)" if r['n'] < 50 else ""
    print(f"  {r['case']:<6} O>{r['o_thr']:.1f} AND G>{r['g_thr']:.1f} {'':<6} {r['n']:>5,} "
          f"{r['roi']:>8.4f} {r['ci_lo']:>7.4f} {r['ci_hi']:>7.4f}{note}")

# ========== [Step 4] 組み合わせ最適化 ==========
print("\n[Step 4] 全型組み合わせ最適化 ...")
# 型1 候補: 現状, Q, S (N >= 500)
t1_candidates = [(r["case"], r["g_thr"], r["o_thr"], r["n"]) for r in t1_sim if r["n"] >= 500]
# 型2: δ 固定 (分析済み)
t2_candidates = [("δ", 0.9, 0.4)]
# 型3 候補: 現状 or P (O>0.5, G>0.4) — 大きい N だけ
t3_candidates = [(r["case"], r["o_thr"], r["g_thr"], r["n"]) for r in t3_sim if r["n"] >= 50]
# 「型3 除外」ケースも追加
t3_candidates.append(("除外", None, None, 0))

comb_rows = []
for t1c in t1_candidates:
    t1_name, t1_g, t1_o, t1_n = t1c
    t1_sub = t1_bet[(t1_bet["G_S"]>t1_g) & (t1_bet["O_S"]<t1_o)]
    for t2c in t2_candidates:
        t2_name, t2_g, t2_o = t2c
        t2_sub = t2_data[(t2_data["G_S"]>t2_g) & (t2_data["O_S"]>t2_o)]
        for t3c in t3_candidates:
            t3_name, t3_o, t3_g, t3_n = t3c
            if t3_name == "除外":
                t3_sub = t3_bet.iloc[:0]  # empty
            else:
                t3_sub = t3_bet[(t3_bet["O_S"]>t3_o) & (t3_bet["G_S"]>t3_g)]
            combined = pd.concat([t1_sub, t2_sub, t3_sub])
            st = combined["total_stake"].sum(); pay = combined["total_payout"].sum()
            n = len(combined); n_hit = int(combined["hit"].sum())
            roi = pay/st if st else 0
            ci = bootstrap_roi_ci(combined["total_stake"].values, combined["total_payout"].values, n_boot=500)
            comb_rows.append({
                "strategy": f"T1={t1_name}/T2={t2_name}/T3={t3_name}",
                "t1": t1_name, "t2": t2_name, "t3": t3_name,
                "n_bet": n, "n_hit": n_hit, "stake": int(st), "payout": int(pay),
                "profit": int(pay-st), "roi": roi,
                "ci_lo": ci[0], "ci_hi": ci[1]
            })
comb_df = pd.DataFrame(comb_rows).sort_values("roi", ascending=False)
comb_df.to_csv(OUT/"all_types_optimization.csv", index=False, encoding="utf-8-sig")
print(f"\n  Top 10 組み合わせ (ROI 降順):")
print(f"  {'戦略':<30} {'N':>5} {'ROI':>7} {'CI下':>7} {'CI上':>7} {'profit':>12}")
for _, rw in comb_df.head(10).iterrows():
    print(f"  {rw['strategy']:<30} {int(rw['n_bet']):>5,} {rw['roi']:>7.4f} "
          f"{rw['ci_lo']:>7.4f} {rw['ci_hi']:>7.4f} ¥{int(rw['profit']):>+11,}")

# 最適戦略: ROI 最大で N>=1500
viable = comb_df[comb_df["n_bet"] >= 1500]
if len(viable) == 0: viable = comb_df
best_row = viable.iloc[0]
best_t1 = best_row["t1"]; best_t3 = best_row["t3"]
t1_best_rec = [r for r in t1_sim if r["case"]==best_t1][0]
t3_best_rec = None if best_t3 == "除外" else [r for r in t3_sim if r["case"]==best_t3][0]
print(f"\n  選定: 型1={best_t1} (G>{t1_best_rec['g_thr']:.1f} AND O<{t1_best_rec['o_thr']:.1f}), "
      f"型2=δ (G>0.9 AND O>0.4), 型3={best_t3}"
      + (f" (O>{t3_best_rec['o_thr']:.1f} AND G>{t3_best_rec['g_thr']:.1f})" if t3_best_rec else ""))
print(f"  test ROI={best_row['roi']:.4f}  CI[{best_row['ci_lo']:.4f}, {best_row['ci_hi']:.4f}]")

# ========== [Step 5] train 汎化検証 ==========
print("\n[Step 5] train 期間 汎化検証 ...")
from scripts.db import get_connection
conn = get_connection()
print("   train 期間のオッズ/結果 ロード中 ...")
odds_tr = pd.read_sql_query("""
    SELECT date, stadium, race_number, combination, odds_1min
    FROM trifecta_odds
    WHERE date BETWEEN '2023-05-01' AND '2025-12-31' AND odds_1min IS NOT NULL
""", conn.native)
odds_tr["date"] = pd.to_datetime(odds_tr["date"]).dt.date
res_tr = pd.read_sql_query("""
    SELECT date, stadium, race_number, rank, boat FROM race_results
    WHERE date BETWEEN '2023-05-01' AND '2025-12-31' AND rank BETWEEN 1 AND 3
""", conn.native)
res_tr["date"] = pd.to_datetime(res_tr["date"]).dt.date
conn.close()
print(f"   train odds={len(odds_tr):,}, results={len(res_tr):,}")

res_map_tr = {}
for (d,s,r),g in res_tr.groupby(["date","stadium","race_number"]):
    g2 = g.sort_values("rank")
    if len(g2) < 3: continue
    res_map_tr[(d,s,r)] = (int(g2.iloc[0]["boat"]), int(g2.iloc[1]["boat"]), int(g2.iloc[2]["boat"]))
odds_map_tr = {}
for (d,s,r),g in odds_tr.groupby(["date","stadium","race_number"]):
    book = {}
    for _,row in g.iterrows():
        try:
            a,b,c = map(int, row["combination"].split("-"))
            o = float(row["odds_1min"])
            if o > 0: book[(a,b,c)] = o
        except Exception: pass
    if book: odds_map_tr[(d,s,r)] = book

# classifier with 選定 thresholds
t1_g = t1_best_rec["g_thr"]; t1_o = t1_best_rec["o_thr"]
t2_g = 0.9; t2_o = 0.4  # δ
if best_t3 == "除外":
    def classify(idx):
        if idx["top1_lane"] == 0:
            if idx["G_S"]>t1_g and idx["O_S"]<t1_o: return "型1"
            if idx["G_S"]>t2_g and idx["O_S"]>t2_o: return "型2"
            return "型4"
        return "型4"  # 型3 を見送り
else:
    t3_o_thr = t3_best_rec["o_thr"]; t3_g_thr = t3_best_rec["g_thr"]
    def classify(idx):
        if idx["top1_lane"] == 0:
            if idx["G_S"]>t1_g and idx["O_S"]<t1_o: return "型1"
            if idx["G_S"]>t2_g and idx["O_S"]>t2_o: return "型2"
            return "型4"
        else:
            if idx["O_S"]>t3_o_thr and idx["G_S"]>t3_g_thr: return "型3"
            return "型4"

def run_backtest(X, keys, res_map, odds_map, label):
    N = len(keys)
    total_stake = total_pay = 0
    bet_races = hit_races = 0
    type_stats = {"型1":{"stake":0,"pay":0,"races":0,"hits":0},
                  "型2":{"stake":0,"pay":0,"races":0,"hits":0},
                  "型3":{"stake":0,"pay":0,"races":0,"hits":0}}
    per_race_st = []; per_race_pay = []
    t0 = time.time()
    for i in range(N):
        k = keys.iloc[i]
        d = pd.Timestamp(k["date"]).date(); s=int(k["stadium"]); r=int(k["race_number"])
        if (d,s,r) not in res_map or (d,s,r) not in odds_map: continue
        book = odds_map[(d,s,r)]; actual = res_map[(d,s,r)]
        Z = sum(1.0/o for o in book.values()) if len(book) >= MIN_COMBOS_FOR_Z else ALPHA_FALLBACK
        S = X[i] @ beta_v4
        idx = compute_indices(S)
        t = classify(idx)
        if t == "型4": continue
        rel = TYPE_RELIABILITY[t]
        s_t = S/TAU; s_t = s_t - s_t.max()
        p_lane = np.exp(s_t)/np.exp(s_t).sum()
        probs = pl_probs(p_lane)
        probs_cal = iso.transform(probs.astype(np.float64))
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
        used = sum(a[2] for a in alloc)
        extra = BUDGET - used
        order = sorted(range(len(alloc)), key=lambda ii: -cands[ii][3])
        for idx_a in order:
            if extra < MIN_UNIT: break
            combo, odds, stake = alloc[idx_a]
            alloc[idx_a] = (combo, odds, stake + MIN_UNIT); extra -= MIN_UNIT
        alloc = [a for a in alloc if a[2] > 0]
        if not alloc: continue
        race_stake = sum(a[2] for a in alloc); race_pay = 0; hit_flag = 0
        for combo, odds, stake in alloc:
            if combo == actual:
                race_pay += int(stake * odds); hit_flag = 1
        total_stake += race_stake; total_pay += race_pay
        bet_races += 1; hit_races += hit_flag
        per_race_st.append(race_stake); per_race_pay.append(race_pay)
        TA = type_stats[t]
        TA["stake"] += race_stake; TA["pay"] += race_pay
        TA["races"] += 1; TA["hits"] += hit_flag
        if (i+1) % 10000 == 0:
            print(f"     [{label}] {i+1}/{N}  elapsed {time.time()-t0:.1f}s")
    print(f"   [{label}] done elapsed {time.time()-t0:.1f}s")
    roi = total_pay/total_stake if total_stake else 0
    ci = bootstrap_roi_ci(np.array(per_race_st), np.array(per_race_pay))
    return {"label":label,"N":N,"bet_races":bet_races,"hit_races":hit_races,
            "stake":total_stake,"payout":total_pay,"roi":roi,
            "ci_lo":ci[0],"ci_hi":ci[1],
            "type_stats":type_stats}

train_result = run_backtest(X_tr, keys_tr, res_map_tr, odds_map_tr, "train")
# test は既存結果から計算
test_stake = case_d["total_stake"].sum() if best_t3 != "除外" else (t1_data[(t1_data["G_S"]>t1_g)&(t1_data["O_S"]<t1_o)]["total_stake"].sum()+t2_delta["total_stake"].sum())
# Actually, 選定された戦略に基づく test 集計を再計算
t1_sub = t1_bet[(t1_bet["G_S"]>t1_g) & (t1_bet["O_S"]<t1_o)]
t2_sub = t2_data[(t2_data["G_S"]>t2_g) & (t2_data["O_S"]>t2_o)]
if best_t3 == "除外":
    t3_sub = t3_bet.iloc[:0]
else:
    t3_sub = t3_bet[(t3_bet["O_S"]>t3_o_thr) & (t3_bet["G_S"]>t3_g_thr)]
test_combined = pd.concat([t1_sub, t2_sub, t3_sub])
test_stake = int(test_combined["total_stake"].sum())
test_pay = int(test_combined["total_payout"].sum())
test_roi = test_pay/test_stake if test_stake else 0
test_ci = bootstrap_roi_ci(test_combined["total_stake"].values, test_combined["total_payout"].values)
test_t1 = {"stake": int(t1_sub["total_stake"].sum()), "pay": int(t1_sub["total_payout"].sum()),
           "races": len(t1_sub), "hits": int(t1_sub["hit"].sum())}
test_t2 = {"stake": int(t2_sub["total_stake"].sum()), "pay": int(t2_sub["total_payout"].sum()),
           "races": len(t2_sub), "hits": int(t2_sub["hit"].sum())}
test_t3 = {"stake": int(t3_sub["total_stake"].sum()), "pay": int(t3_sub["total_payout"].sum()),
           "races": len(t3_sub), "hits": int(t3_sub["hit"].sum())}
test_result = {"label":"test","N":N_te,"bet_races":len(test_combined),
               "hit_races":int(test_combined["hit"].sum()),
               "stake":test_stake,"payout":test_pay,"roi":test_roi,
               "ci_lo":test_ci[0],"ci_hi":test_ci[1],
               "type_stats":{"型1":test_t1,"型2":test_t2,"型3":test_t3}}

# 比較表
gen_rows = []
for res in [train_result, test_result]:
    gen_rows.append({
        "period": res["label"], "N_races": res["N"],
        "bet_races": res["bet_races"], "hit_races": res["hit_races"],
        "stake": res["stake"], "payout": res["payout"],
        "profit": res["payout"]-res["stake"],
        "roi": res["roi"], "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"],
        "t1_races": res["type_stats"]["型1"]["races"],
        "t1_roi": res["type_stats"]["型1"]["pay"]/max(res["type_stats"]["型1"]["stake"],1),
        "t2_races": res["type_stats"]["型2"]["races"],
        "t2_roi": res["type_stats"]["型2"]["pay"]/max(res["type_stats"]["型2"]["stake"],1),
        "t3_races": res["type_stats"]["型3"]["races"],
        "t3_roi": res["type_stats"]["型3"]["pay"]/max(res["type_stats"]["型3"]["stake"],1),
    })
gen_df = pd.DataFrame(gen_rows)
gen_df.to_csv(OUT/"final_strategy_train_vs_test.csv", index=False, encoding="utf-8-sig")

print(f"\n  {'期間':<6} {'bets':>5} {'ROI':>7} {'CI下':>7} {'CI上':>7} {'T1 ROI':>8} "
      f"{'T2 ROI':>8} {'T3 ROI':>8}")
for rw in gen_rows:
    print(f"  {rw['period']:<6} {rw['bet_races']:>5,} {rw['roi']:>7.4f} "
          f"{rw['ci_lo']:>7.4f} {rw['ci_hi']:>7.4f} {rw['t1_roi']:>8.4f} "
          f"{rw['t2_roi']:>8.4f} {rw['t3_roi']:>8.4f}")

# 汎化判定
diff = abs(train_result["roi"] - test_result["roi"])
if diff < 0.05:
    gen_verdict = "汎化成功"
elif train_result["roi"] > test_result["roi"]:
    gen_verdict = "test 偶然の疑い (train < test が普通だが逆)"
    # 実は train < test なら test が過学習、train > test なら test がたまたま良い
    gen_verdict = "test 優位 (train 実性能 < test 見かけ性能 の可能性)"
    if train_result["roi"] >= 1.0:
        gen_verdict = f"train ROI {train_result['roi']:.3f} ≥ 1.0 → train 側でも黒字、汎化良好"
    else:
        gen_verdict = f"train ROI {train_result['roi']:.3f} < 1.0, test {test_result['roi']:.3f} → test 偶然"
else:
    gen_verdict = f"train ROI {train_result['roi']:.3f} > test {test_result['roi']:.3f} → test 過学習"
print(f"\n  汎化判定: {gen_verdict}")

# ========== [Step 6-7] 最終戦略 + レポート ==========
print("\n[Step 6-7] 最終戦略仕様 + レポート ...")
final_spec = {
    "モデル": "v4 (12 features, PL)",
    "キャリブレーション": "Isotonic Regression (train fit)",
    "判定": f"EV >= {EV_THR} AND Edge >= {EDGE_THR}",
    "型信頼度": "型1=1.00, 型2=0.90, 型3=0.80",
    "予算": f"{BUDGET}円/レース",
    "配分": "EV 比例",
    "型1 条件": f"top1_lane=0 AND G_S > {t1_g} AND O_S < {t1_o}",
    "型2 条件": f"top1_lane=0 AND G_S > {t2_g} AND O_S > {t2_o}",
    "型3 条件": (f"top1_lane>0 AND O_S > {t3_o_thr} AND G_S > {t3_g_thr}"
                  if best_t3 != "除外" else "見送り (除外)"),
    "型4": "見送り",
    "test ROI": test_result["roi"],
    "test CI": (test_result["ci_lo"], test_result["ci_hi"]),
    "train ROI": train_result["roi"],
    "train CI": (train_result["ci_lo"], train_result["ci_hi"]),
    "汎化判定": gen_verdict,
}

# 運用判定
if test_result["roi"] >= 1.0 and test_result["ci_lo"] >= 0.97:
    op_verdict = "運用候補確定 (ROI≥1.0 AND CI下≥0.97)"
elif test_result["roi"] >= 1.0:
    op_verdict = "慎重運用 (ROI≥1.0 だが CI下<0.97)"
else:
    op_verdict = "追加検証または諦め (ROI<1.0)"

md = f"""# Phase C 最終判定レポート

## 最終戦略 (数値仕様)

```
モデル: v4 (12 features, PL)
キャリブレーション: Isotonic Regression (train 2023-05〜2025-12 で fit)
判定: EV >= {EV_THR} AND Edge >= {EDGE_THR}
型信頼度 r: 型1=1.00, 型2=0.90, 型3=0.80
予算: {BUDGET}円/レース
配分: EV 比例 (100円単位)

型分類:
  型1 (逃げ本命): top1_lane=0 AND G_S > {t1_g} AND O_S < {t1_o}
  型2 (イン残り): top1_lane=0 AND G_S > {t2_g} AND O_S > {t2_o}
  型3 (頭荒れ):   {'top1_lane>0 AND O_S > ' + str(t3_o_thr) + ' AND G_S > ' + str(t3_g_thr) if best_t3 != '除外' else '見送り (除外)'}
  型4 (ノイズ):   上記以外 → 見送り
```

## Step 1: case δ (型2 G>0.9 AND O>0.4) 詳細

- ベット数: {len(case_d):,} (型1={len(t1_data)}, 型2δ={len(t2_delta)}, 型3={len(t3_data)})
- ROI: **{roi_d:.4f}**
- Bootstrap 95% CI: **[{ci_lo:.4f}, {ci_hi:.4f}]**

### 月別
| 月 | N | hit | ROI | CI下 | CI上 |
|---|---|---|---|---|---|
"""
for r in mon_rows:
    md += (f"| {r['month']} | {r['n_bet']} | {r['n_hit']} | "
           f"{r['roi']:.4f} | {r['ci_lo']:.4f} | {r['ci_hi']:.4f} |\n")

md += """
### 型別
| 型 | N | hit | ROI | CI下 | CI上 |
|---|---|---|---|---|---|
"""
for r in type_rows:
    md += (f"| {r['type']} | {r['n_bet']} | {r['n_hit']} | "
           f"{r['roi']:.4f} | {r['ci_lo']:.4f} | {r['ci_hi']:.4f} |\n")

md += f"""

![ROI分布](case_d_roi_distribution.png)

## Step 2: 型1 内部分解

![type1 heatmap](type1_heatmap.png)

### 型1 閾値案
| 案 | 条件 | N | ROI | CI下 | CI上 |
|---|---|---|---|---|---|
"""
for r in t1_sim:
    md += (f"| {r['case']} | G>{r['g_thr']:.1f} AND O<{r['o_thr']:.1f} | "
           f"{r['n']:,} | {r['roi']:.4f} | {r['ci_lo']:.4f} | {r['ci_hi']:.4f} |\n")

md += f"""

## Step 3: 型3 内部分解

![type3 heatmap](type3_heatmap.png)

### 型3 閾値案
| 案 | 条件 | N | ROI | CI下 | CI上 | 備考 |
|---|---|---|---|---|---|---|
"""
for r in t3_sim:
    note = "**N<50 参考**" if r['n'] < 50 else ""
    md += (f"| {r['case']} | O>{r['o_thr']:.1f} AND G>{r['g_thr']:.1f} | "
           f"{r['n']:,} | {r['roi']:.4f} | {r['ci_lo']:.4f} | {r['ci_hi']:.4f} | {note} |\n")

md += f"""

## Step 4: 全型組み合わせ最適化 (Top 10 by ROI)

| 戦略 | N_bet | ROI | CI下 | CI上 | profit |
|---|---|---|---|---|---|
"""
for _, rw in comb_df.head(10).iterrows():
    md += (f"| {rw['strategy']} | {int(rw['n_bet']):,} | {rw['roi']:.4f} | "
           f"{rw['ci_lo']:.4f} | {rw['ci_hi']:.4f} | ¥{int(rw['profit']):+,} |\n")

md += f"""

### 選定: {best_row['strategy']}

## Step 5: train 汎化検証

| 期間 | bets | ROI | CI下 | CI上 | T1 ROI | T2 ROI | T3 ROI |
|---|---|---|---|---|---|---|---|
"""
for rw in gen_rows:
    md += (f"| {rw['period']} | {rw['bet_races']:,} | {rw['roi']:.4f} | "
           f"{rw['ci_lo']:.4f} | {rw['ci_hi']:.4f} | {rw['t1_roi']:.4f} | "
           f"{rw['t2_roi']:.4f} | {rw['t3_roi']:.4f} |\n")

md += f"""

**train ROI - test ROI 差: {diff:+.4f}**
**汎化判定: {gen_verdict}**

## 期待性能 (test ベース)

- ROI: **{test_result['roi']:.4f}**
- Bootstrap 95% CI: **[{test_result['ci_lo']:.4f}, {test_result['ci_hi']:.4f}]**
- ベット率: {test_result['bet_races']/N_te*100:.2f}%
- 型別:
  - 型1: N={test_t1['races']}, ROI={test_t1['pay']/max(test_t1['stake'],1):.4f}
  - 型2 (δ): N={test_t2['races']}, ROI={test_t2['pay']/max(test_t2['stake'],1):.4f}
  - 型3: N={test_t3['races']}, ROI={test_t3['pay']/max(test_t3['stake'],1):.4f}

## 運用判定: {op_verdict}

### 運用への注意点
1. **サンプル制約**: 型3 は N={test_t3['races']} と少なく、ROI の真値不確実
2. **月別変動**: 4 ヶ月で std={float(np.std([r['roi'] for r in mon_rows])):.3f}
3. **train 汎化**: train ROI {train_result['roi']:.4f} vs test {test_result['roi']:.4f} → {gen_verdict}
4. **ブック vig**: 25% → 構造的な利益幅はタイト

### 次アクション
"""
if test_result["roi"] >= 1.0 and test_result["ci_lo"] >= 0.97:
    md += """
1. **運用候補確定** — 仕様書 (part1-3) に反映
2. ペーパートレード 1 ヶ月
3. 実運用移行
"""
elif test_result["roi"] >= 1.0:
    md += """
1. **慎重運用** — CI 下限が目標 0.97 未満
2. データ期間延長で CI 狭める (2026-05 以降追加)
3. サンプル増強後に再評価
"""
else:
    md += """
1. **追加検証** 2連単・2連複バックテスト
2. 3連単撤退を検討
"""

md += """

## 出力ファイル
- `case_d_confidence_analysis.csv` / `case_d_roi_distribution.png`
- `type1_matrix.csv` / `type1_heatmap.png` / `type1_threshold_scenarios.csv`
- `type3_matrix.csv` / `type3_heatmap.png` / `type3_threshold_scenarios.csv`
- `all_types_optimization.csv`
- `final_strategy_train_vs_test.csv`
"""

with open(OUT/"phase_c_final_verdict.md", "w", encoding="utf-8") as f:
    f.write(md)
print(f"   saved: phase_c_final_verdict.md")
print(f"\n=== 運用判定: {op_verdict} ===")
print(f"    test ROI={test_result['roi']:.4f} CI[{test_result['ci_lo']:.4f},{test_result['ci_hi']:.4f}]")
print(f"    train ROI={train_result['roi']:.4f} CI[{train_result['ci_lo']:.4f},{train_result['ci_hi']:.4f}]")
print("完了")
