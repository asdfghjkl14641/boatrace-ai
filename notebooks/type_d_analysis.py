# -*- coding: utf-8 -*-
"""案 D の検証: 型2 を「1号艇残 + 外枠警戒」で肯定定義"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "notebooks" / "output"
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


# ================================================================
# 入力 (type_c 済み CSV を再利用、互換のため S_i も揃う)
# ================================================================
df = pd.read_csv(OUT / "pl_race_log_with_types_c.csv")
N = len(df)
log(f"入力: {N:,} レース  cols={list(df.columns)[:8]}...")


# ================================================================
# Step 1: 案 D 分類
# ================================================================
def classify_d(r):
    if r["top1_lane"] == 0:
        if r["G_S"] > 1.0 and r["O_S"] < 0.3:
            return "型1_逃げ本命"
        elif r["G_S"] > 0.6 and r["O_S"] > 0.2:
            return "型2_イン残りヒモ荒れ"
        else:
            return "型4_ノイズ"
    else:
        if r["O_S"] > 0.3 and r["G_S"] > 0.4:
            return "型3_頭荒れ"
        else:
            return "型4_ノイズ"


df["type_d"] = df.apply(classify_d, axis=1)

log("\n" + "=" * 60)
log("[Step 1] 案 D 型分布")
log("=" * 60)
dist = df["type_d"].value_counts().reindex(
    ["型1_逃げ本命","型2_イン残りヒモ荒れ","型3_頭荒れ","型4_ノイズ"]).fillna(0).astype(int)
for t, n in dist.items():
    log(f"  {t:18s}: {n:>5,}  ({n/N*100:5.2f}%)")

log("\n  目安との対比:")
for t, target in [("型1_逃げ本命", "35-40%"), ("型2_イン残りヒモ荒れ", "12-22%"),
                  ("型3_頭荒れ", "2-5%"), ("型4_ノイズ", "35-50%")]:
    pct = dist[t] / N * 100
    log(f"    {t:18s}: {pct:5.2f}%  目安 {target}")


# ================================================================
# Step 2: 型別詳細実績
# ================================================================
log("\n" + "=" * 60)
log("[Step 2] 型別 詳細実績 (1着/2着/3着枠分布, top1 hit, log-loss)")
log("=" * 60)


def stats(sub: pd.DataFrame) -> dict:
    n = len(sub)
    row = {"N": n, "share_%": n/N*100}
    for pos, col in [("1st","actual_1"),("2nd","actual_2"),("3rd","actual_3")]:
        for l in range(1, 7):
            row[f"{pos}_lane{l}_%"] = (sub[col]==l).mean()*100
    row["top1_hit_%"] = (sub["pred_top1"]==sub["actual_1"]).mean()*100
    row["log_loss_mean"] = sub["log_loss"].mean()
    # 外枠絡み (2着 or 3着 に 4/5/6 が入るレースの割合)
    outer_in_234 = (sub["actual_2"].isin([4,5,6]) | sub["actual_3"].isin([4,5,6])).mean()*100
    row["outer_in_2nd_or_3rd_%"] = outer_in_234
    return row


rows_d = []
for t in ["型1_逃げ本命","型2_イン残りヒモ荒れ","型3_頭荒れ","型4_ノイズ"]:
    sub = df[df["type_d"]==t]
    r = stats(sub); r["type"] = t
    rows_d.append(r)
dd = pd.DataFrame(rows_d)
core_cols = ["type","N","share_%","top1_hit_%","log_loss_mean","outer_in_2nd_or_3rd_%",
             "1st_lane1_%","1st_lane2_%","1st_lane3_%","1st_lane4_%","1st_lane5_%","1st_lane6_%"]
log("\n  1 着分布・top1 hit・log_loss・外枠絡み率:")
log(dd[core_cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

log("\n  2 着枠分布 (型別):")
c2 = ["type","N","2nd_lane1_%","2nd_lane2_%","2nd_lane3_%","2nd_lane4_%","2nd_lane5_%","2nd_lane6_%"]
log(dd[c2].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

log("\n  3 着枠分布 (型別):")
c3 = ["type","N","3rd_lane1_%","3rd_lane2_%","3rd_lane3_%","3rd_lane4_%","3rd_lane5_%","3rd_lane6_%"]
log(dd[c3].to_string(index=False, float_format=lambda x: f"{x:.2f}"))


# ================================================================
# Step 3: 型1 vs 型2 の三連単 top10
# ================================================================
log("\n" + "=" * 60)
log("[Step 3] 型1 vs 型2 三連単 top10 (外枠絡み比較)")
log("=" * 60)

def trifecta_counts(sub):
    tri = sub["actual_1"].astype(str)+"-"+sub["actual_2"].astype(str)+"-"+sub["actual_3"].astype(str)
    return tri.value_counts()

for t in ["型1_逃げ本命","型2_イン残りヒモ荒れ"]:
    sub = df[df["type_d"]==t]
    n = len(sub)
    log(f"\n  {t}  (N={n:,})  top10 三連単:")
    tc = trifecta_counts(sub).head(10)
    for kk, vv in tc.items():
        flag = ""
        # 2着 or 3着 に 4/5/6 が含まれるか
        parts = kk.split("-")
        if any(int(p) in (4,5,6) for p in parts[1:3]):
            flag = " ← 外枠絡み"
        log(f"    {kk}  {vv:>4}  ({vv/n*100:5.2f}%){flag}")

# 外枠絡み率 (1 着 1 号艇のうち 2or3 着が 4-6 のレース)
for t in ["型1_逃げ本命","型2_イン残りヒモ荒れ"]:
    sub = df[df["type_d"]==t]
    sub1 = sub[sub["actual_1"]==1]  # 1着 1 号艇のみ
    if len(sub1)==0:
        continue
    outer = (sub1["actual_2"].isin([4,5,6]) | sub1["actual_3"].isin([4,5,6])).mean()*100
    log(f"\n  {t}  1着=1号艇 のレース中 2 or 3 着に外枠 (4-6) が入る率: {outer:.2f}% (N={len(sub1):,})")


# ================================================================
# Step 4: 案 C vs 案 D 比較
# ================================================================
log("\n" + "=" * 60)
log("[Step 4] 案 C vs 案 D 直接比較")
log("=" * 60)

comp_rows = []
for rule, col in [("C", "type_c"), ("D", "type_d")]:
    for t in sorted(df[col].unique()):
        sub = df[df[col]==t]
        r = stats(sub)
        r.update({"rule": rule, "type": t})
        comp_rows.append(r)
comp = pd.DataFrame(comp_rows)
cols_show = ["rule","type","N","share_%","top1_hit_%","log_loss_mean",
             "outer_in_2nd_or_3rd_%",
             "1st_lane1_%","1st_lane4_%","1st_lane5_%","1st_lane6_%"]
log(comp[cols_show].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
comp.to_csv(OUT / "type_c_vs_d.csv", index=False, encoding="utf-8-sig")
log(f"\n  saved: {OUT}/type_c_vs_d.csv")


# ================================================================
# Step 5: 型4 の内部構造
# ================================================================
log("\n" + "=" * 60)
log("[Step 5] 型4 の内部サブグループ")
log("=" * 60)

t4 = df[df["type_d"]=="型4_ノイズ"].copy()
# 4a: 1号艇トップだが弱い
t4a = t4[t4["top1_lane"]==0]
# 4b: 1号艇以外トップだが弱い
t4b = t4[t4["top1_lane"]!=0]

for label, sub in [("4a (top1=1号艇 弱)", t4a), ("4b (top1≠1 弱)", t4b)]:
    n = len(sub)
    if n==0: continue
    log(f"\n  {label}  N={n:,}  ({n/len(t4)*100:.2f}% of 型4, {n/N*100:.2f}% of 全体)")
    log(f"    1 枠 hit率   : {(sub['actual_1']==1).mean()*100:6.2f}%")
    log(f"    top1 hit率  : {(sub['pred_top1']==sub['actual_1']).mean()*100:6.2f}%")
    log(f"    log_loss    : {sub['log_loss'].mean():.4f}")
    log(f"    外枠絡み率  : {(sub['actual_2'].isin([4,5,6]) | sub['actual_3'].isin([4,5,6])).mean()*100:6.2f}%")
    # 1着枠の分布
    dist = sub["actual_1"].value_counts().sort_index()
    dist_str = "  ".join(f"{l}枠={dist.get(l,0)/n*100:4.1f}%" for l in range(1,7))
    log(f"    1 着枠分布: {dist_str}")

# CSV 保存
t4_stat_rows = []
for label, sub in [("4a", t4a), ("4b", t4b)]:
    r = stats(sub); r["subgroup"] = label
    t4_stat_rows.append(r)
t4df = pd.DataFrame(t4_stat_rows)
t4df.to_csv(OUT / "type4_subgroup.csv", index=False, encoding="utf-8-sig")
log(f"\n  saved: {OUT}/type4_subgroup.csv")


# ================================================================
# ログ保存 + 最終 CSV
# ================================================================
keep_cols = ["date","stadium","race_number","actual_1","actual_2","actual_3",
             "S_1","S_2","S_3","S_4","S_5","S_6",
             "mean_S","sigma_S","F_S","G_S","O_S","N_S",
             "pred_top1","top1_lane","P_actual_1","log_loss",
             "race_type","type_c","type_c_merged","type_d"]
df[keep_cols].to_csv(OUT / "pl_race_log_with_types_d.csv", index=False, encoding="utf-8-sig")
log(f"\n  saved: {OUT}/pl_race_log_with_types_d.csv (shape={df[keep_cols].shape})")

import tempfile
(Path(tempfile.gettempdir())/"type_d_analysis.log").write_text(
    "\n".join(LOG), encoding="utf-8")
log("\n[tee] /tmp/type_d_analysis.log")
