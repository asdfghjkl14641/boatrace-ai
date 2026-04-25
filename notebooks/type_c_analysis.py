# -*- coding: utf-8 -*-
"""
型分類ルール案 C の検証と型2妥当性チェック。

入力: notebooks/output/pl_race_log_with_scores.csv
     (S_1..S_6, F_S, G_S, O_S, N_S, race_type [旧], pred_top1, actual_*)
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "notebooks" / "output"
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


df = pd.read_csv(OUT / "pl_race_log_with_scores.csv")
S_cols = [f"S_{i}" for i in range(1, 7)]
S = df[S_cols].to_numpy()

# top1_lane (0..5)
df["top1_lane"] = S.argmax(axis=1)
# P_1 の計算: τ=1.0 の素の softmax (型間相対比較には τ cancels)
Ss = S - S.max(axis=1, keepdims=True)
P = np.exp(Ss) / np.exp(Ss).sum(axis=1, keepdims=True)
df["P_actual_1"] = P[np.arange(len(df)), df["actual_1"].to_numpy() - 1]
df["log_loss"] = -np.log(np.clip(df["P_actual_1"], 1e-9, None))


# ================================================================
# Step 1: 案 C 再分類
# ================================================================
def classify_c(r):
    if r["top1_lane"] == 0:
        if r["G_S"] > 1.0 and r["O_S"] < 0.3:
            return "型1_逃げ本命"
        else:
            return "型2_イン残り"
    else:
        if r["O_S"] > 0.3 and r["G_S"] > 0.4:
            return "型3_頭荒れ"
        else:
            return "型4_ノイズ"


df["type_c"] = df.apply(classify_c, axis=1)

log("=" * 60)
log("[Step 1] 案 C による型分布")
log("=" * 60)
N = len(df)
dist = df["type_c"].value_counts().reindex(
    ["型1_逃げ本命", "型2_イン残り", "型3_頭荒れ", "型4_ノイズ"]).fillna(0).astype(int)
log(f"  全 {N:,} レース")
for t, n in dist.items():
    log(f"  {t:14s}: {n:>5,}  ({n/N*100:5.2f}%)")

log("\n目安との対比:")
for t, target in [("型1_逃げ本命", "30-45%"), ("型2_イン残り", "15-25%"),
                  ("型3_頭荒れ", "8-15%"), ("型4_ノイズ", "20-30%")]:
    pct = dist[t] / N * 100
    log(f"  {t:14s}: {pct:5.2f}%  目安 {target}")


# ================================================================
# Step 2: 型別実績 & 旧 vs 案 C 比較
# ================================================================
log("\n" + "=" * 60)
log("[Step 2] 型別実績 & 旧ルールとの直接比較")
log("=" * 60)


def collect_stats(sub: pd.DataFrame) -> dict:
    n = len(sub)
    row = {"N": n, "share_%": n / N * 100}
    # 1着枠分布
    for l in range(1, 7):
        row[f"win_lane{l}_%"] = (sub["actual_1"] == l).mean() * 100
    # top1 hit 率
    row["top1_hit_%"] = (sub["pred_top1"] == sub["actual_1"]).mean() * 100
    # log-loss
    row["log_loss_mean"] = sub["log_loss"].mean()
    # 三連単 top5
    tri = (sub["actual_1"].astype(str) + "-" +
           sub["actual_2"].astype(str) + "-" +
           sub["actual_3"].astype(str))
    top5 = tri.value_counts().head(5)
    row["top5_trifecta"] = " / ".join(f"{k}:{v}" for k, v in top5.items())
    return row


rows = []
for rule, col in [("old (Step 5 prev)", "race_type"), ("C (new)", "type_c")]:
    for t in ["型1_逃げ本命", "型2_イン残り", "型3_頭荒れ", "型4_ノイズ"]:
        sub = df[df[col] == t]
        r = collect_stats(sub)
        r.update({"rule": rule, "type": t})
        rows.append(r)
comp = pd.DataFrame(rows)
# 列順整理
cols_order = ["rule", "type", "N", "share_%", "top1_hit_%", "log_loss_mean",
              "win_lane1_%", "win_lane2_%", "win_lane3_%",
              "win_lane4_%", "win_lane5_%", "win_lane6_%", "top5_trifecta"]
comp = comp[cols_order]
comp.to_csv(OUT / "type_comparison_oldVSc.csv", index=False, encoding="utf-8-sig")
log(f"\n  saved: {OUT}/type_comparison_oldVSc.csv")

show = comp.drop(columns=["top5_trifecta"])
log("\n  比較表 (rule × type):")
log(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

log("\n  三連単 top5 (案 C):")
for _, r in comp[comp["rule"] == "C (new)"].iterrows():
    log(f"    {r['type']}: {r['top5_trifecta']}")

# 全体の参考値
log(f"\n  (参考) 全体 top1 hit率: {(df['pred_top1']==df['actual_1']).mean()*100:.2f}%  "
    f"全体 log_loss: {df['log_loss'].mean():.4f}")


# ================================================================
# Step 3: 型2 妥当性チェック
# ================================================================
log("\n" + "=" * 60)
log("[Step 3] 型2 妥当性チェック (案 C)")
log("=" * 60)
c1 = df[df["type_c"] == "型1_逃げ本命"]
c2 = df[df["type_c"] == "型2_イン残り"]
c3 = df[df["type_c"] == "型3_頭荒れ"]
c4 = df[df["type_c"] == "型4_ノイズ"]

log(f"  型2  N={len(c2):,}  1枠hit={c2['actual_1'].eq(1).mean()*100:.2f}%  "
    f"top1_hit={(c2['pred_top1']==c2['actual_1']).mean()*100:.2f}%  "
    f"log_loss={c2['log_loss'].mean():.4f}")
log(f"  型1  N={len(c1):,}  1枠hit={c1['actual_1'].eq(1).mean()*100:.2f}%  "
    f"top1_hit={(c1['pred_top1']==c1['actual_1']).mean()*100:.2f}%  "
    f"log_loss={c1['log_loss'].mean():.4f}")
log(f"  型4  N={len(c4):,}  1枠hit={c4['actual_1'].eq(1).mean()*100:.2f}%  "
    f"top1_hit={(c4['pred_top1']==c4['actual_1']).mean()*100:.2f}%  "
    f"log_loss={c4['log_loss'].mean():.4f}")

# 統合判定
# 条件: (型2 1枠 hit ≤ 型4 1枠 hit) AND (型2 log_loss ≥ 型4 log_loss)
c2_win1 = c2["actual_1"].eq(1).mean() * 100
c4_win1 = c4["actual_1"].eq(1).mean() * 100
c2_ll = c2["log_loss"].mean()
c4_ll = c4["log_loss"].mean()

should_merge = (c2_win1 <= c4_win1) and (c2_ll >= c4_ll)
log(f"\n  統合判定条件:")
log(f"    型2 1枠hit ({c2_win1:.2f}%)  ≤  型4 1枠hit ({c4_win1:.2f}%) ? → {c2_win1 <= c4_win1}")
log(f"    型2 log_loss ({c2_ll:.4f})  ≥  型4 log_loss ({c4_ll:.4f}) ? → {c2_ll >= c4_ll}")
log(f"  → 型2 を型4 に統合すべき: **{should_merge}**")

# 常に 3 型統合版もサマリ
df["type_c_merged"] = df["type_c"].replace({"型2_イン残り": "型4_ノイズ"})
log("\n  (参考) 型2 を型4 に統合した 3 型版の分布:")
dm = df["type_c_merged"].value_counts()
for t, n in dm.items():
    log(f"    {t:14s}: {n:>5,}  ({n/N*100:5.2f}%)")
log("\n  3 型版の集計:")
for t in sorted(df["type_c_merged"].unique()):
    sub = df[df["type_c_merged"] == t]
    log(f"    {t:14s}: N={len(sub):>5,}  1枠hit={sub['actual_1'].eq(1).mean()*100:6.2f}%  "
        f"top1_hit={(sub['pred_top1']==sub['actual_1']).mean()*100:6.2f}%  "
        f"log_loss={sub['log_loss'].mean():.4f}")


# ================================================================
# Step 4: 散布図 (G_S vs O_S、top1_lane で色分け)
# ================================================================
log("\n" + "=" * 60)
log("[Step 4] G_S vs O_S 散布図 (top1_lane 色分け)")
log("=" * 60)

fig, ax = plt.subplots(figsize=(10, 7))
m0 = df["top1_lane"] == 0
m1 = ~m0
ax.scatter(df.loc[m0, "G_S"], df.loc[m0, "O_S"], s=3, alpha=0.20,
           color="#2a5fa7", label=f"top1=1号艇 (N={m0.sum():,})")
ax.scatter(df.loc[m1, "G_S"], df.loc[m1, "O_S"], s=3, alpha=0.30,
           color="#c04e4e", label=f"top1=2-6号艇 (N={m1.sum():,})")
# 案 C の分類線
ax.axvline(1.0, color="#2a5fa7", linestyle="--", alpha=0.6, label="G_S=1.0 (型1/型2 境界)")
ax.axhline(0.3, color="#c04e4e", linestyle="--", alpha=0.6, label="O_S=0.3 (型3/型4 境界)")
ax.axvline(0.4, color="gray", linestyle=":", alpha=0.5, label="G_S=0.4 (型3 要件)")
ax.set_xlabel("G_S  (決定度)")
ax.set_ylabel("O_S  (外枠 max z)")
ax.set_title("Rule-C boundaries: G_S vs O_S  colored by argmax S lane")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.2)
fig.tight_layout()
png = OUT / "gs_os_by_top1.png"
fig.savefig(png, dpi=130)
plt.close()
log(f"  saved: {png}")

# 各象限の集計も
log("\n  象限別 件数 (top1=1号艇):")
for g_hi, g_label in [(1.0, "G_S>1.0"), (0.0, "G_S<=1.0")]:
    for o_hi, o_label in [(None, "O_S<0.3"), (0.3, "O_S>=0.3")]:
        if g_label == "G_S>1.0":
            mask = m0 & (df["G_S"] > 1.0)
        else:
            mask = m0 & (df["G_S"] <= 1.0)
        if o_hi is None:
            mask = mask & (df["O_S"] < 0.3)
        else:
            mask = mask & (df["O_S"] >= 0.3)
        log(f"    {g_label} & {o_label}: {mask.sum():>5,}")


# ================================================================
# 出力ファイル & ログ保存
# ================================================================
df_out = df[[
    "date","stadium","race_number","actual_1","actual_2","actual_3",
    "S_1","S_2","S_3","S_4","S_5","S_6",
    "mean_S","sigma_S","F_S","G_S","O_S","N_S",
    "pred_top1","top1_lane","P_actual_1","log_loss",
    "race_type","type_c","type_c_merged",
]]
df_out.to_csv(OUT / "pl_race_log_with_types_c.csv", index=False, encoding="utf-8-sig")
log(f"\n  saved: {OUT}/pl_race_log_with_types_c.csv (shape={df_out.shape})")

# tee
import tempfile
(Path(tempfile.gettempdir()) / "type_c_analysis.log").write_text(
    "\n".join(LOG), encoding="utf-8")
log(f"\n[tee] full log → /tmp/type_c_analysis.log")
