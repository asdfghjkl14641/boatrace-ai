# -*- coding: utf-8 -*-
"""v3 学習結果の可視化 — 4 視点グラフ"""
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

# 日本語フォント試行
for font in ["Meiryo","MS Gothic","Yu Gothic","Noto Sans CJK JP","Hiragino Sans"]:
    try:
        matplotlib.rcParams["font.family"] = font
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

LAYER = {
    "theta_ability": "A", "f3_ST": "A",
    "f4_disp": "B", "f5_motor": "B", "f6_form": "B", "f6_nomatch": "B",
    "f7_lane_L": "C", "f8_V": "C", "f9_W": "C", "f10_H_new": "C",
}
LAYER_COLOR = {"A": "#3B82F6", "B": "#10B981", "C": "#F59E0B"}
LAYER_NAME = {"A": "A層 (能力)", "B": "B層 (当日)", "C": "C層 (構造)"}
LAYER_MEMBERS = {
    "A": ["θ_ability", "f3_ST"],
    "B": ["f4_disp", "f5_motor", "f6_form", "f6_nomatch"],
    "C": ["f7_lane_L", "f8_V", "f9_W", "f10_H_new"],
}

# ---------- Step 1: データ整形 ----------
w = pd.read_csv(OUT / "v3_weights.csv")         # feature, beta_old_H, beta_new_H
v2w = pd.read_csv(OUT / "v2_extended_weights.csv")  # READ-ONLY
contrib = pd.read_csv(OUT / "v3_contribution.csv")  # feature, contrib_v2_%, contrib_v3_%, diff
coef = pd.read_csv(OUT / "v3_coef_diff.csv")        # feature, beta_v2, beta_v3, diff
# merge
df = w.merge(v2w[["feature", "beta_new_H"]].rename(columns={"beta_new_H":"beta_v2"}), on="feature", how="left")
df = df.rename(columns={"beta_new_H":"beta_v3"})
df = df.merge(contrib[["feature","contrib_v2_%","contrib_v3_%"]], on="feature", how="left")
df["layer"] = df["feature"].map(LAYER)
# 純粋な |β| の割合 (合計 100 %)
abs_beta = df["beta_v3"].abs()
df["abs_beta_pct"] = abs_beta / abs_beta.sum() * 100
# std(X) を逆算: contrib_v3 = |β|×std / Σ|β_j|×std_j の関係から std ∝ contrib/|β|
# 簡易: 寄与度 / |β|
df["std_X_proxy"] = df["contrib_v3_%"] / (abs_beta + 1e-9)

summary = df[["feature","layer","beta_v2","beta_v3","std_X_proxy",
              "contrib_v2_%","contrib_v3_%","abs_beta_pct"]]
summary.to_csv(OUT / "v3_weights_summary.csv", index=False, encoding="utf-8-sig")
print("=== v3 重み分析 サマリテーブル ===")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# 表示用ラベル
NICE_LABEL = {
    "theta_ability": "θ_ability",
    "f3_ST": "f3_ST",
    "f4_disp": "f4_disp",
    "f5_motor": "f5_motor",
    "f6_form": "f6_form",
    "f6_nomatch": "f6_nomatch",
    "f7_lane_L": "f7_lane_L",
    "f8_V": "f8_V",
    "f9_W": "f9_W",
    "f10_H_new": "f10_H",
}
df["label"] = df["feature"].map(NICE_LABEL)
df["color"] = df["layer"].map(LAYER_COLOR)

# ---------- Step 2: 4 subplot ----------
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("v3_newH 重み分析 (10 features, train 2023-05〜2025-12)",
             fontsize=14, fontweight="bold", y=0.995)

# === 視点 1: 寄与度 (|β|×std) 降順 横棒 ===
ax = axes[0, 0]
d1 = df.sort_values("contrib_v3_%", ascending=True).reset_index(drop=True)
bars = ax.barh(d1["label"], d1["contrib_v3_%"], color=d1["color"])
for i, v in enumerate(d1["contrib_v3_%"]):
    ax.text(v + 0.3, i, f"{v:.2f}%", va="center", fontsize=9)
ax.set_xlabel("寄与度 (%) = |β| × std(X) / 合計")
ax.set_title("① v3 寄与度 (|β|×std) — 予測への実効影響", fontsize=11)
ax.set_xlim(0, max(d1["contrib_v3_%"]) * 1.15)
ax.grid(axis="x", alpha=0.3)

# === 視点 2: 純粋 |β| 割合 降順 ===
ax = axes[0, 1]
d2 = df.sort_values("abs_beta_pct", ascending=True).reset_index(drop=True)
bars = ax.barh(d2["label"], d2["abs_beta_pct"], color=d2["color"])
for i, v in enumerate(d2["abs_beta_pct"]):
    ax.text(v + 0.3, i, f"{v:.2f}%", va="center", fontsize=9)
ax.set_xlabel("|β| の割合 (%) = |β_i| / 合計|β|")
ax.set_title("② 純粋な係数大きさ |β| — 値域を無視した重み", fontsize=11)
ax.set_xlim(0, max(d2["abs_beta_pct"]) * 1.15)
ax.grid(axis="x", alpha=0.3)

# === 視点 3: 層別 寄与度 円グラフ ===
ax = axes[1, 0]
layer_sum = df.groupby("layer")["contrib_v3_%"].sum()
# A, B, C の順
order = ["A", "B", "C"]
sizes = [layer_sum.get(l, 0) for l in order]
labels_pie = [f"{LAYER_NAME[l]}\n{sizes[i]:.1f}%" for i, l in enumerate(order)]
colors_pie = [LAYER_COLOR[l] for l in order]
wedges, texts = ax.pie(sizes, labels=labels_pie, colors=colors_pie,
                        startangle=90, counterclock=False, textprops={"fontsize":10})
ax.set_title("③ 層別 寄与度内訳", fontsize=11)
# legend 中に各層の特徴量
legend_lines = [f"{LAYER_NAME[l]}: {', '.join(LAYER_MEMBERS[l])}" for l in order]
ax.text(-1.4, -1.45, "\n".join(legend_lines), fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="gray"))

# === 視点 4: v2 vs v3 寄与度比較 棒グラフ ===
ax = axes[1, 1]
# v2/v3 を等順 (v3 降順) に並べて縦棒
d4 = df.sort_values("contrib_v3_%", ascending=False).reset_index(drop=True)
x = np.arange(len(d4))
w_bar = 0.35
bars_v2 = ax.bar(x - w_bar/2, d4["contrib_v2_%"], w_bar,
                 color="#9ca3af", label="v2_ext_newH", edgecolor="white")
bars_v3 = ax.bar(x + w_bar/2, d4["contrib_v3_%"], w_bar,
                 color=d4["color"], label="v3_newH", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(d4["label"], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("寄与度 (%)")
ax.set_title("④ v2 vs v3 寄与度比較", fontsize=11)
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
# 各 v3 バーに値ラベル
for i, v in enumerate(d4["contrib_v3_%"]):
    ax.text(i + w_bar/2, v + 0.4, f"{v:.1f}", ha="center", fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
outpng = OUT / "v3_weights_visualization.png"
fig.savefig(outpng, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nsaved: {outpng}")

# ---------- Step 3: 補足テキスト ----------
print("\n" + "=" * 60)
print("v3 寄与度ランキング")
print("=" * 60)
rank = df.sort_values("contrib_v3_%", ascending=False).reset_index(drop=True)
for i, row in rank.iterrows():
    print(f"  {i+1:>2}. {row['label']:<14} {row['contrib_v3_%']:>6.2f}%  ({row['layer']}層)")

print("\n" + "=" * 60)
print("層別寄与度合計")
print("=" * 60)
for l in ["A", "B", "C"]:
    ls = df[df["layer"]==l]["contrib_v3_%"].sum()
    members = ", ".join(df[df["layer"]==l]["label"])
    print(f"  {LAYER_NAME[l]}: {ls:>6.2f}% ({members})")

print("\n" + "=" * 60)
print("v2 → v3 で大きく動いた特徴量 (|差分| 順)")
print("=" * 60)
df["delta_contrib"] = df["contrib_v3_%"] - df["contrib_v2_%"]
df_d = df.reindex(df["delta_contrib"].abs().sort_values(ascending=False).index)
for _, row in df_d.iterrows():
    sign = "+" if row["delta_contrib"] >= 0 else ""
    print(f"  {row['label']:<14} {row['contrib_v2_%']:>6.2f}% → {row['contrib_v3_%']:>6.2f}%  ({sign}{row['delta_contrib']:+.2f}pt)")

print("\n" + "=" * 60)
print("|β| 順位と 寄与度順位の乖離 (値域が狭い特徴量は |β| 大でも寄与度小)")
print("=" * 60)
df["rank_abs"] = df["abs_beta_pct"].rank(ascending=False).astype(int)
df["rank_contrib"] = df["contrib_v3_%"].rank(ascending=False).astype(int)
df["rank_gap"] = df["rank_contrib"] - df["rank_abs"]
df_g = df.reindex(df["rank_gap"].abs().sort_values(ascending=False).index)
for _, row in df_g.head(5).iterrows():
    if row["rank_gap"] == 0: continue
    print(f"  {row['label']:<14} |β|順位 {row['rank_abs']:>2}位、寄与度順位 {row['rank_contrib']:>2}位  "
          f"(差 {row['rank_gap']:+d}、|β|={row['abs_beta_pct']:.1f}%、寄与度={row['contrib_v3_%']:.1f}%)")

print("\n=== §12.1 判定 ===")
f9 = df[df["feature"]=="f9_W"].iloc[0]
print(f"f9_W: v2 寄与度 {f9['contrib_v2_%']:.2f}% → v3 {f9['contrib_v3_%']:.2f}%")
if f9["contrib_v3_%"] < 5:
    print("  → 仕様書の「24.6%」は測定誤りの可能性大、実測では元々低寄与")
