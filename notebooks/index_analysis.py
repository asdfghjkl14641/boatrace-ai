# -*- coding: utf-8 -*-
"""
v2_ext_newH の生スコア S_i ベースの 4 指数検証。

- pl_with_extended_data.py を runpy 経由で実行し、β と X_test_new を取得
- 各レースの S_i (i=1..6) を計算
- Step 2〜5 の分析を実行
- 最終結果は stdout + notebooks/output/* へ
"""
from __future__ import annotations
import io
import sys
import runpy
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "notebooks" / "output"
OUT.mkdir(parents=True, exist_ok=True)
LOG = []


def log(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    LOG.append(s)


# ================================================================
# 0. v2_ext_newH パイプラインを丸ごと実行して β/X_test を取得
# ================================================================
log("=" * 60)
log("[0] pl_with_extended_data.py を runpy 実行し scores 抽出…")
log("=" * 60)

# runpy 実行中の長い stdout を抑える
_dev_null = io.StringIO()
_orig_stdout = sys.stdout
sys.stdout = _dev_null
try:
    ns = runpy.run_path(str(BASE / "notebooks" / "pl_with_extended_data.py"))
finally:
    sys.stdout = _orig_stdout

beta_new = ns["beta_new"]
X_test_new = ns["X_test_new"]
pi_test = ns["pi_test"]
keys_test = ns["keys_test"]
FEATURES_NEW = ns["FEATURES_NEW"]
log(f"  β (newH) = {np.array2string(beta_new, precision=3)}")
log(f"  X_test_new shape = {X_test_new.shape}")
log(f"  test races = {len(keys_test):,}")


# ================================================================
# 1. スコア計算 & ロング形式 DataFrame
# ================================================================
log("\n" + "=" * 60)
log("[1] スコア計算")
log("=" * 60)

# S shape = (N_races, 6 lanes)
S = X_test_new @ beta_new
N = S.shape[0]
log(f"  S shape = {S.shape}")
log(f"  S stats: mean={S.mean():.3f} std={S.std():.3f} min={S.min():.3f} max={S.max():.3f}")

# wide 形式にまとめる (1 レース = 1 行)
rows = []
for i in range(N):
    key = keys_test.iloc[i]
    row = {
        "date": key["date"],
        "stadium": int(key["stadium"]),
        "race_number": int(key["race_number"]),
        "actual_1": int(pi_test[i, 0]) + 1,
        "actual_2": int(pi_test[i, 1]) + 1,
        "actual_3": int(pi_test[i, 2]) + 1,
    }
    for l in range(6):
        row[f"S_{l+1}"] = float(S[i, l])
    rows.append(row)
df = pd.DataFrame(rows)
log(f"  DataFrame shape = {df.shape}")
log(f"  columns = {list(df.columns)}")


# ================================================================
# 2. sigma_S の分布
# ================================================================
log("\n" + "=" * 60)
log("[Step 2] sigma_S_raw (6艇 S の標準偏差 ddof=0) の分布")
log("=" * 60)

S_cols = [f"S_{i}" for i in range(1, 7)]
S_arr = df[S_cols].to_numpy()
mean_S = S_arr.mean(axis=1)
sigma_raw = S_arr.std(axis=1, ddof=0)
df["mean_S"] = mean_S
df["sigma_S_raw"] = sigma_raw

percentiles = [0, 1, 5, 50, 95, 99, 100]
desc = pd.Series({
    "mean": sigma_raw.mean(), "std": sigma_raw.std(),
    **{f"P{p}": np.percentile(sigma_raw, p) for p in percentiles},
})
log("\n  sigma_S_raw 統計:")
for k, v in desc.items():
    log(f"    {k:>6s}: {v:.4f}")

log("\n  sigma_S_raw < 閾値 のレース比率:")
for thr in [0.1, 0.2, 0.3, 0.5]:
    n = int((sigma_raw < thr).sum())
    log(f"    < {thr}: {n:>5,} races ({n/N*100:.2f}%)")


# ================================================================
# 3. sigma_min 感度分析
# ================================================================
log("\n" + "=" * 60)
log("[Step 3] σ_min 感度分析")
log("=" * 60)

def compute_indices(S_arr, sigma_min):
    mean = S_arr.mean(axis=1)
    sigma = np.maximum(S_arr.std(axis=1, ddof=0), sigma_min)
    # top1/top2/top3 の S 値 (降順ソートの先頭 3)
    S_sorted = -np.sort(-S_arr, axis=1)   # 降順
    top1 = S_sorted[:, 0]
    top2 = S_sorted[:, 1]
    top3 = S_sorted[:, 2]
    # 外枠 lane ∈ {4,5,6} は index 3..5
    outer_max = S_arr[:, 3:6].max(axis=1)
    F = (S_arr[:, 0] - mean) / sigma
    G = (top1 - top2) / sigma
    O = (outer_max - mean) / sigma
    Ns = 1.0 - (top1 - top3) / (2.0 * sigma)
    return F, G, O, Ns, sigma

sigma_mins = [0.0, 0.2, 0.3, 0.5]
summary_rows = []
for sm in sigma_mins:
    F, G, O, Ns, sigma = compute_indices(S_arr, sm)
    clip_rate = (sigma == sm).mean() * 100 if sm > 0 else 0.0
    for name, vals in [("F_S", F), ("G_S", G), ("O_S", O), ("N_S", Ns)]:
        summary_rows.append({
            "sigma_min": sm, "index": name,
            "mean": float(vals.mean()), "std": float(vals.std(ddof=0)),
            "P5": float(np.percentile(vals, 5)),
            "P50": float(np.percentile(vals, 50)),
            "P95": float(np.percentile(vals, 95)),
            "|max|": float(np.abs(vals).max()),
            "clip_rate_%": clip_rate,
        })
sens = pd.DataFrame(summary_rows)
log("\n")
log(sens.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
sens.to_csv(OUT / "index_sigma_sensitivity.csv", index=False, encoding="utf-8-sig")
log(f"\n  saved: {OUT}/index_sigma_sensitivity.csv")


# ================================================================
# σ_min = 0.3 で固定して以降の分析
# ================================================================
SIGMA_MIN = 0.3
log(f"\n→ 以降は σ_min = {SIGMA_MIN} で固定")

F_S, G_S, O_S, N_S, sigma_eff = compute_indices(S_arr, SIGMA_MIN)
df["F_S"] = F_S
df["G_S"] = G_S
df["O_S"] = O_S
df["N_S"] = N_S
df["sigma_S"] = sigma_eff


# ================================================================
# 4. 可視化 (2×3 グリッド)
# ================================================================
log("\n" + "=" * 60)
log("[Step 4] 分布可視化")
log("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

axes[0, 0].hist(sigma_raw, bins=80, color="#4c72b0")
axes[0, 0].axvline(SIGMA_MIN, color="red", linestyle="--", label=f"σ_min={SIGMA_MIN}")
axes[0, 0].set_title("sigma_S_raw")
axes[0, 0].set_xlabel("sigma_S_raw"); axes[0, 0].legend()

axes[0, 1].hist(F_S, bins=60, color="#55a868")
axes[0, 1].axvline(0, color="gray", alpha=0.4); axes[0, 1].axvline(1.0, color="red", alpha=0.5, ls="--")
axes[0, 1].set_title("F_S  (1号艇信頼度)")
axes[0, 1].set_xlabel("F_S = (S_1 - mean) / sigma")

axes[0, 2].hist(G_S, bins=60, color="#c44e52")
axes[0, 2].axvline(0.7, color="red", alpha=0.5, ls="--")
axes[0, 2].set_title("G_S  (決定度)")
axes[0, 2].set_xlabel("G_S = (S_top1 - S_top2) / sigma")

axes[1, 0].hist(O_S, bins=60, color="#8172b2")
axes[1, 0].axvline(0.5, color="red", alpha=0.5, ls="--")
axes[1, 0].set_title("O_S  (外枠 max z)")
axes[1, 0].set_xlabel("O_S = max_{4..6}(S - mean)/sigma")

axes[1, 1].hist(N_S, bins=60, color="#ccb974")
axes[1, 1].axvline(0.4, color="red", alpha=0.5, ls="--")
axes[1, 1].set_title("N_S  (top3 接近度)")
axes[1, 1].set_xlabel("N_S = 1 - (S_top1 - S_top3) / (2*sigma)")

axes[1, 2].scatter(G_S, N_S, s=2, alpha=0.15, color="#4c72b0")
axes[1, 2].set_xlabel("G_S"); axes[1, 2].set_ylabel("N_S")
axes[1, 2].set_title("G_S vs N_S")
axes[1, 2].axvline(0.7, color="red", alpha=0.3); axes[1, 2].axhline(0.4, color="red", alpha=0.3)

plt.tight_layout()
out_png = OUT / "index_distributions.png"
plt.savefig(out_png, dpi=130)
plt.close()
log(f"  saved: {out_png}")


# ================================================================
# 5. 仮の型分類 + 実績クロス
# ================================================================
log("\n" + "=" * 60)
log("[Step 5] 型分類 + 実績クロス集計")
log("=" * 60)

def classify(row):
    F, G, O, Ns = row["F_S"], row["G_S"], row["O_S"], row["N_S"]
    if F > 1.0 and G > 0.7:
        return "型1_逃げ本命"
    if F > 0.3 and Ns > 0.4:
        return "型2_イン残り"
    if O > 0.5 and G > 0.4:
        return "型3_頭荒れ"
    return "型4_ノイズ"

df["race_type"] = df.apply(classify, axis=1)

# pred top1 = argmax S
df["pred_top1"] = S_arr.argmax(axis=1) + 1

# メトリクス集計
log("\n  型分布:")
typ_dist = df["race_type"].value_counts().sort_index()
for k, v in typ_dist.items():
    log(f"    {k:16s}: {v:>5,} races ({v/len(df)*100:5.2f}%)")

# 型別集計
agg_rows = []
for t, sub in df.groupby("race_type"):
    n = len(sub)
    # 1着枠番分布 (1..6)
    winner_dist = sub["actual_1"].value_counts().sort_index()
    hit_by_lane = {l: int(winner_dist.get(l, 0)) for l in range(1, 7)}
    hit_rate_by_lane = {l: hit_by_lane[l] / n * 100 for l in range(1, 7)}
    # top1 hit率
    top1_hit = (sub["pred_top1"] == sub["actual_1"]).mean() * 100
    # 上位 3連単
    trifecta = sub.apply(lambda r: f"{r['actual_1']}-{r['actual_2']}-{r['actual_3']}", axis=1)
    top5 = trifecta.value_counts().head(5).to_dict()
    row = {
        "race_type": t, "N": n, "share_%": n / len(df) * 100,
        "top1_hit_%": top1_hit,
    }
    for l in range(1, 7):
        row[f"lane{l}_win_%"] = hit_rate_by_lane[l]
    row["top5_trifecta"] = " / ".join(f"{k}:{v}" for k, v in top5.items())
    agg_rows.append(row)

agg = pd.DataFrame(agg_rows)
log("\n  型別 集計:")
cols_show = ["race_type", "N", "share_%", "top1_hit_%",
             "lane1_win_%", "lane2_win_%", "lane3_win_%",
             "lane4_win_%", "lane5_win_%", "lane6_win_%"]
log(agg[cols_show].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
log("\n  型別 トップ 5 三連単:")
for _, r in agg.iterrows():
    log(f"    {r['race_type']}: {r['top5_trifecta']}")

# 全体の top1 hit率参考
overall_top1 = (df["pred_top1"] == df["actual_1"]).mean() * 100
log(f"\n  (参考) 全体 top1 hit率: {overall_top1:.2f}% / N={len(df):,}")

agg.to_csv(OUT / "type_summary.csv", index=False, encoding="utf-8-sig")
log(f"\n  saved: {OUT}/type_summary.csv")

# --- 拡張ログ保存 (S_1..S_6 + 指数) ---
df.to_csv(OUT / "pl_race_log_with_scores.csv", index=False, encoding="utf-8-sig")
log(f"  saved: {OUT}/pl_race_log_with_scores.csv (shape={df.shape})")

# tee /tmp/index_analysis.log
import tempfile
tmpdir = Path(tempfile.gettempdir())
log_path = tmpdir / "index_analysis.log"
log_path.write_text("\n".join(LOG), encoding="utf-8")
print(f"\n[tee] full log → {log_path}")
