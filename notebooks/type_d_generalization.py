# -*- coding: utf-8 -*-
"""
案 D の汎化検証: train 期間 (2023-05〜2025-12) に同じ閾値を適用し、
test 期間 (2026-01〜04) との結果一致を検証。
"""
from __future__ import annotations
import io
import sys
import runpy
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "notebooks" / "output"
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)


# ================================================================
# Step 0: v2_ext_newH パイプライン実行 → train/test 両方の X, β を取得
# ================================================================
log("=" * 60)
log("[Step 0] pl_with_extended_data.py を runpy 実行 (train/test 両方取得)")
log("=" * 60)
_buf = io.StringIO(); _o = sys.stdout; sys.stdout = _buf
try:
    ns = runpy.run_path(str(BASE / "notebooks" / "pl_with_extended_data.py"))
finally:
    sys.stdout = _o

beta_new = ns["beta_new"]
X_train = ns["X_train_new"]; pi_train = ns["pi_train"]; keys_train = ns["keys_train"]
X_test  = ns["X_test_new"];  pi_test  = ns["pi_test"];  keys_test  = ns["keys_test"]
log(f"  β (newH) shape={beta_new.shape}")
log(f"  train: X={X_train.shape}  pi={pi_train.shape}  races={len(keys_train):,}")
log(f"  test : X={X_test.shape}   pi={pi_test.shape}   races={len(keys_test):,}")


# ================================================================
# Step 0b: S_i 計算 + 指数 + 案 D 分類 を関数化
# ================================================================
def scores_to_df(X, pi, keys, tau=1.0):
    S = X @ beta_new
    Ss = S - S.max(axis=1, keepdims=True)
    P = np.exp(Ss) / np.exp(Ss).sum(axis=1, keepdims=True)
    N = S.shape[0]
    df = pd.DataFrame({
        "date": keys["date"].values,
        "stadium": keys["stadium"].astype(int).values,
        "race_number": keys["race_number"].astype(int).values,
        "actual_1": pi[:, 0] + 1,
        "actual_2": pi[:, 1] + 1,
        "actual_3": pi[:, 2] + 1,
    })
    for i in range(6):
        df[f"S_{i+1}"] = S[:, i]
    df["P_actual_1"] = P[np.arange(N), pi[:, 0]]
    df["log_loss"] = -np.log(np.clip(df["P_actual_1"], 1e-9, None))
    df["pred_top1"] = S.argmax(axis=1) + 1
    df["top1_lane"] = S.argmax(axis=1)
    return df


def compute_indices(df, sigma_min=0.3):
    S_arr = df[[f"S_{i}" for i in range(1, 7)]].to_numpy()
    mean = S_arr.mean(axis=1)
    sigma = np.maximum(S_arr.std(axis=1, ddof=0), sigma_min)
    S_sorted = -np.sort(-S_arr, axis=1)
    top1 = S_sorted[:, 0]; top2 = S_sorted[:, 1]; top3 = S_sorted[:, 2]
    outer_max = S_arr[:, 3:6].max(axis=1)
    df["mean_S"] = mean
    df["sigma_S_raw"] = S_arr.std(axis=1, ddof=0)
    df["sigma_S"] = sigma
    df["F_S"] = (S_arr[:, 0] - mean) / sigma
    df["G_S"] = (top1 - top2) / sigma
    df["O_S"] = (outer_max - mean) / sigma
    df["N_S"] = 1.0 - (top1 - top3) / (2.0 * sigma)


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


df_tr = scores_to_df(X_train, pi_train, keys_train)
df_te = scores_to_df(X_test, pi_test, keys_test)
compute_indices(df_tr); compute_indices(df_te)
df_tr["type_d"] = df_tr.apply(classify_d, axis=1)
df_te["type_d"] = df_te.apply(classify_d, axis=1)
df_tr.to_csv(OUT / "train_S_scores.csv", index=False, encoding="utf-8-sig")
log(f"  saved: {OUT}/train_S_scores.csv  (shape={df_tr.shape})")


# ================================================================
# Step 1: 指数分布 train vs test
# ================================================================
log("\n" + "=" * 60)
log("[Step 1] 指数分布 train vs test")
log("=" * 60)

def describe_one(series):
    return {
        "mean": series.mean(),
        "std": series.std(ddof=0),
        "P5": np.percentile(series, 5),
        "P50": np.percentile(series, 50),
        "P95": np.percentile(series, 95),
    }

rows = []
for split, df in [("train", df_tr), ("test", df_te)]:
    for idx in ["sigma_S_raw", "F_S", "G_S", "O_S", "N_S"]:
        r = describe_one(df[idx]); r["split"] = split; r["index"] = idx
        rows.append(r)
dist = pd.DataFrame(rows)[["split","index","mean","std","P5","P50","P95"]]
dist_wide = dist.pivot(index="index", columns="split",
                       values=["mean","std","P5","P50","P95"]).round(4)
log("\n  分布比較 (train vs test):")
log(dist.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
dist.to_csv(OUT / "index_distribution_train_vs_test.csv",
            index=False, encoding="utf-8-sig")
log(f"\n  saved: {OUT}/index_distribution_train_vs_test.csv")

# 差分チェック
log("\n  主要指標の train-test 差:")
for idx in ["F_S","G_S","O_S","N_S"]:
    tr_m = df_tr[idx].mean(); te_m = df_te[idx].mean()
    tr_s = df_tr[idx].std(ddof=0); te_s = df_te[idx].std(ddof=0)
    log(f"    {idx}: mean Δ={te_m-tr_m:+.4f}  std Δ={te_s-tr_s:+.4f}")


# ================================================================
# Step 2: 型分布 比較
# ================================================================
log("\n" + "=" * 60)
log("[Step 2] 型分布 比較")
log("=" * 60)

def dist_table(df, split):
    N = len(df)
    d = df["type_d"].value_counts().reindex(
        ["型1_逃げ本命","型2_イン残りヒモ荒れ","型3_頭荒れ","型4_ノイズ"]).fillna(0).astype(int)
    return [{"split":split, "type":t, "N":int(n), "pct":n/N*100} for t,n in d.items()], N

tr_d, tr_N = dist_table(df_tr, "train")
te_d, te_N = dist_table(df_te, "test")
type_dist_df = pd.DataFrame(tr_d + te_d)

log(f"\n  train: N_total={tr_N:,}   test: N_total={te_N:,}")
log("\n  | 型 | train N | train % | test N | test % | 差 (pt) |")
log("  |---|---:|---:|---:|---:|---:|")
for t in ["型1_逃げ本命","型2_イン残りヒモ荒れ","型3_頭荒れ","型4_ノイズ"]:
    tr = next(r for r in tr_d if r["type"]==t)
    te = next(r for r in te_d if r["type"]==t)
    log(f"  | {t} | {tr['N']:,} | {tr['pct']:.2f}% | {te['N']:,} | {te['pct']:.2f}% | {te['pct']-tr['pct']:+.2f} |")


# ================================================================
# Step 3: 型別実績 比較
# ================================================================
log("\n" + "=" * 60)
log("[Step 3] 型別実績 比較 (top1_hit / log_loss / 1着枠分布 / 外枠絡み)")
log("=" * 60)


def stats(sub, split, t):
    n = len(sub)
    row = {"split": split, "type": t, "N": n}
    row["top1_hit_%"] = (sub["pred_top1"]==sub["actual_1"]).mean()*100 if n else np.nan
    row["log_loss"] = sub["log_loss"].mean() if n else np.nan
    for l in range(1,7):
        row[f"win_lane{l}_%"] = (sub["actual_1"]==l).mean()*100 if n else 0.0
    row["outer_in_23_%"] = ((sub["actual_2"].isin([4,5,6]) | sub["actual_3"].isin([4,5,6])).mean()*100) if n else 0.0
    # 1着=1号艇 の条件付 外枠絡み
    sub_i1 = sub[sub["actual_1"]==1]
    if len(sub_i1):
        row["outer_in_23_if_1st=1_%"] = (sub_i1["actual_2"].isin([4,5,6]) | sub_i1["actual_3"].isin([4,5,6])).mean()*100
    else:
        row["outer_in_23_if_1st=1_%"] = 0.0
    return row


stats_rows = []
for split, df in [("train", df_tr), ("test", df_te)]:
    for t in ["型1_逃げ本命","型2_イン残りヒモ荒れ","型3_頭荒れ","型4_ノイズ"]:
        stats_rows.append(stats(df[df["type_d"]==t], split, t))
statsdf = pd.DataFrame(stats_rows)
show_cols = ["split","type","N","top1_hit_%","log_loss",
             "win_lane1_%","win_lane4_%","win_lane5_%","win_lane6_%",
             "outer_in_23_%","outer_in_23_if_1st=1_%"]
log("\n  型別実績:")
log(statsdf[show_cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# 三連単 top10 (型別、train / test)
def trifecta_top10(sub):
    tri = sub["actual_1"].astype(str)+"-"+sub["actual_2"].astype(str)+"-"+sub["actual_3"].astype(str)
    return tri.value_counts().head(10)

for t in ["型1_逃げ本命","型2_イン残りヒモ荒れ","型3_頭荒れ"]:
    log(f"\n  ── 三連単 top10 比較: {t} ──")
    for split, df in [("train", df_tr), ("test", df_te)]:
        sub = df[df["type_d"]==t]
        tc = trifecta_top10(sub)
        pattern = " / ".join(f"{k}:{v/len(sub)*100:.1f}%" for k,v in tc.items())
        log(f"    {split} (N={len(sub):,}): {pattern}")

statsdf.to_csv(OUT / "type_d_train_vs_test.csv", index=False, encoding="utf-8-sig")
log(f"\n  saved: {OUT}/type_d_train_vs_test.csv")


# ================================================================
# Step 4: 不一致分析 (判定)
# ================================================================
log("\n" + "=" * 60)
log("[Step 4] 不一致分析")
log("=" * 60)

# 型分布差が 5pt 以上ある型
big_diffs = []
for t in ["型1_逃げ本命","型2_イン残りヒモ荒れ","型3_頭荒れ","型4_ノイズ"]:
    tr_p = next(r for r in tr_d if r["type"]==t)["pct"]
    te_p = next(r for r in te_d if r["type"]==t)["pct"]
    diff = abs(te_p - tr_p)
    if diff >= 5.0:
        big_diffs.append((t, tr_p, te_p, diff))

log(f"\n  型分布差 ≥5pt: {len(big_diffs)} 件")
for t, tr_p, te_p, diff in big_diffs:
    log(f"    {t}: train={tr_p:.2f}%  test={te_p:.2f}%  |Δ|={diff:.2f}pt")

# 型別 top1_hit 差
hit_diffs = []
for t in ["型1_逃げ本命","型2_イン残りヒモ荒れ","型3_頭荒れ","型4_ノイズ"]:
    tr_row = statsdf[(statsdf["split"]=="train") & (statsdf["type"]==t)].iloc[0]
    te_row = statsdf[(statsdf["split"]=="test") & (statsdf["type"]==t)].iloc[0]
    hit_diffs.append((t, tr_row["top1_hit_%"], te_row["top1_hit_%"],
                      te_row["top1_hit_%"] - tr_row["top1_hit_%"]))
log("\n  型別 top1_hit 差 (test - train):")
for t, tr_h, te_h, d in hit_diffs:
    log(f"    {t}: train={tr_h:.2f}%  test={te_h:.2f}%  Δ={d:+.2f}pt")


# 判定ロジック
def judge():
    max_dist_diff = max(abs(te_p - tr_p) for _, tr_p, te_p, _ in big_diffs) if big_diffs else 0
    hit_diff_abs = [abs(d[3]) for d in hit_diffs]
    max_hit_diff = max(hit_diff_abs)
    # (a) 型分布も実績もほぼ一致
    if max_dist_diff < 3 and max_hit_diff < 3:
        return "(a) 型分布も型別実績もほぼ一致 → 案 D は汎化 ✅"
    # (c) 型分布が乖離
    if max_dist_diff >= 5:
        offender = [t for t, tr_p, te_p, d in big_diffs]
        return f"(c) 型分布が乖離 ({offender}) → 閾値 test fit 可能性"
    # (b) 分布一致だが hit 差
    if max_dist_diff < 5 and max_hit_diff >= 5:
        return f"(b) 型分布は一致、hit 率に test バイアス (max |Δ|={max_hit_diff:.1f}pt)"
    # (d) 型3 だけ挙動違い (サンプル少)
    t3 = [d for d in hit_diffs if d[0]=="型3_頭荒れ"][0]
    if abs(t3[3]) >= 5 and all(abs(d[3]) < 5 for d in hit_diffs if d[0]!="型3_頭荒れ"):
        return f"(d) 型3 のみ挙動差 (Δ={t3[3]:+.2f}) — サンプル少 (N={len(df_te[df_te['type_d']=='型3_頭荒れ']):,})"
    return "(中間) 小さな差が複数、詳細議論必要"

verdict = judge()
log(f"\n  判定: {verdict}")


# ================================================================
# ログ保存
# ================================================================
import tempfile
(Path(tempfile.gettempdir())/"type_d_generalization.log").write_text(
    "\n".join(LOG), encoding="utf-8")
log("\n[tee] /tmp/type_d_generalization.log")
