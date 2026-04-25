# -*- coding: utf-8 -*-
"""
Phase B: 型分類 (案 D) を v4 下で再検証
"""
from __future__ import annotations
import io, sys, runpy
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "notebooks" / "output"
sys.path.insert(0, str(BASE))
sys.stdout.reconfigure(encoding="utf-8")
import warnings; warnings.filterwarnings("ignore")

print("=" * 80)
print("Phase B: 型分類を v4 下で再検証")
print("=" * 80)

# === v4 training を runpy で取得 ===
print("\n[0] pl_v4_training.py を runpy ...")
_buf = io.StringIO(); _o = sys.stdout; sys.stdout = _buf
try:
    ns = runpy.run_path(str(BASE / "notebooks" / "pl_v4_training.py"))
finally:
    sys.stdout = _o
beta_v4 = ns["beta_v4"]
X_train_v4 = ns["X_train_v4"]; pi_train_v4 = ns["pi_train_v4"]; keys_train_v4 = ns["keys_train_v4"]
X_test_v4  = ns["X_test_v4"];  pi_test_v4  = ns["pi_test_v4"];  keys_test_v4  = ns["keys_test_v4"]
print(f"  β_v4 = {np.array2string(beta_v4, precision=3)}")
print(f"  train: {X_train_v4.shape}  test: {X_test_v4.shape}")

SIGMA_MIN = 0.3


def compute_indices(X, pi, keys):
    """5 指数 + 型を DataFrame で返す"""
    S = X @ beta_v4  # (N, 6)
    mean = S.mean(axis=1)
    sigma_raw = S.std(axis=1, ddof=0)
    sigma = np.maximum(sigma_raw, SIGMA_MIN)
    S_sorted = -np.sort(-S, axis=1)
    top1 = S_sorted[:, 0]; top2 = S_sorted[:, 1]; top3 = S_sorted[:, 2]
    outer_max = S[:, 3:6].max(axis=1)
    F = (S[:, 0] - mean) / sigma
    G = (top1 - top2) / sigma
    O = (outer_max - mean) / sigma
    Ns = 1.0 - (top1 - top3) / (2.0 * sigma)
    top1_lane = S.argmax(axis=1)
    # actual 1st/2nd/3rd
    df = pd.DataFrame({
        "date": keys["date"].values,
        "stadium": keys["stadium"].astype(int).values,
        "race_number": keys["race_number"].astype(int).values,
        "actual_1": pi[:, 0] + 1, "actual_2": pi[:, 1] + 1, "actual_3": pi[:, 2] + 1,
        "pred_top1": top1_lane + 1,
        "top1_lane": top1_lane,
        "sigma_S_raw": sigma_raw, "sigma_S": sigma, "mean_S": mean,
        "F_S": F, "G_S": G, "O_S": O, "N_S": Ns,
    })
    # 型 D 分類
    def classify(r):
        if r["top1_lane"] == 0:
            if r["G_S"] > 1.0 and r["O_S"] < 0.3: return "型1_逃げ本命"
            elif r["G_S"] > 0.6 and r["O_S"] > 0.2: return "型2_イン残りヒモ荒れ"
            else: return "型4_ノイズ"
        else:
            if r["O_S"] > 0.3 and r["G_S"] > 0.4: return "型3_頭荒れ"
            else: return "型4_ノイズ"
    df["type_d"] = df.apply(classify, axis=1)
    # log loss (τ=0.8 デフォルト、v3 と揃える)
    TAU = 0.7  # v4 最適
    S_tau = S / TAU
    Ss = S_tau - S_tau.max(axis=1, keepdims=True)
    P = np.exp(Ss) / np.exp(Ss).sum(axis=1, keepdims=True)
    df["P_actual_1"] = P[np.arange(len(df)), pi[:, 0]]
    df["log_loss"] = -np.log(np.clip(df["P_actual_1"], 1e-9, None))
    return df


print("\n[1] v4 の S_i + 指数 + 型 D (train + test)")
df_tr = compute_indices(X_train_v4, pi_train_v4, keys_train_v4)
df_te = compute_indices(X_test_v4, pi_test_v4, keys_test_v4)
print(f"  train: {len(df_tr):,}  test: {len(df_te):,}")


# === Step 3: 指数分布比較 ===
print("\n" + "=" * 80)
print("[Step 3] 指数分布比較 v3 vs v4")
print("=" * 80)
# v3 baseline values (from prior work)
v3_stats = {
    "train": {
        "sigma_S_raw": {"mean":0.832,"std":0.234},
        "F_S":         {"mean":1.521,"std":0.452},
        "G_S":         {"mean":0.994,"std":0.596},
        "O_S":         {"mean":0.163,"std":0.465},
        "N_S":         {"mean":0.279,"std":0.289},
    },
    "test": {
        "sigma_S_raw": {"mean":0.853,"std":0.235},
        "F_S":         {"mean":1.538,"std":0.453},
        "G_S":         {"mean":1.016,"std":0.595},
        "O_S":         {"mean":0.154,"std":0.467},
        "N_S":         {"mean":0.264,"std":0.284},
    },
}

dist_rows = []
print(f"\n  {'期間':>6} {'指数':>12} {'v3 mean':>8} {'v3 std':>7} {'v4 mean':>8} {'v4 std':>7} {'Δmean':>7} {'Δstd':>7}")
for period, df in [("train", df_tr), ("test", df_te)]:
    for idx in ["sigma_S_raw","F_S","G_S","O_S","N_S"]:
        v4m = df[idx].mean(); v4s = df[idx].std(ddof=0)
        v3m = v3_stats[period][idx]["mean"]; v3s = v3_stats[period][idx]["std"]
        dm = v4m - v3m; ds = v4s - v3s
        flag = "⚠️" if (abs(dm) > 0.1 or abs(ds) > 0.05) else ""
        print(f"  {period:>6} {idx:>12} {v3m:>+8.3f} {v3s:>7.3f} {v4m:>+8.3f} {v4s:>7.3f} {dm:>+7.3f} {ds:>+7.3f} {flag}")
        dist_rows.append({"period":period,"index":idx,"v3_mean":v3m,"v3_std":v3s,
                          "v4_mean":v4m,"v4_std":v4s,"dmean":dm,"dstd":ds})
pd.DataFrame(dist_rows).to_csv(OUT/"v3_vs_v4_type_comparison.csv", index=False, encoding="utf-8-sig")


# === 型分布比較 ===
print("\n" + "=" * 80)
print("[Step 3b] 型分布比較 v3 vs v4")
print("=" * 80)
# v3 baseline
v3_type = {
    "train": {"型1_逃げ本命":37.39, "型2_イン残りヒモ荒れ":20.04, "型3_頭荒れ":2.54, "型4_ノイズ":40.04},
    "test":  {"型1_逃げ本命":38.08, "型2_イン残りヒモ荒れ":20.48, "型3_頭荒れ":2.68, "型4_ノイズ":38.77},
}
type_rows = []
max_dtype = 0.0
print(f"\n  {'期間':>6} {'型':>20} {'v3 %':>8} {'v4 %':>8} {'Δ':>7}")
for period, df in [("train", df_tr), ("test", df_te)]:
    N = len(df)
    for t in ["型1_逃げ本命","型2_イン残りヒモ荒れ","型3_頭荒れ","型4_ノイズ"]:
        v4p = (df["type_d"]==t).mean()*100
        v3p = v3_type[period][t]
        d = v4p - v3p
        max_dtype = max(max_dtype, abs(d))
        flag = "⚠️" if abs(d) > 5 else ""
        print(f"  {period:>6} {t:>20} {v3p:>7.2f}% {v4p:>7.2f}% {d:>+6.2f}pt {flag}")
        type_rows.append({"period":period,"type":t,"v3_%":v3p,"v4_%":v4p,"diff":d})
pd.DataFrame(type_rows).to_csv(OUT/"v4_type_distribution.csv", index=False, encoding="utf-8-sig")


# === Step 4: 型別実績 (test) ===
print("\n" + "=" * 80)
print("[Step 4] 型別実績 (test 2026-01..04)")
print("=" * 80)
# v3 test baseline
v3_perf_test = {
    "型1_逃げ本命": {"top1":70.62,"ll":1.06,"win1":70.62,"win4":6.32,"win5":4.54,"win6":2.42},
    "型2_イン残りヒモ荒れ": {"top1":56.96,"ll":1.27,"win1":56.96,"win4":13.15,"win5":6.80,"win6":4.03},
    "型3_頭荒れ":        {"top1":38.13,"ll":1.51,"win1":23.06,"win4":19.63,"win5":10.73,"win6":3.42},
    "型4_ノイズ":        {"top1":45.04,"ll":1.38,"win1":42.15,"win4":10.89,"win5":6.43,"win6":3.18},
}

perf_rows = []
print(f"\n  {'type':>20} {'N':>5} {'top1':>7} {'LL':>7} {'w1':>7} {'w4':>6} {'w5':>6} {'w6':>6}")
for t in ["型1_逃げ本命","型2_イン残りヒモ荒れ","型3_頭荒れ","型4_ノイズ"]:
    sub = df_te[df_te["type_d"]==t]
    n = len(sub)
    if n == 0: continue
    top1 = (sub["pred_top1"]==sub["actual_1"]).mean()*100
    ll = sub["log_loss"].mean()
    w1 = (sub["actual_1"]==1).mean()*100
    w4 = (sub["actual_1"]==4).mean()*100
    w5 = (sub["actual_1"]==5).mean()*100
    w6 = (sub["actual_1"]==6).mean()*100
    v3 = v3_perf_test[t]
    dtop = top1 - v3["top1"]; dll = ll - v3["ll"]
    flag = "⚠️" if abs(dtop) > 5 else ""
    print(f"  {t:>20} {n:>5} {top1:>6.2f}% {ll:>7.4f} {w1:>6.2f}% {w4:>5.2f}% {w5:>5.2f}% {w6:>5.2f}% {flag}")
    print(f"    v3: top1={v3['top1']:.2f} LL={v3['ll']:.2f} | Δ top1 {dtop:+.2f}pt LL {dll:+.4f}")
    perf_rows.append({"type":t, "n":n, "top1":top1, "ll":ll, "win1":w1, "win4":w4, "win5":w5, "win6":w6,
                      "v3_top1":v3["top1"], "v3_ll":v3["ll"], "d_top1":dtop, "d_ll":dll})
pd.DataFrame(perf_rows).to_csv(OUT/"v4_type_performance.csv", index=False, encoding="utf-8-sig")


# === Step 4b: 三連単 top10 (test) ===
print("\n" + "=" * 80)
print("[Step 4b] 三連単 top10 (test) v4")
print("=" * 80)
for t in ["型1_逃げ本命","型2_イン残りヒモ荒れ","型3_頭荒れ","型4_ノイズ"]:
    sub = df_te[df_te["type_d"]==t]
    if len(sub)==0: continue
    tri = sub["actual_1"].astype(str)+"-"+sub["actual_2"].astype(str)+"-"+sub["actual_3"].astype(str)
    tc = tri.value_counts().head(10)
    print(f"\n  {t} (N={len(sub):,}) top10:")
    for k, v in tc.items():
        print(f"    {k}  {v:>4} ({v/len(sub)*100:5.2f}%)")


# === Step 5: パターン判定 ===
print("\n" + "=" * 80)
print("[Step 5] パターン判定")
print("=" * 80)
max_index_diff = max(abs(r["dmean"]) for r in dist_rows)
max_type_diff = max_dtype
max_hit_diff = max(abs(r["d_top1"]) for r in perf_rows)
print(f"\n  指数 mean 最大差: {max_index_diff:.3f}")
print(f"  型分布 最大差:    {max_type_diff:.2f}pt")
print(f"  型別 top1 hit 最大差: {max_hit_diff:.2f}pt")

if max_type_diff < 3 and max_hit_diff < 5 and max_index_diff < 0.1:
    pattern = "X"; label = "ほぼ同じ (汎化している ✅) → 案 D 閾値そのまま Phase C へ"
elif max_type_diff < 8 and max_hit_diff < 10:
    pattern = "Y"; label = "微調整推奨"
else:
    pattern = "Z"; label = "大幅再設計必要"
print(f"\n  → パターン {pattern}: {label}")


# === Phase B レポート ===
(OUT/"v4_phase_b_report.md").write_text(f"""# v4 Phase B: 型分類再検証レポート

## 判定
**パターン {pattern}**: {label}

## 指数分布 (test 比較)
| 指標 | v3 mean | v3 std | v4 mean | v4 std |
|---|---:|---:|---:|---:|
""" + "\n".join(
    [f"| {r['index']} | {r['v3_mean']:+.3f} | {r['v3_std']:.3f} | {r['v4_mean']:+.3f} | {r['v4_std']:.3f} |"
     for r in dist_rows if r['period']=='test']
) + f"""

## 型分布 (test)
| 型 | v3 % | v4 % | Δ |
|---|---:|---:|---:|
""" + "\n".join(
    [f"| {r['type']} | {r['v3_%']:.2f} | {r['v4_%']:.2f} | {r['diff']:+.2f} |"
     for r in type_rows if r['period']=='test']
) + f"""

## 型別実績 (test、top1 hit)
| 型 | N | v3 top1 | v4 top1 | Δ |
|---|---:|---:|---:|---:|
""" + "\n".join(
    [f"| {r['type']} | {r['n']} | {r['v3_top1']:.2f}% | {r['top1']:.2f}% | {r['d_top1']:+.2f}pt |"
     for r in perf_rows]
) + f"""

## サマリ
- 指数 mean 最大差: {max_index_diff:.3f}
- 型分布 最大差: {max_type_diff:.2f}pt
- 型別 top1 hit 最大差: {max_hit_diff:.2f}pt

## 次アクション
{"Phase C (期待値計算) に進む" if pattern == "X" else "閾値再調整 or 設計見直し"}
""", encoding="utf-8")

# === df_te + df_tr 保存 ===
df_all = pd.concat([df_tr.assign(split="train"), df_te.assign(split="test")], ignore_index=True)
df_all.to_csv(OUT/"v4_indices.csv", index=False, encoding="utf-8-sig")
print(f"\nsaved: v4_indices.csv ({len(df_all):,} rows)")
print(f"saved: v4_type_distribution.csv")
print(f"saved: v4_type_performance.csv")
print(f"saved: v3_vs_v4_type_comparison.csv")
print(f"saved: v4_phase_b_report.md")
