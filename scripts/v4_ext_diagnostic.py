# -*- coding: utf-8 -*-
"""v4_ext 追加検証 (Phase A-ext-6 前): 3 検証で問題の構造特定."""
from __future__ import annotations
import io, os, sys, runpy, pickle, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import sqlite3

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "notebooks" / "output"
sys.path.insert(0, str(BASE))
sys.stdout.reconfigure(encoding="utf-8")

DB = BASE / "boatrace.db"
print("=" * 80); print("v4_ext 追加検証 — 3 validations"); print("=" * 80)

# ========== v4_ext runpy (tensor + PCA + C-layer 再構築, 同じ env で) ==========
print("\n[0] v4_ext runpy 経由で tensor 等をロード ...")
os.environ["V4_TRAIN_FROM"] = "2020-02-01"
os.environ["V4_TRAIN_TO"]   = "2025-06-30"
os.environ["V4_TEST_FROM"]  = "2025-07-01"
os.environ["V4_TEST_TO"]    = "2026-04-18"
os.environ["V4_OUT_SUFFIX"] = "_ext"

_buf = io.StringIO(); _o = sys.stdout; sys.stdout = _buf
try:
    ns = runpy.run_path(str(BASE / "notebooks" / "pl_v4_training.py"))
finally:
    sys.stdout = _o

beta_v4_ext = ns["beta_v4"]
X_test = ns["X_test_v4"]; pi_test = ns["pi_test_v4"]; keys_test = ns["keys_test_v4"].reset_index(drop=True)
X_train = ns["X_train_v4"]; pi_train = ns["pi_train_v4"]
evaluate = ns["evaluate"]
print(f"  β_v4_ext shape={beta_v4_ext.shape}, X_test={X_test.shape}, X_train={X_train.shape}")

# ========== 検証 1: 同じ test で v4 vs v4_ext ==========
print("\n" + "=" * 80)
print("[検証 1] 同じ test (2025-07〜2026-04) で v4 vs v4_ext 比較")
print("=" * 80)
# v4 の β を読み込み
v4w = pd.read_csv(OUT / "v4_weights.csv")
beta_v4 = v4w["beta_v4"].values
print(f"  v4 β (10 features + f11 + f12): {np.array2string(beta_v4, precision=3)}")
print(f"  v4_ext β                      : {np.array2string(beta_v4_ext, precision=3)}")

# 同じ test tensor (X_test は v4_ext の features で構築されているので、v4 β を当てて評価)
# 厳密には v4 features と v4_ext features は f11/f12 が違うが、同じ test 集合での性能比較として意味あり
print("\n  同じ test_tensor (v4_ext features 基準) で比較:")
m_v4     = evaluate(beta_v4,     X_test, pi_test, tau=0.8, label="v4 (v4 β)")
m_v4_ext = evaluate(beta_v4_ext, X_test, pi_test, tau=0.8, label="v4_ext")

rows = []
for label, m in [("v4 (on ext test)", m_v4), ("v4_ext", m_v4_ext)]:
    rows.append({"model": label, "hit1": m["hit1"], "hit2": m["hit2"], "hit3": m["hit3"],
                 "ll": m["ll"], "pl_ll": m["pl_ll"]})
cmp_df = pd.DataFrame(rows)
cmp_df.to_csv(OUT/"v4_ext_same_test_comparison.csv", index=False, encoding="utf-8-sig")
print(cmp_df.to_string(index=False))

diff_hit1 = m_v4_ext["hit1"] - m_v4["hit1"]
diff_ll = m_v4_ext["ll"] - m_v4["ll"]
print(f"\n  差 (ext - v4): hit1={diff_hit1*100:+.3f}pt, LL={diff_ll:+.4f}")

if abs(diff_hit1) < 0.003:
    v1_verdict = "中立"
elif diff_hit1 > 0.003:
    v1_verdict = "期間拡大は有効"
else:
    v1_verdict = "期間拡大は悪手"
print(f"  検証1 判定: {v1_verdict}")

# ========== 検証 2: f5_motor 激減原因 ==========
print("\n" + "=" * 80)
print("[検証 2] f5_motor 激減 原因分析")
print("=" * 80)

conn = sqlite3.connect(str(DB))
# 2.1 motor_in2nd 年別分布
print("\n  [2.1] motor_in2nd 年別分布 (race_cards):")
q = """
SELECT
  substr(date,1,4) AS year,
  COUNT(*) AS n,
  AVG(motor_in2nd) AS mean,
  MIN(motor_in2nd) AS mn,
  MAX(motor_in2nd) AS mx
FROM race_cards
WHERE date BETWEEN '2020-02-01' AND '2026-04-18' AND motor_in2nd IS NOT NULL
GROUP BY year ORDER BY year
"""
m_df = pd.read_sql_query(q, conn)
# std 別途
std_rows = []
for y in m_df["year"]:
    s = conn.execute(f"SELECT AVG((motor_in2nd - (SELECT AVG(motor_in2nd) FROM race_cards WHERE substr(date,1,4)='{y}'))*(motor_in2nd - (SELECT AVG(motor_in2nd) FROM race_cards WHERE substr(date,1,4)='{y}'))) FROM race_cards WHERE substr(date,1,4)='{y}'").fetchone()[0]
    std_rows.append(np.sqrt(s) if s else 0)
m_df["std"] = std_rows
print(m_df.to_string(index=False))

# 2.2 f5_motor (race 内 z-score) 分布: train 期間の v4_ext features で確認
# X_train[:,:,3] = f5_motor column (3 番目の feature)
print("\n  [2.2] v4_ext の f5_motor (train) 分布:")
f5_train = X_train[:,:,3].flatten()
print(f"    mean={f5_train.mean():.4f}  std={f5_train.std():.4f}  min={f5_train.min():.3f}  max={f5_train.max():.3f}")
# 非ゼロ比率
nonzero_pct = (np.abs(f5_train) > 0.01).mean() * 100
print(f"    |f5|>0.01 比率: {nonzero_pct:.2f}%")
print(f"    |f5|>0.5 比率: {(np.abs(f5_train) > 0.5).mean()*100:.2f}%")

# 2.3 期間絞り学習 (2022-01〜2025-06, 2023-05〜2025-06 の 2 通り)
print("\n  [2.3] 期間絞り学習で f5_motor β がどう変わるか")
# これは pl_v4_training を train 期間違いで 2 回 runpy する必要あり
# 重いのでスキップ、代わりに X_train を期間 subset して f5 を単独評価
# rc_train は ns["rc_train"] にある? 確認
if "rc_train_v4" in ns:
    rc_train_v4 = ns["rc_train_v4"]
    print(f"    rc_train_v4 rows: {len(rc_train_v4):,}")
    # 年別 f5_motor の分布
    rc_train_v4["year"] = pd.to_datetime(rc_train_v4["date"]).dt.year
    yearly = rc_train_v4.groupby("year")["f5_motor"].agg(["count","mean","std","min","max"])
    print("    年別 f5_motor 分布 (train 内):")
    print(yearly.to_string())
else:
    print("    (rc_train_v4 not exported, 期間別学習はスキップ)")
    yearly = pd.DataFrame()

f5_analysis = {
    "f5_train_mean": float(f5_train.mean()),
    "f5_train_std": float(f5_train.std()),
    "f5_nonzero_0.01_pct": float(nonzero_pct),
    "f5_nonzero_0.5_pct": float((np.abs(f5_train)>0.5).mean()*100),
    "beta_v4_f5": float(beta_v4[3]),
    "beta_v4_ext_f5": float(beta_v4_ext[3]),
}
pd.DataFrame([f5_analysis]).to_csv(OUT/"v4_ext_f5_motor_analysis.csv", index=False, encoding="utf-8-sig")
m_df.to_csv(OUT/"v4_ext_motor_yearly.csv", index=False, encoding="utf-8-sig")
if not yearly.empty:
    yearly.to_csv(OUT/"v4_ext_f5_yearly_in_train.csv", encoding="utf-8-sig")

v2_verdict = "正常" if f5_train.std() > 0.5 else "分布が崩壊"
print(f"\n  検証2 判定: {v2_verdict}")

# ========== 検証 3: test 期間内時系列 ==========
print("\n" + "=" * 80)
print("[検証 3] test 期間内の時系列性能")
print("=" * 80)

keys_test["date"] = pd.to_datetime(keys_test["date"])
keys_test["ym"] = keys_test["date"].dt.to_period("M").astype(str)

# 2 ヶ月ビンへ割り当て
def _bin(y):
    m = int(y[-2:]); yr = y[:4]
    if m <= 2: return f"{yr}-01〜02"
    if m <= 4: return f"{yr}-03〜04"
    if m <= 6: return f"{yr}-05〜06"
    if m <= 8: return f"{yr}-07〜08"
    if m <= 10: return f"{yr}-09〜10"
    return f"{yr}-11〜12"
keys_test["bin2m"] = keys_test["ym"].apply(_bin)

print(f"\n  {'period':<14} {'N':>6} {'hit1%':>7} {'LL':>7} {'PL_NLL':>7}")
ts_rows = []
for period in sorted(keys_test["bin2m"].unique()):
    mask = (keys_test["bin2m"] == period).values
    n = mask.sum()
    if n < 500: continue
    X_sub = X_test[mask]; pi_sub = pi_test[mask]
    m_sub = evaluate(beta_v4_ext, X_sub, pi_sub, tau=0.8, label=period)
    print(f"  {period:<14} {n:>6,} {m_sub['hit1']*100:>6.2f}% {m_sub['ll']:>7.4f} {m_sub['pl_ll']:>7.4f}")
    ts_rows.append({"period":period,"N":int(n),"hit1":m_sub["hit1"],"ll":m_sub["ll"],"pl_ll":m_sub["pl_ll"]})
ts_df = pd.DataFrame(ts_rows)
ts_df.to_csv(OUT/"v4_ext_test_time_series.csv", index=False, encoding="utf-8-sig")

hit1_std = ts_df["hit1"].std() * 100
hit1_range = (ts_df["hit1"].max() - ts_df["hit1"].min()) * 100
print(f"\n  hit1 std: {hit1_std:.2f}pt,  range: {hit1_range:.2f}pt")

if hit1_range > 3.0:
    v3_verdict = "大きな変動 → 市場変化 or モデル劣化"
elif hit1_range > 1.5:
    v3_verdict = "中程度の変動"
else:
    v3_verdict = "安定"
print(f"  検証3 判定: {v3_verdict}")

# ========== 統合判断 ==========
print("\n" + "=" * 80)
print("統合判定")
print("=" * 80)

if abs(diff_hit1) < 0.003 and f5_train.std() < 0.5:
    pattern = "I"; desc = "v4_ext ≈ v4、f5_motor 分布崩壊 → train 期間を絞る提案"
elif diff_hit1 < -0.003:
    pattern = "II"; desc = "v4_ext 明確に劣る → train 期間拡大は悪手、v4 現行で運用判断"
elif diff_hit1 > 0.003:
    pattern = "III"; desc = "v4_ext が v4 より良い → Phase A-ext-6 に進める"
elif hit1_range > 3.0:
    pattern = "IV"; desc = "test 期間内で大変動 → モデル再設計必要"
else:
    pattern = "I"; desc = "v4_ext ≈ v4、train 期間絞り込みを検討"

print(f"\n  判定: パターン {pattern}")
print(f"  {desc}")

md = f"""# v4_ext 追加検証レポート

## 検証 1: 同じ test 集合で v4 vs v4_ext

| モデル | hit1 | hit2 | hit3 | Log-loss | PL NLL |
|---|---|---|---|---|---|
| v4 (β v4, features v4_ext, test 2025-07〜2026-04) | {m_v4['hit1']*100:.2f}% | {m_v4['hit2']*100:.2f}% | {m_v4['hit3']*100:.2f}% | {m_v4['ll']:.4f} | {m_v4['pl_ll']:.4f} |
| v4_ext | {m_v4_ext['hit1']*100:.2f}% | {m_v4_ext['hit2']*100:.2f}% | {m_v4_ext['hit3']*100:.2f}% | {m_v4_ext['ll']:.4f} | {m_v4_ext['pl_ll']:.4f} |
| 差 (ext - v4) | {diff_hit1*100:+.3f}pt | | | {diff_ll:+.4f} | |

**検証1 判定**: {v1_verdict}

## 検証 2: f5_motor 激減 原因分析

### motor_in2nd 年別分布 (race_cards)
{m_df.to_markdown(index=False, floatfmt='.3f')}

### v4_ext の f5_motor 分布 (train)
- mean: {f5_train.mean():.4f}
- std: {f5_train.std():.4f}
- |f5|>0.01: {nonzero_pct:.2f}%
- β_v4: {beta_v4[3]:.5f}
- β_v4_ext: {beta_v4_ext[3]:.5f}

**検証2 判定**: {v2_verdict}

## 検証 3: test 期間内の時系列性能 (v4_ext)

| 期間 | N | hit1 | LL | PL NLL |
|---|---|---|---|---|
"""
for _, r in ts_df.iterrows():
    md += f"| {r['period']} | {r['N']:,} | {r['hit1']*100:.2f}% | {r['ll']:.4f} | {r['pl_ll']:.4f} |\n"
md += f"""

hit1 std: {hit1_std:.2f}pt, range: {hit1_range:.2f}pt

**検証3 判定**: {v3_verdict}

## 統合判定: パターン {pattern}

**{desc}**

### 推奨アクション
"""
if pattern == "I":
    md += "1. train 期間を 2022-01 以降 or 2023-05 以降に絞って v4_ext 再学習\n"
    md += "2. f5_motor の分布安定化を確認\n"
elif pattern == "II":
    md += "1. v4 現行 (2023-05〜2025-12 train) で運用判断を進める\n"
    md += "2. 期間拡大は諦める\n"
elif pattern == "III":
    md += "1. Phase A-ext-6 (6 戦略過学習検証) に進める\n"
elif pattern == "IV":
    md += "1. モデル再設計 (v5) の方針議論\n"
md += """

## 出力ファイル
- v4_ext_same_test_comparison.csv
- v4_ext_f5_motor_analysis.csv
- v4_ext_motor_yearly.csv
- v4_ext_test_time_series.csv
"""

(OUT/"v4_ext_diagnostic.md").write_text(md, encoding="utf-8")
print(f"\nsaved: v4_ext_diagnostic.md")
print(f"\n=== パターン {pattern}: {desc} ===")
conn.close()
