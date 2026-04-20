# -*- coding: utf-8 -*-
# ---
# # 波高 × 枠番 の実データ分析 (f10_H 符号問題の原因究明)
#
# 期間: 2025-08-01 〜 2026-04-18 (race_conditions カバー期間)
#
# ## 目的
# 現行の f10_H = h_coef × k(lane), k = [0,-0.05,-0.10,-0.15,-0.20,-0.25]
# は「波が高いほど外枠不利」を仮定するが、PL推定では β_H = -1.49 と強い負値。
# これは「波補正そのものが全体的にマイナス効果」と解釈され不自然。
# 実データで波×枠のパターンを検証し、再設計方針を決める。
# ---

# %% imports
import sqlite3
import math
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
DB   = BASE / "boatrace.db"
OUTD = BASE / "notebooks" / "output"
OUTD.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE))


def logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p))


# %% 1. データ取得
print("[1] データ取得中...")
conn = sqlite3.connect(DB)
df = pd.read_sql_query("""
    SELECT rr.stadium, rr.race_number, rr.boat, rr.rank,
           rc.wave_height, rc.wind_speed, rc.wind_direction
    FROM race_results rr
    JOIN race_conditions rc
      ON rr.date=rc.date AND rr.stadium=rc.stadium AND rr.race_number=rc.race_number
    WHERE rr.rank IS NOT NULL
      AND rr.boat BETWEEN 1 AND 6
      AND rc.wave_height IS NOT NULL
""", conn)
# 全期間1枠勝率 (全国平均、race_results 全データから)
rres_all = pd.read_sql_query("""
    SELECT stadium, boat, rank FROM race_results
    WHERE rank IS NOT NULL AND boat BETWEEN 1 AND 6 AND rank BETWEEN 1 AND 6
""", conn)
conn.close()
print(f"  JOIN可能レコード: {len(df):,}")

# 波高ビン化
def wave_bin(h):
    if h <= 1: return 0      # 0-1cm (波なし)
    if h <= 3: return 1      # 2-3cm (軽微)
    if h <= 5: return 2      # 4-5cm (中)
    if h <= 9: return 3      # 6-9cm (高)
    return 4                 # 10cm以上 (激しい)

df["wave_bin"] = df["wave_height"].apply(wave_bin)
BIN_LABEL = {0:"0-1cm", 1:"2-3cm", 2:"4-5cm", 3:"6-9cm", 4:"10cm+"}


# %% 2. 全国平均 1着率 (全race_resultsから、ベースライン)
g_nat = rres_all.groupby("boat")
nat_win = g_nat.apply(lambda s: (s["rank"]==1).mean())
print("\n全国平均 1着率 (ベースライン):")
for b in range(1, 7):
    print(f"  枠{b}: {nat_win[b]*100:.2f}%")


# %% 3. 波ビン × 枠番 1着率
print("\n[2] 波×枠 クロス集計...")
rows = []
for wb in range(5):
    sub = df[df["wave_bin"] == wb]
    total_races = sub[sub["boat"]==1].shape[0]
    for b in range(1, 7):
        lane_sub = sub[sub["boat"]==b]
        n = len(lane_sub)
        w1 = int((lane_sub["rank"]==1).sum())
        p = w1/n if n else 0.0
        p_nat = nat_win[b]
        dlogit = logit(p) - logit(p_nat) if 0 < p < 1 else np.nan
        se = math.sqrt(p*(1-p)/n) if n else 1.0
        rows.append({
            "wave_bin": wb, "wave_label": BIN_LABEL[wb],
            "lane": b, "N": n, "wins": w1,
            "p_win": p*100, "p_nat": p_nat*100,
            "delta_pct": (p - p_nat)*100,
            "delta_logit": dlogit,
            "se_pct": se*100,
            "total_races_this_bin": total_races,
        })
tab = pd.DataFrame(rows)

# ピボット表示 (勝率%)
pivot_p = tab.pivot(index="wave_bin", columns="lane", values="p_win")
pivot_p.index = [BIN_LABEL[i] for i in pivot_p.index]
pivot_n = tab.pivot(index="wave_bin", columns="lane", values="N")
pivot_n.index = [BIN_LABEL[i] for i in pivot_n.index]
pivot_dl = tab.pivot(index="wave_bin", columns="lane", values="delta_logit")
pivot_dl.index = [BIN_LABEL[i] for i in pivot_dl.index]

print("\n## 波高ビン × 枠番 1着率 (%)")
print(pivot_p.round(2).to_string())
print("\n## 各セルのサンプル数 N")
print(pivot_n.to_string())
print("\n## logit補正 Δ = logit(p_bin_l) - logit(p_national_l)")
print(pivot_dl.round(3).to_string())


# %% 4. パターン判定
print("\n[3] パターン判定...")
print("\n各波ビンにおける、枠番別1着率の全国平均との差 Δ% :")
print()
print("| 波ビン | 枠1Δ | 枠2Δ | 枠3Δ | 枠4Δ | 枠5Δ | 枠6Δ | 合計 |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for wb in range(5):
    sub = tab[tab["wave_bin"]==wb]
    deltas = [sub[sub["lane"]==b]["delta_pct"].iloc[0] for b in range(1,7)]
    total = sum(deltas)
    line = f"| {BIN_LABEL[wb]} | " + " | ".join(f"{d:+.2f}" for d in deltas) + f" | {total:+.2f} |"
    print(line)
print()
# 解釈: wave_bin 高いほど 枠1 Δ がマイナス、外枠 Δ がプラスなら パターンA (sacrifice_outer 逆)
# 外枠 Δ がマイナスなら パターンB (現在の設計通り)


# %% 5. 「波あり vs 波なし」比較
print("\n[4] 波なし (0-1cm) vs 波あり (4cm以上) 比較:")
no_wave = tab[tab["wave_bin"]==0].set_index("lane")["p_win"]
high_wave = tab[tab["wave_bin"]>=2].groupby("lane").apply(
    lambda g: (g["wins"].sum()/g["N"].sum()*100) if g["N"].sum() else 0
)
print()
print("| 枠 | 波なし(0-1cm) | 波あり(4cm+) | Δ (pt) |")
print("|---:|---:|---:|---:|")
for b in range(1, 7):
    nw = no_wave.get(b, 0)
    hw = high_wave.get(b, 0)
    print(f"| {b} | {nw:.2f}% | {hw:.2f}% | {hw-nw:+.2f} |")


# %% 6. H 再設計の提案
print("\n## 再設計の提案 (データに基づく)")
# パターン判定を自動化
dw1 = float(high_wave.get(1, 0) - no_wave.get(1, 0))  # 1枠の変化
dw6 = float(high_wave.get(6, 0) - no_wave.get(6, 0))  # 6枠の変化
total_change = sum(float(high_wave.get(b,0) - no_wave.get(b,0)) for b in range(1,7))

if abs(total_change) > 3.0 and dw1 < -2.0:
    pattern = "A"  # 全枠低下、特に1枠
elif dw6 < dw1 and dw6 < -1.0 and dw1 > -0.5:
    pattern = "B"  # 外枠ほど下がる
elif abs(dw1) < 1.0 and abs(dw6) < 1.0:
    pattern = "C"  # ほぼ中立
else:
    pattern = "A/B mix"

print(f"\n検出パターン: **{pattern}**")
print(f"  1枠の変化 (波なし→波あり): {dw1:+.2f} pt")
print(f"  6枠の変化 (波なし→波あり): {dw6:+.2f} pt")
print(f"  1-6枠合計変化: {total_change:+.2f} pt")


# %% 7. CSV 出力
tab.to_csv(OUTD / "wave_lane_crosstab.csv", index=False, encoding="utf-8-sig")

logit_df = pivot_dl.copy()
logit_df.columns = [f"lane_{b}" for b in logit_df.columns]
logit_df.reset_index(names="wave_bin").to_csv(
    OUTD / "wave_lane_logit_correction.csv", index=False, encoding="utf-8-sig")

# Markdown 提案書
md_out = OUTD / "h_redesign_proposal.md"
with open(md_out, "w", encoding="utf-8") as f:
    f.write(f"""# f10_H 再設計提案

## 波×枠 クロス集計 (期間: 2025-08-01 〜 2026-04-18)

### 1着率 (%)
{pivot_p.round(2).to_markdown()}

### 各セルのサンプル数 N
{pivot_n.to_markdown()}

### logit補正 Δ = logit(p_bin_l) - logit(p_national_l)
{pivot_dl.round(3).to_markdown()}

## 「波なし」vs「波あり」の枠別変化

| 枠 | 波なし(0-1cm) | 波あり(4cm+) | Δ (pt) |
|---:|---:|---:|---:|
""")
    for b in range(1, 7):
        nw = no_wave.get(b, 0)
        hw = high_wave.get(b, 0)
        f.write(f"| {b} | {nw:.2f}% | {hw:.2f}% | {hw-nw:+.2f} |\n")

    f.write(f"""
## 検出パターン: **{pattern}**

- 1枠の波なし→波あり 変化: **{dw1:+.2f} pt**
- 6枠の波なし→波あり 変化: **{dw6:+.2f} pt**
- 全枠合計変化: {total_change:+.2f} pt (≒0 なら「配分」、大きく負なら「全体低下」)

## 推奨設計

""")
    if pattern == "A":
        f.write("""### パターンA: 波で全枠が乱れる、1枠が強く崩れる

風補正と同じ **カーネル方式** を採用する:

```python
# 波強度に応じた logit シフトを 1枠に与え、
# kernel で他枠に配分
H_MAGNITUDE_MAP = {
    (0,): 0.00,   # 0-1cm: 補正なし
    (1,): 0.00,   # 2-3cm: 補正なし
    (2,): -0.10,  # 4-5cm: 1枠がやや崩れる
    (3,): -0.25,  # 6-9cm: 1枠が大きく崩れる
    (4,): -0.40,  # 10cm+: 1枠がかなり崩れる
}
# カーネル: 1枠崩れの分を配分
KERNEL_H = [-1.00, +0.30, +0.25, +0.20, +0.15, +0.10]  # 内寄り分配 (差し気味)
```
""")
    elif pattern == "B":
        f.write("""### パターンB: 外枠ほど下がる (現在の設計方向性は正しい)

係数を実測値で再調整:

```python
def f10_H_new(wave_bin, lane):
    h_coef = [0.0, 0.0, 0.3, 0.7, 1.0][wave_bin]
    # k(lane) を実測 Δlogit から再計算済みの値にする
    # ... (実測値は wave_lane_logit_correction.csv を参照)
    return h_coef * k_empirical[lane]
```

現行 k = [0, -0.05, -0.10, -0.15, -0.20, -0.25] は線形減衰を仮定しているが、
実測では非線形 (例: 1-3枠はほぼ変わらず、4-6枠のみ大きく下がる) の可能性。
""")
    elif pattern == "C":
        f.write("""### パターンC: 波はほぼ中立 → f10_H を削除

- サンプル内で波の影響が有意でない
- f10_H を特徴量から外す (9特徴量モデルへ)
""")
    else:
        f.write("""### パターン A/B 混合: 詳細に応じて場合分け

1枠の変化 (Δ<0 なら風と同じ)、外枠の変化 (Δ<0 なら現行設計) を見ながら
ハイブリッドな設計に。
""")

print(f"\nCSV/Markdown 保存:")
print(f"  {OUTD}/wave_lane_crosstab.csv")
print(f"  {OUTD}/wave_lane_logit_correction.csv")
print(f"  {md_out}")
