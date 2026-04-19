# -*- coding: utf-8 -*-
# ---
# # MVP バックテスト（叩き台重み）
#
# A層(3) + B層(4: 展示/モーター/今節a,b) + C層(4: L,V,W,H) の計11特徴量 + 0-flagで**12特徴量**を合成したスコアで1-3着を予想し、train/testで評価する。
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

# プロジェクトルートを path に追加 (scripts/ を import するため)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

BASE  = Path(__file__).resolve().parent.parent
DB    = BASE / "boatrace.db"
OUTD  = BASE / "notebooks" / "output"
OUTD.mkdir(parents=True, exist_ok=True)

# %% config
TRAIN_START, TRAIN_END = "2025-08-01", "2025-12-31"
TEST_START,  TEST_END  = "2026-01-01", "2026-04-18"

# スコア重み (叩き台)
W_WEIGHTS = {
    "f1_global":  1.0,
    "f2_local":   0.4,
    "f3_ST":      0.6,
    "f4_disp":    1.0,
    "f5_motor":   0.5,
    "f6_form":    0.3,
    "f6_nomatch": 0.0,
    "f7_lane_L":  1.0,
    "f8_V":       1.0,
    "f9_W":       1.0,
    "f10_H":      1.0,
}
TAU = 1.0

# 枠番 L(l) 叩き台 (1..6) — 前回の logit_3基準フィットより
LANE_L = {1: +2.06, 2: +0.12, 3: 0.0, 4: -0.26, 5: -0.82, 6: -1.53}

# 風補正 W のカーネル・マッピング
KERNEL = {
    "differential":    [-1.00, +0.35, +0.35, +0.15, +0.10, +0.05],
    "dashout":         [-1.00, +0.10, +0.10, +0.30, +0.30, +0.20],
    "even":            [-1.00, +0.20, +0.20, +0.20, +0.20, +0.20],
    "sacrifice_2":     [+1.00, -1.00,   0.0,   0.0,   0.0,   0.0],
    "sacrifice_outer": [+1.00,   0.0,   0.0,   0.0, -0.50, -0.50],
}
WIND_MAP = {
    (2, 6):   ("differential",    -6.78),
    (17, 7):  ("differential",    -6.61),
    (24, 7):  ("differential",   -13.87),
    (6, 17):  ("differential",   -10.62),
    (22, 12): ("differential",    -9.14),
    (20, 6):  ("differential",    -6.76),
    (11, 13): ("dashout",         -9.76),
    (14, 11): ("dashout",         -8.07),
    (18, 15): ("dashout",        -13.04),
    (4, 13):  ("even",            -4.41),
    (20, 14): ("even",            -8.48),
    (24, 1):  ("sacrifice_2",     +5.84),
    (23, 12): ("sacrifice_2",     +7.17),
    (17, 17): ("sacrifice_outer", +7.07),
    (4, 5):   ("sacrifice_outer", +5.98),
}
# 1枠ベース勝率 (会場) - logit変換用
P_BASE_LANE1 = {
    1:0.4839, 2:0.4440, 3:0.4827, 4:0.4547, 5:0.5339, 6:0.5180,
    7:0.5878, 8:0.5662, 9:0.5598, 10:0.5193, 11:0.5413, 12:0.5878,
    13:0.5952, 14:0.4807, 15:0.5592, 16:0.5686, 17:0.5737, 18:0.6261,
    19:0.6073, 20:0.5944, 21:0.6024, 22:0.5668, 23:0.5483, 24:0.6336,
}

def logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p))

# %% 1. データ読み込み
print("[1] データ読み込み中...")
conn = sqlite3.connect(DB)

period_filter = f"date >= '{TRAIN_START}' AND date <= '{TEST_END}'"

rc = pd.read_sql_query(f"""
    SELECT date, stadium, race_number, lane, racerid,
           global_win_pt, local_win_pt, aveST, motor_in2nd
    FROM race_cards WHERE {period_filter}
""", conn)
rcond = pd.read_sql_query(f"""
    SELECT date, stadium, race_number,
           wind_direction, wind_speed, wave_height,
           display_time_1, display_time_2, display_time_3,
           display_time_4, display_time_5, display_time_6
    FROM race_conditions WHERE {period_filter}
""", conn)
rres = pd.read_sql_query(f"""
    SELECT date, stadium, race_number, boat, rank
    FROM race_results WHERE {period_filter} AND rank IS NOT NULL
""", conn)
# current_series は EWMA のため、期間の手前も含め7日分余裕をもって取る
cs = pd.read_sql_query(f"""
    SELECT date, stadium, racerid, race_number, rank
    FROM current_series
    WHERE date >= '2025-07-15' AND date <= '{TEST_END}' AND rank IS NOT NULL
""", conn)
# V(stadium, lane) 算出用に全期間の結果も取る
rres_all = pd.read_sql_query("""
    SELECT stadium, boat, rank FROM race_results
    WHERE rank IS NOT NULL AND boat BETWEEN 1 AND 6 AND rank BETWEEN 1 AND 6
""", conn)
conn.close()
print(f"  race_cards={len(rc):,}  race_conditions={len(rcond):,}  race_results={len(rres):,}  current_series={len(cs):,}")

# racerid NULL 除外
rc = rc.dropna(subset=["racerid"]).copy()
rc["racerid"] = rc["racerid"].astype("int64")
rc["stadium"] = rc["stadium"].astype("int64")
cs = cs.dropna(subset=["racerid"]).copy()
cs["racerid"] = cs["racerid"].astype("int64")
cs["stadium"] = cs["stadium"].astype("int64")

# %% 2. 特徴量計算: A層
print("[2] A層 (全国/当地/ST)...")
rc["f1_global"] = (rc["global_win_pt"] - 5.3) / 1.3
f2_raw = (rc["local_win_pt"] - 5.4) / 1.3
rc["f2_local"] = np.where(rc["local_win_pt"] >= 2.4, f2_raw, rc["f1_global"])
rc["f3_ST"]    = -(rc["aveST"] - 0.16) / 0.023

# %% 2.2 B層 (展示、モーター、今節)
print("[2] B層 (展示/モーター/今節)...")

# B-4: display_time をレース内z-score
dt_long = rcond.melt(
    id_vars=["date","stadium","race_number"],
    value_vars=[f"display_time_{i}" for i in range(1,7)],
    var_name="lane", value_name="dt")
dt_long["lane"] = dt_long["lane"].str.extract(r"(\d)").astype(int)
dt_v = dt_long.dropna(subset=["dt"])
dt_v = dt_v[dt_v["dt"] > 0].copy()
g = dt_v.groupby(["date","stadium","race_number"])["dt"]
dt_v["race_mean"] = g.transform("mean")
dt_v["race_std"]  = g.transform("std")
sig_min = 0.025
dt_v["sigma_eff"] = dt_v["race_std"].fillna(sig_min).clip(lower=sig_min)
# スペック: clip(-(dt-mean)/max(std,sig_min), -2.5, +2.5)
# 符号: 展示タイム小さい方が有利 → 符号反転
dt_v["f4_disp_raw"] = -(dt_v["dt"] - dt_v["race_mean"]) / dt_v["sigma_eff"]
dt_v["f4_disp"]     = dt_v["f4_disp_raw"].clip(-2.5, 2.5)
rc = rc.merge(
    dt_v[["date","stadium","race_number","lane","f4_disp"]],
    on=["date","stadium","race_number","lane"], how="left")

# B-5: motor_in2nd
rc["f5_motor"] = (rc["motor_in2nd"] - 33) / 11

# B-6: EWMA 今節調子
cs = cs.sort_values(["stadium","racerid","date","race_number"]).reset_index(drop=True)
cs["ewma_here"] = cs.groupby(["stadium","racerid"])["rank"].transform(
    lambda s: s.ewm(alpha=0.35, adjust=False).mean())
# 直近N=10に丸めるため、シンプルに10件ローリングは実装が重い。
# ewm は自然に重みが減衰するので、事実上 N≤10 と同等の寄与。

# pos = date序数 × 13 + race_number で一意化して merge_asof
epoch = pd.Timestamp("2020-01-01")
rc["rc_date"] = pd.to_datetime(rc["date"])
cs["cs_date"] = pd.to_datetime(cs["date"])
rc["pos"] = (rc["rc_date"] - epoch).dt.days.astype("int64") * 13 + rc["race_number"].astype("int64")
cs["pos"] = (cs["cs_date"] - epoch).dt.days.astype("int64") * 13 + cs["race_number"].astype("int64")

cs_keep = cs[["stadium","racerid","pos","cs_date","ewma_here"]].rename(columns={"ewma_here":"cs_ewma"})
rc_sorted = rc.sort_values("pos")
cs_sorted = cs_keep.sort_values("pos")
merged = pd.merge_asof(
    rc_sorted, cs_sorted,
    on="pos", by=["stadium","racerid"],
    direction="backward", allow_exact_matches=False)
merged["days_diff"] = (merged["rc_date"] - merged["cs_date"]).dt.days
valid = merged["days_diff"].notna() & (merged["days_diff"] >= 0) & (merged["days_diff"] <= 7)
merged.loc[~valid, "cs_ewma"] = np.nan
merged["f6_form_raw"] = -(merged["cs_ewma"] - 3.5) / 1.0
merged["f6_form"]     = merged["f6_form_raw"].fillna(0.0)
merged["f6_nomatch"]  = merged["f6_form_raw"].isna().astype(int)

# 並び順を元に戻す
rc = merged.sort_values(["date","stadium","race_number","lane"]).reset_index(drop=True)

# %% 2.3 C層: L(枠), V(会場×枠), W(風), H(波)
print("[2] C層 (L/V/W/H)...")

# L: lane 固定効果
rc["f7_lane_L"] = rc["lane"].map(LANE_L)

# V: 会場×枠番 logit差 (race_results 全期間から計算)
res_all = rres_all.copy()
g_sl = res_all.groupby(["stadium","boat"])
win_rate = g_sl.apply(lambda df: (df["rank"]==1).mean())
nat_lane = res_all.groupby("boat").apply(lambda df: (df["rank"]==1).mean())
# V(stadium, lane) = logit(p_sl) - logit(p_national_lane)
V_table = {}
for (s, l), p in win_rate.items():
    if p <= 0 or p >= 1:
        V_table[(s, l)] = 0.0
    else:
        V_table[(s, l)] = logit(p) - logit(nat_lane[l])
rc["f8_V"] = rc.apply(lambda r: V_table.get((r["stadium"], r["lane"]), 0.0), axis=1)

# W: 風補正
def compute_W_row(stadium, lane, wind_direction, wind_speed):
    if pd.isna(wind_direction):
        return 0.0
    wd = int(wind_direction)
    key = (int(stadium), wd)
    if key not in WIND_MAP:
        return 0.0
    # 無風(17)以外で風速 < 2 なら効果なし
    if wd != 17 and (pd.isna(wind_speed) or wind_speed < 2):
        return 0.0
    kern_name, delta_pct = WIND_MAP[key]
    kernel = KERNEL[kern_name]
    p_base = P_BASE_LANE1[int(stadium)]
    p_new = p_base + delta_pct / 100.0
    W1 = logit(p_new) - logit(p_base)
    # W_l = W1 * kernel[l-1] / kernel[0]
    return float(W1 * kernel[lane - 1] / kernel[0])

# race_conditions をmerge
cond_keep = rcond[["date","stadium","race_number","wind_direction","wind_speed","wave_height"]]
rc = rc.merge(cond_keep, on=["date","stadium","race_number"], how="left")

print("  W(風補正) 計算中...")
rc["f9_W"] = [compute_W_row(s, l, wd, ws) for s, l, wd, ws in
              zip(rc["stadium"], rc["lane"], rc["wind_direction"], rc["wind_speed"])]

# H: 波補正
def h_coef(h):
    if pd.isna(h): return 0.0
    if h <= 3: return 0.0
    if h <= 5: return 0.3
    if h <= 9: return 0.7
    return 1.0

K_LANE_WAVE = {1:0.0, 2:-0.05, 3:-0.10, 4:-0.15, 5:-0.20, 6:-0.25}
rc["h_coef"] = rc["wave_height"].apply(h_coef)
rc["f10_H"]  = rc["h_coef"] * rc["lane"].map(K_LANE_WAVE)

# 欠損は 0 埋め (モデル側に伝わらないようにする)
FEATURES = list(W_WEIGHTS.keys())
for f in FEATURES:
    rc[f] = rc[f].fillna(0.0)

# %% 3. スコア計算と予想
print("[3] スコア計算...")
# S_i = Σ w_k * X_k
W_arr = np.array([W_WEIGHTS[f] for f in FEATURES])
X = rc[FEATURES].to_numpy(dtype=float)
rc["S"] = X @ W_arr

# レース単位で softmax(S/tau) → P
def race_softmax(s):
    s = s.values if hasattr(s, "values") else s
    s_arr = np.asarray(s, dtype=float) / TAU
    m = s_arr.max()
    e = np.exp(s_arr - m)
    return e / e.sum()

rc["P"] = rc.groupby(["date","stadium","race_number"])["S"].transform(race_softmax)

# 予想: 各レースでSの大きい上位3艇
rc["pred_rank"] = rc.groupby(["date","stadium","race_number"])["S"].rank(
    method="min", ascending=False).astype(int)

# %% 4. 実結果との照合
print("[4] 実結果との照合...")
res = rres.rename(columns={"boat":"lane", "rank":"actual_rank"})
rc = rc.merge(res[["date","stadium","race_number","lane","actual_rank"]],
              on=["date","stadium","race_number","lane"], how="left")

# レース単位集計用
def agg_race(g):
    # predicted 1st/2nd/3rd
    g2 = g.sort_values("pred_rank")
    pred_order = g2.drop_duplicates("pred_rank")
    p1 = g2[g2["pred_rank"]==1]["lane"].iloc[0] if (g2["pred_rank"]==1).any() else None
    p2 = g2[g2["pred_rank"]==2]["lane"].iloc[0] if (g2["pred_rank"]==2).any() else None
    p3 = g2[g2["pred_rank"]==3]["lane"].iloc[0] if (g2["pred_rank"]==3).any() else None
    # actual 1,2,3
    g3 = g.sort_values("actual_rank")
    a1 = g[g["actual_rank"]==1]["lane"].iloc[0] if (g["actual_rank"]==1).any() else None
    a2 = g[g["actual_rank"]==2]["lane"].iloc[0] if (g["actual_rank"]==2).any() else None
    a3 = g[g["actual_rank"]==3]["lane"].iloc[0] if (g["actual_rank"]==3).any() else None
    # log-loss: -log(P of actual winner)
    if a1 is not None:
        p_actual = g[g["lane"]==a1]["P"].iloc[0]
        ll = -math.log(max(p_actual, 1e-9))
    else:
        ll = None
    return pd.Series({
        "pred_1": p1, "pred_2": p2, "pred_3": p3,
        "actual_1": a1, "actual_2": a2, "actual_3": a3,
        "log_loss": ll,
        "hit_1st": (p1 == a1) if (p1 is not None and a1 is not None) else False,
        "hit_top2": any(p in {a1, a2} for p in [p1, p2] if p is not None) if (a1 is not None) else False,
        "hit_top3": any(p in {a1, a2, a3} for p in [p1, p2, p3] if p is not None) if (a1 is not None) else False,
    })

print("  レース単位集計中 (数万レースあるので数秒〜数十秒)...")
race_log = rc.groupby(["date","stadium","race_number"]).apply(agg_race).reset_index()

# 期間分割
def split_period(df, start, end):
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()

train_log = split_period(race_log, TRAIN_START, TRAIN_END)
test_log  = split_period(race_log, TEST_START, TEST_END)

# %% 5. 集計
def summarize(df, label):
    df_valid = df.dropna(subset=["actual_1"])
    n = len(df_valid)
    hit1 = df_valid["hit_1st"].mean() * 100
    hit2 = df_valid["hit_top2"].mean() * 100
    hit3 = df_valid["hit_top3"].mean() * 100
    ll = df_valid["log_loss"].mean()
    print(f"\n## {label} (N={n:,})")
    print(f"| 指標 | 値 |")
    print(f"|---|---|")
    print(f"| 1着的中率 | {hit1:.2f}% |")
    print(f"| 2連対的中率 (上位2艇に1-2着が含まれる) | {hit2:.2f}% |")
    print(f"| 3連対的中率 (上位3艇に1-3着が含まれる) | {hit3:.2f}% |")
    print(f"| Log-loss (-log P_1着実艇) | {ll:.4f} |")
    return {"N": n, "hit1": hit1, "hit2": hit2, "hit3": hit3, "ll": ll}

print("\n========== 結果サマリ ==========")
s_train = summarize(train_log, f"Train ({TRAIN_START} 〜 {TRAIN_END})")
s_test  = summarize(test_log,  f"Test  ({TEST_START}  〜 {TEST_END})")

# ベースライン (常に 1-2-3 予想)
def baseline_stats(df, label):
    df_valid = df.dropna(subset=["actual_1"])
    hit1 = (df_valid["actual_1"] == 1).mean() * 100
    hit2 = df_valid.apply(lambda r: any(x in {r["actual_1"], r["actual_2"]} for x in [1,2]), axis=1).mean() * 100
    hit3 = df_valid.apply(lambda r: any(x in {r["actual_1"], r["actual_2"], r["actual_3"]} for x in [1,2,3]), axis=1).mean() * 100
    print(f"\n## ベースライン {label}: 常に 1-2-3 予想 (N={len(df_valid):,})")
    print(f"| 指標 | 値 |")
    print(f"|---|---|")
    print(f"| 1着的中率 | {hit1:.2f}% |")
    print(f"| 2連対的中率 | {hit2:.2f}% |")
    print(f"| 3連対的中率 | {hit3:.2f}% |")
    return {"hit1": hit1, "hit2": hit2, "hit3": hit3}

b_train = baseline_stats(train_log, f"[Train]")
b_test  = baseline_stats(test_log,  f"[Test]")

# %% 5.1 会場別サマリ
def by_stadium(df, label):
    df = df.dropna(subset=["actual_1"])
    from scripts.stadiums import STADIUMS
    rows = []
    for s, g in df.groupby("stadium"):
        n = len(g)
        rows.append({
            "stadium": int(s),
            "name": STADIUMS.get(int(s), "?"),
            "N": n,
            "hit_1st": g["hit_1st"].mean() * 100,
            "hit_top2": g["hit_top2"].mean() * 100,
            "hit_top3": g["hit_top3"].mean() * 100,
            "log_loss": g["log_loss"].mean(),
        })
    out = pd.DataFrame(rows).sort_values("stadium")
    print(f"\n## 会場別 ({label})")
    print(out.to_string(index=False))
    return out

by_stadium(train_log, "Train")
by_stadium(test_log,  "Test")

# %% 6. CSV 出力
race_log.to_csv(OUTD / "race_log.csv", index=False, encoding="utf-8-sig")
train_log.to_csv(OUTD / "train_log.csv", index=False, encoding="utf-8-sig")
test_log.to_csv(OUTD / "test_log.csv", index=False, encoding="utf-8-sig")
print(f"\nCSV 書込: {OUTD}/race_log.csv, train_log.csv, test_log.csv")

# %% 完了
print("\nバックテスト完了")
