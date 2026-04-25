# -*- coding: utf-8 -*-
"""
v4 新特徴量 f11 / f12 の計算

f11_lane_player: 過去 12 ヶ月の (racer_id × lane) 別 1 着率 logit
                 - コース全体平均 logit (リーク防止のため各 target_date より前のみ)
f12_motor_recent: 過去 3 ヶ月の (motor × stadium) 別 2 連対率 z-score
                  (stadium 全モーター基準)

最低サンプル N=5、未満は 0 フォールバック。
"""
from __future__ import annotations
import sys, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
from bisect import bisect_left

BASE = Path(__file__).resolve().parent.parent
OUT  = BASE / "notebooks" / "output"
sys.path.insert(0, str(BASE))
sys.stdout.reconfigure(encoding="utf-8")
from scripts.db import get_connection

import os as _os
TRAIN_FROM = _os.environ.get("V4_TRAIN_FROM", "2023-05-01")
TRAIN_TO   = _os.environ.get("V4_TRAIN_TO",   "2025-12-31")
TEST_FROM  = _os.environ.get("V4_TEST_FROM",  "2026-01-01")
TEST_TO    = _os.environ.get("V4_TEST_TO",    "2026-04-18")
OUT_SUFFIX = _os.environ.get("V4_OUT_SUFFIX", "")

LOOKBACK_F11_DAYS = 365
LOOKBACK_F12_DAYS = 90
MIN_N = 5


def logit(p: float) -> float:
    p = max(min(p, 1 - 1e-6), 1e-6)
    return math.log(p / (1 - p))


print("=" * 80)
print("v4 features 計算: f11_lane_player + f12_motor_recent")
print("=" * 80)

# --- データロード ---
conn = get_connection()
t0 = time.time()
print("\n[1] データロード...")
rh = pd.read_sql_query("""
    SELECT racer_id, race_date, stadium, race_no, lane, finish_pos
    FROM racer_history
    WHERE lane BETWEEN 1 AND 6 AND finish_pos IS NOT NULL
""", conn.native if hasattr(conn, "native") else conn)
rh["race_date"] = pd.to_datetime(rh["race_date"])
rh = rh.astype({"racer_id": int, "stadium": int, "race_no": int,
                "lane": int, "finish_pos": int})
print(f"   racer_history: {len(rh):,} 行 ({time.time()-t0:.1f}s)")

rc = pd.read_sql_query("""
    SELECT date AS race_date, stadium, race_number AS race_no, lane, racerid, motor
    FROM race_cards
    WHERE motor IS NOT NULL AND lane BETWEEN 1 AND 6
""", conn.native if hasattr(conn, "native") else conn)
rc["race_date"] = pd.to_datetime(rc["race_date"])
rc = rc.astype({"stadium": int, "race_no": int, "lane": int,
                "racerid": int, "motor": int})
print(f"   race_cards: {len(rc):,} 行 ({time.time()-t0:.1f}s)")


# --- 全国 lane 1 着率 (logit) — train 期間固定 ---
print("\n[2] 全国 lane 1 着率 (train 2023-05〜2025-12 固定)")
train_rh = rh[(rh["race_date"] >= TRAIN_FROM) & (rh["race_date"] <= TRAIN_TO)]
p_lane = {}
for l in range(1, 7):
    s = train_rh[train_rh["lane"] == l]
    p_lane[l] = (s["finish_pos"] == 1).mean()
    print(f"   lane {l}: p={p_lane[l]:.4f}  logit={logit(p_lane[l]):+.4f}")
logit_lane = {l: logit(p) for l, p in p_lane.items()}


# --- f11: (racer_id, lane) 別 時系列インデックス ---
print("\n[3] (racer_id, lane) 別インデックス構築 ...")
rh_sorted = rh.sort_values(["racer_id", "lane", "race_date"]).reset_index(drop=True)
rh_dates = rh_sorted["race_date"].values.astype("datetime64[D]").astype(np.int64)  # days
rh_wins  = (rh_sorted["finish_pos"].values == 1).astype(np.int8)

# Group index: for each (racer_id, lane), start/end indices
keys = list(zip(rh_sorted["racer_id"].values, rh_sorted["lane"].values))
# Build group start indices
from itertools import groupby
group_ranges = {}
idx = 0
for key, g in groupby(keys):
    cnt = sum(1 for _ in g)
    group_ranges[key] = (idx, idx + cnt)
    idx += cnt
print(f"   グループ数 (racer × lane): {len(group_ranges):,}")


def f11_lookup(racer_id: int, lane: int, target_day: int,
               lookback: int = LOOKBACK_F11_DAYS,
               ewma_half_life_days: int | None = None) -> tuple[float, int]:
    """過去 lookback 日の 1 着率 logit - global lane logit"""
    rg = group_ranges.get((racer_id, lane))
    if rg is None:
        return 0.0, 0
    lo_day = target_day - lookback
    hi_day = target_day  # exclusive
    # binary search within group
    start, end = rg
    dates_slice = rh_dates[start:end]
    l = np.searchsorted(dates_slice, lo_day, side="left")
    r = np.searchsorted(dates_slice, hi_day, side="left")
    n = r - l
    if n < MIN_N:
        return 0.0, int(n)
    wins = rh_wins[start + l : start + r]
    if ewma_half_life_days is None:
        p = wins.mean()
    else:
        # EWMA — weight = 0.5^((target - t)/half_life)
        ds = dates_slice[l:r]
        w = np.power(0.5, (target_day - ds) / ewma_half_life_days)
        p = float((wins * w).sum() / w.sum())
    return logit(p) - logit_lane[lane], int(n)


# --- f12 素材: motor × stadium × 時系列 (2 連対 = finish_pos <= 2) ---
print("\n[4] (motor, stadium) の 2 連対率の素材構築 ...")
# Need to join race_cards (motor) with racer_history (finish_pos) on date/stadium/race_no/racer
mf = rc.merge(
    rh[["racer_id","race_date","stadium","race_no","finish_pos"]].rename(
        columns={"racer_id":"racerid"}),
    on=["race_date","stadium","race_no","racerid"], how="inner")
mf = mf.sort_values(["motor","stadium","race_date"]).reset_index(drop=True)
mf_dates = mf["race_date"].values.astype("datetime64[D]").astype(np.int64)
mf_top2  = (mf["finish_pos"].values <= 2).astype(np.int8)
mf_keys = list(zip(mf["motor"].values, mf["stadium"].values))
motor_group_ranges = {}
idx = 0
for key, g in groupby(mf_keys):
    cnt = sum(1 for _ in g)
    motor_group_ranges[key] = (idx, idx + cnt)
    idx += cnt
print(f"   (motor, stadium) グループ: {len(motor_group_ranges):,}, 行数 {len(mf):,}")


# stadium 別 motor 2 連対率の mean/std (train 期間全 motor 平均基準)
print("\n[5] stadium 別 motor 2 連対率の mean/std (train 固定) ...")
mf_train = mf[(mf["race_date"] >= TRAIN_FROM) & (mf["race_date"] <= TRAIN_TO)]
p_motor_stadium = mf_train.groupby(["motor","stadium"]).agg(
    n=("finish_pos","count"),
    p_top2=("finish_pos", lambda s: (s <= 2).mean()))
p_motor_stadium = p_motor_stadium[p_motor_stadium["n"] >= 30]
# stadium 統計 (全 motor 集合から)
stad_stats = p_motor_stadium.reset_index().groupby("stadium")["p_top2"].agg(
    ["mean", "std"]).to_dict("index")
print(f"   stadium 統計件数: {len(stad_stats)}")
for s in sorted(stad_stats)[:5]:
    print(f"     stadium {s}: mean={stad_stats[s]['mean']:.4f} std={stad_stats[s]['std']:.4f}")


def f12_lookup(motor: int, stadium: int, target_day: int,
               lookback: int = LOOKBACK_F12_DAYS) -> tuple[float, int]:
    rg = motor_group_ranges.get((motor, stadium))
    if rg is None:
        return 0.0, 0
    lo_day = target_day - lookback
    hi_day = target_day
    s, e = rg
    dates_slice = mf_dates[s:e]
    l = np.searchsorted(dates_slice, lo_day, side="left")
    r = np.searchsorted(dates_slice, hi_day, side="left")
    n = r - l
    if n < MIN_N:
        return 0.0, int(n)
    p = mf_top2[s + l : s + r].mean()
    ss = stad_stats.get(stadium)
    if ss is None or ss["std"] is None or ss["std"] == 0 or np.isnan(ss["std"]):
        return 0.0, int(n)
    return float((p - ss["mean"]) / ss["std"]), int(n)


# --- 対象レース (train + test) の race_cards 行に対して f11/f12 を計算 ---
print("\n[6] 対象レースで f11/f12 計算 ...")
t_start = time.time()
mask_target = ((rc["race_date"] >= TRAIN_FROM) & (rc["race_date"] <= TEST_TO))
tgt = rc[mask_target].copy().reset_index(drop=True)
print(f"   計算対象行: {len(tgt):,}")

tgt_days = tgt["race_date"].values.astype("datetime64[D]").astype(np.int64)
racers = tgt["racerid"].values
lanes = tgt["lane"].values
motors = tgt["motor"].values
stadiums = tgt["stadium"].values

f11_vals_simple = np.zeros(len(tgt), dtype=np.float32)
f11_n = np.zeros(len(tgt), dtype=np.int32)
f11_vals_ewma = np.zeros(len(tgt), dtype=np.float32)
f12_vals = np.zeros(len(tgt), dtype=np.float32)
f12_n = np.zeros(len(tgt), dtype=np.int32)

for i in range(len(tgt)):
    td = tgt_days[i]
    v, n = f11_lookup(int(racers[i]), int(lanes[i]), td)
    f11_vals_simple[i] = v; f11_n[i] = n
    v_e, _ = f11_lookup(int(racers[i]), int(lanes[i]), td, ewma_half_life_days=180)
    f11_vals_ewma[i] = v_e
    v2, n2 = f12_lookup(int(motors[i]), int(stadiums[i]), td)
    f12_vals[i] = v2; f12_n[i] = n2
    if (i+1) % 50000 == 0:
        print(f"   {i+1:,}/{len(tgt):,} ({(i+1)/len(tgt)*100:.1f}%)  {time.time()-t_start:.1f}s")

print(f"   計算完了 {time.time()-t_start:.1f}s")

tgt["f11_lane_player_simple"] = f11_vals_simple
tgt["f11_lane_player_ewma"]   = f11_vals_ewma
tgt["f11_sample_n"]           = f11_n
tgt["f12_motor_recent"]       = f12_vals
tgt["f12_sample_n"]           = f12_n

# 保存
out_parquet = OUT / f"v4{OUT_SUFFIX}_features.parquet"
try:
    tgt.to_parquet(out_parquet, index=False)
    print(f"   saved: {out_parquet}")
except Exception as e:
    out_csv = OUT / f"v4{OUT_SUFFIX}_features.csv"
    tgt.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"   (parquet 失敗 {e}) saved CSV: {out_csv}")


# --- 検証 ---
print("\n" + "=" * 80)
print("Step 3 & 4: 検証 + 方式比較")
print("=" * 80)

# サンプル数分布
def pct_bins(arr, bins=[0, 1, 5, 10, 30, 100, 500, 10_000]):
    import numpy as np
    cnts = np.histogram(arr, bins=bins)[0]
    total = len(arr)
    return [(bins[i], bins[i+1], cnts[i], cnts[i]/total*100) for i in range(len(cnts))]

print("\n### f11 sample_n 分布")
print(f"  {'bin':>15s} {'count':>8s} {'率':>7s}")
for lo, hi, cnt, p in pct_bins(f11_n):
    print(f"  {f'[{lo:>4},{hi:>5})':>15s} {cnt:>8,} {p:>6.2f}%")
print(f"\n  N=0 (フォールバック): {(f11_n == 0).sum():,} ({(f11_n==0).mean()*100:.2f}%)")
print(f"  N >= 5:              {(f11_n >= 5).sum():,} ({(f11_n>=5).mean()*100:.2f}%)")

print("\n### f12 sample_n 分布")
print(f"  {'bin':>15s} {'count':>8s} {'率':>7s}")
for lo, hi, cnt, p in pct_bins(f12_n, bins=[0,1,5,10,30,100,500,10_000]):
    print(f"  {f'[{lo:>4},{hi:>5})':>15s} {cnt:>8,} {p:>6.2f}%")
print(f"\n  N=0: {(f12_n==0).sum():,} ({(f12_n==0).mean()*100:.2f}%)")
print(f"  N >= 5: {(f12_n>=5).sum():,} ({(f12_n>=5).mean()*100:.2f}%)")

# 値の分布 (train のみ、N>=5 行)
print("\n### 値の分布 (train 期間、N>=5 行のみ)")
train_mask = ((tgt["race_date"] >= TRAIN_FROM) & (tgt["race_date"] <= TRAIN_TO))
f11s_train = tgt.loc[train_mask & (tgt["f11_sample_n"]>=5), "f11_lane_player_simple"]
f11e_train = tgt.loc[train_mask & (tgt["f11_sample_n"]>=5), "f11_lane_player_ewma"]
f12_train  = tgt.loc[train_mask & (tgt["f12_sample_n"]>=5), "f12_motor_recent"]
print(f"  f11 simple: mean={f11s_train.mean():+.4f} std={f11s_train.std():.4f}  "
      f"P5={f11s_train.quantile(.05):+.3f} P95={f11s_train.quantile(.95):+.3f}")
print(f"  f11 ewma:   mean={f11e_train.mean():+.4f} std={f11e_train.std():.4f}  "
      f"P5={f11e_train.quantile(.05):+.3f} P95={f11e_train.quantile(.95):+.3f}")
print(f"  f12:        mean={f12_train.mean():+.4f} std={f12_train.std():.4f}  "
      f"P5={f12_train.quantile(.05):+.3f} P95={f12_train.quantile(.95):+.3f}")

# 既存特徴量との相関 (race_cards の aveST, global_win_pt, local_win_pt, motor_in2nd で近似)
print("\n### 既存特徴量との相関 (train)")
rc_ext = pd.read_sql_query(f"""
    SELECT date AS race_date, stadium, race_number AS race_no, lane, racerid,
           aveST, global_win_pt, local_win_pt, motor_in2nd, motor_in3rd
    FROM race_cards
    WHERE date BETWEEN '{TRAIN_FROM}' AND '{TRAIN_TO}'
""", conn.native if hasattr(conn, "native") else conn)
rc_ext["race_date"] = pd.to_datetime(rc_ext["race_date"])
rc_ext = rc_ext.astype({"stadium":int, "race_no":int, "lane":int, "racerid":int})
merged = tgt[train_mask].merge(rc_ext,
    on=["race_date","stadium","race_no","lane","racerid"], how="inner")
merged_f11 = merged[merged["f11_sample_n"]>=5]
merged_f12 = merged[merged["f12_sample_n"]>=5]

for f11_col in ["f11_lane_player_simple", "f11_lane_player_ewma"]:
    print(f"\n  {f11_col}:")
    for c in ["global_win_pt","local_win_pt","aveST","motor_in2nd"]:
        sub = merged_f11.dropna(subset=[f11_col, c])
        if len(sub) > 100:
            corr = sub[f11_col].corr(sub[c])
            print(f"    vs {c}: r = {corr:+.3f}  (N={len(sub):,})")

print(f"\n  f12_motor_recent:")
for c in ["motor_in2nd","motor_in3rd","global_win_pt"]:
    sub = merged_f12.dropna(subset=["f12_motor_recent", c])
    if len(sub) > 100:
        corr = sub["f12_motor_recent"].corr(sub[c])
        print(f"    vs {c}: r = {corr:+.3f}  (N={len(sub):,})")

# f11 simple vs ewma の相関
corr_se = merged_f11["f11_lane_player_simple"].corr(merged_f11["f11_lane_player_ewma"])
print(f"\n  f11 simple vs ewma: r = {corr_se:+.4f}")

# Step 5: 推奨設定
print("\n" + "=" * 80)
print("推奨設定")
print("=" * 80)
simple_std = f11s_train.std()
ewma_std = f11e_train.std()
print(f"  f11 simple std: {simple_std:.3f}, ewma std: {ewma_std:.3f}")
print(f"  推奨集計: {'シンプル平均' if abs(simple_std-1.0) < abs(ewma_std-1.0) else 'EWMA (半減期 6ヶ月)'}")
print(f"  f12 std: {f12_train.std():.3f}")
print(f"  f11 N=0 率: {(f11_n==0).mean()*100:.2f}%  f12 N=0 率: {(f12_n==0).mean()*100:.2f}%")

conn.close()
print("\n完了")
