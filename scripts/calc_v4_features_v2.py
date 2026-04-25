# -*- coding: utf-8 -*-
"""
v4 features v2: f12 Beta prior smoothing + prior_strength 感度分析
既存 v4_features.csv (f11 simple/ewma) を読み、f12 を smoothing して書き直す。
"""
from __future__ import annotations
import sys, math, time
from pathlib import Path
import numpy as np
import pandas as pd
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
LOOKBACK_F12_DAYS = 90

print("=" * 80)
print("Step 1: f12 Beta prior smoothing + 感度分析")
print("=" * 80)

conn = get_connection()
# データロード
print("\n[1] 素材ロード ...")
rh = pd.read_sql_query("""
    SELECT racer_id, race_date, stadium, race_no, finish_pos
    FROM racer_history WHERE finish_pos IS NOT NULL
""", conn.native)
rh["race_date"] = pd.to_datetime(rh["race_date"])
rh = rh.astype({"racer_id":int, "stadium":int, "race_no":int, "finish_pos":int})

rc = pd.read_sql_query("""
    SELECT date AS race_date, stadium, race_number AS race_no, lane, racerid, motor
    FROM race_cards WHERE motor IS NOT NULL AND lane BETWEEN 1 AND 6 AND racerid IS NOT NULL
""", conn.native)
rc["race_date"] = pd.to_datetime(rc["race_date"])
rc = rc.astype({"stadium":int, "race_no":int, "lane":int, "racerid":int, "motor":int})
conn.close()
print(f"   racer_history {len(rh):,} / race_cards {len(rc):,}")

# motor × stadium ごとの時系列 (top2 = finish <= 2)
mf = rc.merge(
    rh.rename(columns={"racer_id":"racerid"})[["racerid","race_date","stadium","race_no","finish_pos"]],
    on=["race_date","stadium","race_no","racerid"], how="inner")
mf = mf.sort_values(["motor","stadium","race_date"]).reset_index(drop=True)
mf_dates = mf["race_date"].values.astype("datetime64[D]").astype(np.int64)
mf_top2  = (mf["finish_pos"].values <= 2).astype(np.int8)

from itertools import groupby
motor_keys = list(zip(mf["motor"].values, mf["stadium"].values))
motor_group_ranges = {}
idx = 0
for key, g in groupby(motor_keys):
    cnt = sum(1 for _ in g)
    motor_group_ranges[key] = (idx, idx + cnt)
    idx += cnt

# stadium 全モーター 2 連対率 の mean/std (train 固定)
mf_train = mf[(mf["race_date"]>=TRAIN_FROM) & (mf["race_date"]<=TRAIN_TO)]
p_motor_stadium = mf_train.groupby(["motor","stadium"]).agg(
    n=("finish_pos","count"),
    p_top2=("finish_pos", lambda s: (s <= 2).mean()))
p_motor_stadium = p_motor_stadium[p_motor_stadium["n"] >= 30]
stad_stats = p_motor_stadium.reset_index().groupby("stadium")["p_top2"].agg(["mean","std"]).to_dict("index")
# stadium_mean (Beta prior の中心値)
stadium_prior_mean = {s: stad_stats[s]["mean"] for s in stad_stats}
print(f"\n[2] stadium prior mean (2連対率 平均): {len(stadium_prior_mean)} stadiums")


def f12_smoothed_lookup(motor, stadium, target_day, prior_strength=10):
    rg = motor_group_ranges.get((motor, stadium))
    if rg is None:
        return 0.0, 0
    lo = target_day - LOOKBACK_F12_DAYS; hi = target_day
    s, e = rg
    dates_slice = mf_dates[s:e]
    L = np.searchsorted(dates_slice, lo, side="left")
    R = np.searchsorted(dates_slice, hi, side="left")
    n = R - L
    prior_mean = stadium_prior_mean.get(stadium)
    ss = stad_stats.get(stadium)
    if prior_mean is None or ss is None or ss["std"] is None or ss["std"] == 0 or np.isnan(ss["std"]):
        return 0.0, int(n)
    wins = int(mf_top2[s+L:s+R].sum())
    alpha = prior_mean * prior_strength
    beta  = (1 - prior_mean) * prior_strength
    p_smoothed = (wins + alpha) / (n + alpha + beta)
    return float((p_smoothed - ss["mean"]) / ss["std"]), int(n)


# 対象行
print("\n[3] f12 smoothing 計算 (prior_strength = 5, 10, 20) ...")
t0 = time.time()
mask_target = ((rc["race_date"]>=TRAIN_FROM) & (rc["race_date"]<=TEST_TO))
tgt = rc[mask_target].copy().reset_index(drop=True)
tgt_days = tgt["race_date"].values.astype("datetime64[D]").astype(np.int64)
motors = tgt["motor"].values.astype(int)
stadiums = tgt["stadium"].values.astype(int)

for ps in [5, 10, 20]:
    f12 = np.zeros(len(tgt), dtype=np.float32)
    n_arr = np.zeros(len(tgt), dtype=np.int32)
    for i in range(len(tgt)):
        v, n = f12_smoothed_lookup(int(motors[i]), int(stadiums[i]), int(tgt_days[i]), prior_strength=ps)
        f12[i] = v; n_arr[i] = n
    tgt[f"f12_s{ps}"] = f12
    tgt[f"f12_s{ps}_n"] = n_arr
    # train 期間で評価 (N >= 5)
    tm = (tgt["race_date"]>=TRAIN_FROM) & (tgt["race_date"]<=TRAIN_TO)
    v_train = tgt.loc[tm & (tgt[f"f12_s{ps}_n"] >= 5), f"f12_s{ps}"]
    raw_std = v_train.std(); clip_std = v_train.clip(-2.5,2.5).std()
    clip_rate = (v_train.abs() > 2.5).mean() * 100
    print(f"   prior_strength={ps}: raw std={raw_std:.3f}, clip±2.5 std={clip_std:.3f}, "
          f"clip発動率={clip_rate:.1f}%")
print(f"   計算時間 {time.time()-t0:.1f}s")

# 既存の v4_features.csv を読み込んで f12 を差し替え
print("\n[4] 既存 v4_features.csv に smoothing f12 を追記 ...")
old = pd.read_csv(OUT / f"v4{OUT_SUFFIX}_features.csv")
old["race_date"] = pd.to_datetime(old["race_date"])
# tgt と old をキーで merge
keys = ["race_date","stadium","race_no","lane","racerid","motor"]
merged = old.merge(tgt[keys + [f"f12_s{p}" for p in [5,10,20]] + [f"f12_s{p}_n" for p in [5,10,20]]],
                    on=keys, how="left")

# 推奨 prior_strength を clip std が 1.0 に近いもので選択
best_ps = None
best_diff = 1e9
for ps in [5, 10, 20]:
    v = merged[f"f12_s{ps}"]
    if v.notna().sum() > 1000:
        nm = merged[f"f12_s{ps}_n"] >= 5
        clip_std = v[nm].clip(-2.5, 2.5).std()
        if abs(clip_std - 1.0) < best_diff:
            best_diff = abs(clip_std - 1.0); best_ps = ps
print(f"\n   推奨 prior_strength: {best_ps} (clip std が 1.0 に最も近い)")

merged["f12_final"] = merged[f"f12_s{best_ps}"].fillna(0).clip(-2.5, 2.5)
merged["f12_final_n"] = merged[f"f12_s{best_ps}_n"].fillna(0).astype(int)
merged["f11_final"]   = merged["f11_lane_player_simple"].fillna(0).clip(-2.5, 2.5)
merged["f11_final_n"] = merged["f11_sample_n"].fillna(0).astype(int)

out_csv = OUT / f"v4{OUT_SUFFIX}_features_v2.csv"
merged.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"   saved: {out_csv}")

# 最終値の分布
print("\n[5] 最終特徴量 (clip±2.5 後) の統計 (train, N>=5):")
tm = (merged["race_date"]>=TRAIN_FROM) & (merged["race_date"]<=TRAIN_TO)
for col, ncol in [("f11_final","f11_final_n"), ("f12_final","f12_final_n")]:
    v = merged.loc[tm & (merged[ncol]>=5), col]
    print(f"   {col}: mean={v.mean():+.4f} std={v.std():.4f} "
          f"P5={v.quantile(.05):+.3f} P95={v.quantile(.95):+.3f}")
