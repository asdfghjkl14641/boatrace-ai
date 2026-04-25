# -*- coding: utf-8 -*-
"""MVP 6特徴量の正規化後統計を出す一回限りの集計スクリプト。"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / 'boatrace.db'
conn = sqlite3.connect(DB)

print('[1/5] race_cards 読み込み中...')
rc = pd.read_sql_query('''
    SELECT date, stadium, race_number, lane, racerid,
           global_win_pt, local_win_pt, aveST, motor_in2nd
    FROM race_cards
''', conn)
print(f'  race_cards: {len(rc):,} 行')

print('[2/5] race_conditions (display_time_*) 読み込み中...')
rcond = pd.read_sql_query('''
    SELECT date, stadium, race_number,
           display_time_1, display_time_2, display_time_3,
           display_time_4, display_time_5, display_time_6
    FROM race_conditions
''', conn)
print(f'  race_conditions: {len(rcond):,} 行')

print('[3/5] current_series 読み込み中...')
cs = pd.read_sql_query('''
    SELECT date, stadium, racerid, race_number, rank
    FROM current_series
    WHERE rank IS NOT NULL
''', conn)
print(f'  current_series: {len(cs):,} 行')
conn.close()


def desc(name, s, target_std=1.0):
    s = pd.Series(s).dropna()
    q = s.quantile
    print(f"\n### {name}")
    print('| 統計量 | 値 |')
    print('|---|---|')
    print(f'| N | {len(s):,} |')
    print(f'| 平均 mean | {s.mean():+.4f} |')
    print(f'| 中央値 median | {s.median():+.4f} |')
    print(f'| **std** | **{s.std():.4f}** (target {target_std}) |')
    print(f'| 最小 min | {s.min():+.4f} |')
    print(f'| 最大 max | {s.max():+.4f} |')
    print(f'| P5 | {q(0.05):+.4f} |')
    print(f'| P95 | {q(0.95):+.4f} |')
    print(f'| Q1 | {q(0.25):+.4f} |')
    print(f'| Q3 | {q(0.75):+.4f} |')


print('[4/5] A層1-3 と B-5 計算中 ...')
# A-1
rc['f1_global'] = (rc['global_win_pt'] - 5.3) / 1.3
# A-2 条件分岐
f2_local_val = (rc['local_win_pt'] - 5.4) / 1.3
use_local = rc['local_win_pt'] >= 2.4
rc['f2_local'] = np.where(use_local, f2_local_val, rc['f1_global'])
# A-3
rc['f3_ST'] = -(rc['aveST'] - 0.16) / 0.023
# B-5
rc['f5_motor'] = (rc['motor_in2nd'] - 33) / 11

print('[5/5] B-4 (レース内 z-score) と B-6 (EWMA) 計算中 ...')
# B-4
dt_long = rcond.melt(
    id_vars=['date', 'stadium', 'race_number'],
    value_vars=[f'display_time_{i}' for i in range(1, 7)],
    var_name='lane', value_name='dt',
)
dt_long['lane'] = dt_long['lane'].str.extract(r'(\d)').astype(int)
dt_valid = dt_long.dropna(subset=['dt'])
dt_valid = dt_valid[dt_valid['dt'] > 0].copy()
g = dt_valid.groupby(['date', 'stadium', 'race_number'])['dt']
dt_valid['race_mean'] = g.transform('mean')
dt_valid['race_std'] = g.transform('std')
sig_min = 0.025
dt_valid['sigma_eff'] = dt_valid['race_std'].fillna(sig_min).clip(lower=sig_min)
dt_valid['f4_disp_raw'] = (dt_valid['dt'] - dt_valid['race_mean']) / dt_valid['sigma_eff']
dt_valid['f4_disp'] = dt_valid['f4_disp_raw'].clip(-2.0, 2.0)
rc = rc.merge(
    dt_valid[['date', 'stadium', 'race_number', 'lane', 'f4_disp']],
    on=['date', 'stadium', 'race_number', 'lane'], how='left',
)

# B-6: EWMA
# racerid 不明 (NULL) の行は EWMA 計算できないので除外 + 型を揃える
rc = rc.dropna(subset=['racerid']).copy()
rc['racerid'] = rc['racerid'].astype('int64')
rc['stadium'] = rc['stadium'].astype('int64')
cs = cs.dropna(subset=['racerid']).copy()
cs['racerid'] = cs['racerid'].astype('int64')
cs['stadium'] = cs['stadium'].astype('int64')

cs = cs.sort_values(['stadium', 'racerid', 'date', 'race_number']).reset_index(drop=True)
cs['ewma_here'] = cs.groupby(['stadium', 'racerid'])['rank'].transform(
    lambda s: s.ewm(alpha=0.35, adjust=False).mean()
)
epoch = pd.Timestamp('2020-01-01')


def pos_key(df):
    dt = pd.to_datetime(df['date'])
    return (dt - epoch).dt.days.astype('int64') * 13 + df['race_number'].astype('int64')


rc['pos'] = pos_key(rc)
cs['pos'] = pos_key(cs)
cs['cs_date'] = pd.to_datetime(cs['date'])
cs_keep = cs[['stadium', 'racerid', 'pos', 'cs_date', 'ewma_here']].rename(
    columns={'ewma_here': 'cs_ewma'}
)
rc_sorted = rc.sort_values('pos').copy()
rc_sorted['rc_date'] = pd.to_datetime(rc_sorted['date'])
cs_sorted_m = cs_keep.sort_values('pos')
merged = pd.merge_asof(
    rc_sorted, cs_sorted_m,
    on='pos', by=['stadium', 'racerid'],
    direction='backward', allow_exact_matches=False,
)
merged['days_diff'] = (merged['rc_date'] - merged['cs_date']).dt.days
valid = merged['days_diff'].notna() & (merged['days_diff'] >= 0) & (merged['days_diff'] <= 7)
merged.loc[~valid, 'cs_ewma'] = np.nan
merged['f6_form_raw'] = -(merged['cs_ewma'] - 3.5) / 1.0
merged['f6_form'] = merged['f6_form_raw'].fillna(0.0)

print()
print('## 結果: 6特徴量の正規化後統計')
print()
print(f'期間: {rc["date"].min()} 〜 {rc["date"].max()}')
print(f'race_cards 総 N: {len(rc):,}')

desc('A-1 正規化全国勝率', rc['f1_global'])
desc('A-2 正規化当地勝率 (条件分岐)', rc['f2_local'])
desc('A-3 正規化ST', rc['f3_ST'])
desc('B-4 正規化展示タイム (z-score, σ_min=0.025, clip±2)', merged['f4_disp'])
desc('B-5 正規化モーター (motor_in2nd)', rc['f5_motor'])
desc('B-6 正規化今節調子 (全行、未出走=0)', merged['f6_form'])

use_cnt = int(use_local.sum())
print(f'\n---\n補足: f2_local 計算内訳')
print(f'  local_win_pt>=2.4 (当地式適用): {use_cnt:,} ({use_cnt/len(rc)*100:.2f}%)')
print(f'  それ以外 (全国式fallback):     {len(rc)-use_cnt:,} ({(len(rc)-use_cnt)/len(rc)*100:.2f}%)')

ewma_cnt = int(merged['f6_form_raw'].notna().sum())
print(f'\n補足: f6_form 計算内訳')
print(f'  EWMA計算可 (今節出走あり): {ewma_cnt:,} ({ewma_cnt/len(merged)*100:.2f}%)')
print(f'  未出走→0 埋め:            {len(merged)-ewma_cnt:,} ({(len(merged)-ewma_cnt)/len(merged)*100:.2f}%)')

desc('B-6 [参考: EWMA計算できた行のみ]', merged['f6_form_raw'].dropna())
