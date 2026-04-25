# -*- coding: utf-8 -*-
"""v3 再学習 Phase A-1: データ readiness 確認 (v2 ファイル一切触らず)"""
from __future__ import annotations
import sys
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "notebooks" / "output"
sys.path.insert(0, str(BASE))
from scripts.db import get_connection

c = get_connection().cursor()

TRAIN_FROM, TRAIN_TO = "2023-05-01", "2025-12-31"
TEST_FROM, TEST_TO = "2026-01-01", "2026-04-18"

print("=" * 80)
print("v3 Phase A-1: データ readiness")
print("=" * 80)


def count_races(table, date_cond, extra=""):
    c.execute(f"SELECT COUNT(DISTINCT date||'-'||stadium||'-'||race_number) "
              f"FROM {table} WHERE {date_cond} {extra}")
    return c.fetchone()[0]


def count_race_conditions_by_source(date_cond):
    c.execute(f"""SELECT source, COUNT(DISTINCT date||'-'||stadium||'-'||race_number),
                  SUM(CASE WHEN display_time_1 IS NOT NULL AND display_time_2 IS NOT NULL
                           AND display_time_3 IS NOT NULL AND display_time_4 IS NOT NULL
                           AND display_time_5 IS NOT NULL AND display_time_6 IS NOT NULL
                           THEN 1 ELSE 0 END)
                  FROM race_conditions WHERE {date_cond} GROUP BY source""")
    return c.fetchall()


rows = []
for label, cond in [("train", f"date BETWEEN '{TRAIN_FROM}' AND '{TRAIN_TO}'"),
                    ("test",  f"date BETWEEN '{TEST_FROM}' AND '{TEST_TO}'")]:
    print(f"\n【{label} 期間】{cond}")
    # race_cards
    n_rc = count_races("race_cards", cond)
    rows.append({"period": label, "table": "race_cards", "races": n_rc, "notes": ""})
    print(f"  race_cards          : {n_rc:>7,}")
    # race_results
    n_rr = count_races("race_results", cond, "AND rank IS NOT NULL")
    rows.append({"period": label, "table": "race_results", "races": n_rr, "notes": "rank非NULLのみ"})
    print(f"  race_results        : {n_rr:>7,} (rank 非 NULL)")
    # race_conditions (source 別)
    source_info = count_race_conditions_by_source(cond)
    total_rc = sum(r[1] for r in source_info)
    total_full6 = sum(r[2] or 0 for r in source_info)
    print(f"  race_conditions    : {total_rc:>7,} (合計), {total_full6:,} 6艇揃い ({total_full6/total_rc*100 if total_rc else 0:.1f}%)")
    for src, n, full6 in source_info:
        print(f"    - {src}: {n:,} races, 6艇揃い {full6 or 0:,} ({(full6 or 0)/n*100 if n else 0:.1f}%)")
        rows.append({"period": label, "table": f"race_conditions/{src}",
                     "races": n, "notes": f"6艇揃い {full6 or 0:,} ({(full6 or 0)/n*100 if n else 0:.1f}%)"})
    # current_series
    n_cs = count_races("current_series", cond)
    rows.append({"period": label, "table": "current_series", "races": n_cs, "notes": ""})
    print(f"  current_series      : {n_cs:>7,}")
    # race_meta_murao
    n_rmm = count_races("race_meta_murao", cond)
    rows.append({"period": label, "table": "race_meta_murao", "races": n_rmm, "notes": ""})
    print(f"  race_meta_murao     : {n_rmm:>7,}")
    # 実使用: race_cards ∩ race_results ∩ race_conditions (f4_disp 使える範囲)
    c.execute(f"""SELECT COUNT(*) FROM (
                  SELECT rc.date, rc.stadium, rc.race_number FROM race_cards rc
                  WHERE rc.date BETWEEN '{cond.split("'")[1]}' AND '{cond.split("'")[3]}'
                  INTERSECT
                  SELECT date, stadium, race_number FROM race_results
                  WHERE {cond} AND rank IS NOT NULL GROUP BY date, stadium, race_number)""")
    n_rc_rr = c.fetchone()[0]
    rows.append({"period": label, "table": "race_cards ∩ race_results",
                 "races": n_rc_rr, "notes": "v3 学習に使える最大"})
    print(f"  race_cards ∩ race_results: {n_rc_rr:>7,}  ← 学習可能レース")

    # f4_disp 完備率 (race_conditions で 6 艇揃い) ÷ (race_cards ∩ race_results)
    c.execute(f"""SELECT COUNT(DISTINCT rc.date||'-'||rc.stadium||'-'||rc.race_number)
                  FROM race_conditions rc
                  WHERE {cond.replace('date', 'rc.date')}
                    AND rc.display_time_1 IS NOT NULL AND rc.display_time_6 IS NOT NULL""")
    n_f4 = c.fetchone()[0]
    rows.append({"period": label, "table": "race_conditions (display_time 6艇揃い)",
                 "races": n_f4, "notes": "f4_disp 計算可能レース"})
    print(f"  → f4_disp 計算可能   : {n_f4:>7,} ({n_f4/n_rc_rr*100 if n_rc_rr else 0:.1f}% of 学習可能)")

# racer_profile / racer_history は Phase D 用なのでカウントのみ
print("\n【Phase D 用 (今回未使用)】")
c.execute("SELECT COUNT(*) FROM racer_profile")
rp = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM racer_history")
rh = c.fetchone()[0]
rows.append({"period": "-", "table": "racer_profile", "races": rp, "notes": "Phase D 用"})
rows.append({"period": "-", "table": "racer_history", "races": rh, "notes": "Phase D 用"})
print(f"  racer_profile: {rp:,} / racer_history: {rh:,}")

# race_conditions の期間ごとの 6 艇揃い率 (月別)
print("\n【train 期間の f4_disp 月別完備率 (最重要)】")
c.execute(f"""SELECT strftime('%Y-%m', date) ym,
              COUNT(DISTINCT date||'-'||stadium||'-'||race_number) total,
              SUM(CASE WHEN display_time_1 IS NOT NULL AND display_time_2 IS NOT NULL
                       AND display_time_3 IS NOT NULL AND display_time_4 IS NOT NULL
                       AND display_time_5 IS NOT NULL AND display_time_6 IS NOT NULL
                       THEN 1 ELSE 0 END) f6
              FROM race_conditions
              WHERE date BETWEEN '{TRAIN_FROM}' AND '{TRAIN_TO}'
              GROUP BY ym ORDER BY ym""")
print(f"  {'月':>8} {'total':>8} {'6艇揃い':>10} {'率':>7}")
for ym, tot, f6 in c.fetchall():
    pct = (f6 or 0) / tot * 100 if tot else 0
    print(f"  {ym:>8} {tot:>8,} {(f6 or 0):>10,} {pct:>6.1f}%")

# CSV 保存
df = pd.DataFrame(rows)
out_csv = OUT / "v3_data_coverage.csv"
df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\nsaved: {out_csv}")
