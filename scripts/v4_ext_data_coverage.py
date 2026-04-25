# -*- coding: utf-8 -*-
"""Phase A-ext-1: 新 train/test 期間のデータカバー率確認."""
import sys, sqlite3
from pathlib import Path
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "notebooks" / "output"
OUT.mkdir(parents=True, exist_ok=True)
DB = BASE / "boatrace.db"

TRAIN_FROM = "2020-02-01"; TRAIN_TO = "2025-06-30"
TEST_FROM  = "2025-07-01"; TEST_TO  = "2026-04-18"

conn = sqlite3.connect(str(DB))
print("=" * 90)
print(f"Phase A-ext-1: データカバー確認")
print(f"  train: {TRAIN_FROM} 〜 {TRAIN_TO}")
print(f"  test : {TEST_FROM} 〜 {TEST_TO}")
print("=" * 90)

# 1.1 年別 race_cards カバー + race_results JOIN
print("\n[1.1] 年別 race_cards / race_results / race_conditions カバー (train)")
q = f"""
SELECT
  strftime('%Y', rc.date) AS year,
  COUNT(DISTINCT rc.date || '-' || rc.stadium || '-' || rc.race_number) AS n_races,
  COUNT(DISTINCT CASE WHEN rr.boat IS NOT NULL THEN rc.date || '-' || rc.stadium || '-' || rc.race_number END) AS n_with_results,
  COUNT(DISTINCT CASE WHEN cond.display_time_1 IS NOT NULL THEN rc.date || '-' || rc.stadium || '-' || rc.race_number END) AS n_with_disp,
  COUNT(DISTINCT CASE WHEN cond.wind_speed IS NOT NULL THEN rc.date || '-' || rc.stadium || '-' || rc.race_number END) AS n_with_wind
FROM race_cards rc
LEFT JOIN race_results rr ON rc.date = rr.date AND rc.stadium = rr.stadium AND rc.race_number = rr.race_number
LEFT JOIN race_conditions cond ON rc.date = cond.date AND rc.stadium = cond.stadium AND rc.race_number = cond.race_number
WHERE rc.date BETWEEN '{TRAIN_FROM}' AND '{TRAIN_TO}'
GROUP BY year
ORDER BY year
"""
tr = pd.read_sql_query(q, conn)
tr["result_pct"] = (tr["n_with_results"]/tr["n_races"]*100).round(2)
tr["disp_pct"]   = (tr["n_with_disp"]/tr["n_races"]*100).round(2)
tr["wind_pct"]   = (tr["n_with_wind"]/tr["n_races"]*100).round(2)
tr["period"] = "train"
print(tr.to_string(index=False))

# 1.2 test
print("\n[1.2] test 期間 (2025-07〜2026-04)")
q2 = q.replace(TRAIN_FROM, TEST_FROM).replace(TRAIN_TO, TEST_TO)
te = pd.read_sql_query(q2, conn)
te["result_pct"] = (te["n_with_results"]/te["n_races"]*100).round(2)
te["disp_pct"]   = (te["n_with_disp"]/te["n_races"]*100).round(2)
te["wind_pct"]   = (te["n_with_wind"]/te["n_races"]*100).round(2)
te["period"] = "test"
print(te.to_string(index=False))

# 1.3 f4_disp カバー率 (display_time_1 の有無で判定)
print("\n[1.3] 年別 f4_disp カバー率 (display_time_1 NOT NULL 率)")
all_df = pd.concat([tr, te], ignore_index=True)
all_df[["year","period","n_races","result_pct","disp_pct","wind_pct"]].to_csv(
    OUT/"v4_ext_data_coverage.csv", index=False, encoding="utf-8-sig")
print(f"\nsaved: v4_ext_data_coverage.csv")

# まとめ
train_total = int(tr["n_races"].sum())
train_result_pct = int(tr["n_with_results"].sum()) / train_total * 100
train_disp_pct = int(tr["n_with_disp"].sum()) / train_total * 100
test_total = int(te["n_races"].sum())
test_result_pct = int(te["n_with_results"].sum()) / test_total * 100 if test_total else 0
test_disp_pct = int(te["n_with_disp"].sum()) / test_total * 100 if test_total else 0
print()
print("=" * 90)
print(f"  train: {train_total:,} races,  result {train_result_pct:.2f}%,  disp {train_disp_pct:.2f}%")
print(f"  test : {test_total:,} races,  result {test_result_pct:.2f}%,  disp {test_disp_pct:.2f}%")
print("=" * 90)
conn.close()
