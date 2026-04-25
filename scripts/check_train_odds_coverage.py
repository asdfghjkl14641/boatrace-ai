# -*- coding: utf-8 -*-
"""Step 0: train 期間の trifecta_odds 可用性を月別に確認."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")
from scripts.db import get_connection
import pandas as pd

conn = get_connection()
q = """
SELECT
  strftime('%Y-%m', rr.date) AS month,
  COUNT(DISTINCT rr.date || '-' || rr.stadium || '-' || rr.race_number) AS with_results,
  COUNT(DISTINCT CASE WHEN tr.odds_1min IS NOT NULL
        THEN rr.date || '-' || rr.stadium || '-' || rr.race_number END) AS with_odds_1min
FROM race_results rr
LEFT JOIN trifecta_odds tr
  ON rr.date = tr.date AND rr.stadium = tr.stadium AND rr.race_number = tr.race_number
WHERE rr.date BETWEEN '2023-05-01' AND '2025-12-31'
GROUP BY month
ORDER BY month
"""
df = pd.read_sql_query(q, conn.native)
df["coverage_pct"] = (df["with_odds_1min"] / df["with_results"] * 100).round(2)
print(df.to_string(index=False))
print(f"\nTotal: with_results={df['with_results'].sum():,}, with_odds_1min={df['with_odds_1min'].sum():,}")
print(f"Overall coverage: {df['with_odds_1min'].sum()/df['with_results'].sum()*100:.2f}%")
conn.close()
