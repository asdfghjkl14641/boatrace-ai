# -*- coding: utf-8 -*-
"""Step 4-5: マージ後の検証 + レポート."""
import sys, sqlite3
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent
DB = BASE / "boatrace.db"
LOG = BASE.parent / "logs" / "race_cards_historical_final.md"

conn = sqlite3.connect(str(DB))

print("=" * 80)
print("race_cards 履歴マージ検証")
print("=" * 80)

# 4.1 全体
print("\n[4.1] 全体統計 ...")
total = conn.execute("SELECT COUNT(*) FROM race_cards").fetchone()[0]
min_d, max_d = conn.execute("SELECT MIN(date), MAX(date) FROM race_cards").fetchone()
print(f"  race_cards 総行数: {total:,}  期間 {min_d} ~ {max_d}")

# 既存 2023-05 以降のデータ整合性
since_may = conn.execute("SELECT COUNT(*) FROM race_cards WHERE date >= '2023-05-01'").fetchone()[0]
print(f"  2023-05 以降 (既存): {since_may:,}")

# 2020-02〜2023-04 の新規
new_period = conn.execute("SELECT COUNT(*) FROM race_cards WHERE date BETWEEN '2020-02-01' AND '2023-04-30'").fetchone()[0]
print(f"  2020-02〜2023-04: {new_period:,}")

# 4.2 月別カバー
print("\n[4.2] 月別カバー (2020-02〜2023-04):")
q = """
SELECT
  substr(date,1,7) AS ym,
  COUNT(*) AS n,
  COUNT(DISTINCT date||'|'||stadium||'|'||race_number) AS races,
  SUM(CASE WHEN motor IS NULL THEN 1 ELSE 0 END) AS motor_null,
  SUM(CASE WHEN global_win_pt IS NULL THEN 1 ELSE 0 END) AS gw_null
FROM race_cards
WHERE date BETWEEN '2020-02-01' AND '2023-04-30'
GROUP BY ym ORDER BY ym
"""
rows = conn.execute(q).fetchall()
print(f"  {'月':<10} {'rows':>7} {'races':>6} {'motor NL':>9} {'gw NL':>7}")
total_motor_null = 0; total_gw_null = 0
for ym, n, races, mn, gn in rows:
    total_motor_null += mn; total_gw_null += gn
    pct_m = mn/n*100 if n else 0
    pct_g = gn/n*100 if n else 0
    warn = " ⚠" if pct_m > 0 else ""
    print(f"  {ym:<10} {n:>7,} {races:>6,} {mn:>8,} ({pct_m:.2f}%) {gn:>6,} ({pct_g:.2f}%){warn}")
print(f"  (計 {len(rows)} 月)")

# 4.3 trifecta_odds との JOIN
print("\n[4.3] trifecta_odds × race_cards JOIN:")
q = """
SELECT
  COUNT(DISTINCT t.date||'|'||t.stadium||'|'||t.race_number) AS odds_races,
  COUNT(DISTINCT CASE WHEN rc.date IS NOT NULL THEN t.date||'|'||t.stadium||'|'||t.race_number END) AS joined_races
FROM trifecta_odds t
LEFT JOIN race_cards rc
  ON t.date = rc.date AND t.stadium = rc.stadium AND t.race_number = rc.race_number
WHERE t.date BETWEEN '2020-02-01' AND '2023-04-30'
"""
odds_races, joined = conn.execute(q).fetchone()
pct = joined/odds_races*100 if odds_races else 0
print(f"  trifecta_odds レース数: {odds_races:,}")
print(f"  race_cards JOIN 成功: {joined:,} ({pct:.2f}%)")
success = pct >= 95.0

# 4.4 既存データ破壊していないか (2023-05 以降)
print("\n[4.4] 2023-05 以降の既存データ整合性:")
existing = conn.execute("SELECT COUNT(*) FROM race_cards WHERE date >= '2023-05-01'").fetchone()[0]
expected_existing = 997488  # 前回確認した値 (bulk merge 前)
print(f"  2023-05 以降: {existing:,} (前回 {expected_existing:,})")
if existing == expected_existing:
    print("  ✅ 既存データ無変化")
elif existing > expected_existing:
    print(f"  ⚠ 増加 (+{existing-expected_existing:,}) — 2023-04 artifact で追加の可能性")
else:
    print(f"  ❌ 減少 ({existing-expected_existing})")

conn.close()

# レポート
LOG.parent.mkdir(parents=True, exist_ok=True)
md = f"""# race_cards 履歴取得 最終レポート

## 概要
- **取得期間**: 2020-02-01 〜 2023-04-30
- **手法**: GH Actions 並列 (39 ジョブ) + mbrace.or.jp LZH B ダウンロード + b_parser_v2

## 総合結果
| 指標 | 値 |
|---|---|
| race_cards 総行数 | {total:,} |
| 期間カバー | {min_d} 〜 {max_d} |
| 2020-02〜2023-04 新規 | {new_period:,} |
| 2023-05 以降 既存 | {since_may:,} |
| motor NULL 率 | {total_motor_null/new_period*100:.4f}% |
| global_win_pt NULL 率 | {total_gw_null/new_period*100:.4f}% |
| trifecta_odds JOIN 成功率 | **{pct:.2f}%** |

## 月別カバー (期待 vs 実績)

| 月 | rows | races | motor NULL | gw NULL |
|---|---|---|---|---|
"""
for ym, n, races, mn, gn in rows:
    md += f"| {ym} | {n:,} | {races:,} | {mn:,} | {gn:,} |\n"

md += f"""

## 判定
- JOIN 成功率 {pct:.2f}% {'✅ (95% 以上)' if pct >= 95 else '⚠ 95% 未満'}
- motor NULL 率 {total_motor_null/new_period*100:.4f}% {'✅ (5% 以下)' if total_motor_null/new_period < 0.05 else '⚠'}
- 既存データ (2023-05 以降) {'保持 ✅' if existing == expected_existing else '変化あり ⚠'}

## 次ステップ
1. v4 モデル再学習: train 2020-02〜2025-12 (約 68 ヶ月に拡大可能)
2. train/test 分割再設計: test を 2026-01〜04 (現状) + 2020-02〜2020-06 (過去) で分散
3. 過学習問題の再検証: より大きなサンプルで CI を狭める

## 生成ファイル
- [boatrace-ai/scripts/b_parser_v2.py](boatrace-ai/scripts/b_parser_v2.py)
- [boatrace-ai/scripts/b_parser_test.py](boatrace-ai/scripts/b_parser_test.py)
- [boatrace-ai/scripts/batch_merge_artifacts.py](boatrace-ai/scripts/batch_merge_artifacts.py)
- boatrace-ai/boatrace.db (merged)
- boatrace_backup_20260423_pre_historical_merge.db (バックアップ)
"""
LOG.write_text(md, encoding="utf-8")
print(f"\nreport saved: {LOG}")
print(f"\n=== JOIN成功率 {pct:.2f}%  {'✅ OK' if success else '⚠ 要調査'} ===")
