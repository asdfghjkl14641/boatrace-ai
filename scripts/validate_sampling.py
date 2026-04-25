# -*- coding: utf-8 -*-
"""Step 1 サンプリング検証: 4 月の artifact の品質チェック."""
import sys, sqlite3
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")

SAMPLES = [
    ("2020-02", "sampling_db/2020-02-01-2020-02-29.db"),
    ("2021-10", "sampling_db/2021-10-01-2021-10-31.db"),
    ("2022-06", "sampling_db/2022-06-01-2022-06-30.db"),
    ("2023-04", "sampling_db/2023-04-01-2023-04-30.db"),
]

print("="*90)
print(f"{'月':<10} {'rows':>7} {'races':>6} {'motor NL%':>10} "
      f"{'gw NL%':>8} {'age NL%':>8} {'A1':>5} {'A2':>5} {'B1':>5} {'B2':>5}")
print("="*90)

all_rows = 0; all_motor_null = 0; all_gw_null = 0
any_fail = False
for month, path in SAMPLES:
    if not Path(path).exists():
        print(f"  {month} MISSING: {path}")
        any_fail = True; continue
    conn = sqlite3.connect(path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM race_cards").fetchone()[0]
        races = conn.execute("SELECT COUNT(DISTINCT date||'|'||stadium||'|'||race_number) FROM race_cards").fetchone()[0]
        motor_null = conn.execute("SELECT COUNT(*) FROM race_cards WHERE motor IS NULL").fetchone()[0]
        gw_null = conn.execute("SELECT COUNT(*) FROM race_cards WHERE global_win_pt IS NULL").fetchone()[0]
        age_null = conn.execute("SELECT COUNT(*) FROM race_cards WHERE age IS NULL").fetchone()[0]
        class_dist = {c: n for c, n in conn.execute(
            "SELECT class, COUNT(*) FROM race_cards GROUP BY class").fetchall()}
        a1 = class_dist.get("A1",0); a2 = class_dist.get("A2",0)
        b1 = class_dist.get("B1",0); b2 = class_dist.get("B2",0)
        motor_pct = motor_null/total*100 if total else 0
        gw_pct = gw_null/total*100 if total else 0
        age_pct = age_null/total*100 if total else 0
        print(f"  {month:<10} {total:>7,} {races:>6,} {motor_pct:>9.2f}% "
              f"{gw_pct:>7.2f}% {age_pct:>7.2f}% {a1:>5,} {a2:>5,} {b1:>5,} {b2:>5,}")
        all_rows += total; all_motor_null += motor_null; all_gw_null += gw_null
        # 異常チェック
        if motor_pct > 5.0:
            print(f"    ⚠ motor NULL 率 > 5%"); any_fail = True
        if total == 0:
            print(f"    ⚠ 0 rows"); any_fail = True
        if total < 15000 or total > 35000:
            print(f"    ⚠ 行数想定外 (15000-35000 期待)"); any_fail = True
    except Exception as e:
        print(f"  {month} ERROR: {e}")
        any_fail = True
    finally:
        conn.close()

print("="*90)
print(f"  {'合計':<10} {all_rows:>7,}  motor NULL: {all_motor_null} "
      f"({all_motor_null/all_rows*100:.2f}%)  gw NULL: {all_gw_null} "
      f"({all_gw_null/all_rows*100:.2f}%)")
print()
if any_fail:
    print("**異常あり** — Step 2 に進まない. 上記の警告を確認.")
    sys.exit(1)
else:
    print("全サンプル OK — Step 2 (35 ヶ月取得) に進める.")
