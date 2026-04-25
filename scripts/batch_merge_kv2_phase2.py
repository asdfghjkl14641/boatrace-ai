# -*- coding: utf-8 -*-
"""Phase 2 (2023-05〜2025-06) のみをマージ."""
from __future__ import annotations
import sys, sqlite3, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parent.parent
DB = BASE / "boatrace.db"
PHASE2 = BASE / "phase2_db"

artifacts = sorted(PHASE2.glob("*.db"))
print(f"artifacts: {len(artifacts)}")

SQLS = {
    "race_cards": "INSERT OR IGNORE INTO race_cards SELECT NULL, date, stadium, stadium_name, race_number, lane, racerid, name, class, branch, birthplace, age, weight, f, l, aveST, global_win_pt, global_in2nd, global_in3rd, local_win_pt, local_in2nd, local_in3rd, motor, motor_in2nd, motor_in3rd, boat, boat_in2nd, boat_in3rd, created_at FROM src.race_cards",
    "race_results": """INSERT OR IGNORE INTO race_results (date, stadium, stadium_name, race_number, rank, boat, racerid, name, time, time_sec) SELECT date, stadium, stadium_name, race_number, rank, boat, racerid, name, time, time_sec FROM src.race_results""",
    "current_series": """INSERT OR IGNORE INTO current_series (date, stadium, stadium_name, racerid, race_number, boat_number, course, st, rank) SELECT date, stadium, stadium_name, racerid, race_number, boat_number, course, st, rank FROM src.current_series""",
}

t_start = time.time()
conn = sqlite3.connect(str(DB))
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("PRAGMA synchronous=NORMAL;")
conn.execute("PRAGMA cache_size=-200000;")

before = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in SQLS}
print(f"Before: {before}")

# Sentinel: 2025-08 以降は touch しないが、Phase 2 データは 2025-07 含む
# 2025-07 の既存 race_results の 3,588 races は残したい + 1,608 races 追加で 5,196 に
sent = conn.execute("SELECT COUNT(*) FROM race_results WHERE date >= '2025-08-01'").fetchone()[0]
print(f"Sentinel (2025-08+): {sent:,}")

totals = {t: 0 for t in SQLS}
for i, art in enumerate(artifacts, 1):
    t0 = time.time()
    print(f"[{i}/{len(artifacts)}] {art.name}", end=" ")
    conn.execute(f"ATTACH DATABASE '{art.as_posix()}' AS src;")
    for t, sql in SQLS.items():
        try:
            c = conn.execute(f"SELECT COUNT(*) FROM src.{t}").fetchone()[0]
            if c == 0: continue
            bef = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            # race_cards のスキーマ合わせに素直なアプローチ:
            if t == "race_cards":
                sql2 = """INSERT OR IGNORE INTO race_cards (date, stadium, stadium_name, race_number, lane, racerid, name, class, branch, birthplace, age, weight, f, l, aveST, global_win_pt, global_in2nd, global_in3rd, local_win_pt, local_in2nd, local_in3rd, motor, motor_in2nd, motor_in3rd, boat, boat_in2nd, boat_in3rd) SELECT date, stadium, stadium_name, race_number, lane, racerid, name, class, branch, birthplace, age, weight, f, l, aveST, global_win_pt, global_in2nd, global_in3rd, local_win_pt, local_in2nd, local_in3rd, motor, motor_in2nd, motor_in3rd, boat, boat_in2nd, boat_in3rd FROM src.race_cards"""
                conn.execute(sql2)
            else:
                conn.execute(sql)
            aft = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            totals[t] += (aft-bef)
            print(f"+{aft-bef:>5,} {t}", end=" ")
        except Exception as e:
            print(f"ERROR {t}: {e}", end=" ")
    conn.commit()
    conn.execute("DETACH DATABASE src;")
    print(f"({time.time()-t0:.1f}s)")

sent_after = conn.execute("SELECT COUNT(*) FROM race_results WHERE date >= '2025-08-01'").fetchone()[0]
print(f"\nSentinel after: {sent_after:,} (期待 {sent:,} から変化なし)")

print(f"\n{'='*60}\nマージ完了 ({time.time()-t_start:.1f}s)")
for t in SQLS:
    aft = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t:<20} {before[t]:>10,} → {aft:>10,} (+{totals[t]:,})")
conn.close()
print("done")
