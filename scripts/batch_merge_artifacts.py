# -*- coding: utf-8 -*-
"""複数の artifact.db を本 DB に一括マージ (VACUUM は最後に 1 回のみ)."""
from __future__ import annotations
import sys, sqlite3, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parent.parent
DB = BASE / "boatrace.db"
SAMPLING = BASE / "sampling_db"
BULK = BASE / "bulk_db"

if not DB.exists():
    print(f"ERROR: {DB} not found"); sys.exit(1)

# 全 artifact 収集
artifacts = sorted(list(SAMPLING.glob("*.db")) + list(BULK.glob("*.db")))
print(f"見つかった artifact: {len(artifacts)} 件")
for a in artifacts:
    print(f"  {a.name} ({a.stat().st_size/1024/1024:.1f} MB)")

SQLS = {
    "race_cards": """
        INSERT OR IGNORE INTO race_cards (
            date, stadium, stadium_name, race_number, lane,
            racerid, name, class, branch, birthplace, age, weight,
            f, l, aveST,
            global_win_pt, global_in2nd, global_in3rd,
            local_win_pt,  local_in2nd,  local_in3rd,
            motor, motor_in2nd, motor_in3rd,
            boat, boat_in2nd, boat_in3rd
        )
        SELECT
            date, stadium, stadium_name, race_number, lane,
            racerid, name, class, branch, birthplace, age, weight,
            f, l, aveST,
            global_win_pt, global_in2nd, global_in3rd,
            local_win_pt,  local_in2nd,  local_in3rd,
            motor, motor_in2nd, motor_in3rd,
            boat, boat_in2nd, boat_in3rd
        FROM src.race_cards
    """,
    "race_results": """
        INSERT INTO race_results (
            date, stadium, stadium_name, race_number,
            rank, boat, racerid, name, time, time_sec
        )
        SELECT
            date, stadium, stadium_name, race_number,
            rank, boat, racerid, name, time, time_sec
        FROM src.race_results
        WHERE true
        ON CONFLICT (date, stadium, race_number, boat) DO UPDATE SET
            time     = COALESCE(race_results.time,     excluded.time),
            time_sec = COALESCE(race_results.time_sec, excluded.time_sec),
            rank     = COALESCE(race_results.rank,     excluded.rank),
            racerid  = COALESCE(race_results.racerid,  excluded.racerid),
            name     = COALESCE(race_results.name,     excluded.name)
    """,
    "current_series": """
        INSERT OR IGNORE INTO current_series (
            date, stadium, stadium_name, racerid, race_number,
            boat_number, course, st, rank
        )
        SELECT
            date, stadium, stadium_name, racerid, race_number,
            boat_number, course, st, rank
        FROM src.current_series
    """,
}

# 一括マージ
t_start = time.time()
conn = sqlite3.connect(str(DB))
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("PRAGMA synchronous=NORMAL;")
conn.execute("PRAGMA temp_store=MEMORY;")
conn.execute("PRAGMA cache_size=-200000;")  # 200MB cache

totals = {t: 0 for t in SQLS}
before_counts = {}
for t in SQLS:
    before_counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
print(f"\nBefore merge:")
for t, n in before_counts.items():
    print(f"  {t}: {n:,}")

for i, art in enumerate(artifacts, 1):
    t0 = time.time()
    print(f"\n[{i}/{len(artifacts)}] {art.name}")
    conn.execute(f"ATTACH DATABASE '{art.as_posix()}' AS src;")
    for t, sql in SQLS.items():
        cur = conn.cursor()
        try:
            c = conn.execute(f"SELECT COUNT(*) FROM src.{t}").fetchone()[0]
            if c == 0:
                continue
            before = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            cur.execute(sql)
            after = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            delta = after - before
            totals[t] += delta
            print(f"  {t:<18} src={c:>6,}  +{delta:>6,}  (dst now {after:>10,})")
        except Exception as e:
            print(f"  {t}: ERROR {e}")
    conn.commit()
    conn.execute("DETACH DATABASE src;")
    print(f"  elapsed: {time.time()-t0:.1f}s")

print(f"\n{'='*60}")
print(f"マージ完了 ({time.time()-t_start:.1f}s)")
print(f"挿入合計:")
for t, n in totals.items():
    print(f"  {t:<20} +{n:,}")

print(f"\nAfter merge:")
for t in SQLS:
    after = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t}: {after:,} (was {before_counts[t]:,})")

# VACUUM せず終了 (巨大 DB では VACUUM は別タイミングで手動実施推奨)
conn.close()
print("\ndone (VACUUM はスキップ)")
