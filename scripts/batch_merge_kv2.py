# -*- coding: utf-8 -*-
"""K v2 artifacts を本 DB にマージ (39 月, INSERT OR IGNORE で既存保護)."""
from __future__ import annotations
import sys, sqlite3, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parent.parent
DB = BASE / "boatrace.db"
SAMPLING = BASE / "sampling_db_kv2"
BULK = BASE / "bulk_db_kv2"
PHASE2 = BASE / "phase2_db"

dirs = []
for d in [SAMPLING, BULK, PHASE2]:
    if d.exists():
        dirs.append(d)
artifacts = sorted([a for d in dirs for a in d.glob("*.db")])
print(f"artifacts: {len(artifacts)}")

# race_results: INSERT OR IGNORE (既存保護 — 2023-05 以降の 30% も壊さない)
# race_cards: INSERT OR IGNORE (既存保護)
# current_series: INSERT OR IGNORE
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
        INSERT OR IGNORE INTO race_results (
            date, stadium, stadium_name, race_number,
            rank, boat, racerid, name, time, time_sec
        )
        SELECT
            date, stadium, stadium_name, race_number,
            rank, boat, racerid, name, time, time_sec
        FROM src.race_results
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

t_start = time.time()
conn = sqlite3.connect(str(DB))
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("PRAGMA synchronous=NORMAL;")
conn.execute("PRAGMA cache_size=-200000;")

before = {}
for t in SQLS:
    before[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
print(f"Before: {before}")

# 2025-07 以降の保護確認用 sentinel
sentinel_2025_07 = conn.execute(
    "SELECT COUNT(*) FROM race_results WHERE date >= '2025-08-01'").fetchone()[0]
print(f"Sentinel (2025-08+ race_results): {sentinel_2025_07:,}")

totals = {t: 0 for t in SQLS}
for i, art in enumerate(artifacts, 1):
    t0 = time.time()
    print(f"\n[{i}/{len(artifacts)}] {art.name}")
    conn.execute(f"ATTACH DATABASE '{art.as_posix()}' AS src;")
    for t, sql in SQLS.items():
        c = conn.execute(f"SELECT COUNT(*) FROM src.{t}").fetchone()[0]
        if c == 0: continue
        bef = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        conn.execute(sql)
        aft = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        delta = aft - bef
        totals[t] += delta
        print(f"  {t:<18} src={c:>7,}  +{delta:>7,}  now {aft:>10,}")
    conn.commit()
    conn.execute("DETACH DATABASE src;")
    print(f"  elapsed: {time.time()-t0:.1f}s")

# センチネル再確認
sentinel_after = conn.execute(
    "SELECT COUNT(*) FROM race_results WHERE date >= '2025-08-01'").fetchone()[0]
print(f"\nSentinel after: {sentinel_after:,} (期待 変化なし: {sentinel_2025_07:,})")
if sentinel_after != sentinel_2025_07:
    print(f"**WARNING**: 2025-08 以降 race_results が変化 ({sentinel_after - sentinel_2025_07:+,})")

# 最終確認
print(f"\n{'='*60}")
print(f"マージ完了 ({time.time()-t_start:.1f}s)")
for t in SQLS:
    after = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t:<20} {before[t]:>10,} → {after:>10,} (+{totals[t]:,})")
conn.close()
print("done")
