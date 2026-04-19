# -*- coding: utf-8 -*-
"""
GitHub Actions で生成された artifact.db を、ローカルの boatrace.db に
マージする。

■ 背景
    lhafile は Windows Python 3.14 ではビルドできないため、LZH取り込みは
    GH Actions 上で行い、結果を artifact.db (SQLite) として配布する。
    このスクリプトでそれをローカル boatrace.db に INSERT/UPSERT する。

■ マージ方針
    race_cards       : INSERT OR IGNORE (既存レコードは保持)
    race_results     : UPSERT。time/time_sec が NULL の既存行には値を埋める
    current_series   : INSERT OR IGNORE
    race_conditions  : INSERT OR IGNORE

■ 使い方
    python -m scripts.merge_sqlite path/to/artifact.db
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )


MERGE_SQL = {
    # race_cards: 既存行を保持する (後から入れた方を捨てる)
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
    # race_results: UPSERT (time/time_sec が NULL の場合は埋める)
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
    # current_series: 既存保持
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
    # race_conditions: 既存保持
    "race_conditions": """
        INSERT OR IGNORE INTO race_conditions (
            date, stadium, stadium_name, race_number,
            weather, temperature, wind_direction, wind_speed,
            water_temperature, wave_height, stabilizer,
            display_time_1, display_time_2, display_time_3,
            display_time_4, display_time_5, display_time_6
        )
        SELECT
            date, stadium, stadium_name, race_number,
            weather, temperature, wind_direction, wind_speed,
            water_temperature, wave_height, stabilizer,
            display_time_1, display_time_2, display_time_3,
            display_time_4, display_time_5, display_time_6
        FROM src.race_conditions
    """,
}


def merge(src_path: Path, dst_path: Path) -> None:
    if not src_path.exists():
        raise FileNotFoundError(f"ソースが見つかりません: {src_path}")
    if not dst_path.exists():
        raise FileNotFoundError(f"ターゲットが見つかりません: {dst_path}")

    logging.info(f"マージ: {src_path}  ➜  {dst_path}")
    src_size = src_path.stat().st_size / 1024 / 1024
    dst_size_before = dst_path.stat().st_size / 1024 / 1024
    logging.info(f"  src  = {src_size:.2f} MB")
    logging.info(f"  dst(前) = {dst_size_before:.2f} MB")

    conn = sqlite3.connect(dst_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(f"ATTACH DATABASE '{src_path.as_posix()}' AS src;")

    # 移行元のテーブル一覧を確認
    cur = conn.cursor()
    cur.execute("SELECT name FROM src.sqlite_master WHERE type='table'")
    src_tables = {r[0] for r in cur.fetchall()}
    logging.info(f"  src のテーブル: {sorted(src_tables)}")

    totals = {}
    for table, sql in MERGE_SQL.items():
        if table not in src_tables:
            logging.info(f"  {table}: (src に無いのでスキップ)")
            continue
        # src の該当テーブルが空ならスキップ
        cur.execute(f"SELECT COUNT(*) FROM src.{table}")
        src_count = cur.fetchone()[0]
        if src_count == 0:
            logging.info(f"  {table}: src が空 (0行)")
            totals[table] = 0
            continue
        t0 = time.time()
        cur.execute("SELECT COUNT(*) FROM " + table)
        before = cur.fetchone()[0]
        cur.execute(sql)
        cur.execute("SELECT COUNT(*) FROM " + table)
        after = cur.fetchone()[0]
        conn.commit()
        delta = after - before
        dt_s = time.time() - t0
        logging.info(
            f"  {table:20s} src={src_count:>10,}  dst {before:>10,} → {after:>10,}  "
            f"(+{delta:>6,})  {dt_s:.1f}s"
        )
        totals[table] = delta

    conn.execute("DETACH DATABASE src;")
    logging.info("VACUUM 中...")
    conn.execute("VACUUM")
    conn.close()

    dst_size_after = dst_path.stat().st_size / 1024 / 1024
    logging.info(f"  dst(後) = {dst_size_after:.2f} MB (差 {dst_size_after - dst_size_before:+.2f} MB)")
    logging.info("=" * 60)
    logging.info(f"マージ完了。INSERT:")
    for k, v in totals.items():
        logging.info(f"  {k:20s} +{v:>8,}")
    logging.info("=" * 60)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("source", help="マージ元 SQLite ファイル (artifact.db)")
    p.add_argument("--target", default=str(BASE / "boatrace.db"),
                   help="マージ先 SQLite (既定: boatrace.db)")
    args = p.parse_args()
    setup_logging()
    merge(Path(args.source), Path(args.target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
