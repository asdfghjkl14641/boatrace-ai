# -*- coding: utf-8 -*-
"""
GitHub Actions ワークフロー `kyotei-murao-a2.yml` が生成した artifact.db を
ローカルの boatrace.db にマージする。

■ マージ戦略
    trifecta_odds       : UPSERT。新しい 5min/1min/pop_* を上書き、
                          既存 `odds` カラム (旧スクレイパー由来) は保持。
    odds_exacta         : UPSERT (combination ごとの odds を最新化)
    odds_quinella       : UPSERT
    odds_trio           : UPSERT
    race_meta_murao     : INSERT OR IGNORE (1 レース 1 行、変化しない前提)

■ 使い方
    python -m scripts.merge_kyotei_murao path/to/artifact.db
    python -m scripts.merge_kyotei_murao artifact.db --target boatrace.db
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
    # trifecta_odds: 既存行の `odds` は保持しつつ、5min/1min/pop_* を上書き
    "trifecta_odds": """
        INSERT INTO trifecta_odds (
            date, stadium, stadium_name, race_number, combination,
            odds_5min, pop_5min, odds_1min, pop_1min
        )
        SELECT
            date, stadium, stadium_name, race_number, combination,
            odds_5min, pop_5min, odds_1min, pop_1min
        FROM src.trifecta_odds
        WHERE true
        ON CONFLICT (date, stadium, race_number, combination) DO UPDATE SET
            odds_5min = excluded.odds_5min,
            pop_5min  = excluded.pop_5min,
            odds_1min = excluded.odds_1min,
            pop_1min  = excluded.pop_1min
    """,
    "odds_exacta": """
        INSERT INTO odds_exacta (
            date, stadium, stadium_name, race_number, combination,
            odds_5min, pop_5min, odds_1min, pop_1min
        )
        SELECT
            date, stadium, stadium_name, race_number, combination,
            odds_5min, pop_5min, odds_1min, pop_1min
        FROM src.odds_exacta
        WHERE true
        ON CONFLICT (date, stadium, race_number, combination) DO UPDATE SET
            odds_5min = excluded.odds_5min,
            pop_5min  = excluded.pop_5min,
            odds_1min = excluded.odds_1min,
            pop_1min  = excluded.pop_1min
    """,
    "odds_quinella": """
        INSERT INTO odds_quinella (
            date, stadium, stadium_name, race_number, combination,
            odds_5min, pop_5min, odds_1min, pop_1min
        )
        SELECT
            date, stadium, stadium_name, race_number, combination,
            odds_5min, pop_5min, odds_1min, pop_1min
        FROM src.odds_quinella
        WHERE true
        ON CONFLICT (date, stadium, race_number, combination) DO UPDATE SET
            odds_5min = excluded.odds_5min,
            pop_5min  = excluded.pop_5min,
            odds_1min = excluded.odds_1min,
            pop_1min  = excluded.pop_1min
    """,
    "odds_trio": """
        INSERT INTO odds_trio (
            date, stadium, stadium_name, race_number, combination,
            odds_5min, pop_5min, odds_1min, pop_1min
        )
        SELECT
            date, stadium, stadium_name, race_number, combination,
            odds_5min, pop_5min, odds_1min, pop_1min
        FROM src.odds_trio
        WHERE true
        ON CONFLICT (date, stadium, race_number, combination) DO UPDATE SET
            odds_5min = excluded.odds_5min,
            pop_5min  = excluded.pop_5min,
            odds_1min = excluded.odds_1min,
            pop_1min  = excluded.pop_1min
    """,
    "race_meta_murao": """
        INSERT OR IGNORE INTO race_meta_murao (
            date, stadium, stadium_name, race_number,
            weekday, series, grade, rank_count, rank_lineup,
            race_type, time_zone, entry_fixed, schedule_day, final_race,
            weather, wind_direction, wind_speed, wave,
            result_order, kimarite, payout
        )
        SELECT
            date, stadium, stadium_name, race_number,
            weekday, series, grade, rank_count, rank_lineup,
            race_type, time_zone, entry_fixed, schedule_day, final_race,
            weather, wind_direction, wind_speed, wave,
            result_order, kimarite, payout
        FROM src.race_meta_murao
    """,
}


def ensure_schema(dst: sqlite3.Connection) -> None:
    """マージ先 DB に murao 用テーブルが無ければ migration 004 を流す。"""
    cur = dst.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    need_init = not ({"odds_exacta", "odds_quinella", "odds_trio", "race_meta_murao"} <= tables)
    if need_init:
        logging.info("マージ先に murao 用テーブルが無いので migration 004 を適用します")
        # 冪等適用 (ALTER TABLE は列存在チェック、CREATE は IF NOT EXISTS)
        import importlib
        sys.path.insert(0, str(BASE))
        mod = importlib.import_module("scripts.apply_migration_004")
        mod.main()


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

    # まず dst のスキーマが最新かを確認
    dst_preconn = sqlite3.connect(dst_path)
    ensure_schema(dst_preconn)
    dst_preconn.close()

    conn = sqlite3.connect(dst_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(f"ATTACH DATABASE '{src_path.as_posix()}' AS src;")

    cur = conn.cursor()
    cur.execute("SELECT name FROM src.sqlite_master WHERE type='table'")
    src_tables = {r[0] for r in cur.fetchall()}
    logging.info(f"  src のテーブル: {sorted(src_tables)}")

    totals: dict[str, int] = {}
    for table, sql in MERGE_SQL.items():
        if table not in src_tables:
            logging.info(f"  {table}: (src に無いのでスキップ)")
            continue
        cur.execute(f"SELECT COUNT(*) FROM src.{table}")
        src_count = cur.fetchone()[0]
        if src_count == 0:
            logging.info(f"  {table}: src が空 (0 行)")
            totals[table] = 0
            continue
        t0 = time.time()
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        before = cur.fetchone()[0]
        cur.execute(sql)
        conn.commit()
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        after = cur.fetchone()[0]
        delta = after - before
        dt_s = time.time() - t0
        logging.info(
            f"  {table:18s} src={src_count:>10,}  dst {before:>10,} → {after:>10,}  "
            f"(+{delta:>7,})  {dt_s:.1f}s"
        )
        totals[table] = delta

    conn.execute("DETACH DATABASE src;")
    logging.info("VACUUM 中...")
    conn.execute("VACUUM")
    conn.close()

    dst_size_after = dst_path.stat().st_size / 1024 / 1024
    logging.info(f"  dst(後) = {dst_size_after:.2f} MB (差 {dst_size_after - dst_size_before:+.2f} MB)")
    logging.info("=" * 60)
    logging.info("マージ完了。INSERT 件数 (既存更新は含まない):")
    for k, v in totals.items():
        logging.info(f"  {k:18s} +{v:>8,}")
    logging.info("=" * 60)


def main() -> int:
    p = argparse.ArgumentParser(
        description="kyotei_murao_scraper が書き出した artifact.db を boatrace.db にマージ",
    )
    p.add_argument("source", help="マージ元 SQLite (artifact.db)")
    p.add_argument("--target", default=str(BASE / "boatrace.db"),
                   help="マージ先 SQLite (default: boatrace.db)")
    args = p.parse_args()
    setup_logging()
    merge(Path(args.source), Path(args.target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
