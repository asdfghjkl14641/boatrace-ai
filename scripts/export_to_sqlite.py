# -*- coding: utf-8 -*-
"""
Supabase の全テーブルを SQLite (boatrace.db) に移行する。

■ 使い方
    python -m scripts.export_to_sqlite
        (既存の boatrace.db がある場合は上書き確認あり)
    python -m scripts.export_to_sqlite --force
        (確認なしで上書き)

■ 動作
    1. Supabase に接続 (DATABASE_URL)
    2. sqlite_schema.sql で SQLite テーブルを作成
    3. 各テーブルを SELECT → ページング (10000件ずつ) → SQLite へ INSERT
    4. 件数とサイズを表示

■ 注意
    本スクリプト実行時は psycopg も sqlite3 も両方使うので、
    db.py の DB_MODE 環境変数には依存しない (直接両方に接続する)。
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from tqdm import tqdm

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")

SQLITE_PATH = Path(os.getenv("SQLITE_PATH", str(BASE / "boatrace.db")))
SCHEMA_PATH = BASE / "sql" / "sqlite_schema.sql"

# 移行対象テーブル。id/created_at は自動生成なので除外する。
TABLES = {
    "race_cards": [
        "date", "stadium", "stadium_name", "race_number", "lane",
        "racerid", "name", "class", "branch", "birthplace", "age", "weight",
        "f", "l", "aveST",
        "global_win_pt", "global_in2nd", "global_in3rd",
        "local_win_pt", "local_in2nd", "local_in3rd",
        "motor", "motor_in2nd", "motor_in3rd",
        "boat", "boat_in2nd", "boat_in3rd",
    ],
    "race_results": [
        "date", "stadium", "stadium_name", "race_number",
        "rank", "boat", "racerid", "name", "time", "time_sec",
    ],
    "race_conditions": [
        "date", "stadium", "stadium_name", "race_number",
        "weather", "temperature", "wind_direction", "wind_speed",
        "water_temperature", "wave_height", "stabilizer",
        "display_time_1", "display_time_2", "display_time_3",
        "display_time_4", "display_time_5", "display_time_6",
    ],
    "trifecta_odds": [
        "date", "stadium", "stadium_name", "race_number", "combination", "odds",
    ],
    "win_odds": [
        "date", "stadium", "stadium_name", "race_number", "boat_number", "odds",
    ],
    "place_odds": [
        "date", "stadium", "stadium_name", "race_number", "boat_number", "odds",
    ],
    "current_series": [
        "date", "stadium", "stadium_name", "racerid", "race_number",
        "boat_number", "course", "st", "rank",
    ],
}

PAGE_SIZE = 10000


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="既存 boatrace.db を確認なしで上書き")
    args = parser.parse_args()

    setup_logging()
    logging.info(f"SQLite出力先: {SQLITE_PATH}")
    logging.info(f"スキーマ: {SCHEMA_PATH}")

    # 既存DBの処理
    if SQLITE_PATH.exists():
        if not args.force:
            ans = input(f"{SQLITE_PATH} は既に存在します。上書きしますか? [y/N]: ")
            if ans.strip().lower() not in ("y", "yes"):
                logging.info("中断しました。")
                return 1
        SQLITE_PATH.unlink()
        logging.info("既存DBを削除しました。")

    # Supabase 接続 (pgBouncer対策で prepare_threshold=None)
    pg_url = os.getenv("DATABASE_URL")
    if not pg_url:
        logging.error("DATABASE_URL が設定されていません。")
        return 2
    if "sslmode=" not in pg_url:
        sep = "&" if "?" in pg_url else "?"
        pg_url = f"{pg_url}{sep}sslmode=require"

    logging.info("Supabase に接続...")
    pg_conn = psycopg.connect(pg_url, prepare_threshold=None)

    # SQLite 初期化
    logging.info("SQLite を作成中...")
    sq_conn = sqlite3.connect(SQLITE_PATH)
    sq_conn.execute("PRAGMA journal_mode=WAL;")
    sq_conn.execute("PRAGMA synchronous=OFF;")   # 移行中のみ高速化
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        sq_conn.executescript(f.read())
    sq_conn.commit()
    logging.info("テーブル作成完了")

    # 各テーブルを転送
    totals = {}
    start_ts = datetime.now()
    for table, cols in TABLES.items():
        t0 = datetime.now()
        col_list = ", ".join(cols)
        # 件数取得
        with pg_conn.cursor() as pg_cur:
            pg_cur.execute(f"SELECT COUNT(*) FROM {table}")
            total_rows = pg_cur.fetchone()[0]

        if total_rows == 0:
            logging.info(f"  {table}: 0 行 (スキップ)")
            totals[table] = 0
            continue

        placeholders = ", ".join(["?"] * len(cols))
        insert_sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"

        # サーバサイドカーソル (named) でストリーム取得
        cur_name = f"_export_{table}"
        copied = 0
        bar = tqdm(total=total_rows, desc=f"{table:18s}", unit="row")
        with pg_conn.cursor(name=cur_name) as pg_cur:
            pg_cur.itersize = PAGE_SIZE
            pg_cur.execute(f"SELECT {col_list} FROM {table} ORDER BY id")
            batch = []
            for row in pg_cur:
                batch.append(tuple(row))
                if len(batch) >= PAGE_SIZE:
                    sq_conn.executemany(insert_sql, batch)
                    sq_conn.commit()
                    copied += len(batch)
                    bar.update(len(batch))
                    batch.clear()
            if batch:
                sq_conn.executemany(insert_sql, batch)
                sq_conn.commit()
                copied += len(batch)
                bar.update(len(batch))
        bar.close()

        totals[table] = copied
        dt = (datetime.now() - t0).total_seconds()
        logging.info(f"  ✔ {table}: {copied:,} 行 ({dt:.1f}秒)")

    # VACUUM で最終圧縮
    logging.info("VACUUM (最終圧縮) ...")
    sq_conn.execute("PRAGMA synchronous=NORMAL;")
    sq_conn.commit()
    sq_conn.close()
    # VACUUM は独立モードで
    conn2 = sqlite3.connect(SQLITE_PATH)
    conn2.execute("VACUUM")
    conn2.close()

    pg_conn.close()

    # サイズ
    size_mb = SQLITE_PATH.stat().st_size / 1024 / 1024
    total_dt = (datetime.now() - start_ts).total_seconds()
    logging.info("=" * 60)
    logging.info(f"移行完了: {SQLITE_PATH}  /  所要 {total_dt:.1f}秒")
    logging.info(f"SQLiteファイルサイズ: {size_mb:.2f} MB")
    logging.info("件数:")
    for k, v in totals.items():
        logging.info(f"  {k:20s} {v:>12,}")
    logging.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
