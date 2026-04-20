# -*- coding: utf-8 -*-
"""
Migration 004 を boatrace.db (SQLite) に冪等適用する。

  python -m scripts.apply_migration_004
"""
from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from scripts.db import get_connection  # noqa: E402

MIG_SQL = BASE / "scripts" / "migrations" / "004_kyotei_murao_tables.sql"

NEW_TRIFECTA_COLS = [
    ("odds_5min", "REAL"),
    ("pop_5min",  "INTEGER"),
    ("odds_1min", "REAL"),
    ("pop_1min",  "INTEGER"),
]


def table_columns(cur, table: str) -> list[str]:
    cur.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]


def ensure_column(cur, table: str, name: str, type_: str) -> bool:
    cols = table_columns(cur, table)
    if name in cols:
        return False
    cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {type_}")
    return True


def main() -> int:
    conn = get_connection()
    added = []
    try:
        cur = conn.cursor()

        # trifecta_odds 拡張 (冪等)
        for name, tp in NEW_TRIFECTA_COLS:
            if ensure_column(cur, "trifecta_odds", name, tp):
                added.append(f"trifecta_odds.{name} {tp}")

        # 新規テーブル (CREATE IF NOT EXISTS 入りのSQL一括実行)
        sql = MIG_SQL.read_text(encoding="utf-8")
        # SQLite 複数文一括実行は native の executescript が使える
        native = conn.native
        native.executescript(sql)

        conn.commit()
    finally:
        conn.close()

    print("migration 004 applied.")
    if added:
        print("  ALTER TABLE additions:")
        for a in added:
            print(f"    + {a}")
    else:
        print("  (no ALTER TABLE changes needed — already up-to-date)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
