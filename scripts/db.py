# -*- coding: utf-8 -*-
"""
DB接続ユーティリティ。SQLite と Supabase (PostgreSQL) を切り替えられる。

■ 切替方法
    環境変数 DB_MODE:
        'sqlite'   (デフォルト): プロジェクト直下の boatrace.db に保存
        'supabase'               : DATABASE_URL 経由で Supabase に接続

■ 使い方
    from scripts.db import get_connection, placeholder
    conn = get_connection()
    ph = placeholder()              # '?' (sqlite) or '%s' (postgres)
    cur = conn.cursor()
    cur.execute(f"INSERT INTO t (a,b) VALUES ({ph},{ph})", (1,2))
    conn.commit()

SQLスクリプト側は placeholder() を使って書けば両対応になる。
ON CONFLICT は (カラム名) 形式を使うこと (制約名は SQLite で不可)。
"""
from __future__ import annotations

import datetime as _dt
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

# プロジェクトルートの .env を読む
_BASE = Path(__file__).resolve().parent.parent
load_dotenv(_BASE / ".env")

DB_MODE = os.getenv("DB_MODE", "sqlite").strip().lower()
SQLITE_PATH = Path(os.getenv("SQLITE_PATH", str(_BASE / "boatrace.db")))


# ---- sqlite3 の date adapter/converter を登録 (Python 3.12+ で必須) ----
def _adapt_date(d: _dt.date) -> str:
    return d.isoformat()


def _convert_date(b: bytes) -> _dt.date:
    s = b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else str(b)
    # SQLiteにはDATETIMEが入る事もあり得るので、日付部分だけを取る
    s = s.split(" ")[0]
    return _dt.date.fromisoformat(s)


sqlite3.register_adapter(_dt.date, _adapt_date)
sqlite3.register_adapter(_dt.datetime, lambda d: d.isoformat(sep=" "))
sqlite3.register_converter("DATE", _convert_date)


def placeholder() -> str:
    """現在のDB_MODEに合ったパラメータプレースホルダを返す。"""
    return "?" if DB_MODE == "sqlite" else "%s"


# ---- 共通の接続オブジェクトラッパ ----
# psycopg3 の connection/cursor と、sqlite3 のそれを同一インターフェースで使えるように薄くラップする。
class _CursorWrap:
    def __init__(self, cur, mode: str):
        self._cur = cur
        self._mode = mode

    # with ステートメントで使えるようにしておく (sqlite3.Cursor は元々非対応)
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._cur.close()
        return False

    # 実行系
    def execute(self, sql, params=None):
        if self._mode == "sqlite":
            sql = sql.replace("%s", "?")
        return self._cur.execute(sql, params or ())

    def executemany(self, sql, seq):
        if self._mode == "sqlite":
            sql = sql.replace("%s", "?")
        return self._cur.executemany(sql, seq)

    # 取得系
    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    def fetchmany(self, size=None):
        return self._cur.fetchmany(size) if size is not None else self._cur.fetchmany()

    def close(self):
        self._cur.close()

    @property
    def rowcount(self):
        return self._cur.rowcount

    @property
    def description(self):
        return self._cur.description


class _ConnWrap:
    def __init__(self, native, mode: str):
        self._native = native
        self._mode = mode

    def cursor(self):
        return _CursorWrap(self._native.cursor(), self._mode)

    def commit(self):
        self._native.commit()

    def rollback(self):
        try:
            self._native.rollback()
        except Exception:
            pass

    def close(self):
        try:
            self._native.close()
        except Exception:
            pass

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def native(self):
        return self._native


# ------------------------------------------------------------
# 接続取得
# ------------------------------------------------------------
def _connect_sqlite() -> _ConnWrap:
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        SQLITE_PATH,
        detect_types=sqlite3.PARSE_DECLTYPES,
        timeout=30.0,
    )
    # 高速化: WALモード + 同期弱め (単一プロセス利用ならOK)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return _ConnWrap(conn, "sqlite")


def _connect_supabase() -> _ConnWrap:
    import psycopg  # 遅延import (sqlite-only運用時に不要)
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DB_MODE=supabase だが環境変数 DATABASE_URL が設定されていません。"
        )
    if "sslmode=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    # pgBouncer (Transaction Pooler) 対策で prepared statement を無効化
    conn = psycopg.connect(url, prepare_threshold=None)
    return _ConnWrap(conn, "supabase")


def get_connection() -> _ConnWrap:
    """DB_MODE に応じて接続を返す。呼び出し側は with 文か明示 close する。"""
    if DB_MODE == "sqlite":
        return _connect_sqlite()
    elif DB_MODE == "supabase":
        return _connect_supabase()
    else:
        raise ValueError(f"DB_MODE は 'sqlite' か 'supabase' を指定してください (指定値: {DB_MODE!r})")
