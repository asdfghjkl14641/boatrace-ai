# -*- coding: utf-8 -*-
"""
Supabase に7テーブルを作成する初期セットアップスクリプト。
一度だけ実行すればOK。再実行しても IF NOT EXISTS なので安全。

実行例:
    python -m scripts.setup_db
"""
import os
from scripts.db import get_connection


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    schema_path = os.path.join(base, "sql", "schema.sql")

    with open(schema_path, "r", encoding="utf-8") as f:
        sql = f.read()

    print(f"[setup_db] schema.sql を読み込みました: {schema_path}")
    print("[setup_db] Supabase に接続してテーブルを作成します...")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print("[setup_db] ✔ テーブル作成完了")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
