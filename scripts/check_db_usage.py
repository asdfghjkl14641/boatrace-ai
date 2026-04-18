# -*- coding: utf-8 -*-
"""
Supabase (無料枠 500MB) の現在使用量を確認する。
80% (400MB) を超えたら警告を出力し、exit code 1 を返す。

実行例:
    python -m scripts.check_db_usage
"""
import sys
from scripts.db import get_connection

# 無料枠の上限 (MB)
FREE_LIMIT_MB = 500
WARN_RATIO = 0.80  # 80% で警告


def format_mb(bytes_: int) -> float:
    return bytes_ / (1024 * 1024)


def main() -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # データベース全体のサイズ
            cur.execute("SELECT pg_database_size(current_database());")
            db_size_bytes = cur.fetchone()[0]

            # 各テーブルのサイズ上位
            cur.execute("""
                SELECT
                    relname AS table_name,
                    pg_total_relation_size(C.oid) AS total_bytes
                FROM pg_class C
                LEFT JOIN pg_namespace N ON N.oid = C.relnamespace
                WHERE nspname = 'public' AND C.relkind = 'r'
                ORDER BY total_bytes DESC;
            """)
            tables = cur.fetchall()
    finally:
        conn.close()

    db_mb = format_mb(db_size_bytes)
    ratio = db_mb / FREE_LIMIT_MB
    bar_len = 30
    filled = min(bar_len, int(bar_len * ratio))
    bar = "█" * filled + "░" * (bar_len - filled)

    print("=" * 60)
    print("Supabase 使用量チェック")
    print("=" * 60)
    print(f"  DB全体サイズ: {db_mb:.2f} MB / {FREE_LIMIT_MB} MB  ({ratio*100:.1f}%)")
    print(f"  [{bar}]")
    print()
    print("  テーブル別サイズ:")
    for name, size in tables:
        print(f"    {name:25s} {format_mb(size):>8.2f} MB")
    print("=" * 60)

    if ratio >= WARN_RATIO:
        print()
        print(f"⚠️  警告: 無料枠の {WARN_RATIO*100:.0f}% ({FREE_LIMIT_MB*WARN_RATIO:.0f} MB) を超えています！")
        print(f"    古いデータの削除や、不要なテーブルの整理を検討してください。")
        return 1

    if ratio >= 0.5:
        print()
        print(f"ℹ️  情報: 無料枠の50%を超えました。今後の増加に注意。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
