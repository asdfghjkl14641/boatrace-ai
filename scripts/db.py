# -*- coding: utf-8 -*-
"""
Supabase (PostgreSQL) への接続ユーティリティ。
環境変数 DATABASE_URL から接続情報を取得する。
DATABASE_URL はSupabaseダッシュボード →
  Project Settings → Database → Connection string → URI から取得可能。
"""
import os
import psycopg
from dotenv import load_dotenv

# プロジェクトルートの .env を読み込む
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_BASE, ".env"))


def get_connection():
    """psycopg (v3) の接続オブジェクトを返す。
    呼び出し側は with で使うか、明示的に close すること。
    """
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "環境変数 DATABASE_URL が設定されていません。"
            ".env か GitHub Secrets を確認してください。"
        )
    # sslmode=require を強制 (Supabaseは常にSSL必須)
    if "sslmode=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    return psycopg.connect(url)
