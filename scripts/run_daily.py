# -*- coding: utf-8 -*-
"""
毎日のボートレースデータ取得 (タスクスケジューラから呼ばれる想定)。

■ 何をするか
    1. SQLite から最後に取り込んだ日付を調べる
    2. その翌日 〜 昨日 までの範囲を fetch_all で一括取得
    3. 既に最新ならスキップ

PC が数日間起動していなくても、次回起動時にまとめて取得されるよう
ギャップ検出を入れている。

■ 使い方
    python -m scripts.run_daily            # 通常運用
    python -m scripts.run_daily --dry-run  # 対象日付だけ表示して終了
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import subprocess
import sys
from pathlib import Path

from scripts.db import get_connection

# race_cards が最古日付のテーブルなので、race_cards の MAX(date) を基準にする。
# race_cards がまだ無い (初回実行) なら、この日から開始。
BOOTSTRAP_START = dt.date(2023, 5, 1)

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_daily_{ts}.log"
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    logging.info(f"ログファイル: {log_path}")
    return log_path


def last_fetched_date() -> dt.date | None:
    """race_cards から MAX(date) を取り出す。空なら None を返す。"""
    try:
        conn = get_connection()
    except Exception as e:
        logging.warning(f"DB接続失敗 (初回運用?): {e}")
        return None
    try:
        cur = conn.cursor()
        cur.execute("SELECT MAX(date) FROM race_cards")
        row = cur.fetchone()
    finally:
        conn.close()
    if not row or row[0] is None:
        return None
    v = row[0]
    if isinstance(v, dt.date):
        return v
    # SQLite が str で返す場合 (ISO形式)
    return dt.date.fromisoformat(str(v).split(" ")[0])


def main() -> int:
    p = argparse.ArgumentParser(description="ボートレース日次取得 (ギャップ補完あり)")
    p.add_argument("--dry-run", action="store_true",
                   help="取得対象の日付範囲を表示するだけで終了")
    p.add_argument("--max-gap-days", type=int, default=60,
                   help="過去ギャップをどこまで遡るか上限日数 (デフォルト60日)。"
                        "これより大きいギャップは backfill に委ねる。")
    args = p.parse_args()

    setup_logging()

    today = dt.date.today()
    yesterday = today - dt.timedelta(days=1)

    last = last_fetched_date()
    if last is None:
        start = BOOTSTRAP_START
        logging.info(f"race_cards が空。初回として {start} から取得します。")
    elif last >= yesterday:
        logging.info(f"最新まで取得済み (last={last}, yesterday={yesterday})。終了。")
        return 0
    else:
        start = last + dt.timedelta(days=1)
        gap_days = (yesterday - last).days
        logging.info(f"最後の取得日: {last} / ギャップ: {gap_days} 日")
        if gap_days > args.max_gap_days:
            logging.warning(
                f"ギャップが大きい ({gap_days} 日)。"
                f"直近 {args.max_gap_days} 日分だけ取得します。"
                f"残りは backfill を別途実行してください。"
            )
            start = yesterday - dt.timedelta(days=args.max_gap_days - 1)

    end = yesterday
    logging.info(f"★ 取得範囲: {start} 〜 {end} ({(end - start).days + 1} 日分)")

    if args.dry_run:
        logging.info("(--dry-run のため取得はスキップ)")
        return 0

    # fetch_all を子プロセスで呼び出す (ログもそちらに集約される)
    cmd = [
        sys.executable, "-m", "scripts.fetch_all",
        "--start", start.isoformat(),
        "--end", end.isoformat(),
    ]
    logging.info(f"実行: {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    rc = subprocess.call(cmd, env=env)
    logging.info(f"fetch_all 終了コード: {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
