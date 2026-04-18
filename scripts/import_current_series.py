# -*- coding: utf-8 -*-
"""
BoatraceOpenAPI results から current_series を再構成する。

■ 背景
    import_openapi.py では race_results を生成したが、course と ST カラムは
    既存の race_results スキーマに無いため、current_series を埋められなかった。
    一方 OpenAPI results JSON には各艇の racer_course_number / racer_start_timing /
    racer_place_number が含まれる。ここから current_series を直接生成する。

■ カバー期間
    results API のカバレッジ = 2025-07-15 〜 現在
    それより前の期間 (2023-05-01 〜 2025-07-14) は import_lzh.py で補完される。

■ 使い方
    python -m scripts.import_current_series --start 2025-07-15 --end 2026-04-18
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Iterable

import requests
from tqdm import tqdm

from scripts.db import get_connection
from scripts.stadiums import stadium_name

BASE_URL = "https://boatraceopenapi.github.io"
REQUEST_INTERVAL = float(os.getenv("OPENAPI_INTERVAL_SEC", "0.3"))
RESULTS_START = date(2025, 7, 15)

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def setup_logging() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(LOG_DIR, f"import_current_series_{ts}.log")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    return logfile


_last = [0.0]


def _throttle():
    e = time.time() - _last[0]
    if e < REQUEST_INTERVAL:
        time.sleep(REQUEST_INTERVAL - e)
    _last[0] = time.time()


def fetch_results(d: date, session: requests.Session) -> dict | None:
    _throttle()
    url = f"{BASE_URL}/results/v2/{d.year}/{d.strftime('%Y%m%d')}.json"
    r = session.get(url, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def rows_from_results(data: dict) -> list[tuple]:
    """current_series にINSERTするタプル列を作成。"""
    rows: list[tuple] = []
    for race in data.get("results", []):
        d = race.get("race_date")
        stadium = race.get("race_stadium_number")
        race_number = race.get("race_number")
        if not (d and stadium and race_number):
            continue
        s_name = stadium_name(stadium)
        for b in race.get("boats", []) or []:
            racerid = b.get("racer_number")
            if not racerid:
                continue
            rank = b.get("racer_place_number")
            if isinstance(rank, int) and not (1 <= rank <= 6):
                rank = None
            rows.append((
                d, stadium, s_name, racerid, race_number,
                b.get("racer_boat_number"),
                b.get("racer_course_number"),
                b.get("racer_start_timing"),
                rank,
            ))
    return rows


INSERT = """
INSERT INTO current_series (date, stadium, stadium_name, racerid, race_number,
                            boat_number, course, "ST", rank)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT ON CONSTRAINT uq_current_series DO NOTHING
;
"""


def daterange(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def main() -> int:
    p = argparse.ArgumentParser(description="OpenAPI results から current_series を生成")
    p.add_argument("--start", default=RESULTS_START.isoformat(),
                   help=f"開始日 (デフォルト: {RESULTS_START})")
    p.add_argument("--end", required=True, help="終了日 YYYY-MM-DD")
    args = p.parse_args()
    logfile = setup_logging()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if start < RESULTS_START:
        logging.warning(f"--start が API カバー外 ({RESULTS_START}〜)。{RESULTS_START} に調整します。")
        start = RESULTS_START

    logging.info(f"current_series 生成: {start} 〜 {end}")
    session = requests.Session()
    session.headers.update({"User-Agent": "boatrace-ai/current-series/1.0"})

    total_inserted = 0
    total_days = 0
    total_404 = 0

    for d in tqdm(list(daterange(start, end)), desc="days", unit="day"):
        try:
            data = fetch_results(d, session)
        except Exception as e:
            logging.warning(f"  {d}: fetch失敗: {e}")
            continue
        if data is None:
            total_404 += 1
            continue
        rows = rows_from_results(data)
        if not rows:
            continue
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.executemany(INSERT, rows)
                total_inserted += cur.rowcount
            conn.commit()
            total_days += 1
        except Exception as e:
            logging.error(f"  {d}: DB書込失敗: {e}")
            try: conn.rollback()
            except Exception: pass
        finally:
            try: conn.close()
            except Exception: pass

    logging.info("=" * 60)
    logging.info("current_series 生成サマリー")
    logging.info(f"  処理日数:         {total_days}")
    logging.info(f"  データなし(404):  {total_404}")
    logging.info(f"  INSERT累計:       {total_inserted}")
    logging.info(f"  ログ: {logfile}")
    logging.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
