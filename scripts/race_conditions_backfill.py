# -*- coding: utf-8 -*-
"""
race_conditions (直前情報) 専用バックフィルスクリプト

既存 BoatraceScraper.get_just_before_info() を並列 (ThreadPoolExecutor)
で呼び出し、race_conditions テーブルに INSERT OR IGNORE する。

■ 倫理
  * 公式 boatrace.jp は商用サイト。負荷をかけないこと。
  * デフォルト並列 3、1.0〜1.5 秒間隔 (合計 約 2〜3 req/sec)
  * 429/503 が出たら即停止・間隔拡大・並列度減

■ 対象
  race_cards に存在するが race_conditions に無いレース

■ 使い方
  python -m scripts.race_conditions_backfill \
      --from-date 2023-05-01 --to-date 2025-07-31 \
      --parallel 3 --sleep-min 1.0 --sleep-max 1.5
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from scripts.db import get_connection, placeholder  # noqa: E402
from scripts.scraper import BoatraceScraper  # noqa: E402


HOURLY_LIMIT_PER_WORKER = 1200
PROGRESS_PATH = BASE / "scripts" / "race_conditions_backfill_progress.json"
LOG_DIR = BASE / "logs"


def _build_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d")
    log_path = LOG_DIR / f"race_conditions_backfill_{stamp}.log"
    logger = logging.getLogger("race_conditions_backfill")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); logger.addHandler(sh)
    return logger


# ----------------------------------------------------------------
# 共有レートリミッタ (並列ワーカーが共有)
# ----------------------------------------------------------------
@dataclass
class RateLimiter:
    min_sleep: float = 1.0
    max_sleep: float = 1.5
    hourly_limit: int = 3600
    _last_req: float = 0.0
    _window_start: float = 0.0
    _window_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def throttle(self, sleep_fn=time.sleep) -> float:
        with self._lock:
            now = time.time()
            if self._window_start == 0.0 or now - self._window_start > 3600:
                self._window_start = now
                self._window_count = 0
            if self._window_count >= self.hourly_limit:
                wait = 3600 - (now - self._window_start) + 1
                if wait > 0:
                    sleep_fn(wait)
                self._window_start = time.time()
                self._window_count = 0
            now = time.time()
            elapsed = now - self._last_req if self._last_req else float("inf")
            target = random.uniform(self.min_sleep, self.max_sleep)
            slept = 0.0
            if elapsed < target:
                slept = target - elapsed
                sleep_fn(slept)
            self._last_req = time.time()
            self._window_count += 1
            return slept


# ----------------------------------------------------------------
# チェックポイント (key = "YYYY-MM-DD:stadium:race")
# ----------------------------------------------------------------
@dataclass
class Progress:
    completed: set[str] = field(default_factory=set)
    failed: dict[str, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @staticmethod
    def key(d: dt.date, stadium: int, race: int) -> str:
        return f"{d.isoformat()}:{stadium}:{race}"

    def is_done(self, d, s, r) -> bool:
        with self._lock:
            return self.key(d, s, r) in self.completed

    def mark_done(self, d, s, r) -> None:
        k = self.key(d, s, r)
        with self._lock:
            self.completed.add(k)
            self.failed.pop(k, None)

    def mark_failed(self, d, s, r, reason: str) -> None:
        k = self.key(d, s, r)
        with self._lock:
            self.failed[k] = reason

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "completed": sorted(self.completed),
                "failed": dict(self.failed),
                "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            }


_save_lock = threading.Lock()

def save_progress(p: Progress, path: Path = PROGRESS_PATH) -> None:
    with _save_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        retries = 0
        while True:
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(p.to_dict(), f, ensure_ascii=False, indent=2)
                tmp.replace(path)
                return
            except (PermissionError, OSError) as e:
                retries += 1
                if retries >= 5:
                    # 書けなかったら諦めて続行 (メモリ上の state はロスト)
                    return
                time.sleep(0.5 * retries)


def load_progress(path: Path = PROGRESS_PATH) -> Progress:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            p = Progress()
            p.completed = set(d.get("completed", []))
            p.failed = dict(d.get("failed", {}))
            return p
        except Exception:
            pass
    return Progress()


# ----------------------------------------------------------------
# DB: 未取得レース取得 + INSERT OR IGNORE
# ----------------------------------------------------------------
def get_pending_races(date_from: dt.date, date_to: dt.date) -> list[tuple]:
    """race_cards にあるが race_conditions に無い (date, stadium, race_number)"""
    ph = placeholder()
    sql = f"""
    SELECT rc.date, rc.stadium, rc.race_number
    FROM race_cards rc
    LEFT JOIN race_conditions rco
      ON rc.date = rco.date AND rc.stadium = rco.stadium
     AND rc.race_number = rco.race_number
    WHERE rc.date BETWEEN {ph} AND {ph}
      AND rco.date IS NULL
    GROUP BY rc.date, rc.stadium, rc.race_number
    ORDER BY rc.date, rc.stadium, rc.race_number
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql, (date_from, date_to))
        rows = cur.fetchall()
        out = []
        for r in rows:
            d = r[0]
            if isinstance(d, str):
                d = dt.date.fromisoformat(d)
            out.append((d, int(r[1]), int(r[2])))
        return out
    finally:
        conn.close()


def insert_condition(row: dict, logger: logging.Logger) -> int:
    """1 レース分の race_conditions を INSERT OR IGNORE。"""
    if not row:
        return 0
    ph = placeholder()
    cols = [
        "date","stadium","stadium_name","race_number",
        "weather","temperature","wind_direction","wind_speed",
        "water_temperature","wave_height","stabilizer",
        "display_time_1","display_time_2","display_time_3",
        "display_time_4","display_time_5","display_time_6",
    ]
    placeholders = ",".join([ph] * len(cols))
    sql = (f"INSERT INTO race_conditions ({','.join(cols)}) "
           f"VALUES ({placeholders}) "
           f"ON CONFLICT (date, stadium, race_number) DO NOTHING")
    values = tuple(row.get(c) for c in cols)
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql, values)
        conn.commit()
        return cur.rowcount
    except Exception as e:
        conn.rollback()
        logger.exception("insert_condition failed: %s", e)
        raise
    finally:
        conn.close()


# ----------------------------------------------------------------
# 取得 + パース + 挿入
# ----------------------------------------------------------------
def fetch_and_store(
    d: dt.date, stadium: int, race: int,
    scraper: BoatraceScraper, rate: RateLimiter,
    progress: Progress, logger: logging.Logger,
) -> tuple[bool, str]:
    rate.throttle()
    try:
        raw = scraper.get_just_before_info(d, stadium, race)
    except Exception as e:
        return (False, f"fetch err: {e!r}"[:200])

    if not raw:
        return (False, "empty raw")

    # fetch_all.py と同じ正規化
    from scripts.stadiums import stadium_name
    w = raw.get("weather_information") or {}
    display = raw.get("display_times") or {}

    def _to_int(v):
        try: return int(v) if v is not None and str(v).strip() != "" else None
        except (ValueError, TypeError): return None

    def _to_float(v):
        try: return float(v) if v is not None and str(v).strip() != "" else None
        except (ValueError, TypeError): return None

    def _to_bool(v):
        if v is None: return None
        if isinstance(v, bool): return int(v)
        s = str(v).lower()
        return 1 if s in ("true", "1", "yes", "on", "有") else 0

    row = {
        "date": d,
        "stadium": stadium,
        "stadium_name": stadium_name(stadium),
        "race_number": race,
        "weather": w.get("weather"),
        "temperature": _to_float(w.get("temperature")),
        "wind_direction": _to_int(w.get("wind_direction")),
        "wind_speed": _to_float(w.get("wind_speed")),
        "water_temperature": _to_float(w.get("water_temperature")),
        "wave_height": _to_float(w.get("wave_height")),
        "stabilizer": _to_bool(raw.get("stabilizer"))
                     if raw.get("stabilizer") is not None
                     else (1 if "安定板" in str(raw) else 0),
    }
    # display_time_1..6
    if isinstance(display, dict):
        for i in range(1, 7):
            row[f"display_time_{i}"] = _to_float(display.get(str(i)) or display.get(i))
    elif isinstance(display, list):
        for i in range(1, 7):
            row[f"display_time_{i}"] = _to_float(display[i-1] if i-1 < len(display) else None)
    else:
        for i in range(1, 7):
            row[f"display_time_{i}"] = None

    # 内容が全部 None ならレース無し/キャンセル
    if all(row.get(k) is None for k in ("weather","wind_speed","wave_height")):
        return (False, "no useful data (empty race?)")

    try:
        insert_condition(row, logger)
    except Exception as e:
        return (False, f"db err: {e!r}"[:200])
    return (True, "ok")


# ----------------------------------------------------------------
# メインループ
# ----------------------------------------------------------------
def run(
    date_from: dt.date, date_to: dt.date,
    parallel: int, sleep_min: float, sleep_max: float,
    dry_run: bool, logger: logging.Logger,
) -> None:
    logger.info("pending 計算中...")
    pending_all = get_pending_races(date_from, date_to)
    logger.info("race_cards にあって race_conditions に無いレース: %d 件", len(pending_all))

    progress = load_progress()
    # 成功済 + 過去に empty/中止扱いの race もスキップ (再試行しても取れない)
    def _skip(d, s, r):
        if progress.is_done(d, s, r):
            return True
        k = progress.key(d, s, r)
        reason = progress.failed.get(k, "")
        return reason.startswith("empty") or reason.startswith("no useful")
    pending = [t for t in pending_all if not _skip(*t)]
    logger.info("checkpoint 除外後 pending: %d 件 (empty/中止 skip 込み)", len(pending))

    if dry_run:
        for d, s, r in pending[:10]:
            logger.info("[dry] %s stadium=%d race=%d", d, s, r)
        logger.info("dry-run: 先頭 10 件のみ表示")
        return

    rate = RateLimiter(
        min_sleep=sleep_min, max_sleep=sleep_max,
        hourly_limit=HOURLY_LIMIT_PER_WORKER * parallel,
    )

    # ワーカーごとにセッション分離
    thread_local = threading.local()
    def _get_scraper():
        if not hasattr(thread_local, "scraper"):
            thread_local.scraper = BoatraceScraper(min_interval=0.0, timeout=30)
        return thread_local.scraper

    def _task(task):
        d, s, r = task
        sc = _get_scraper()
        ok, reason = fetch_and_store(d, s, r, sc, rate, progress, logger)
        return (task, ok, reason)

    done = 0; fail = 0
    t0 = time.time()
    if parallel == 1:
        for task in pending:
            res_task, ok, reason = _task(task)
            if ok:
                progress.mark_done(*res_task); done += 1
            else:
                progress.mark_failed(*res_task, reason); fail += 1
            if (done + fail) % 100 == 0:
                save_progress(progress)
                logger.info("progress: done=%d fail=%d / %d  elapsed=%.1fs",
                            done, fail, len(pending), time.time() - t0)
    else:
        # 一括 submit だとメモリを食い過ぎるのでチャンクで流す
        CHUNK = max(parallel * 4, 32)
        with ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="rc") as ex:
            idx = 0
            while idx < len(pending):
                batch = pending[idx: idx + CHUNK]
                idx += len(batch)
                futs = [ex.submit(_task, t) for t in batch]
                for fut in as_completed(futs):
                    try:
                        res_task, ok, reason = fut.result()
                    except Exception as e:
                        logger.exception("worker err: %s", e); fail += 1; continue
                    if ok:
                        progress.mark_done(*res_task); done += 1
                    else:
                        progress.mark_failed(*res_task, reason); fail += 1
                    if (done + fail) % 100 == 0:
                        save_progress(progress)
                        logger.info("progress: done=%d fail=%d / %d  elapsed=%.1fs  rate=%.1f/min",
                                    done, fail, len(pending), time.time() - t0,
                                    (done + fail) / max(time.time() - t0, 1) * 60)
    save_progress(progress)
    logger.info("done. completed=%d failed=%d", done, fail)


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------
def _parse_date(s): return dt.date.fromisoformat(s)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="race_conditions_backfill")
    p.add_argument("--from-date", type=_parse_date, required=True)
    p.add_argument("--to-date", type=_parse_date, required=True)
    p.add_argument("--parallel", type=int, default=3,
                   help="並列ワーカー数 1..16 (公式サイト配慮で大きくしすぎない)")
    p.add_argument("--sleep-min", type=float, default=1.0)
    p.add_argument("--sleep-max", type=float, default=1.5)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.sleep_min < 0.5:
        raise SystemExit("ERROR: --sleep-min >= 0.5 (公式サイト配慮)")
    if not (1 <= args.parallel <= 16):
        raise SystemExit(f"ERROR: --parallel は 1..16 (got {args.parallel})")
    logger = _build_logger()
    # 無名 uncaught exception も必ず log に残す
    try:
        run(args.from_date, args.to_date, args.parallel,
            args.sleep_min, args.sleep_max, args.dry_run, logger)
    except Exception:
        import traceback
        logger.error("fatal uncaught:\n%s", traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
