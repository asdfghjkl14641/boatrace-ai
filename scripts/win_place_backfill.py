# -*- coding: utf-8 -*-
"""
win_odds / place_odds 専用バックフィルスクリプト

scraper.BoatraceScraper.get_odds_win_place_show() を並列で呼び出し
公式 boatrace.jp から単勝・複勝オッズを取り込む。

使い方 (他ジョブと並行起動推奨):
  python -m scripts.win_place_backfill \
      --from-date 2023-05-01 --to-date 2026-04-18 \
      --parallel 10 --sleep-min 1.0 --sleep-max 1.5
"""
from __future__ import annotations
import argparse, datetime as dt, json, logging, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from scripts.db import get_connection, placeholder  # noqa
from scripts.scraper import BoatraceScraper  # noqa
from scripts.stadiums import stadium_name  # noqa


PROGRESS_PATH = BASE / "scripts" / "win_place_backfill_progress.json"
LOG_DIR = BASE / "logs"


def _build_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d")
    log_path = LOG_DIR / f"win_place_backfill_{stamp}.log"
    logger = logging.getLogger("win_place_backfill")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); logger.addHandler(sh)
    return logger


@dataclass
class RateLimiter:
    min_sleep: float = 1.0
    max_sleep: float = 1.5
    _last_req: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def throttle(self, sleep_fn=time.sleep) -> float:
        with self._lock:
            now = time.time()
            elapsed = now - self._last_req if self._last_req else float("inf")
            target = random.uniform(self.min_sleep, self.max_sleep)
            slept = 0.0
            if elapsed < target:
                slept = target - elapsed
                sleep_fn(slept)
            self._last_req = time.time()
            return slept


@dataclass
class Progress:
    completed: set = field(default_factory=set)
    failed: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @staticmethod
    def key(d, s, r) -> str:
        return f"{d.isoformat()}:{s}:{r}"

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
        for retry in range(5):
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(p.to_dict(), f, ensure_ascii=False, indent=2)
                tmp.replace(path)
                return
            except (PermissionError, OSError):
                time.sleep(0.5 * (retry + 1))


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


def get_pending(from_date: dt.date, to_date: dt.date) -> list:
    """race_cards にあって win_odds にまだ無い (date, stadium, race) を返す。
    place_odds も原則同じ範囲なので、win_odds を基準にする。"""
    ph = placeholder()
    sql = f"""
    SELECT rc.date, rc.stadium, rc.race_number
    FROM race_cards rc
    LEFT JOIN win_odds w
      ON rc.date = w.date AND rc.stadium = w.stadium
     AND rc.race_number = w.race_number
    WHERE rc.date BETWEEN {ph} AND {ph}
      AND w.date IS NULL
    GROUP BY rc.date, rc.stadium, rc.race_number
    ORDER BY rc.date, rc.stadium, rc.race_number
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql, (from_date, to_date))
        out = []
        for r in cur.fetchall():
            d = r[0]
            if isinstance(d, str):
                d = dt.date.fromisoformat(d)
            out.append((d, int(r[1]), int(r[2])))
        return out
    finally:
        conn.close()


def _to_float(v):
    try:
        return float(v) if v is not None and str(v).strip() not in ("", "-", "−") else None
    except (ValueError, TypeError):
        return None


def insert_win_place(d, stadium, race, raw, logger):
    ph = placeholder()
    conn = get_connection()
    try:
        cur = conn.cursor()
        s_name = stadium_name(stadium)
        # win_odds: boat 1..6 × 1 odds
        win = raw.get("win") or {}
        rows_w = []
        for boat_s in ["1","2","3","4","5","6"]:
            v = _to_float(win.get(boat_s))
            if v is None:
                continue
            rows_w.append((d, stadium, s_name, race, int(boat_s), v))
        if rows_w:
            sql_w = (f"INSERT INTO win_odds (date, stadium, stadium_name, race_number, boat_number, odds) "
                     f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph}) "
                     f"ON CONFLICT DO NOTHING")
            cur.executemany(sql_w, rows_w)
        # place_odds: [下限, 上限] の 2 レンジ → odds=平均値 1 つだけ保存
        place = raw.get("place_show") or {}
        rows_p = []
        for boat_s in ["1","2","3","4","5","6"]:
            arr = place.get(boat_s) or []
            vals = [_to_float(a) for a in arr if _to_float(a) is not None]
            if not vals:
                continue
            # 代表値は下限 (安全側、着内最悪オッズ) を保存
            rows_p.append((d, stadium, s_name, race, int(boat_s), min(vals)))
        if rows_p:
            sql_p = (f"INSERT INTO place_odds (date, stadium, stadium_name, race_number, boat_number, odds) "
                     f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph}) "
                     f"ON CONFLICT DO NOTHING")
            cur.executemany(sql_p, rows_p)
        conn.commit()
        return (len(rows_w), len(rows_p))
    except Exception as e:
        conn.rollback()
        logger.exception("db err: %s", e)
        raise
    finally:
        conn.close()


def fetch_and_store(d, stadium, race, scraper, rate, logger) -> tuple[bool, str]:
    rate.throttle()
    try:
        raw = scraper.get_odds_win_place_show(d, stadium, race)
    except Exception as e:
        return (False, f"fetch err: {e!r}"[:200])
    if not raw or not raw.get("win"):
        return (False, "empty")
    try:
        insert_win_place(d, stadium, race, raw, logger)
    except Exception as e:
        return (False, f"db err: {e!r}"[:200])
    return (True, "ok")


def run(from_date, to_date, parallel, sleep_min, sleep_max, dry_run, logger):
    logger.info("pending 計算中...")
    pending_all = get_pending(from_date, to_date)
    logger.info("pending (race_cards にあり win_odds に無い): %d 件", len(pending_all))

    progress = load_progress()
    # 成功済 + 過去に「empty」「中止」扱いで失敗した race はスキップ
    def _skip(t):
        if progress.is_done(*t):
            return True
        k = progress.key(*t)
        r = progress.failed.get(k, "")
        # 中止レース (empty) は再試行しても取れないのでスキップ
        return r.startswith("empty") or r.startswith("no useful")
    pending = [t for t in pending_all if not _skip(t)]
    logger.info("checkpoint 除外後: %d 件 (empty/中止 skip 込み)", len(pending))

    if dry_run:
        for d, s, r in pending[:10]:
            logger.info("[dry] %s stadium=%d race=%d", d, s, r)
        return

    rate = RateLimiter(min_sleep=sleep_min, max_sleep=sleep_max)
    thread_local = threading.local()
    def _get_scraper():
        if not hasattr(thread_local, "scraper"):
            thread_local.scraper = BoatraceScraper(min_interval=0.0, timeout=30)
        return thread_local.scraper

    def _task(task):
        d, s, r = task
        sc = _get_scraper()
        ok, reason = fetch_and_store(d, s, r, sc, rate, logger)
        return (task, ok, reason)

    done = 0; fail = 0; t0 = time.time()
    CHUNK = max(parallel * 4, 32)
    with ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="wp") as ex:
        idx = 0
        while idx < len(pending):
            batch = pending[idx:idx+CHUNK]; idx += len(batch)
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
                                done, fail, len(pending), time.time()-t0,
                                (done+fail)/max(time.time()-t0,1)*60)
    save_progress(progress)
    logger.info("done. completed=%d failed=%d", done, fail)


def _parse_date(s): return dt.date.fromisoformat(s)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="win_place_backfill")
    p.add_argument("--from-date", type=_parse_date, required=True)
    p.add_argument("--to-date", type=_parse_date, required=True)
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--sleep-min", type=float, default=1.0)
    p.add_argument("--sleep-max", type=float, default=1.5)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.sleep_min < 0.5:
        raise SystemExit("ERROR: --sleep-min >= 0.5")
    if not (1 <= args.parallel <= 16):
        raise SystemExit(f"ERROR: --parallel 1..16")
    logger = _build_logger()
    try:
        run(args.from_date, args.to_date, args.parallel, args.sleep_min, args.sleep_max,
            args.dry_run, logger)
    except Exception:
        import traceback
        logger.error("fatal uncaught:\n%s", traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
