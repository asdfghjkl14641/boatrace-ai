# -*- coding: utf-8 -*-
"""
Part 4a + 4b: kyotei.murao111.net から 選手プロフィール + 選手履歴 を取得

使い方:
  # Phase 4a (選手プロフィール 1,956 人を 1 req)
  python -m scripts.racer_backfill --profiles-only

  # Phase 4b (各選手の履歴 1,956 req)
  python -m scripts.racer_backfill \
      --from-date 2020-02-01 --to-date 2026-04-18 \
      --parallel 3 --sleep 1.0 --no-night-pause
"""
from __future__ import annotations
import argparse, datetime as dt, json, logging, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode
import requests
from bs4 import BeautifulSoup
import warnings
try:
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except Exception: pass

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from scripts.db import get_connection, placeholder  # noqa

BASE_URL = "https://kyotei.murao111.net"
PROGRESS_PATH = BASE / "scripts" / "racer_backfill_progress.json"
LOG_DIR = BASE / "logs"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Accept": "text/html,*/*;q=0.8",
}


def _build_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d")
    log_path = LOG_DIR / f"racer_backfill_{stamp}.log"
    logger = logging.getLogger("racer_backfill")
    if logger.handlers: return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    return logger


# ---------------------------- Schema ----------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS racer_profile (
    racer_id INTEGER PRIMARY KEY,
    racer_name TEXT, racer_name_kana TEXT,
    branch TEXT, birthplace TEXT,
    grade TEXT,
    birth_date DATE, gender TEXT,
    height REAL, weight REAL, blood_type TEXT,
    snapshot_date DATE
);

CREATE TABLE IF NOT EXISTS racer_history (
    racer_id INTEGER NOT NULL,
    race_date DATE NOT NULL,
    stadium INTEGER NOT NULL,
    race_no INTEGER NOT NULL,
    weekday TEXT, schedule_day TEXT, final_race TEXT,
    time_zone TEXT, series TEXT, race_grade TEXT,
    rank_count TEXT, rank_lineup TEXT, race_type TEXT,
    entry_fixed TEXT,
    weather TEXT, wind_direction TEXT, wind_speed REAL, wave REAL,
    lane INTEGER, course INTEGER,
    st REAL, st_rank INTEGER, st_diff_from_first REAL,
    display_time REAL, display_rank INTEGER,
    finish_pos INTEGER,
    result_order TEXT, kimarite TEXT,
    payout_trifecta INTEGER, payout_exacta INTEGER,
    PRIMARY KEY (racer_id, race_date, stadium, race_no)
);
CREATE INDEX IF NOT EXISTS idx_rh_drs ON racer_history (race_date, stadium, race_no);
CREATE INDEX IF NOT EXISTS idx_rh_racer ON racer_history (racer_id);
"""


def ensure_schema(logger: logging.Logger) -> None:
    conn = get_connection()
    try:
        native = conn.native
        native.executescript(SCHEMA_SQL)
        conn.commit()
        logger.info("schema ensured (racer_profile + racer_history)")
    finally:
        conn.close()


# ---------------------------- RateLimiter ----------------------------
@dataclass
class RateLimiter:
    min_sleep: float = 1.0
    max_sleep: float = 1.0
    _last: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def throttle(self, sleep_fn=time.sleep) -> float:
        with self._lock:
            now = time.time()
            elapsed = now - self._last if self._last else float("inf")
            target = random.uniform(self.min_sleep, self.max_sleep)
            slept = 0.0
            if elapsed < target:
                slept = target - elapsed; sleep_fn(slept)
            self._last = time.time()
            return slept


# ---------------------------- Progress ----------------------------
@dataclass
class Progress:
    completed: set = field(default_factory=set)
    failed: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def is_done(self, key) -> bool:
        with self._lock: return key in self.completed

    def mark_done(self, key) -> None:
        with self._lock:
            self.completed.add(key); self.failed.pop(key, None)

    def mark_failed(self, key, reason: str) -> None:
        with self._lock: self.failed[key] = reason

    def to_dict(self) -> dict:
        with self._lock:
            return {"completed": sorted(self.completed),
                    "failed": dict(self.failed),
                    "updated_at": dt.datetime.now().isoformat(timespec="seconds")}


_save_lock = threading.Lock()

def save_progress(p: Progress, path: Path = PROGRESS_PATH) -> None:
    with _save_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        for _ in range(5):
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(p.to_dict(), f, ensure_ascii=False, indent=2)
                tmp.replace(path)
                return
            except (PermissionError, OSError):
                time.sleep(0.5)


def load_progress(path: Path = PROGRESS_PATH) -> Progress:
    if path.exists():
        try:
            d = json.load(open(path, "r", encoding="utf-8"))
            p = Progress()
            p.completed = set(d.get("completed", []))
            p.failed = dict(d.get("failed", {}))
            return p
        except Exception: pass
    return Progress()


# ---------------------------- Phase 4a: Profile ----------------------------
def fetch_profiles(logger: logging.Logger) -> list[dict]:
    """GET /players 1 req で 1,956 人取得"""
    r = requests.get(f"{BASE_URL}/players",
                     params={"display_num": "FIVE_THOUSAND"},
                     headers=HEADERS, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    tables = [t for t in soup.find_all("table") if len(t.find_all("tr")) > 100]
    if not tables:
        raise RuntimeError("no player table found")
    t = tables[0]
    head = t.find("thead")
    cols = [th.get_text(strip=True) for th in head.find_all(["th","td"])] if head else []
    logger.info(f"/players thead: {cols}")
    body = t.find("tbody") or t
    out = []
    for tr in body.find_all("tr"):
        if tr.parent and tr.parent.name == "thead": continue
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td","th"])]
        if len(cells) < 11: continue
        def _clean(s): return s.replace("\u3000", "").replace(" ", "").strip() if s else None
        def _f(s):
            try: return float(s) if s and s not in ("","-") else None
            except ValueError: return None
        out.append({
            "racer_id": int(cells[0]),
            "racer_name": _clean(cells[1]),
            "racer_name_kana": _clean(cells[2]),
            "branch": _clean(cells[3]),
            "birthplace": _clean(cells[4]),
            "grade": _clean(cells[5]),
            "birth_date": cells[6] if cells[6] else None,
            "gender": _clean(cells[7]),
            "height": _f(cells[8]),
            "weight": _f(cells[9]),
            "blood_type": _clean(cells[10]),
        })
    return out


def insert_profiles(rows: list[dict], snapshot_date: dt.date, logger: logging.Logger) -> int:
    if not rows: return 0
    ph = placeholder()
    sql = (f"INSERT INTO racer_profile "
           f"(racer_id, racer_name, racer_name_kana, branch, birthplace, grade, "
           f" birth_date, gender, height, weight, blood_type, snapshot_date) "
           f"VALUES ({','.join([ph]*12)}) "
           f"ON CONFLICT(racer_id) DO UPDATE SET "
           f" racer_name=excluded.racer_name, grade=excluded.grade, "
           f" weight=excluded.weight, height=excluded.height, "
           f" snapshot_date=excluded.snapshot_date")
    conn = get_connection()
    try:
        cur = conn.cursor()
        data = [(r["racer_id"], r["racer_name"], r["racer_name_kana"],
                 r["branch"], r["birthplace"], r["grade"],
                 r["birth_date"], r["gender"], r["height"], r["weight"],
                 r["blood_type"], snapshot_date) for r in rows]
        cur.executemany(sql, data)
        conn.commit()
        return len(data)
    finally:
        conn.close()


# ---------------------------- Phase 4b: History ----------------------------
HISTORY_COLS_EXPECTED = [
    "ﾚｰｽ日付","曜日","開催場","日程","最終R","時間帯","ｼﾘｰｽﾞ","ｸﾞﾚｰﾄﾞ",
    "ﾗﾝｸ人数","ﾗﾝｸ並び","ﾚｰｽ種別","ﾚｰｽ","進入固定","天気","風向","風速","波",
    "枠","進入","ST","ST順位","ST1位との差","展示ﾀｲﾑ","展示T順位",
    "着順","結果","決まり手","3連単配当","2連単配当",
]


def parse_history(txt: str) -> list[dict]:
    soup = BeautifulSoup(txt, "lxml")
    target = None
    for t in soup.find_all("table"):
        head = t.find("thead")
        cols = [th.get_text(strip=True) for th in head.find_all(["th","td"])] if head else []
        if "ﾚｰｽ日付" in cols and "展示ﾀｲﾑ" in cols:
            target = (t, cols); break
    if not target: return []
    t, cols = target
    out = []
    for tr in t.find_all("tr"):
        if tr.parent and tr.parent.name == "thead": continue
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td","th"])]
        if len(cells) == len(cols):
            out.append(dict(zip(cols, cells)))
    return out


_STADIUM_RE = re.compile(r"^(\d{1,2})#")
_R_RE = re.compile(r"(\d{1,2})R")
_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _f(s):
    if not s or s in ("-","−","",None): return None
    try: return float(s)
    except ValueError: return None


def _i(s):
    if not s or s in ("-","−","",None): return None
    m = re.search(r"-?\d+", s)
    try: return int(m.group(0)) if m else None
    except ValueError: return None


def _payout_i(s):
    if not s: return None
    s = s.replace(",", "").strip()
    return _i(s)


def history_to_db_rows(racer_id: int, recs: list[dict]) -> list[tuple]:
    rows = []
    for rec in recs:
        try:
            md = _DATE_RE.match(rec.get("ﾚｰｽ日付",""))
            if not md: continue
            race_date = dt.date(int(md.group(1)), int(md.group(2)), int(md.group(3)))
            m_stad = _STADIUM_RE.match(rec.get("開催場","").strip())
            if not m_stad: continue
            stadium = int(m_stad.group(1))
            m_r = _R_RE.match(rec.get("ﾚｰｽ","").strip())
            if not m_r: continue
            race_no = int(m_r.group(1))
            rows.append((
                racer_id, race_date, stadium, race_no,
                rec.get("曜日") or None, rec.get("日程") or None,
                rec.get("最終R") or None, rec.get("時間帯") or None,
                rec.get("ｼﾘｰｽﾞ") or None, rec.get("ｸﾞﾚｰﾄﾞ") or None,
                rec.get("ﾗﾝｸ人数") or None, rec.get("ﾗﾝｸ並び") or None,
                rec.get("ﾚｰｽ種別") or None, rec.get("進入固定") or None,
                rec.get("天気") or None, rec.get("風向") or None,
                _f(rec.get("風速")), _f(rec.get("波")),
                _i(rec.get("枠")), _i(rec.get("進入")),
                _f(rec.get("ST")), _i(rec.get("ST順位")),
                _f(rec.get("ST1位との差")),
                _f(rec.get("展示ﾀｲﾑ")), _i(rec.get("展示T順位")),
                _i(rec.get("着順")),
                rec.get("結果") or None, rec.get("決まり手") or None,
                _payout_i(rec.get("3連単配当")),
                _payout_i(rec.get("2連単配当")),
            ))
        except Exception:
            continue
    return rows


def upsert_history(racer_id: int, rows: list[tuple], logger: logging.Logger) -> int:
    if not rows: return 0
    ph = placeholder()
    sql = (f"INSERT INTO racer_history "
           f"(racer_id, race_date, stadium, race_no, weekday, schedule_day, final_race, "
           f" time_zone, series, race_grade, rank_count, rank_lineup, race_type, entry_fixed, "
           f" weather, wind_direction, wind_speed, wave, lane, course, "
           f" st, st_rank, st_diff_from_first, display_time, display_rank, "
           f" finish_pos, result_order, kimarite, payout_trifecta, payout_exacta) "
           f"VALUES ({','.join([ph]*30)}) "
           f"ON CONFLICT(racer_id, race_date, stadium, race_no) DO NOTHING")
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.executemany(sql, rows)
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def fetch_racer_history(
    racer_id: int, from_date: dt.date, to_date: dt.date,
    rate: RateLimiter, timeout: int = 60,
) -> tuple[bool, int, str]:
    """1 選手分の履歴を 1 req で取得"""
    rate.throttle()
    try:
        r = requests.get(f"{BASE_URL}/player_records",
            params={"conditions[player_code]": str(racer_id),
                    "conditions[race_date][from]": from_date.isoformat(),
                    "conditions[race_date][to]": to_date.isoformat(),
                    "display_num": "FIVE_THOUSAND"},
            headers=HEADERS, timeout=timeout)
        if r.status_code != 200:
            return (False, 0, f"status={r.status_code}")
        recs = parse_history(r.text)
        rows = history_to_db_rows(racer_id, recs)
        n = upsert_history(racer_id, rows, None) if rows else 0
        return (True, n, "ok")
    except requests.Timeout:
        return (False, 0, "timeout")
    except Exception as e:
        return (False, 0, f"err: {e!r}"[:200])


# ---------------------------- runners ----------------------------
def run_profiles(logger: logging.Logger) -> int:
    ensure_schema(logger)
    logger.info("Phase 4a: fetching /players ...")
    profs = fetch_profiles(logger)
    logger.info(f"/players parsed {len(profs)} rows")
    n = insert_profiles(profs, dt.date.today(), logger)
    logger.info(f"Phase 4a done. inserted/updated {n} profiles")
    return n


def run_histories(from_date, to_date, parallel, sleep_val,
                  night_pause, dry_run, logger) -> None:
    ensure_schema(logger)
    # 対象: racer_profile の racer_id
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT racer_id FROM racer_profile ORDER BY racer_id")
        racers = [r[0] for r in cur.fetchall()]
    finally:
        conn.close()
    if not racers:
        logger.error("racer_profile empty; run --profiles-only first")
        return
    logger.info(f"Phase 4b: {len(racers)} racers, range {from_date}..{to_date}, parallel={parallel}")

    progress = load_progress()
    pending = [rid for rid in racers
               if not progress.is_done(rid) and str(rid) not in progress.completed]
    logger.info(f"pending after checkpoint: {len(pending)}")
    if dry_run:
        for rid in pending[:5]:
            logger.info(f"[dry] racer {rid}")
        return

    rate = RateLimiter(min_sleep=sleep_val, max_sleep=sleep_val)
    if not night_pause:
        pass  # no-op, we don't implement night pause here

    def _task(rid):
        ok, n, reason = fetch_racer_history(rid, from_date, to_date, rate)
        return (rid, ok, n, reason)

    done = 0; fail = 0; inserted = 0
    t0 = time.time()
    CHUNK = max(parallel * 8, 32)
    with ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="rb") as ex:
        idx = 0
        while idx < len(pending):
            batch = pending[idx:idx+CHUNK]; idx += len(batch)
            futs = [ex.submit(_task, rid) for rid in batch]
            for fut in as_completed(futs):
                try:
                    rid, ok, n, reason = fut.result()
                except Exception as e:
                    logger.exception("worker err: %s", e); fail += 1; continue
                if ok:
                    progress.mark_done(rid); done += 1; inserted += n
                else:
                    progress.mark_failed(rid, reason); fail += 1
                if (done + fail) % 50 == 0:
                    save_progress(progress)
                    logger.info(f"progress: done={done} fail={fail}/{len(pending)}  "
                                f"inserted={inserted}  elapsed={time.time()-t0:.1f}s  "
                                f"rate={(done+fail)/max(time.time()-t0,1)*60:.1f}/min")
    save_progress(progress)
    logger.info(f"Phase 4b done. done={done} fail={fail} inserted={inserted}")


def _parse_date(s): return dt.date.fromisoformat(s)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="racer_backfill")
    p.add_argument("--profiles-only", action="store_true", help="Phase 4a のみ")
    p.add_argument("--from-date", type=_parse_date, default=dt.date(2020,2,1))
    p.add_argument("--to-date", type=_parse_date, default=dt.date(2026,4,18))
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--sleep", type=float, default=1.0)
    p.add_argument("--no-night-pause", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.sleep < 0.5:
        raise SystemExit("ERROR: --sleep >= 0.5 (murao 配慮)")
    if not (1 <= args.parallel <= 10):
        raise SystemExit("ERROR: --parallel 1..10")
    logger = _build_logger()
    try:
        if args.profiles_only:
            run_profiles(logger)
        else:
            run_profiles(logger)   # プロフィールは必須なので常に先に
            run_histories(args.from_date, args.to_date, args.parallel, args.sleep,
                          not args.no_night_pause, args.dry_run, logger)
    except Exception:
        import traceback
        logger.error("fatal uncaught:\n%s", traceback.format_exc())
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
