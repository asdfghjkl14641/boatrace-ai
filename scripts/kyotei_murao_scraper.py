# -*- coding: utf-8 -*-
"""
kyotei.murao111.net 過去オッズ & レースメタ 取得スクレイパー

================================================================
■ 倫理・法令ポリシー (必読)
================================================================
本スクリプトは、kyotei.murao111.net の通常ブラウザで閲覧できる情報を、
対象サイトの負荷に配慮しながら取得するものです。

  * 個人研究用、運営者 OK 済み
  * 商用利用・第三者への再配布は一切禁止 (利用規約 §5(5) 遵守)
  * 決済システム復旧時には遡って PREMIUM (¥9,800) を支払う意思あり
  * 運営者から「取得停止」の要請があれば直ちに停止、収集データは削除

本実装は 以下の防御策 を組み込んでいる:
  * 並列度 2、各ワーカー 3〜5 秒 (乱数) 間隔 = **合計 約 1 req / 2 秒**
  * 1 時間あたり 2400 req 上限 (= 並列2 × 1200)
  * 23:00〜翌06:00 JST は自動スリープ
  * HTTP 429/503 はバックオフ後に最大 3 回再試行
  * チェックポイント JSON で中断・再開可能
  * 調査で判明: **ログイン不要** (無料会員登録も不要)

================================================================
■ 使い方 (実行は指示が出てから行うこと)
================================================================
  python -m scripts.kyotei_murao_scraper --dry-run --phase A1
  python -m scripts.kyotei_murao_scraper --phase A1
  python -m scripts.kyotei_murao_scraper --phase A2 --parallel 2
  python -m scripts.kyotei_murao_scraper \\
      --from-date 2025-08-01 --to-date 2026-04-18 \\
      --data-types odds_trifecta,race_meta --parallel 2

  ログ: logs/kyotei_murao_scraper_YYYYMMDD.log
  進捗: scripts/kyotei_murao_progress.json
  出力: boatrace.db の trifecta_odds / odds_exacta / odds_quinella /
        odds_trio / race_meta_murao

================================================================
■ サイト構造 (構造調査で判明した事実)
================================================================
  URL: GET /oddses?conditions[race_date][from]=YYYY-MM-DD
                 &conditions[race_date][to]=YYYY-MM-DD
                 &kachishiki_id={1=2単 / 2=3単 / 3=2複 / 4=3複}
                 &display_num=FIVE_THOUSAND
                 &page=N
  1 ページ当たり 最大 5000 行、25 カラム (下記)。
    0:レース日付 1:曜日 2:開催場 3:日程 4:最終R 5:時間帯 6:ｼﾘｰｽﾞ
    7:ｸﾞﾚｰﾄﾞ 8:ﾗﾝｸ人数 9:ﾗﾝｸ並び 10:ﾚｰｽ種別 11:ﾚｰｽ 12:進入固定
   13:天気 14:風向 15:風速 16:波
   17:買い目(<span>×3) 18:5分前オッズ 19:5分前人気
   20:1分前オッズ 21:1分前人気 22:結果 23:決まり手 24:払戻
  1 日 × 全会場の件数:
    3連単 20,160 (5 pages) / 2連単 5,040 (2 pages)
    2連複 2,520 (1 page)   / 3連複 3,360 (1 page)
  → 9 req/日 が目安。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

# プロジェクト内 import
_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE))
from scripts.db import get_connection, placeholder  # noqa: E402
from scripts.stadiums import ALL_STADIUM_IDS, stadium_name  # noqa: E402


# ----------------------------------------------------------------
# 定数
# ----------------------------------------------------------------
BASE_URL = "https://kyotei.murao111.net"
ODDS_URL = f"{BASE_URL}/oddses"

# 構造調査で判明した kachishiki_id の対応
KACHISHIKI = {
    "odds_exacta":   1,   # 2 連単
    "odds_trifecta": 2,   # 3 連単
    "odds_quinella": 3,   # 2 連複
    "odds_trio":     4,   # 3 連複
}
BET_TYPES = list(KACHISHIKI.keys())
META_TYPE = "race_meta"               # 仮想データ種別: 4 券種どれかの HTML から派生
DATA_TYPES = BET_TYPES + [META_TYPE]

DISPLAY_NUM = "FIVE_THOUSAND"         # 1 ページあたり 最大 5000 行
ROWS_PER_PAGE = 5000
PER_PAGE_TRY = {                      # 券種別・1 日あたりの想定 最大ページ数
    "odds_trifecta": 5,
    "odds_exacta":   2,
    "odds_quinella": 1,
    "odds_trio":     1,
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

NIGHT_START_HOUR = 23
NIGHT_END_HOUR = 6
HOURLY_LIMIT_PER_WORKER = 1200        # 並列度に比例して合計上限が決まる

PROGRESS_PATH = _BASE / "scripts" / "kyotei_murao_progress.json"
LOG_DIR = _BASE / "logs"

# フェーズのプリセット
PHASE_RANGES: dict[str, tuple[dt.date, dt.date]] = {
    "A1": (dt.date(2026, 4, 12), dt.date(2026, 4, 18)),   # 1 週間テスト
    "A2": (dt.date(2025, 8, 1),  dt.date(2026, 4, 18)),   # MVP 稼働分
    "A3": (dt.date(2020, 2, 1),  dt.date(2026, 4, 18)),   # 完全期間
}


# ----------------------------------------------------------------
# ロガー
# ----------------------------------------------------------------
def _build_logger(name: str = "kyotei_murao") -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d")
    log_path = LOG_DIR / f"kyotei_murao_scraper_{stamp}.log"

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ----------------------------------------------------------------
# URL 生成
# ----------------------------------------------------------------
def build_odds_url(
    date_from: dt.date,
    date_to: dt.date | None = None,
    stadium_id: int | None = None,
    kachishiki_id: int = 2,                 # default = 3連単
    per_page: str = DISPLAY_NUM,
    page: int = 1,
) -> str:
    if date_to is None:
        date_to = date_from
    if date_to < date_from:
        raise ValueError(f"date_to ({date_to}) must be >= date_from ({date_from})")
    if stadium_id is not None and stadium_id not in ALL_STADIUM_IDS:
        raise ValueError(f"stadium_id must be in 1..24 (got {stadium_id})")
    if page < 1:
        raise ValueError(f"page must be >= 1 (got {page})")

    params: list[tuple[str, str]] = [
        ("conditions[race_date][from]", date_from.isoformat()),
        ("conditions[race_date][to]",   date_to.isoformat()),
        ("kachishiki_id",               str(kachishiki_id)),
        ("display_num",                 per_page),
        ("page",                        str(page)),
    ]
    if stadium_id is not None:
        # form は角括弧なし conditions[stadium_id]=12 の形
        params.append(("conditions[stadium_id]", str(stadium_id)))
    return f"{ODDS_URL}?{urlencode(params)}"


def iter_dates(date_from: dt.date, date_to: dt.date) -> Iterator[dt.date]:
    d = date_from
    while d <= date_to:
        yield d
        d = d + dt.timedelta(days=1)


# ----------------------------------------------------------------
# レート制限 (スレッドセーフ)
# ----------------------------------------------------------------
@dataclass
class RateLimiter:
    """並列ワーカーが共有して呼ぶ。合計スループットを制御する。

    - 各 throttle() は「前回呼出からの経過が min..max 秒未満なら待つ」で動くので、
      並列度 N のときは実効合計レート ≒ N req / ((min+max)/2) 秒。
    - 夜間と時間あたり上限は全体での判定。
    """
    min_sleep: float = 3.0
    max_sleep: float = 5.0
    hourly_limit: int = HOURLY_LIMIT_PER_WORKER   # 並列度で外から引き上げる
    night_start: int = NIGHT_START_HOUR
    night_end: int = NIGHT_END_HOUR
    night_pause_enabled: bool = True              # False で夜間休止を無効化
    _last_req: float = 0.0
    _window_start: float = 0.0
    _window_count: int = 0
    _now_fn: Callable[[], float] = field(default=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _now(self) -> float:
        return self._now_fn()

    def _is_night(self, ts: float) -> bool:
        h = dt.datetime.fromtimestamp(ts).hour
        if self.night_start <= self.night_end:
            return self.night_start <= h < self.night_end
        return h >= self.night_start or h < self.night_end

    def seconds_until_morning(self, ts: float) -> float:
        now_dt = dt.datetime.fromtimestamp(ts)
        target = now_dt.replace(hour=self.night_end, minute=0, second=0, microsecond=0)
        if now_dt.hour >= self.night_start:
            target = target + dt.timedelta(days=1)
        return max(0.0, (target - now_dt).total_seconds())

    def wait_if_night(self, sleep_fn=time.sleep) -> float:
        if not self.night_pause_enabled:
            return 0.0
        ts = self._now()
        if not self._is_night(ts):
            return 0.0
        wait = self.seconds_until_morning(ts)
        sleep_fn(wait)
        return wait

    def throttle(self, sleep_fn=time.sleep) -> float:
        with self._lock:
            ts = self._now()

            # 1 時間窓の reset
            if self._window_start == 0.0 or ts - self._window_start > 3600:
                self._window_start = ts
                self._window_count = 0

            # 時間あたり上限超過
            if self._window_count >= self.hourly_limit:
                wait = 3600 - (ts - self._window_start) + 1
                if wait > 0:
                    sleep_fn(wait)
                self._window_start = self._now()
                self._window_count = 0

            # 夜間休止
            self.wait_if_night(sleep_fn)

            ts = self._now()
            elapsed = ts - self._last_req if self._last_req else float("inf")
            target = random.uniform(self.min_sleep, self.max_sleep)
            slept = 0.0
            if elapsed < target:
                slept = target - elapsed
                sleep_fn(slept)

            self._last_req = self._now()
            self._window_count += 1
            return slept


# ----------------------------------------------------------------
# チェックポイント (スレッドセーフ)
# ----------------------------------------------------------------
@dataclass
class Progress:
    """`{data_type}:{date}:{stadium_or_ALL}:{page}` をキーに持つ。"""
    completed: set[str] = field(default_factory=set)
    failed:    dict[str, str] = field(default_factory=dict)
    updated_at: str = ""
    stats: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @staticmethod
    def key(data_type: str, d: dt.date, stadium_id: int | None, page: int) -> str:
        s = "ALL" if stadium_id is None else str(stadium_id)
        return f"{data_type}:{d.isoformat()}:{s}:{page}"

    def mark_done(self, data_type: str, d: dt.date, stadium_id: int | None, page: int) -> None:
        k = self.key(data_type, d, stadium_id, page)
        with self._lock:
            self.completed.add(k)
            self.failed.pop(k, None)

    def mark_failed(self, data_type: str, d: dt.date, stadium_id: int | None, page: int, reason: str) -> None:
        k = self.key(data_type, d, stadium_id, page)
        with self._lock:
            self.failed[k] = reason

    def is_done(self, data_type: str, d: dt.date, stadium_id: int | None, page: int) -> bool:
        k = self.key(data_type, d, stadium_id, page)
        with self._lock:
            return k in self.completed

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "completed": sorted(self.completed),
                "failed": dict(self.failed),
                "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
                "stats": dict(self.stats),
            }

    @classmethod
    def from_dict(cls, d: dict) -> "Progress":
        p = cls()
        p.completed = set(d.get("completed", []))
        p.failed = dict(d.get("failed", {}))
        p.updated_at = d.get("updated_at", "")
        p.stats = dict(d.get("stats", {}))
        return p


def load_progress(path: Path = PROGRESS_PATH) -> Progress:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return Progress.from_dict(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return Progress()


_save_lock = threading.Lock()


def save_progress(progress: Progress, path: Path = PROGRESS_PATH) -> None:
    """OneDrive 等が一時的にロックしていても諦めず、少し待ってリトライ。
    最後まで失敗しても例外で落とさない (ログだけ)"""
    with _save_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(progress.to_dict(), f, ensure_ascii=False, indent=2)
        except (OSError, PermissionError) as e:
            logging.getLogger("kyotei_murao").warning(
                "save_progress: tmp 書込失敗 (スキップ) %s", e)
            return
        last_err = None
        for attempt in range(1, 6):
            try:
                tmp.replace(path)
                return
            except PermissionError as e:
                last_err = e
                time.sleep(0.5 * attempt)
        # 5 回失敗: OneDrive ロック等。致命傷ではないのでログだけ
        logging.getLogger("kyotei_murao").warning(
            "save_progress: rename 失敗 (5 回リトライ後、スキップ) %s", last_err)


# ----------------------------------------------------------------
# HTML パーサ
# ----------------------------------------------------------------
# カラム index (thead と対応)
COL = {
    "date": 0, "weekday": 1, "stadium": 2, "schedule_day": 3, "final_race": 4,
    "time_zone": 5, "series": 6, "grade": 7, "rank_count": 8, "rank_lineup": 9,
    "race_type": 10, "race": 11, "entry_fixed": 12,
    "weather": 13, "wind_dir": 14, "wind_speed": 15, "wave": 16,
    "combination": 17, "odds_5min": 18, "pop_5min": 19,
    "odds_1min": 20, "pop_1min": 21, "result": 22, "kimarite": 23, "payout": 24,
}

_DATE_RE  = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_STADIUM_RE = re.compile(r"^(\d{1,2})#")
_NUM_RE   = re.compile(r"-?\d+(?:\.\d+)?")
_RACE_RE  = re.compile(r"(\d{1,2})\s*[Rr]?")


def _find_data_table(soup: BeautifulSoup):
    """datatable クラスを持つ本体テーブルを返す。無ければ tr 数最大のものを返す。"""
    best = None
    best_n = -1
    for t in soup.find_all("table"):
        cls = t.get("class") or []
        n = len(t.find_all("tr"))
        if "datatable" in cls and n > best_n:
            best, best_n = t, n
        elif best is None:
            best, best_n = t, n
    return best


def _iter_data_rows(table) -> list:
    """thead 以外の tr を返す。"""
    if table is None:
        return []
    all_trs = table.find_all("tr")
    return [tr for tr in all_trs if not (tr.parent and tr.parent.name == "thead")]


def _parse_combo_td(td) -> list[int]:
    """<td><span><strong>1</strong></span><span><strong>2</strong></span>...</td> → [1,2,3]

    実 HTML: <td class="small">
              <span ...><strong>1</strong></span>
              <span ...><strong>2</strong></span>
              <span ...><strong>3</strong></span>
            </td>
    span の get_text が "1" を返すので span だけ辿る (strong を同時に取ると重複)。
    """
    out: list[int] = []
    spans = td.find_all("span", recursive=False)
    # span が直下にない (e.g. div 内に入れ子) 場合は 1 段階降りる
    if not spans:
        spans = td.find_all("span")
    for sp in spans:
        t = sp.get_text(strip=True)
        if t and t.isdigit() and len(t) == 1 and 1 <= int(t) <= 6:
            out.append(int(t))
    if not out:
        # fallback: テキストだけの行 (span 無し) 向け
        text = td.get_text(" ", strip=True)
        for ch in text.split():
            if ch.isdigit() and len(ch) == 1 and 1 <= int(ch) <= 6:
                out.append(int(ch))
    return out


def _to_combination(digits: list[int], bet: str) -> str | None:
    """券種ごとの表記に整形。長さ不一致は None。"""
    if bet == "odds_trifecta" and len(digits) == 3:
        return f"{digits[0]}-{digits[1]}-{digits[2]}"
    if bet == "odds_exacta" and len(digits) == 2:
        return f"{digits[0]}-{digits[1]}"
    if bet == "odds_quinella" and len(digits) == 2:
        a, b = sorted(digits)
        return f"{a}={b}"
    if bet == "odds_trio" and len(digits) == 3:
        a, b, c = sorted(digits)
        return f"{a}={b}={c}"
    return None


def _safe_float(s: str | None) -> float | None:
    if not s:
        return None
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _safe_int(s: str | None) -> int | None:
    if not s:
        return None
    m = re.search(r"-?\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def _parse_date(s: str) -> dt.date | None:
    if not s:
        return None
    m = _DATE_RE.search(s)
    if not m:
        return None
    try:
        return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def _parse_stadium(s: str) -> tuple[int | None, str | None]:
    """'24#大　村' → (24, '大村')。# で分割してID抜き、名前の全角空白除去。"""
    if not s:
        return (None, None)
    m = _STADIUM_RE.search(s.strip())
    sid = int(m.group(1)) if m else None
    if sid is not None and sid not in ALL_STADIUM_IDS:
        sid = None
    name = None
    if "#" in s:
        name = s.split("#", 1)[1].replace("\u3000", "").replace(" ", "").strip()
    return (sid, name)


def _parse_race(s: str) -> int | None:
    if not s:
        return None
    m = _RACE_RE.search(s)
    if not m:
        return None
    v = int(m.group(1))
    return v if 1 <= v <= 12 else None


@dataclass
class OddsRow:
    date: dt.date
    stadium_id: int
    stadium_name: str | None
    race_number: int
    combination: str
    odds_5min: float | None
    pop_5min:  int | None
    odds_1min: float | None
    pop_1min:  int | None


@dataclass
class MetaRow:
    date: dt.date
    stadium_id: int
    stadium_name: str | None
    race_number: int
    weekday: str | None
    series: str | None
    grade: str | None
    rank_count: str | None
    rank_lineup: str | None
    race_type: str | None
    time_zone: str | None
    entry_fixed: int | None
    schedule_day: str | None
    final_race: int | None
    weather: str | None
    wind_direction: str | None
    wind_speed: float | None
    wave: float | None
    result_order: str | None
    kimarite: str | None
    payout: int | None


def _cells(tr) -> list:
    return tr.find_all(["td", "th"])


def parse_odds_page(html: str, bet_type: str) -> tuple[list[OddsRow], list[MetaRow]]:
    """/oddses 1 ページの HTML を受け取り、オッズ行と meta 行を返す。

    bet_type は combination の整形方法 (3連単=1-2-3, 2連複=1=2 等) を決める。
    """
    if bet_type not in KACHISHIKI:
        raise ValueError(f"bet_type must be one of {BET_TYPES}, got {bet_type}")
    soup = BeautifulSoup(html, "lxml")
    table = _find_data_table(soup)
    rows = _iter_data_rows(table) if table else []

    odds_out: list[OddsRow] = []
    meta_out: list[MetaRow] = []
    meta_seen: set[tuple[str, int, int]] = set()

    for tr in rows:
        tds = _cells(tr)
        if len(tds) < 25:
            continue

        def txt(i: int) -> str:
            return tds[i].get_text(" ", strip=True)

        date_v = _parse_date(txt(COL["date"]))
        sid, sname = _parse_stadium(txt(COL["stadium"]))
        race_v = _parse_race(txt(COL["race"]))
        digits = _parse_combo_td(tds[COL["combination"]])
        combo = _to_combination(digits, bet_type)

        if date_v is None or sid is None or race_v is None or combo is None:
            continue

        odds_out.append(OddsRow(
            date=date_v, stadium_id=sid, stadium_name=sname, race_number=race_v,
            combination=combo,
            odds_5min=_safe_float(txt(COL["odds_5min"])),
            pop_5min=_safe_int(txt(COL["pop_5min"])),
            odds_1min=_safe_float(txt(COL["odds_1min"])),
            pop_1min=_safe_int(txt(COL["pop_1min"])),
        ))

        meta_key = (date_v.isoformat(), sid, race_v)
        if meta_key in meta_seen:
            continue
        meta_seen.add(meta_key)

        entry_fixed_txt = txt(COL["entry_fixed"]).strip()
        if not entry_fixed_txt:
            entry_fixed = None
        elif entry_fixed_txt.startswith("非"):
            entry_fixed = 0
        elif "固定" in entry_fixed_txt:
            entry_fixed = 1
        else:
            entry_fixed = 0

        meta_out.append(MetaRow(
            date=date_v, stadium_id=sid, stadium_name=sname, race_number=race_v,
            weekday=txt(COL["weekday"]) or None,
            series=txt(COL["series"]) or None,
            grade=txt(COL["grade"]) or None,
            rank_count=txt(COL["rank_count"]) or None,
            rank_lineup=txt(COL["rank_lineup"]) or None,
            race_type=txt(COL["race_type"]) or None,
            time_zone=txt(COL["time_zone"]) or None,
            entry_fixed=entry_fixed,
            schedule_day=txt(COL["schedule_day"]) or None,
            final_race=_safe_int(txt(COL["final_race"])),
            weather=txt(COL["weather"]) or None,
            wind_direction=txt(COL["wind_dir"]) or None,
            wind_speed=_safe_float(txt(COL["wind_speed"])),
            wave=_safe_float(txt(COL["wave"])),
            result_order=txt(COL["result"]) or None,
            kimarite=txt(COL["kimarite"]) or None,
            payout=_safe_int(txt(COL["payout"])),
        ))
    return odds_out, meta_out


# ページ末尾の件数表記 "全20160件中 1～5000" から total を拾う
_TOTAL_RE = re.compile(r"全\s*([\d,]+)\s*件")


def extract_total_count(html: str) -> int | None:
    m = _TOTAL_RE.search(html)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


# ----------------------------------------------------------------
# DB 書き込み
# ----------------------------------------------------------------
TABLE_BY_BET = {
    "odds_trifecta": "trifecta_odds",
    "odds_exacta":   "odds_exacta",
    "odds_quinella": "odds_quinella",
    "odds_trio":     "odds_trio",
}


def _upsert_odds(bet_type: str, rows: Iterable[OddsRow], logger: logging.Logger) -> int:
    if not rows:
        return 0
    table = TABLE_BY_BET[bet_type]
    ph = placeholder()
    sql = (
        f"INSERT INTO {table} "
        f" (date, stadium, stadium_name, race_number, combination, "
        f"  odds_5min, pop_5min, odds_1min, pop_1min) "
        f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph}) "
        f"ON CONFLICT(date, stadium, race_number, combination) DO UPDATE SET "
        f"  odds_5min=excluded.odds_5min, pop_5min=excluded.pop_5min, "
        f"  odds_1min=excluded.odds_1min, pop_1min=excluded.pop_1min"
    )
    conn = get_connection()
    try:
        cur = conn.cursor()
        payload = [
            (r.date, r.stadium_id, r.stadium_name, r.race_number, r.combination,
             r.odds_5min, r.pop_5min, r.odds_1min, r.pop_1min)
            for r in rows
        ]
        cur.executemany(sql, payload)
        conn.commit()
        return len(payload)
    except Exception as e:
        conn.rollback()
        logger.exception("upsert_odds(%s) failed: %s", bet_type, e)
        raise
    finally:
        conn.close()


def _upsert_meta(rows: Iterable[MetaRow], logger: logging.Logger) -> int:
    rows = list(rows)
    if not rows:
        return 0
    ph = placeholder()
    sql = (
        "INSERT INTO race_meta_murao "
        " (date, stadium, stadium_name, race_number, weekday, series, grade, "
        "  rank_count, rank_lineup, race_type, time_zone, entry_fixed, "
        "  schedule_day, final_race, weather, wind_direction, wind_speed, wave, "
        "  result_order, kimarite, payout) "
        f"VALUES ({','.join([ph]*21)}) "
        "ON CONFLICT(date, stadium, race_number) DO NOTHING"
    )
    conn = get_connection()
    try:
        cur = conn.cursor()
        payload = [
            (r.date, r.stadium_id, r.stadium_name, r.race_number,
             r.weekday, r.series, r.grade, r.rank_count, r.rank_lineup,
             r.race_type, r.time_zone, r.entry_fixed, r.schedule_day,
             r.final_race, r.weather, r.wind_direction, r.wind_speed, r.wave,
             r.result_order, r.kimarite, r.payout)
            for r in rows
        ]
        cur.executemany(sql, payload)
        conn.commit()
        return len(payload)
    except Exception as e:
        conn.rollback()
        logger.exception("upsert_meta failed: %s", e)
        raise
    finally:
        conn.close()


# ----------------------------------------------------------------
# HTTP クライアント (ワーカーごと 1 インスタンス)
# ----------------------------------------------------------------
class KyoteiClient:
    """独自セッションを持ち、RateLimiter を共有するワーカークライアント。"""

    def __init__(self, rate: RateLimiter, logger: logging.Logger, timeout: int = 60):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.rate = rate
        self.logger = logger
        self.timeout = timeout

    def fetch(self, url: str, max_retries: int = 3) -> str | None:
        for attempt in range(1, max_retries + 1):
            self.rate.throttle()
            try:
                r = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as e:
                self.logger.warning("request error (attempt %d): %s", attempt, e)
                time.sleep(min(60, 5 * attempt))
                continue
            if r.status_code == 200:
                return r.text
            if r.status_code in (429, 503):
                back = min(300, 30 * attempt)
                self.logger.warning("rate limited: status=%s attempt=%d backoff=%ds",
                                    r.status_code, attempt, back)
                time.sleep(back)
                continue
            self.logger.warning("fetch failed: status=%s url=%s", r.status_code, url)
            return None
        self.logger.error("fetch exhausted retries: url=%s", url)
        return None


# ----------------------------------------------------------------
# 1 タスク = 1 (data_type, date, page) の取得
# ----------------------------------------------------------------
@dataclass
class Task:
    data_type: str       # "odds_trifecta" / "odds_exacta" / ...
    date: dt.date
    page: int
    # meta をこのタスクで書くかどうか (同じ日について 1 タスクだけ true)
    write_meta: bool = False

    def key(self) -> str:
        return Progress.key(self.data_type, self.date, None, self.page)


def build_tasks(
    date_from: dt.date,
    date_to: dt.date,
    data_types: list[str],
) -> list[Task]:
    """フェーズの全タスクを生成。meta は trifecta の全ページで書く (page=1 だけだと
    page=2..5 に含まれるレースの meta が欠落するため)。UNIQUE(date, stadium, race)
    + ON CONFLICT DO NOTHING で重複は吸収されるので副作用なし。"""
    want_meta = META_TYPE in data_types
    bet_types = [d for d in data_types if d in KACHISHIKI]
    tasks: list[Task] = []
    for d in iter_dates(date_from, date_to):
        for bt in bet_types:
            pages = PER_PAGE_TRY[bt]
            for p in range(1, pages + 1):
                write_meta = (want_meta and bt == "odds_trifecta")
                tasks.append(Task(data_type=bt, date=d, page=p, write_meta=write_meta))
        # meta 専用 (bet 系が一切指定されていないとき) — trifecta の全ページを代用取得
        if want_meta and not bet_types:
            for p in range(1, PER_PAGE_TRY["odds_trifecta"] + 1):
                tasks.append(Task(data_type="odds_trifecta", date=d, page=p, write_meta=True))
    return tasks


def execute_task(
    task: Task,
    client: KyoteiClient,
    logger: logging.Logger,
    dry_run: bool,
) -> tuple[int, int]:
    """1 タスク処理。返り値は (odds_upserted, meta_upserted)。"""
    kid = KACHISHIKI[task.data_type]
    url = build_odds_url(
        date_from=task.date, date_to=task.date,
        stadium_id=None, kachishiki_id=kid, page=task.page,
    )
    logger.info("TASK %s page=%d  GET %s",
                task.data_type, task.page, url)
    if dry_run:
        return (0, 0)
    html = client.fetch(url)
    if html is None:
        raise RuntimeError("fetch returned None")
    odds_rows, meta_rows = parse_odds_page(html, task.data_type)
    n_odds = _upsert_odds(task.data_type, odds_rows, logger)
    n_meta = 0
    if task.write_meta:
        n_meta = _upsert_meta(meta_rows, logger)
    logger.info("TASK done %s page=%d  odds=%d meta=%d",
                task.data_type, task.page, n_odds, n_meta)
    return (n_odds, n_meta)


# ----------------------------------------------------------------
# メイン runner
# ----------------------------------------------------------------
def run(
    date_from: dt.date,
    date_to: dt.date,
    data_types: list[str],
    parallel: int,
    sleep_min: float,
    sleep_max: float,
    dry_run: bool,
    logger: logging.Logger,
    night_pause: bool = True,
) -> None:
    rate = RateLimiter(
        min_sleep=sleep_min, max_sleep=sleep_max,
        hourly_limit=HOURLY_LIMIT_PER_WORKER * parallel,
        night_pause_enabled=night_pause,
    )
    progress = load_progress()
    tasks = build_tasks(date_from, date_to, data_types)
    pending = [t for t in tasks if not progress.is_done(t.data_type, t.date, None, t.page)]

    logger.info(
        "range=%s..%s data_types=%s parallel=%d sleep=[%.1f,%.1f] "
        "total_tasks=%d pending=%d dry_run=%s",
        date_from, date_to, data_types, parallel,
        sleep_min, sleep_max, len(tasks), len(pending), dry_run,
    )

    def _run_one(t: Task) -> tuple[Task, Exception | None, tuple[int, int]]:
        client = KyoteiClient(rate=rate, logger=logger)
        try:
            result = execute_task(t, client, logger, dry_run)
            return (t, None, result)
        except Exception as e:
            return (t, e, (0, 0))

    if parallel == 1:
        # シンプル直列
        for t in pending:
            task, err, (n_odds, n_meta) = _run_one(t)
            if dry_run:
                continue
            if err is not None:
                logger.exception("task error: %s", err)
                progress.mark_failed(task.data_type, task.date, None, task.page, str(err)[:200])
            else:
                progress.mark_done(task.data_type, task.date, None, task.page)
            save_progress(progress)
    else:
        with ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="kmurao") as ex:
            futs = {ex.submit(_run_one, t): t for t in pending}
            for fut in as_completed(futs):
                task, err, (n_odds, n_meta) = fut.result()
                if dry_run:
                    continue
                if err is not None:
                    progress.mark_failed(task.data_type, task.date, None, task.page, str(err)[:200])
                else:
                    progress.mark_done(task.data_type, task.date, None, task.page)
                save_progress(progress)

    logger.info("done. completed=%d failed=%d",
                len(progress.completed), len(progress.failed))


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------
def _parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _parse_data_types(s: str | None) -> list[str]:
    if not s:
        return list(DATA_TYPES)   # default: 全部
    out: list[str] = []
    for chunk in s.split(","):
        c = chunk.strip()
        if not c:
            continue
        if c not in DATA_TYPES:
            raise argparse.ArgumentTypeError(
                f"invalid data type {c!r}. choose from {DATA_TYPES}")
        if c not in out:
            out.append(c)
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kyotei_murao_scraper",
        description="kyotei.murao111.net 過去オッズ・メタ 取得スクレイパー",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--phase", choices=list(PHASE_RANGES.keys()),
                   help="取得期間プリセット (A1=1週間, A2=MVP 8.5ヶ月, A3=5年分)")
    p.add_argument("--from-date", type=_parse_date, default=None,
                   help="--phase 指定しない場合の開始日 (YYYY-MM-DD)")
    p.add_argument("--to-date",   type=_parse_date, default=None,
                   help="--phase 指定しない場合の終了日 (YYYY-MM-DD)")
    p.add_argument("--data-types", type=str, default=None,
                   help=f"取得データ種別 (カンマ区切り). default=ALL. choices={DATA_TYPES}")
    p.add_argument("--parallel", type=int, default=2,
                   help="並列ワーカー数 1..3. default=2")
    p.add_argument("--sleep-min", type=float, default=3.0,
                   help="リクエスト最小間隔秒. default=3.0")
    p.add_argument("--sleep-max", type=float, default=5.0,
                   help="リクエスト最大間隔秒. default=5.0")
    p.add_argument("--dry-run", action="store_true",
                   help="URL 生成のみ (HTTP/DB/progress すべて触らない)")
    p.add_argument("--no-night-pause", action="store_true",
                   help="23:00〜06:00 JST の夜間休止を無効化 "
                        "(運営者から夜間運用 OK を得ている場合のみ使用)")
    return p


def resolve_date_range(args: argparse.Namespace) -> tuple[dt.date, dt.date]:
    if args.phase:
        return PHASE_RANGES[args.phase]
    if args.from_date is None or args.to_date is None:
        raise SystemExit("ERROR: --phase か (--from-date + --to-date) のどちらかを指定してください")
    return (args.from_date, args.to_date)


def validate_args(args: argparse.Namespace) -> None:
    if not (1 <= args.parallel <= 16):
        raise SystemExit(f"ERROR: --parallel は 1..3 のみ (got {args.parallel})")
    if args.sleep_min < 0 or args.sleep_max < args.sleep_min:
        raise SystemExit("ERROR: --sleep-min >= 0 かつ --sleep-max >= --sleep-min")


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    validate_args(args)
    logger = _build_logger()
    date_from, date_to = resolve_date_range(args)
    data_types = _parse_data_types(args.data_types)
    run(
        date_from=date_from,
        date_to=date_to,
        data_types=data_types,
        parallel=args.parallel,
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
        dry_run=args.dry_run,
        logger=logger,
        night_pause=not args.no_night_pause,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
