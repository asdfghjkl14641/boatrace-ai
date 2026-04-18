# -*- coding: utf-8 -*-
"""
BoatraceOpenAPI (https://boatraceopenapi.github.io) から
過去データを一括取得して Supabase にインポートする。

■ データ元 (MITライセンスの非公式API、GitHub Pages上の静的JSON)
    programs/v3   出走表       カバレッジ: 2023-05-01 〜 現在
    results/v2    レース結果   カバレッジ: 2025-07-15 〜 現在
    previews/v2   直前情報     カバレッジ: 2025-08-01 〜 現在

■ 本スクリプトの役割
    指定した日付範囲について:
      - programs JSON → race_cards テーブル
      - results JSON  → race_results テーブル (タイム情報は API にないので NULL)
      - previews JSON → race_conditions テーブル
    既存データは ON CONFLICT DO NOTHING でスキップする。
    ダウンロード元にその日のデータが無ければ 404 で自動スキップ。

■ 使い方
    # 3年分を一括インポート (programs API の最古日から今日まで)
    python -m scripts.import_openapi --start 2023-05-01 --end 2026-04-17

    # programs だけ取り込む
    python -m scripts.import_openapi --start 2023-05-01 --only programs
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Any, Iterable

import requests
from tqdm import tqdm

from scripts.db import get_connection
from scripts.stadiums import stadium_name

# ------------------------------------------------------------
# 設定
# ------------------------------------------------------------
BASE_URL = "https://boatraceopenapi.github.io"

# リクエスト間隔 (秒)。GitHub Pages の静的ファイルなので控えめに。
REQUEST_INTERVAL = float(os.getenv("OPENAPI_INTERVAL_SEC", "0.3"))

# API カバレッジ (これより前の日付は 404 が確定なのでデフォルト起点に使う)
PROGRAMS_START = date(2023, 5, 1)
RESULTS_START  = date(2025, 7, 15)
PREVIEWS_START = date(2025, 8, 1)

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

# ------------------------------------------------------------
# マッピング: API の数字コード → 既存DBで使っている文字列 (スクレイパーと統一)
# ------------------------------------------------------------
CLASS_MAP = {1: "A1", 2: "A2", 3: "B1", 4: "B2"}

# 天候: スクレイパーの _WEATHER_MAP と同じ文字列を使う
WEATHER_MAP = {1: "晴", 2: "曇り", 3: "雨", 4: "雪", 5: "霧"}

# 都道府県コード (支部・出身地)。boatrace APIは全国地方公共団体コード 1-47 を採用。
PREFECTURES = {
    1: "北海道", 2: "青森", 3: "岩手", 4: "宮城", 5: "秋田",
    6: "山形", 7: "福島", 8: "茨城", 9: "栃木", 10: "群馬",
    11: "埼玉", 12: "千葉", 13: "東京", 14: "神奈川", 15: "新潟",
    16: "富山", 17: "石川", 18: "福井", 19: "山梨", 20: "長野",
    21: "岐阜", 22: "静岡", 23: "愛知", 24: "三重", 25: "滋賀",
    26: "京都", 27: "大阪", 28: "兵庫", 29: "奈良", 30: "和歌山",
    31: "鳥取", 32: "島根", 33: "岡山", 34: "広島", 35: "山口",
    36: "徳島", 37: "香川", 38: "愛媛", 39: "高知", 40: "福岡",
    41: "佐賀", 42: "長崎", 43: "熊本", 44: "大分", 45: "宮崎",
    46: "鹿児島", 47: "沖縄",
}

# ------------------------------------------------------------
# ロギング
# ------------------------------------------------------------
def setup_logging() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(LOG_DIR, f"import_openapi_{ts}.log")
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
    logging.info(f"ログファイル: {logfile}")
    return logfile


# ------------------------------------------------------------
# API fetcher
# ------------------------------------------------------------
class OpenAPIFetcher:
    def __init__(self, interval: float = REQUEST_INTERVAL, timeout: int = 30):
        self.interval = interval
        self.timeout = timeout
        self._last = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "boatrace-ai-importer/1.0",
            "Accept": "application/json",
        })

    def _throttle(self) -> None:
        elapsed = time.time() - self._last
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last = time.time()

    def _fetch(self, path: str) -> Any:
        self._throttle()
        url = f"{BASE_URL}/{path}"
        r = self.session.get(url, timeout=self.timeout)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def programs(self, d: date) -> Any:
        return self._fetch(f"programs/v3/{d.year}/{d.strftime('%Y%m%d')}.json")

    def results(self, d: date) -> Any:
        return self._fetch(f"results/v2/{d.year}/{d.strftime('%Y%m%d')}.json")

    def previews(self, d: date) -> Any:
        return self._fetch(f"previews/v2/{d.year}/{d.strftime('%Y%m%d')}.json")


# ------------------------------------------------------------
# 正規化: JSON → DB行 (fetch_all.py と同じキー形式)
# ------------------------------------------------------------
def programs_to_cards(data: dict) -> list[dict]:
    rows: list[dict] = []
    for race in data.get("programs", []):
        d = race.get("date")
        stadium = race.get("stadium_number")
        race_number = race.get("number")
        if not (d and stadium and race_number):
            continue
        s_name = stadium_name(stadium)
        for b in race.get("boats", []) or []:
            rows.append({
                "date": d, "stadium": stadium, "stadium_name": s_name,
                "race_number": race_number,
                "lane": b.get("racer_boat_number"),
                "racerid": b.get("racer_number"),
                "name": b.get("racer_name"),
                "class": CLASS_MAP.get(b.get("racer_class_number")),
                "branch": PREFECTURES.get(b.get("racer_branch_number")),
                "birthplace": PREFECTURES.get(b.get("racer_birthplace_number")),
                "age": b.get("racer_age"),
                "weight": b.get("racer_weight"),
                "f": b.get("racer_flying_count"),
                "l": b.get("racer_late_count"),
                "aveST": b.get("racer_average_start_timing"),
                "global_win_pt": b.get("racer_national_top_1_percent"),
                "global_in2nd": b.get("racer_national_top_2_percent"),
                "global_in3rd": b.get("racer_national_top_3_percent"),
                "local_win_pt":  b.get("racer_local_top_1_percent"),
                "local_in2nd":   b.get("racer_local_top_2_percent"),
                "local_in3rd":   b.get("racer_local_top_3_percent"),
                "motor":       b.get("racer_assigned_motor_number"),
                "motor_in2nd": b.get("racer_assigned_motor_top_2_percent"),
                "motor_in3rd": b.get("racer_assigned_motor_top_3_percent"),
                "boat":        b.get("racer_assigned_boat_number"),
                "boat_in2nd":  b.get("racer_assigned_boat_top_2_percent"),
                "boat_in3rd":  b.get("racer_assigned_boat_top_3_percent"),
            })
    return rows


def results_to_rows(data: dict) -> list[dict]:
    rows: list[dict] = []
    for race in data.get("results", []):
        d = race.get("race_date")
        stadium = race.get("race_stadium_number")
        race_number = race.get("race_number")
        if not (d and stadium and race_number):
            continue
        s_name = stadium_name(stadium)
        for b in race.get("boats", []) or []:
            rank = b.get("racer_place_number")
            # 失格/転覆などで place_number が 7以上や None になる場合、rank は NULL
            if isinstance(rank, int) and not (1 <= rank <= 6):
                rank = None
            rows.append({
                "date": d, "stadium": stadium, "stadium_name": s_name,
                "race_number": race_number,
                "rank": rank,
                "boat": b.get("racer_boat_number"),
                "racerid": b.get("racer_number"),
                "name": b.get("racer_name"),
                # API には race time (1'49"3 の形式) が含まれない。将来 LZH で補完する。
                "time": None,
                "time_sec": None,
            })
    return rows


def previews_to_conditions(data: dict) -> list[dict]:
    rows: list[dict] = []
    for race in data.get("previews", []):
        d = race.get("race_date")
        stadium = race.get("race_stadium_number")
        race_number = race.get("race_number")
        if not (d and stadium and race_number):
            continue
        s_name = stadium_name(stadium)
        # previews.boats は日付によって list と dict の2形式がある。
        #   - 古い日 (〜2025年頃): [{"racer_boat_number":1, ...}, ...] のlist
        #   - 新しい日           : {"1": {"racer_boat_number":1, ...}, ...} のdict
        # どちらでも lane は boat 自身の racer_boat_number から取り出すのが安全。
        raw_boats = race.get("boats")
        if isinstance(raw_boats, list):
            boat_iter = raw_boats
        elif isinstance(raw_boats, dict):
            boat_iter = list(raw_boats.values())
        else:
            boat_iter = []
        display = [None] * 6
        for b in boat_iter:
            if not isinstance(b, dict):
                continue
            lane = b.get("racer_boat_number")
            if isinstance(lane, int) and 1 <= lane <= 6:
                display[lane - 1] = b.get("racer_exhibition_time")
        rows.append({
            "date": d, "stadium": stadium, "stadium_name": s_name,
            "race_number": race_number,
            "weather": WEATHER_MAP.get(race.get("race_weather_number")),
            "temperature": race.get("race_temperature"),
            "wind_direction": race.get("race_wind_direction_number"),
            "wind_speed": race.get("race_wind"),
            "water_temperature": race.get("race_water_temperature"),
            "wave_height": race.get("race_wave"),
            # API には安定板情報が含まれないため一律 False にする。
            # 本当に使用された日はスクレイパー側で上書きされる。
            "stabilizer": False,
            "display_time_1": display[0],
            "display_time_2": display[1],
            "display_time_3": display[2],
            "display_time_4": display[3],
            "display_time_5": display[4],
            "display_time_6": display[5],
        })
    return rows


# ------------------------------------------------------------
# DB 挿入
# ------------------------------------------------------------
COLUMNS = {
    "race_cards": [
        "date", "stadium", "stadium_name", "race_number", "lane",
        "racerid", "name", "class", "branch", "birthplace", "age", "weight",
        "f", "l", "aveST",
        "global_win_pt", "global_in2nd", "global_in3rd",
        "local_win_pt", "local_in2nd", "local_in3rd",
        "motor", "motor_in2nd", "motor_in3rd",
        "boat", "boat_in2nd", "boat_in3rd",
    ],
    "race_results": [
        "date", "stadium", "stadium_name", "race_number",
        "rank", "boat", "racerid", "name", "time", "time_sec",
    ],
    "race_conditions": [
        "date", "stadium", "stadium_name", "race_number",
        "weather", "temperature", "wind_direction", "wind_speed",
        "water_temperature", "wave_height", "stabilizer",
        "display_time_1", "display_time_2", "display_time_3",
        "display_time_4", "display_time_5", "display_time_6",
    ],
}


def bulk_insert(conn, table: str, columns: list[str], rows: list[dict]) -> int:
    if not rows:
        return 0
    placeholders = ", ".join(["%s"] * len(columns))
    col_sql = ", ".join(columns)
    sql = (
        f"INSERT INTO {table} ({col_sql}) "
        f"VALUES ({placeholders}) ON CONFLICT DO NOTHING"
    )
    values = [tuple(r.get(c) for c in columns) for r in rows]
    with conn.cursor() as cur:
        cur.executemany(sql, values)
        return cur.rowcount


def flush_day(buckets: dict[str, list[dict]]) -> dict[str, int]:
    """1日分のバッファをDBに書き込んで結果件数を返す。
    各日処理するたびに接続を開閉する (アイドルタイムアウト対策)。"""
    counts: dict[str, int] = {}
    if not any(buckets.values()):
        return {k: 0 for k in buckets}
    conn = get_connection()
    try:
        for table, rows in buckets.items():
            try:
                counts[table] = bulk_insert(conn, table, COLUMNS[table], rows)
            except Exception as e:
                logging.error(f"  {table} INSERT失敗: {e}")
                try:
                    conn.rollback()
                except Exception:
                    pass
                counts[table] = 0
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return counts


# ------------------------------------------------------------
# 日付の列挙
# ------------------------------------------------------------
def daterange(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> int:
    p = argparse.ArgumentParser(
        description="BoatraceOpenAPI から過去データを一括インポート"
    )
    p.add_argument("--start", help="取得開始日 YYYY-MM-DD (省略時はprograms最古日 2023-05-01)")
    p.add_argument("--end", help="取得終了日 YYYY-MM-DD (省略時は今日)")
    p.add_argument("--only", choices=["programs", "results", "previews"],
                   help="このAPIだけ取得 (指定しないと3つ全部)")
    args = p.parse_args()

    logfile = setup_logging()

    start = parse_date(args.start) if args.start else PROGRAMS_START
    end = parse_date(args.end) if args.end else date.today()
    if start > end:
        logging.error(f"--start ({start}) が --end ({end}) より後です")
        return 2

    want = (args.only,) if args.only else ("programs", "results", "previews")

    logging.info("=" * 60)
    logging.info(f"BoatraceOpenAPI インポート: {start} 〜 {end}")
    logging.info(f"取得対象API: {list(want)}")
    logging.info(f"リクエスト間隔: {REQUEST_INTERVAL}秒")
    logging.info("=" * 60)

    api = OpenAPIFetcher()
    dates = list(daterange(start, end))

    totals: dict[str, int] = {k: 0 for k in COLUMNS}
    days_with_data = 0
    days_404 = 0
    days_error = 0

    with tqdm(dates, desc="インポート中", unit="day") as bar:
        for d in bar:
            bar.set_postfix_str(d.isoformat())
            buckets: dict[str, list[dict]] = {k: [] for k in COLUMNS}
            any_data = False

            # 各APIを(カバレッジ内なら)取得
            if "programs" in want and d >= PROGRAMS_START:
                try:
                    data = api.programs(d)
                    if data:
                        buckets["race_cards"] = programs_to_cards(data)
                        any_data = any_data or bool(buckets["race_cards"])
                except Exception as e:
                    logging.warning(f"  {d} programs 取得エラー: {e}")
                    days_error += 1
                    continue

            if "results" in want and d >= RESULTS_START:
                try:
                    data = api.results(d)
                    if data:
                        buckets["race_results"] = results_to_rows(data)
                        any_data = any_data or bool(buckets["race_results"])
                except Exception as e:
                    logging.warning(f"  {d} results 取得エラー: {e}")

            if "previews" in want and d >= PREVIEWS_START:
                try:
                    data = api.previews(d)
                    if data:
                        buckets["race_conditions"] = previews_to_conditions(data)
                        any_data = any_data or bool(buckets["race_conditions"])
                except Exception as e:
                    logging.warning(f"  {d} previews 取得エラー: {e}")

            if any_data:
                try:
                    counts = flush_day(buckets)
                    for k, v in counts.items():
                        totals[k] += v
                    days_with_data += 1
                except Exception as e:
                    days_error += 1
                    logging.error(f"✖ {d} INSERT致命的エラー: {e}")
            else:
                days_404 += 1

    logging.info("=" * 60)
    logging.info("インポート完了サマリー")
    logging.info(f"  処理日数:      {len(dates)} 日")
    logging.info(f"  データあり:    {days_with_data} 日")
    logging.info(f"  データなし:    {days_404} 日 (404または開催なし)")
    logging.info(f"  エラー:        {days_error} 日")
    logging.info(f"  INSERT件数:")
    for k, v in totals.items():
        logging.info(f"    {k:20s} {v:>8d}")
    logging.info(f"  ログ: {logfile}")
    logging.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
