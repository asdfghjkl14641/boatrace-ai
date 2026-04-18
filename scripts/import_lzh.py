# -*- coding: utf-8 -*-
"""
公式 LZH K ファイル (競走成績) から race_results と current_series を
補完・再構成する。

■ データ元
    https://www1.mbrace.or.jp/od2/K/{YYYYMM}/k{YYMMDD}.lzh
    2002年〜 の日別 Shift_JIS 固定長テキストを LH5圧縮した LZH ファイル。

■ Windowsで動かない理由
    lhafile (LZH展開) は C拡張のビルドが必要で、
    Windows Python 3.14 ではビルドが失敗する。
    Linux (GitHub Actions ubuntu-latest) では gcc で自動ビルド → 動作OK。
    したがって本スクリプトは GitHub Actions 専用。

■ やること
    指定期間の K ファイルを順に:
      1. ダウンロード
      2. LZH 展開 → Shift_JIS TXT
      3. パース (会場セクション → レースブロック → 各艇行)
      4. race_results に INSERT (既存行で time/time_sec が NULL なら UPDATE で埋める)
         current_series に INSERT (UNIQUE (date, stadium, racerid, race_number))

■ ファイル内のテキスト位置 (cstenmt/boatrace 参考)
    各艇行:
      c[2:4]   着順 (01〜06, または失/欠)
      c[6]     枠番 (1〜6)
      c[8:12]  選手登録番号
      c[13:20] 選手名 (Shift_JIS、途中に全角空白)
      c[22:24] モーター番号
      c[27:29] ボート番号
      c[31:36] 展示タイム
      c[38]    進入コース
      c[43:47] スタートタイミング (例 ".15")
      c[52:58] レースタイム (例 "1.50.6")

■ 使い方
    python -m scripts.import_lzh --start 2023-05-01 --end 2025-07-14
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from typing import Any, Iterable

import requests
from tqdm import tqdm

try:
    import lhafile
except ImportError as e:
    print(
        "lhafile がインストールされていません。Linux 環境でのみ動作します。\n"
        "requirements.txt に lhafile が含まれていることを確認し、"
        "`pip install -r requirements.txt` を実行してください。",
        file=sys.stderr,
    )
    raise

from scripts.db import get_connection
from scripts.stadiums import STADIUMS, stadium_name

# ------------------------------------------------------------
# 設定
# ------------------------------------------------------------
BASE_URL = "https://www1.mbrace.or.jp/od2/K/"
REQUEST_INTERVAL = float(os.getenv("LZH_INTERVAL_SEC", "0.5"))  # 公式サーバに優しく

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

# 会場名(日本語) → ID の逆引き。TXT のヘッダから拾うため。
STADIUM_NAME_TO_ID = {v: k for k, v in STADIUMS.items()}


# ------------------------------------------------------------
# ロギング
# ------------------------------------------------------------
def setup_logging() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(LOG_DIR, f"import_lzh_{ts}.log")
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
# ダウンロード & 展開
# ------------------------------------------------------------
_last_req = [0.0]


def _throttle():
    e = time.time() - _last_req[0]
    if e < REQUEST_INTERVAL:
        time.sleep(REQUEST_INTERVAL - e)
    _last_req[0] = time.time()


def download_and_extract(d: date, session: requests.Session) -> str | None:
    """指定日の K ファイルをダウンロードして展開し、Shift_JIS→utf-8文字列で返す。
    非開催日 (404) は None を返す。
    """
    yyyymm = d.strftime("%Y%m")
    yymmdd = d.strftime("%y%m%d")
    url = f"{BASE_URL}{yyyymm}/k{yymmdd}.lzh"

    _throttle()
    try:
        r = session.get(url, timeout=30)
    except requests.RequestException as e:
        logging.warning(f"  {d}: ダウンロード失敗 {e}")
        return None
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        logging.warning(f"  {d}: HTTP {r.status_code}")
        return None

    # LZH 展開 (lhafile は Lhafile(ファイルパス または BytesIO) を受け取る)
    try:
        buf = io.BytesIO(r.content)
        lha = lhafile.Lhafile(buf)
        info = lha.infolist()
        if not info:
            logging.warning(f"  {d}: LZH 内にファイルなし")
            return None
        raw = lha.read(info[0].filename)
    except Exception as e:
        logging.warning(f"  {d}: LZH 展開失敗 {e}")
        return None

    # Shift_JIS → str
    try:
        return raw.decode("shift_jis", errors="replace")
    except Exception as e:
        logging.warning(f"  {d}: Shift_JIS デコード失敗 {e}")
        return None


# ------------------------------------------------------------
# テキストパーサ
#   K TXT の構造:
#     1つのファイルに全24会場 (その日開催分) が順に並ぶ。
#     各会場セクションのフォーマット例:
#
#       競走成績 ...
#       (中略、数行 ヘッダ)
#       第1日目 20XX年MM月DD日  ボートレース住之江
#       (略)
#        1R 予選 ... H1800m
#       (6艇の結果行)
#
# 本パーサは「日付、会場、レース番号」をステートフルに追い、
# 空行またはヘッダ行で切り替える。
# ------------------------------------------------------------
RE_DATE = re.compile(r"(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日")
RE_DATE_SHORT = re.compile(r"(\d{4})/\s*(\d{1,2})/\s*(\d{1,2})")
RE_RACE_HEADER = re.compile(r"\s*([0-9]{1,2})\s*R\s")
RE_BOAT_LINE = re.compile(
    r"^\s*([0-9]{2}|[^\s]{1,2})\s+"   # 着順 (01..06 または 失,転 等)
    r"([1-6])\s+"                        # 枠番
    r"(\d{3,5})\s+"                      # 登録番号 (通常4桁、まれに3桁/5桁)
)
RE_TIME = re.compile(r"(\d)\s*[.\s']\s*(\d{2})\s*[.\s\"]\s*(\d)")


def time_str_to_sec(s: str) -> float | None:
    """K ファイルのレースタイム文字列を秒に変換。
    典型形式: "1.50.6" / "1'50\"6" / " 1 50 6" (スペース区切り) 等。
    1分50秒6 → 110.6。
    """
    if not s:
        return None
    m = RE_TIME.search(s)
    if not m:
        return None
    mm, ss, ds = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return float(mm * 60 + ss + ds / 10.0)


def _parse_field(line: str, start: int, end: int | None = None) -> str:
    """固定位置フィールド取り出し (境界はみ出しOK)。"""
    if end is None:
        s = line[start:start + 1]
    else:
        s = line[start:end]
    return s.strip()


def parse_k_text(text: str) -> list[dict]:
    """K テキスト全体をパースし、艇1行 = dict 1つ の list を返す。
    dict のキー:
      date, stadium, race_number, boat, rank, racerid, name,
      time, time_sec, course, ST
    date/stadium は直前のヘッダから引き継ぐ。
    """
    rows: list[dict] = []
    current_date: str | None = None
    current_stadium: int | None = None
    current_race: int | None = None

    lines = text.splitlines()
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        stripped = line.strip()

        # --- 会場セクションヘッダ検出 ---
        # ここ辺りの行に 20XX年MM月DD日 と会場名が並ぶ。
        m = RE_DATE.search(stripped) or RE_DATE_SHORT.search(stripped)
        if m:
            y, mo, dy = (int(m.group(i)) for i in (1, 2, 3))
            try:
                current_date = date(y, mo, dy).isoformat()
            except ValueError:
                current_date = None
            # 同じ行 (または近隣行) に「ボートレース{会場名}」や会場名がある
            # STADIUM_NAME_TO_ID から最長一致で検索
            hay = stripped
            # 近隣5行も連結して探す
            hay2 = " ".join(lines[max(0, i-1):min(n, i+3)])
            found_id = None
            # 長い名前から優先してマッチ (「江戸川」が「戸田」に誤マッチしないため)
            for name in sorted(STADIUM_NAME_TO_ID.keys(), key=len, reverse=True):
                if name in hay or name in hay2:
                    found_id = STADIUM_NAME_TO_ID[name]
                    break
            if found_id is not None:
                current_stadium = found_id
                current_race = None  # 会場切替時はレースもリセット
            i += 1
            continue

        # --- レース番号ヘッダ検出 ---
        # 例 " 1R 予選..." / "12R 一般戦..."
        m2 = RE_RACE_HEADER.match(line)
        if m2 and (current_date is not None and current_stadium is not None):
            current_race = int(m2.group(1))
            i += 1
            continue

        # --- 艇行検出 ---
        if current_race is not None and len(line) >= 48:
            # 典型: 先頭付近に「01 1 5325...」のような数字パターン
            # 上記の正規表現で判定
            if RE_BOAT_LINE.match(line):
                try:
                    rank_raw = _parse_field(line, 2, 4)
                    lane = _parse_field(line, 6, 7)
                    racerid = _parse_field(line, 8, 12)
                    name = line[13:20].replace("\u3000", " ").strip() if len(line) >= 20 else ""
                    course = _parse_field(line, 38, 39)
                    st = _parse_field(line, 43, 47)
                    time_str = _parse_field(line, 52, 58)

                    # rank: 数字なら int、失/転 等は None
                    rank_int: int | None
                    if rank_raw.isdigit():
                        r = int(rank_raw)
                        rank_int = r if 1 <= r <= 6 else None
                    else:
                        rank_int = None

                    try:
                        lane_int = int(lane) if lane else None
                    except ValueError:
                        lane_int = None

                    try:
                        course_int = int(course) if course else None
                    except ValueError:
                        course_int = None

                    try:
                        st_float = float(st) if st and re.fullmatch(r"-?\.?\d+(?:\.\d+)?", st) else None
                    except ValueError:
                        st_float = None

                    rows.append({
                        "date": current_date,
                        "stadium": current_stadium,
                        "race_number": current_race,
                        "boat": lane_int,
                        "rank": rank_int,
                        "racerid": int(racerid) if racerid.isdigit() else None,
                        "name": name or None,
                        "time": time_str or None,
                        "time_sec": time_str_to_sec(time_str),
                        "course": course_int,
                        "ST": st_float,
                    })
                except Exception as e:
                    logging.debug(f"艇行パース失敗: {line!r}: {e}")
        i += 1

    # 有効な行だけ返す (date/stadium/race 必須)
    return [r for r in rows
            if r["date"] and r["stadium"] and r["race_number"] and r["boat"]]


# ------------------------------------------------------------
# DB 書き込み
#   race_results: UPSERT (time/time_sec が NULL の既存行は埋める)
#   current_series: ON CONFLICT DO NOTHING
# ------------------------------------------------------------
UPSERT_RESULTS = """
INSERT INTO race_results (date, stadium, stadium_name, race_number,
                          rank, boat, racerid, name, time, time_sec)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT ON CONSTRAINT uq_race_results DO UPDATE SET
    time     = COALESCE(race_results.time,     EXCLUDED.time),
    time_sec = COALESCE(race_results.time_sec, EXCLUDED.time_sec),
    rank     = COALESCE(race_results.rank,     EXCLUDED.rank),
    racerid  = COALESCE(race_results.racerid,  EXCLUDED.racerid),
    name     = COALESCE(race_results.name,     EXCLUDED.name)
;
"""

INSERT_SERIES = """
INSERT INTO current_series (date, stadium, stadium_name, racerid, race_number,
                            boat_number, course, "ST", rank)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT ON CONSTRAINT uq_current_series DO NOTHING
;
"""


def write_day(conn, rows: list[dict]) -> tuple[int, int]:
    """1日分の行を 2テーブルに書き込む。(results件数, series件数) を返す。"""
    results_values = []
    series_values = []
    for r in rows:
        s_name = stadium_name(r["stadium"])
        results_values.append((
            r["date"], r["stadium"], s_name, r["race_number"],
            r["rank"], r["boat"], r["racerid"], r["name"],
            r["time"], r["time_sec"],
        ))
        # current_series は racerid があるものだけ
        if r["racerid"] is not None:
            series_values.append((
                r["date"], r["stadium"], s_name, r["racerid"], r["race_number"],
                r["boat"], r["course"], r["ST"], r["rank"],
            ))

    cnt_results = 0
    cnt_series = 0
    with conn.cursor() as cur:
        if results_values:
            cur.executemany(UPSERT_RESULTS, results_values)
            cnt_results = cur.rowcount
        if series_values:
            cur.executemany(INSERT_SERIES, series_values)
            cnt_series = cur.rowcount
    conn.commit()
    return cnt_results, cnt_series


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def parse_date_arg(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def daterange(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def main() -> int:
    p = argparse.ArgumentParser(
        description="公式LZH K ファイルから race_results/current_series を補完"
    )
    p.add_argument("--start", required=True, help="開始日 YYYY-MM-DD")
    p.add_argument("--end", required=True, help="終了日 YYYY-MM-DD")
    args = p.parse_args()

    logfile = setup_logging()
    start = parse_date_arg(args.start)
    end = parse_date_arg(args.end)
    if start > end:
        logging.error(f"--start ({start}) が --end ({end}) より後です")
        return 2

    logging.info("=" * 60)
    logging.info(f"LZH K インポート: {start} 〜 {end}")
    logging.info(f"リクエスト間隔: {REQUEST_INTERVAL}秒")
    logging.info("=" * 60)

    session = requests.Session()
    session.headers.update({"User-Agent": "boatrace-ai-lzh-importer/1.0"})

    dates = list(daterange(start, end))
    tot_days = 0
    tot_404 = 0
    tot_parse = 0
    tot_results = 0
    tot_series = 0

    with tqdm(dates, desc="LZH取得中", unit="day") as bar:
        for d in bar:
            bar.set_postfix_str(d.isoformat())
            text = download_and_extract(d, session)
            if text is None:
                tot_404 += 1
                continue
            try:
                rows = parse_k_text(text)
            except Exception as e:
                logging.warning(f"  {d}: パース中エラー: {e}")
                continue
            if not rows:
                logging.info(f"  {d}: パース結果 0行 (形式変化の可能性)")
                tot_parse += 1
                continue

            # 1日分まとめてINSERT (接続を毎回開閉)
            conn = get_connection()
            try:
                cnt_r, cnt_s = write_day(conn, rows)
                tot_results += cnt_r
                tot_series += cnt_s
                tot_days += 1
            except Exception as e:
                logging.error(f"  {d}: DB書込エラー: {e}")
                try: conn.rollback()
                except Exception: pass
            finally:
                try: conn.close()
                except Exception: pass

    logging.info("=" * 60)
    logging.info("LZH インポート完了サマリー")
    logging.info(f"  期間中日数:     {len(dates)}")
    logging.info(f"  処理成功:       {tot_days}")
    logging.info(f"  データなし:     {tot_404}")
    logging.info(f"  パース0件:      {tot_parse}")
    logging.info(f"  race_results  UPSERT累計: {tot_results}")
    logging.info(f"  current_series INSERT累計: {tot_series}")
    logging.info(f"  ログ: {logfile}")
    logging.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
