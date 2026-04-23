# -*- coding: utf-8 -*-
"""
公式 LZH B ファイル (番組表) から race_cards を補完する。

■ データ元
    https://www1.mbrace.or.jp/od2/B/{YYYYMM}/b{YYMMDD}.lzh
    2002年〜 の日別 Shift_JIS 固定長テキストを LH5圧縮した LZH ファイル。

■ 使い方 (GitHub Actions 推奨)
    python -m scripts.import_lzh_b --start 2020-01-01 --end 2020-01-31

■ フィールド位置 (cstenmt/boatrace の txtTocsv_timetable.py を参考)
    B ファイルの1選手行は Shift_JIS 固定幅。おおむね以下:
        [0]     枠番 (1〜6)
        [2:6]   登録番号 (4桁)
        [6:10]  氏名 (4 Unicode 文字ぶん、実際は 2バイト文字主体)
        [10:13] 級別 + 支部 混在領域 (例 " B1" + 支部コード)
        ... 位置は実データで微調整必要

    今回は cstenmt のオフセット近似 + 正規表現フォールバックで吸収する。
    フォーマット検出失敗時は WARNING を出してスキップし、他日は継続。

■ race_cards の必須キー: (date, stadium, race_number, lane)
    これだけ埋まっていれば INSERT OR IGNORE で安全に取り込める。
    統計値 (勝率など) は部分的にでも入れる best-effort 方針。
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
except ImportError:
    print("lhafile が必要です (Linux/GitHub Actions環境で動作)。", file=sys.stderr)
    raise

from scripts.db import get_connection, placeholder as _ph
from scripts.stadiums import STADIUMS, stadium_name

BASE_URL = "https://www1.mbrace.or.jp/od2/B/"
REQUEST_INTERVAL = float(os.getenv("LZH_INTERVAL_SEC", "3.0"))
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

STADIUM_NAME_TO_ID = {v: k for k, v in STADIUMS.items()}

_PH = _ph()
INSERT_CARDS = f"""
INSERT OR IGNORE INTO race_cards (
    date, stadium, stadium_name, race_number, lane,
    racerid, name, class, branch, birthplace, age, weight,
    f, l, aveST,
    global_win_pt, global_in2nd, global_in3rd,
    local_win_pt,  local_in2nd,  local_in3rd,
    motor, motor_in2nd, motor_in3rd,
    boat,  boat_in2nd,  boat_in3rd
) VALUES ({', '.join([_PH]*27)})
"""


def setup_logging() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(LOG_DIR, f"import_lzh_b_{ts}.log")
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


_last_req = [0.0]


def _throttle():
    e = time.time() - _last_req[0]
    if e < REQUEST_INTERVAL:
        time.sleep(REQUEST_INTERVAL - e)
    _last_req[0] = time.time()


def download_and_extract_b(d: date, session: requests.Session) -> str | None:
    yyyymm = d.strftime("%Y%m")
    yymmdd = d.strftime("%y%m%d")
    url = f"{BASE_URL}{yyyymm}/b{yymmdd}.lzh"
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
    try:
        lha = lhafile.Lhafile(io.BytesIO(r.content))
        info = lha.infolist()
        if not info:
            return None
        raw = lha.read(info[0].filename)
        return raw.decode("shift_jis", errors="replace")
    except Exception as e:
        logging.warning(f"  {d}: LZH 展開失敗 {e}")
        return None


# ------------------------------------------------------------
# パーサ — 2020-2023 対応版に差し替え
# ------------------------------------------------------------
from scripts.b_parser_v2 import parse_b_text as parse_b_text_v2

RE_DATE = re.compile(r"(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日")
RE_DATE_SHORT = re.compile(r"(\d{4})/\s*(\d{1,2})/\s*(\d{1,2})")
RE_RACE_HEADER = re.compile(r"\s*(\d{1,2})\s*[RＲ]\s")
# 選手行の先頭は「枠番(1-6) 登番(4桁)」で始まることが多い
RE_BOAT_LINE = re.compile(r"^([1-6])\s+(\d{4})\s")
# 固定位置が違う可能性があるのでバックアップで空白区切りも試す
RE_SPACE_FIELDS = re.compile(r"\s+")

DEBUG_FIRST_N_LINES = 30   # デバッグ: 最初のファイルの先頭30行をログに出す


def _num_or_none(s: str, cast=float) -> Any:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return cast(s)
    except ValueError:
        return None


def parse_b_text(text: str, debug: bool = False) -> list[dict]:
    """B テキスト全体をパースし、選手1行分 = dict 1つ のlistを返す。
    キー: race_cards のカラム一式 (date, stadium, race_number, lane, racerid, name,
          class, branch, birthplace, age, weight, f, l, aveST,
          global_*, local_*, motor, motor_in2nd/3rd, boat, boat_in2nd/3rd)
    """
    rows: list[dict] = []
    current_date: str | None = None
    current_stadium: int | None = None
    current_race: int | None = None

    lines = text.splitlines()
    if debug:
        logging.info(f"--- B file debug first {min(DEBUG_FIRST_N_LINES, len(lines))} lines ---")
        for i, l in enumerate(lines[:DEBUG_FIRST_N_LINES]):
            logging.info(f"  [{i:3d}] {l.rstrip()!r}")
        logging.info(f"--- total lines: {len(lines)} ---")

    for i, line in enumerate(lines):
        stripped = line.strip()

        # 会場ヘッダ
        m = RE_DATE.search(stripped) or RE_DATE_SHORT.search(stripped)
        if m:
            try:
                current_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
            except ValueError:
                current_date = None
            # 前後3行で会場名を検索 (長い順に試して誤マッチを避ける)
            hay = " ".join(lines[max(0, i - 1):min(len(lines), i + 3)])
            current_stadium = None
            for name in sorted(STADIUM_NAME_TO_ID.keys(), key=len, reverse=True):
                if name in hay:
                    current_stadium = STADIUM_NAME_TO_ID[name]
                    break
            if current_stadium:
                current_race = None
            continue

        # レース番号ヘッダ
        m2 = RE_RACE_HEADER.match(line)
        if m2 and current_date and current_stadium:
            current_race = int(m2.group(1))
            continue

        # 選手行
        if current_race is None:
            continue
        m3 = RE_BOAT_LINE.match(line)
        if not m3:
            continue

        lane = int(m3.group(1))
        racerid = int(m3.group(2))

        # 固定幅をこちらが決め打ち。cstenmtから少し幅広く取る。
        # 氏名: 4-8 文字目あたりを広めに (全半角混在を許容)
        name = line[7:21].replace("\u3000", " ").strip() if len(line) >= 21 else ""
        # 級別: 氏名の直後にあると仮定 (2文字)
        cls = line[21:23].strip() if len(line) >= 23 else ""
        # 以下は空白区切りで数値トークンを取り出す方式 (位置がズレても頑健)
        # lane/racerid/name/class を除いた後半部分を抽出
        tail = line[23:] if len(line) > 23 else ""
        tokens = [t for t in RE_SPACE_FIELDS.split(tail) if t]

        # 典型的な出現順 (番号, 比率, 番号, 比率...):
        #   支部, 出身地, 年齢, 体重,
        #   F回数, L回数, aveST,
        #   全国勝率, 全国2連率, 全国3連率,
        #   当地勝率, 当地2連率, 当地3連率,
        #   モーター番号, モーター2連率, モーター3連率,
        #   ボート番号, ボート2連率, ボート3連率
        # 実際は文字列トークンが混じるため、数値のみ拾って順番に割り当てる best-effort。
        num_re = re.compile(r"^-?\d+(?:\.\d+)?$")
        num_toks = [t for t in tokens if num_re.match(t)]
        str_toks = [t for t in tokens if not num_re.match(t)]

        branch = str_toks[0] if len(str_toks) > 0 else None
        birthplace = str_toks[1] if len(str_toks) > 1 else None

        def _g(i):
            return num_toks[i] if i < len(num_toks) else None

        age    = _num_or_none(_g(0), int)
        weight = _num_or_none(_g(1), float)
        f_cnt  = _num_or_none(_g(2), int)
        l_cnt  = _num_or_none(_g(3), int)
        aveST  = _num_or_none(_g(4), float)
        g_win  = _num_or_none(_g(5), float)
        g_2nd  = _num_or_none(_g(6), float)
        g_3rd  = _num_or_none(_g(7), float)
        l_win  = _num_or_none(_g(8), float)
        l_2nd  = _num_or_none(_g(9), float)
        l_3rd  = _num_or_none(_g(10), float)
        mt     = _num_or_none(_g(11), int)
        m_2nd  = _num_or_none(_g(12), float)
        m_3rd  = _num_or_none(_g(13), float)
        bt     = _num_or_none(_g(14), int)
        b_2nd  = _num_or_none(_g(15), float)
        b_3rd  = _num_or_none(_g(16), float)

        rows.append({
            "date": current_date,
            "stadium": current_stadium,
            "stadium_name": stadium_name(current_stadium),
            "race_number": current_race,
            "lane": lane,
            "racerid": racerid,
            "name": name or None,
            "class": cls or None,
            "branch": branch,
            "birthplace": birthplace,
            "age": age,
            "weight": weight,
            "f": f_cnt,
            "l": l_cnt,
            "aveST": aveST,
            "global_win_pt": g_win,
            "global_in2nd": g_2nd,
            "global_in3rd": g_3rd,
            "local_win_pt": l_win,
            "local_in2nd": l_2nd,
            "local_in3rd": l_3rd,
            "motor": mt,
            "motor_in2nd": m_2nd,
            "motor_in3rd": m_3rd,
            "boat": bt,
            "boat_in2nd": b_2nd,
            "boat_in3rd": b_3rd,
        })
    return rows


# ------------------------------------------------------------
# 書き込み
# ------------------------------------------------------------
COLS = [
    "date", "stadium", "stadium_name", "race_number", "lane",
    "racerid", "name", "class", "branch", "birthplace", "age", "weight",
    "f", "l", "aveST",
    "global_win_pt", "global_in2nd", "global_in3rd",
    "local_win_pt",  "local_in2nd",  "local_in3rd",
    "motor", "motor_in2nd", "motor_in3rd",
    "boat",  "boat_in2nd",  "boat_in3rd",
]


def write_day(conn, rows: list[dict]) -> int:
    if not rows:
        return 0
    values = [tuple(r.get(c) for c in COLS) for r in rows]
    with conn.cursor() as cur:
        cur.executemany(INSERT_CARDS, values)
        n = cur.rowcount
    conn.commit()
    return n


def daterange(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def parse_date_arg(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> int:
    p = argparse.ArgumentParser(description="公式LZH B ファイルから race_cards を補完")
    p.add_argument("--start", required=True, help="開始日 YYYY-MM-DD")
    p.add_argument("--end", required=True, help="終了日 YYYY-MM-DD")
    args = p.parse_args()

    setup_logging()
    start = parse_date_arg(args.start)
    end = parse_date_arg(args.end)
    if start > end:
        logging.error(f"--start ({start}) が --end ({end}) より後です")
        return 2

    logging.info("=" * 60)
    logging.info(f"LZH B インポート: {start} 〜 {end}")
    logging.info(f"リクエスト間隔: {REQUEST_INTERVAL}秒")
    logging.info("=" * 60)

    session = requests.Session()
    session.headers.update({"User-Agent": "boatrace-ai-lzh-b/1.0"})

    dates = list(daterange(start, end))
    tot_success = 0
    tot_404 = 0
    tot_rows = 0
    debug_done = False

    with tqdm(dates, desc="LZH(B)取得中", unit="day") as bar:
        for d in bar:
            bar.set_postfix_str(d.isoformat())
            text = download_and_extract_b(d, session)
            if text is None:
                tot_404 += 1
                continue
            try:
                debug = not debug_done
                rows = parse_b_text_v2(text, debug=debug)
                debug_done = True
            except Exception as e:
                logging.warning(f"  {d}: パースエラー: {e}")
                continue
            if not rows:
                logging.info(f"  {d}: パース結果 0行")
                continue

            conn = get_connection()
            try:
                n = write_day(conn, rows)
                tot_rows += n
                tot_success += 1
            except Exception as e:
                logging.error(f"  {d}: DB書込エラー: {e}")
                try: conn.rollback()
                except Exception: pass
            finally:
                try: conn.close()
                except Exception: pass

    logging.info("=" * 60)
    logging.info("B インポート完了サマリー")
    logging.info(f"  期間中日数: {len(dates)}")
    logging.info(f"  処理成功:   {tot_success}")
    logging.info(f"  データなし: {tot_404}")
    logging.info(f"  race_cards INSERT累計: {tot_rows}")
    logging.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
