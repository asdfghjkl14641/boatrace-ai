# -*- coding: utf-8 -*-
"""K ファイル parser v2 — 2020-2023 旧フォーマット対応 + 2024 以降新フォーマット.

旧フォーマット (2020-2023):
  - 日付ヘッダに全角数字: ２０２０年 ２月 １日
  - レース番号ヘッダに全角 R: １Ｒ
  - 会場名に fullwidth space: 大\u3000村

新フォーマット (2024-):
  - 半角数字の可能性あり (非破壊対応)

## ファイル内のテキスト位置 (cstenmt/boatrace 参考, import_lzh.py から流用)
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
"""
from __future__ import annotations
import re
import logging
from datetime import date as _date_cls

try:
    from scripts.stadiums import STADIUMS, stadium_name
    STADIUM_NAME_TO_ID = {v: k for k, v in STADIUMS.items()}
except ImportError:
    STADIUM_NAME_TO_ID = {}
    def stadium_name(s): return str(s)


# 全角対応版 regex (b_parser_v2 から流用)
RE_DATE      = re.compile(r"([0-9０-９]{4})[年]\s*[\u3000]?\s*([0-9０-９]{1,2})[月]\s*[\u3000]?\s*([0-9０-９]{1,2})[日]")
RE_DATE_ALT  = re.compile(r"([0-9０-９]{4})/\s*([0-9０-９]{1,2})/\s*([0-9０-９]{1,2})")
RE_RACE_HDR  = re.compile(r"^[\s\u3000]*([0-9０-９]{1,2})\s*[RＲ]")
RE_BOAT_LINE = re.compile(
    r"^\s*([0-9]{2}|[^\s]{1,2})\s+"      # 着順 (01..06 または 失, 転 等)
    r"([1-6])\s+"                         # 枠番
    r"(\d{3,5})\s+"                       # 登録番号
)
RE_TIME = re.compile(r"(\d)\s*[.\s']\s*(\d{2})\s*[.\s\"]\s*(\d)")


def _fw2hw_digit(s: str) -> str:
    return s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))


def time_str_to_sec(s: str) -> float | None:
    if not s:
        return None
    m = RE_TIME.search(s)
    if not m:
        return None
    mm, ss, ds = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return float(mm * 60 + ss + ds / 10.0)


def _parse_field(line: str, start: int, end: int | None = None) -> str:
    if end is None:
        s = line[start:start + 1]
    else:
        s = line[start:end]
    return s.strip()


def _detect_stadium(hay: str) -> int | None:
    """会場名 (fullwidth space 許容) を hay から検出."""
    # 直接マッチ
    for name in sorted(STADIUM_NAME_TO_ID.keys(), key=len, reverse=True):
        if name in hay:
            return STADIUM_NAME_TO_ID[name]
    # fullwidth space 除去後にマッチ
    clean = hay.replace("\u3000", "").replace(" ", "")
    for name in sorted(STADIUM_NAME_TO_ID.keys(), key=len, reverse=True):
        clean_name = name.replace("\u3000", "").replace(" ", "")
        if clean_name and clean_name in clean:
            return STADIUM_NAME_TO_ID[name]
    return None


def parse_k_text(text: str) -> list[dict]:
    """K ファイル全体をパースし、艇 1 行 = dict 1 つ の list を返す.

    キー: date, stadium, race_number, boat, rank, racerid, name,
          time, time_sec, course, ST

    date/stadium は直前のヘッダから引き継ぐ.
    stadium=None 時は艇行を無視 (重複衝突防止).
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

        # --- 日付検出 ---
        m = RE_DATE.search(stripped) or RE_DATE_ALT.search(stripped)
        if m:
            try:
                y = int(_fw2hw_digit(m.group(1)))
                mo = int(_fw2hw_digit(m.group(2)))
                dy = int(_fw2hw_digit(m.group(3)))
                current_date = _date_cls(y, mo, dy).isoformat()
            except ValueError:
                current_date = None
            # 会場名検出 (前後 5 行範囲)
            hay = " ".join(lines[max(0, i - 1):min(n, i + 3)])
            found = _detect_stadium(hay)
            if found is not None:
                current_stadium = found
                current_race = None
            i += 1
            continue

        # --- レース番号検出 (全角 R 対応) ---
        m2 = RE_RACE_HDR.match(line)
        if m2 and current_date is not None and current_stadium is not None:
            try:
                current_race = int(_fw2hw_digit(m2.group(1)))
            except ValueError:
                current_race = None
            i += 1
            continue

        # --- 艇行検出 ---
        if (current_race is not None
                and current_date is not None
                and current_stadium is not None
                and len(line) >= 48
                and RE_BOAT_LINE.match(line)):
            try:
                rank_raw = _parse_field(line, 2, 4)
                lane = _parse_field(line, 6, 7)
                racerid = _parse_field(line, 8, 12)
                name = line[13:20].replace("\u3000", " ").strip() if len(line) >= 20 else ""
                course = _parse_field(line, 38, 39)
                st = _parse_field(line, 43, 47)
                time_str = _parse_field(line, 52, 58)

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
                    "stadium_name": stadium_name(current_stadium),
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

    # 有効な行だけ返す
    return [r for r in rows
            if r["date"] and r["stadium"] and r["race_number"] and r["boat"]]
