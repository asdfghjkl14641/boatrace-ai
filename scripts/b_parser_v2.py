# -*- coding: utf-8 -*-
"""B ファイル parser v2 — 2020-2023 旧形式 + 2023-05 以降新形式 両対応.

## 旧形式 (2020-2023)
固定幅:
  [0:1]   lane (1-6)
  [2:6]   racerid (4 digits)
  [6:10]  name (4 chars, 全半角混在、不足は fullwidth space padding)
  [10:12] age
  [12:14] branch (prefecture, 2 chars)
  [14:16] weight
  [16:18] class (A1/A2/B1/B2)
  [18:]   空白区切りで 10 数値:
          global_win, global_in2nd, local_win, local_in2nd,
          motor_no, motor_in2nd, boat_no, boat_in2nd,
          session_results (ignored), early_look (ignored)
  **3率系 + F/L/aveST/birthplace は None**

## 新形式 (2023-05 以降, 推定)
列数が増えた場合 (14+ tokens):
  global_win, global_in2nd, global_in3rd,
  local_win, local_in2nd, local_in3rd,
  motor_no, motor_in2nd, motor_in3rd,
  boat_no, boat_in2nd, boat_in3rd,
  ...
自動判定: stat tokens の数 == 10 なら旧形式, >= 12 なら新形式.

## 例
旧: `1 4664東口\\u3000晃36福井54B1 4.10 14.94 3.50  0.00 74  0.00 54 31.08 356 55`
"""
from __future__ import annotations
import re
import logging
from datetime import date
from typing import Any, Iterable

try:
    from scripts.stadiums import STADIUMS, stadium_name
    STADIUM_NAME_TO_ID = {v: k for k, v in STADIUMS.items()}
except ImportError:
    # test 用フォールバック
    STADIUM_NAME_TO_ID = {}
    def stadium_name(s): return str(s)


RE_DATE      = re.compile(r"([0-9０-９]{4})[年]\s*[\u3000]?\s*([0-9０-９]{1,2})[月]\s*[\u3000]?\s*([0-9０-９]{1,2})[日]")
RE_DATE_ALT  = re.compile(r"([0-9０-９]{4})/\s*([0-9０-９]{1,2})/\s*([0-9０-９]{1,2})")
RE_RACE_HDR  = re.compile(r"^[\s\u3000]*([0-9０-９]{1,2})\s*[RＲ]")
# 選手行の緩和正規表現: 先頭に lane + space + 4 digit racerid (後続は何でも)
RE_BOAT_LINE = re.compile(r"^([1-6])\s+(\d{4})")

DEBUG_FIRST_N_LINES = 30
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def _to_int(s: str) -> int | None:
    s = (s or "").strip()
    if not s: return None
    try: return int(s)
    except ValueError:
        try: return int(float(s))
        except ValueError: return None


def _to_float(s: str) -> float | None:
    s = (s or "").strip()
    if not s: return None
    try: return float(s)
    except ValueError: return None


def _fw2hw_digit(s: str) -> str:
    """全角数字を半角に."""
    trans = str.maketrans("０１２３４５６７８９", "0123456789")
    return s.translate(trans)


def parse_boat_line(line: str) -> dict | None:
    """1 選手行を parse.

    2020-2023 旧形式固定幅を優先, だめなら空白区切り best-effort.
    """
    m = RE_BOAT_LINE.match(line)
    if not m:
        return None
    lane = int(m.group(1))
    racerid = int(m.group(2))

    # --- 固定幅抽出 (旧形式前提) ---
    if len(line) < 18:
        return None  # 短すぎる

    name_raw = line[6:10]
    # 名前の fullwidth space を regular space に変換
    name = name_raw.replace("\u3000", " ").strip()

    age_raw = line[10:12]
    branch_raw = line[12:14].strip()
    weight_raw = line[14:16]
    class_raw = line[16:18].strip()

    age = _to_int(age_raw)
    weight = _to_float(weight_raw)

    # class は "A1" "A2" "B1" "B2" のどれか
    if class_raw not in ("A1", "A2", "B1", "B2"):
        # 固定幅が合ってない可能性 → class 再探索
        cls_m = re.search(r"(A[12]|B[12])", line[:30])
        if cls_m:
            class_raw = cls_m.group(1)
        else:
            class_raw = None

    # branch は漢字 2 文字 (都道府県コード)
    if branch_raw and len(branch_raw) < 2:
        branch_raw = None
    if branch_raw and not all(ord(c) >= 0x3000 for c in branch_raw):
        # 漢字以外が混ざってる → 固定幅ずれてる可能性
        branch_raw = None

    # --- 統計値は空白区切り ---
    tail = line[18:].strip()
    tokens = [t for t in re.split(r"\s+", tail) if t]
    num_toks = [t for t in tokens if NUM_RE.match(t)]

    # フォーマット判定: 旧形式は position 4 が motor_no (整数), 新形式は local_in2nd (小数)
    # → num_toks[4] に "." があれば新形式、なければ旧形式
    n = len(num_toks)
    if n < 8:
        return None  # 数値不足

    is_new_format = n >= 12 and "." in num_toks[4]

    if is_new_format:
        try:
            g_win = _to_float(num_toks[0]);  g_2nd = _to_float(num_toks[1])
            g_3rd = _to_float(num_toks[2])
            l_win = _to_float(num_toks[3]);  l_2nd = _to_float(num_toks[4])
            l_3rd = _to_float(num_toks[5])
            mt    = _to_int(num_toks[6]);    m_2nd = _to_float(num_toks[7])
            m_3rd = _to_float(num_toks[8])
            bt    = _to_int(num_toks[9]);    b_2nd = _to_float(num_toks[10])
            b_3rd = _to_float(num_toks[11])
        except (IndexError, ValueError):
            return None
    else:
        # 旧形式 (3率なし): g_win g_2nd l_win l_2nd motor_no m_2nd boat_no b_2nd [junk...]
        try:
            g_win = _to_float(num_toks[0]);  g_2nd = _to_float(num_toks[1])
            l_win = _to_float(num_toks[2]);  l_2nd = _to_float(num_toks[3])
            mt    = _to_int(num_toks[4]);    m_2nd = _to_float(num_toks[5])
            bt    = _to_int(num_toks[6]);    b_2nd = _to_float(num_toks[7])
            g_3rd = l_3rd = m_3rd = b_3rd = None
        except (IndexError, ValueError):
            return None

    return {
        "lane": lane,
        "racerid": racerid,
        "name": name or None,
        "class": class_raw,
        "branch": branch_raw,
        "birthplace": None,  # 旧形式では抽出不可
        "age": age,
        "weight": weight,
        "f": None,
        "l": None,
        "aveST": None,
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
    }


def parse_b_text(text: str, debug: bool = False) -> list[dict]:
    """B ファイル全体 (Shift_JIS decode 済) を parse.

    返り値: race_cards 行の dict リスト.
    各 dict には date, stadium, stadium_name, race_number, lane, ... を含む.
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

        # 日付ヘッダ
        m = RE_DATE.search(stripped) or RE_DATE_ALT.search(stripped)
        if m:
            try:
                y = int(_fw2hw_digit(m.group(1)))
                mo = int(_fw2hw_digit(m.group(2)))
                d = int(_fw2hw_digit(m.group(3)))
                current_date = date(y, mo, d).isoformat()
            except ValueError:
                current_date = None
            # 同じ / 周辺行で会場名検索 (長い順優先)
            hay = " ".join(lines[max(0, i - 1):min(len(lines), i + 3)])
            current_stadium = None
            for name in sorted(STADIUM_NAME_TO_ID.keys(), key=len, reverse=True):
                if name in hay:
                    current_stadium = STADIUM_NAME_TO_ID[name]
                    break
            # 会場名が 「大　村」 など fullwidth space 含みの場合に対応
            if current_stadium is None:
                clean_hay = hay.replace("\u3000", "").replace(" ", "")
                for name in sorted(STADIUM_NAME_TO_ID.keys(), key=len, reverse=True):
                    clean_name = name.replace("\u3000", "").replace(" ", "")
                    if clean_name and clean_name in clean_hay:
                        current_stadium = STADIUM_NAME_TO_ID[name]
                        break
            if current_stadium:
                current_race = None
            continue

        # レース番号ヘッダ (全角数字対応)
        m2 = RE_RACE_HDR.match(line)
        if m2 and current_date and current_stadium:
            rno_raw = _fw2hw_digit(m2.group(1))
            try:
                current_race = int(rno_raw)
            except ValueError:
                current_race = None
            continue

        # 選手行
        if current_race is None:
            continue
        parsed = parse_boat_line(line)
        if parsed is None:
            continue

        parsed["date"] = current_date
        parsed["stadium"] = current_stadium
        parsed["stadium_name"] = stadium_name(current_stadium) if current_stadium else None
        parsed["race_number"] = current_race
        rows.append(parsed)

    return rows
