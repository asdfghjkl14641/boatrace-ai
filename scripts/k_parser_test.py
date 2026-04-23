# -*- coding: utf-8 -*-
"""k_parser_v2 単体テスト."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from scripts.k_parser_v2 import parse_k_text, RE_DATE, RE_RACE_HDR, _detect_stadium, _fw2hw_digit

print("=" * 70); print("k_parser_v2 unit test"); print("=" * 70)

fails = 0
def check(label, got, expected):
    global fails
    ok = got == expected
    mark = "OK" if ok else "**FAIL**"
    print(f"  {mark}  {label}: got={got!r}  expected={expected!r}")
    if not ok: fails += 1

# ===== Case 1: 全角数字 regex =====
print("\n[Case 1] 全角数字の日付 regex")
m = RE_DATE.search("２０２０年\u3000２月\u30001日")
check("match", m is not None, True)
if m:
    y = _fw2hw_digit(m.group(1))
    mo = _fw2hw_digit(m.group(2))
    d = _fw2hw_digit(m.group(3))
    check("year", y, "2020")
    check("month", mo, "2")
    check("day", d, "1")

# ===== Case 2: 半角数字の日付 (後方互換) =====
print("\n[Case 2] 半角数字の日付 (従来)")
m = RE_DATE.search("2024年5月3日")
check("match", m is not None, True)
if m:
    check("year", m.group(1), "2024")

# ===== Case 3: 全角 R レースヘッダ =====
print("\n[Case 3] 全角 Ｒ レースヘッダ")
m = RE_RACE_HDR.match(" １Ｒ 予選")
check("match", m is not None, True)
if m:
    check("race", _fw2hw_digit(m.group(1)), "1")

m = RE_RACE_HDR.match("12R 一般")
check("match (ASCII)", m is not None, True)
if m:
    check("race", m.group(1), "12")

# ===== Case 4: 会場名検出 (fullwidth space 対応) =====
print("\n[Case 4] 会場名検出")
check("大村 (with fullwidth space)", _detect_stadium("ボートレース大\u3000村"), 24)
check("大村 (normal)", _detect_stadium("大村"), 24)
check("桐生", _detect_stadium("桐生競艇場"), 1)
check("戸田 (江戸川誤マッチ防止)", _detect_stadium("ボートレース戸田"), 2)

# ===== Case 5: 統合テスト — 2020 年の K ファイル模擬 =====
# 固定幅: [2:4]=rank [6]=lane [8:12]=racerid [13:20]=name
#         [22:24]=motor [27:29]=boat [31:36]=disp [38]=course
#         [43:47]=ST [52:58]=race_time
# 艇行は最低 48 文字必要 + 2 leading spaces + rank[2:4] (01-06)
print("\n[Case 5] 2020 K 統合テスト (全角日付 + 複数会場)")

def build_boat_line(rank, lane, racerid, name_padded_7, motor, boat, disp, course, st, race_time):
    """固定幅艇行を構築."""
    # name_padded_7 は Unicode で 7 code points
    s = (
        "  "                      # 0-1
        + f"{rank:02d}"           # 2-3
        + "  "                    # 4-5
        + str(lane)               # 6
        + " "                     # 7
        + f"{racerid:04d}"        # 8-11
        + " "                     # 12
        + name_padded_7           # 13-19
        + "  "                    # 20-21
        + f"{motor:02d}"          # 22-23
        + "   "                   # 24-26
        + f"{boat:02d}"           # 27-28
        + "  "                    # 29-30
        + f"{disp:<5}"            # 31-35
        + "  "                    # 36-37
        + str(course)             # 38
        + "    "                  # 39-42
        + f"{st:<4}"              # 43-46
        + "     "                 # 47-51
        + f"{race_time:<6}"       # 52-57
    )
    return s

def pad_name(name):
    """name を 7 code points にパディング."""
    name_len = len(name)
    if name_len >= 7:
        return name[:7]
    return name + " " * (7 - name_len)

sample_lines = [
    "STARTK",
    "24KBGN",
    "ボートレース大\u3000村   \u3000２月\u30001日  カップ  第\u30004日",
    "",
    "   第\u30004日          ２０２０年\u3000２月\u30001日                  ボートレース大\u3000村",
    "",
    "\u3000１Ｒ  予\u3000選\u3000\u3000\u3000          Ｈ１８００ｍ",
    "-" * 79,
    build_boat_line(1, 1, 4664, pad_name("東口\u3000晃"), 74, 54, "6.72", 1, ".18", "1.50.6"),
    build_boat_line(2, 2, 3521, pad_name("今村賢二"),     13, 70, "6.71", 2, ".15", "1.50.7"),
    build_boat_line(3, 3, 4705, pad_name("吉川勇作"),     35, 63, "6.70", 3, ".14", "1.51.5"),
    build_boat_line(4, 4, 4970, pad_name("井町\u3000泰"), 22, 38, "6.73", 4, ".16", "1.52.3"),
    build_boat_line(5, 5, 5104, pad_name("山田\u3000丈"), 15, 43, "6.75", 5, ".17", "1.53.4"),
    build_boat_line(6, 6, 3283, pad_name("窪田好弘"),     34, 35, "6.80", 6, ".19", "1.54.1"),
    "",
    "\u3000２Ｒ  予\u3000選\u3000\u3000\u3000          Ｈ１８００ｍ",
    "-" * 79,
    build_boat_line(1, 1, 5000, pad_name("選手名Ａ"), 10, 20, "6.70", 1, ".15", "1.50.5"),
    "",
    "ボートレース桐\u3000生   \u3000２月\u30001日  カップ",
    "",
    "   第\u30004日          ２０２０年\u3000２月\u30001日                  ボートレース桐\u3000生",
    "",
    "\u3000１Ｒ  予\u3000選\u3000\u3000\u3000          Ｈ１８００ｍ",
    "-" * 79,
    build_boat_line(1, 1, 4000, pad_name("桐生選手Ａ"), 20, 30, "6.80", 1, ".16", "1.51.2"),
    build_boat_line(2, 2, 4001, pad_name("桐生選手Ｂ"), 21, 31, "6.81", 2, ".17", "1.51.3"),
]
sample = "\n".join(sample_lines)
rows = parse_k_text(sample)
check("総行数 (6+1+2 = 9)", len(rows), 9)
if rows:
    # 大村 1R の 1 着
    r0 = rows[0]
    check("r0.date", r0["date"], "2020-02-01")
    check("r0.stadium (大村=24)", r0["stadium"], 24)
    check("r0.race_number", r0["race_number"], 1)
    check("r0.boat", r0["boat"], 1)
    check("r0.rank", r0["rank"], 1)
    check("r0.racerid", r0["racerid"], 4664)

    # 桐生 1R の 1 着 (会場が切り替わってる)
    kiryu_rows = [r for r in rows if r["stadium"] == 1]
    check("桐生行数 (2)", len(kiryu_rows), 2)
    if kiryu_rows:
        check("桐生 1R 1着 racerid", kiryu_rows[0]["racerid"], 4000)

# ===== Case 6: stadium 追跡失敗時は艇行スキップ =====
print("\n[Case 6] stadium 未設定時は艇行を出さない")
bad_sample = """
some random header without stadium

\u3000１Ｒ  予選
-------------------------------------------------------------------------------
 01  1 4664 東口\u3000晃      74      54    6.72       1     .18    1.50.6
"""
rows = parse_k_text(bad_sample)
check("stadium なし → 0 行", len(rows), 0)

print("\n" + "=" * 70)
if fails == 0:
    print(f"全テストパス 🎉")
else:
    print(f"失敗: {fails} 件")
print("=" * 70)
sys.exit(1 if fails else 0)
