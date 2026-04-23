# -*- coding: utf-8 -*-
"""b_parser_v2 単体テスト."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from scripts.b_parser_v2 import parse_boat_line, parse_b_text

print("=" * 70); print("b_parser_v2 unit test"); print("=" * 70)

fails = 0
def check(label, got, expected, tol=1e-6):
    global fails
    if isinstance(expected, float):
        ok = got is not None and abs(got - expected) < tol
    else:
        ok = got == expected
    mark = "OK" if ok else "**FAIL**"
    print(f"  {mark}  {label}: got={got!r}  expected={expected!r}")
    if not ok: fails += 1

# ===== Case 1: 旧形式 (2020) =====
print("\n[Case 1] 2020 旧形式 (短名 + fullwidth space)")
line1 = "1 4664東口\u3000晃36福井54B1 4.10 14.94 3.50  0.00 74  0.00 54 31.08 356 55"
r = parse_boat_line(line1)
assert r is not None, "should parse"
check("lane", r["lane"], 1)
check("racerid", r["racerid"], 4664)
check("name", r["name"], "東口 晃")
check("age", r["age"], 36)
check("branch", r["branch"], "福井")
check("weight", r["weight"], 54.0)
check("class", r["class"], "B1")
check("global_win_pt", r["global_win_pt"], 4.10)
check("global_in2nd", r["global_in2nd"], 14.94)
check("global_in3rd (NULL)", r["global_in3rd"], None)
check("local_win_pt", r["local_win_pt"], 3.50)
check("local_in2nd", r["local_in2nd"], 0.00)
check("local_in3rd (NULL)", r["local_in3rd"], None)
check("motor", r["motor"], 74)
check("motor_in2nd", r["motor_in2nd"], 0.00)
check("motor_in3rd (NULL)", r["motor_in3rd"], None)
check("boat", r["boat"], 54)
check("boat_in2nd", r["boat_in2nd"], 31.08)
check("boat_in3rd (NULL)", r["boat_in3rd"], None)
check("f (NULL 旧形式)", r["f"], None)
check("aveST (NULL 旧形式)", r["aveST"], None)

# ===== Case 2: 旧形式 4文字名 =====
print("\n[Case 2] 2020 旧形式 (4文字名)")
line2 = "2 3521今村賢二51福岡59B2 3.25  5.45 0.00  0.00 13 58.33 70 34.19 234 56"
r = parse_boat_line(line2)
assert r is not None
check("name (4文字)", r["name"], "今村賢二")
check("age", r["age"], 51)
check("branch", r["branch"], "福岡")
check("weight", r["weight"], 59.0)
check("class", r["class"], "B2")
check("global_win_pt", r["global_win_pt"], 3.25)
check("motor", r["motor"], 13)
check("motor_in2nd", r["motor_in2nd"], 58.33)
check("boat", r["boat"], 70)

# ===== Case 3: 3号艇 =====
print("\n[Case 3] 2020 lane=3")
line3 = "3 4705吉川勇作31長崎53B1 4.57 27.78 5.07 32.09 35 33.33 63 34.21 6 253        8"
r = parse_boat_line(line3)
assert r is not None
check("lane", r["lane"], 3)
check("racerid", r["racerid"], 4705)
check("name", r["name"], "吉川勇作")
check("age", r["age"], 31)
check("class", r["class"], "B1")
check("global_win_pt", r["global_win_pt"], 4.57)
check("global_in2nd", r["global_in2nd"], 27.78)
check("local_win_pt", r["local_win_pt"], 5.07)
check("local_in2nd", r["local_in2nd"], 32.09)

# ===== Case 4: 非艇行は None =====
print("\n[Case 4] ヘッダ行/コメント行は None を返す")
for nonbl in [
    "艇 選手 選手  年 支 体級    全国      当地     モーター   ボート   今節成績  早",
    "-------------------------------------------------------------------------------",
    "",
    "ボートレース大　村",
    "\u3000１Ｒ  予\u3000選",
]:
    r = parse_boat_line(nonbl)
    check(f"line={nonbl[:30]!r}", r, None)

# ===== Case 5: 新形式 (3率あり) 仮想行 =====
print("\n[Case 5] 新形式 (3率ありを想定した仮想行)")
# 新形式の実サンプルは未所持のため、統計値の tokens 数だけ変えたテスト
line5 = "1 5325福田望夫28福岡54B1 4.95 25.53 47.87 3.50 15.00 30.00 28 12.34 23.45 74 0.00 28.57"
r = parse_boat_line(line5)
assert r is not None
check("lane", r["lane"], 1)
check("racerid", r["racerid"], 5325)
check("global_win_pt", r["global_win_pt"], 4.95)
check("global_in2nd", r["global_in2nd"], 25.53)
check("global_in3rd (ある!)", r["global_in3rd"], 47.87)
check("local_win_pt", r["local_win_pt"], 3.50)
check("local_in2nd", r["local_in2nd"], 15.00)
check("local_in3rd", r["local_in3rd"], 30.00)
check("motor", r["motor"], 28)
check("motor_in2nd", r["motor_in2nd"], 12.34)
check("motor_in3rd", r["motor_in3rd"], 23.45)

# ===== Case 6: parse_b_text — 日付/会場/レース/選手の統合 =====
print("\n[Case 6] parse_b_text 統合テスト")
b_text = """STARTB
24BBGN
ボートレース大\u3000村   \u3000３月\u3000１日  第１５回公営レーシン  第\u3000４日

                            ＊＊＊\u3000番組表\u3000＊＊＊

          第１５回公営レーシングプレスカップ

   第\u3000４日          ２０２０年\u3000３月\u3000１日                  ボートレース大\u3000村

               −内容については主催者発行のものと照合して下さい−


\u3000１Ｒ  予\u3000選\u3000\u3000\u3000          Ｈ１８００ｍ  電話投票締切予定１５：１６
-------------------------------------------------------------------------------
艇 選手 選手  年 支 体級    全国      当地     モーター   ボート   今節成績  早
番 登番  名   齢 部 重別 勝率  2率  勝率  2率  NO  2率  NO  2率  １２３４５６見
-------------------------------------------------------------------------------
1 4664東口\u3000晃36福井54B1 4.10 14.94 3.50  0.00 74  0.00 54 31.08 356 55
2 3521今村賢二51福岡59B2 3.25  5.45 0.00  0.00 13 58.33 70 34.19 234 56
3 4705吉川勇作31長崎53B1 4.57 27.78 5.07 32.09 35 33.33 63 34.21 6 253        8
4 4970井町\u3000泰28山口48B1 1.80  2.82 1.60  2.86 22 10.00 38 35.81 6 466        6
5 5104山田\u3000丈21福岡54B2 1.80  2.27 0.00  0.00 15 18.18 43 37.33 6 6 64       7
6 3283窪田好弘52福井54B1 4.50 27.62 4.29 26.47 34 30.77 35 31.25 456 56

\u3000２Ｒ  予\u3000選\u3000\u3000\u3000          Ｈ１８００ｍ  電話投票締切予定１５：３９
-------------------------------------------------------------------------------
艇 選手 選手  年 支 体級    全国      当地     モーター   ボート   今節成績  早
番 登番  名   齢 部 重別 勝率  2率  勝率  2率  NO  2率  NO  2率  １２３４５６見
-------------------------------------------------------------------------------
1 5000選手名A30東京50A1 5.50 40.00 4.50 30.00 10 40.00 20 30.00 123 45
"""
rows = parse_b_text(b_text, debug=False)
check("総行数", len(rows), 7)  # 1R (6 艇) + 2R (1 艇) = 7
if rows:
    check("row0.date", rows[0]["date"], "2020-03-01")
    check("row0.stadium (大村=24)", rows[0]["stadium"], 24)
    check("row0.race_number", rows[0]["race_number"], 1)
    check("row0.lane", rows[0]["lane"], 1)
    check("row0.racerid", rows[0]["racerid"], 4664)
    # 2R の row
    r2 = [r for r in rows if r["race_number"] == 2]
    check("2R row count", len(r2), 1)
    if r2:
        check("2R.racerid", r2[0]["racerid"], 5000)

print("\n" + "=" * 70)
if fails == 0:
    print(f"全テストパス 🎉")
else:
    print(f"失敗: {fails} 件")
print("=" * 70)
sys.exit(1 if fails else 0)
