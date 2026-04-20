# -*- coding: utf-8 -*-
"""
kyotei_murao_scraper の単体テスト。ネットワーク・実 DB には一切触れない。

 - URL 生成 / CLI / チェックポイント I/O
 - HTML パース (4 券種別 + race_meta)
 - RateLimiter (並列時の相互排他も含む)
 - ThreadPoolExecutor 並列書き込み時の Progress のスレッドセーフ性
"""
from __future__ import annotations

import datetime as dt
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from scripts import kyotei_murao_scraper as mod  # noqa: E402
from scripts.kyotei_murao_scraper import (  # noqa: E402
    KACHISHIKI,
    MetaRow,
    OddsRow,
    PHASE_RANGES,
    Progress,
    RateLimiter,
    Task,
    _parse_data_types,
    build_argparser,
    build_odds_url,
    build_tasks,
    extract_total_count,
    iter_dates,
    load_progress,
    parse_odds_page,
    resolve_date_range,
    save_progress,
)


# ================================================================
# サンプル HTML (構造調査で確認した実構造に準拠)
# ================================================================
def _row_html(combo_digits, date="2026-04-18", stadium="24#大　村", race="12R",
              weekday="土", schedule="1日目", final_r="12R", time_zone="ﾃﾞｲ",
              series="一般", grade="一般", rank_count="A6B0",
              rank_lineup="A1A1A1A1A1A1", race_type="特賞|特選|選抜",
              entry_fixed="固定", weather="雨", wind_dir="北",
              wind_speed="1", wave="1",
              odds5="13.3", pop5="2", odds1="13.5", pop1="3",
              result="1 5 3", kimarite="逃げ", payout="0"):
    spans = "".join(f"<span><strong>{d}</strong></span>" for d in combo_digits)
    return f"""
    <tr>
      <td>{date}</td><td>{weekday}</td><td>{stadium}</td>
      <td>{schedule}</td><td>{final_r}</td><td>{time_zone}</td>
      <td>{series}</td><td>{grade}</td><td>{rank_count}</td>
      <td>{rank_lineup}</td><td>{race_type}</td><td>{race}</td>
      <td>{entry_fixed}</td>
      <td>{weather}</td><td>{wind_dir}</td><td>{wind_speed}</td><td>{wave}</td>
      <td class="small">{spans}</td>
      <td>{odds5}</td><td>{pop5}</td><td>{odds1}</td><td>{pop1}</td>
      <td>{result}</td><td>{kimarite}</td><td>{payout}</td>
    </tr>"""


_THEAD = """
<thead><tr>
<th>ﾚｰｽ日付</th><th>曜日</th><th>開催場</th><th>日程</th><th>最終R</th>
<th>時間帯</th><th>ｼﾘｰｽﾞ</th><th>ｸﾞﾚｰﾄﾞ</th><th>ﾗﾝｸ人数</th><th>ﾗﾝｸ並び</th>
<th>ﾚｰｽ種別</th><th>ﾚｰｽ</th><th>進入固定</th>
<th>天気</th><th>風向</th><th>風速</th><th>波</th>
<th>オッズ対象</th><th>5分前オッズ</th><th>5分前人気</th>
<th>1分前オッズ</th><th>1分前人気</th><th>結果</th><th>決まり手</th><th>払戻</th>
</tr></thead>
"""


def make_html(rows_html: str, total: int | None = None) -> str:
    totalline = f'<div>全{total}件中 1～{min(total,5000)}</div>' if total is not None else ""
    return f"""
    <html><body>
    <table class="datatable">
      {_THEAD}
      {rows_html}
    </table>
    {totalline}
    </body></html>"""


# ================================================================
# URL 生成
# ================================================================
class TestBuildOddsUrl(unittest.TestCase):
    def test_default_kid_is_trifecta(self):
        url = build_odds_url(dt.date(2025, 8, 1))
        qs = parse_qs(urlparse(url).query)
        self.assertEqual(qs["kachishiki_id"], ["2"])   # 3連単 = 2
        self.assertEqual(qs["display_num"], ["FIVE_THOUSAND"])

    def test_each_bet_type_id(self):
        self.assertEqual(KACHISHIKI["odds_exacta"],   1)
        self.assertEqual(KACHISHIKI["odds_trifecta"], 2)
        self.assertEqual(KACHISHIKI["odds_quinella"], 3)
        self.assertEqual(KACHISHIKI["odds_trio"],     4)

    def test_with_stadium_no_brackets(self):
        url = build_odds_url(dt.date(2025, 8, 1), stadium_id=12)
        qs = parse_qs(urlparse(url).query)
        # フォームは角括弧なしの `conditions[stadium_id]=12` 形
        self.assertEqual(qs.get("conditions[stadium_id]"), ["12"])

    def test_invalid_stadium(self):
        with self.assertRaises(ValueError):
            build_odds_url(dt.date(2025, 8, 1), stadium_id=25)

    def test_invalid_date_range(self):
        with self.assertRaises(ValueError):
            build_odds_url(dt.date(2025, 8, 10), date_to=dt.date(2025, 8, 1))


class TestIterDates(unittest.TestCase):
    def test_range(self):
        out = list(iter_dates(dt.date(2025, 8, 1), dt.date(2025, 8, 3)))
        self.assertEqual(len(out), 3)

    def test_reverse_empty(self):
        self.assertEqual(list(iter_dates(dt.date(2025, 8, 3), dt.date(2025, 8, 1))), [])


# ================================================================
# HTML パース (4 券種)
# ================================================================
class TestParseOddsPage(unittest.TestCase):
    def test_trifecta(self):
        html = make_html(
            _row_html([1, 2, 3]) + _row_html([1, 3, 2], race="11R"),
            total=20160,
        )
        odds, meta = parse_odds_page(html, "odds_trifecta")
        self.assertEqual(len(odds), 2)
        self.assertEqual(odds[0].combination, "1-2-3")
        self.assertEqual(odds[0].date, dt.date(2026, 4, 18))
        self.assertEqual(odds[0].stadium_id, 24)
        self.assertEqual(odds[0].stadium_name, "大村")
        self.assertEqual(odds[0].race_number, 12)
        self.assertAlmostEqual(odds[0].odds_5min, 13.3)
        self.assertAlmostEqual(odds[0].odds_1min, 13.5)
        self.assertEqual(odds[0].pop_5min, 2)
        self.assertEqual(odds[0].pop_1min, 3)
        self.assertEqual(odds[1].combination, "1-3-2")
        # meta は 1 レース = 1 行のみ (race_number が重複しない 2 レースなので 2 行)
        self.assertEqual(len(meta), 2)
        self.assertEqual(meta[0].kimarite, "逃げ")
        self.assertEqual(meta[0].weather, "雨")
        self.assertEqual(meta[0].entry_fixed, 1)

    def test_exacta_has_2_digits(self):
        html = make_html(_row_html([3, 5]))    # span ×2
        odds, _ = parse_odds_page(html, "odds_exacta")
        self.assertEqual(len(odds), 1)
        self.assertEqual(odds[0].combination, "3-5")

    def test_quinella_sorted(self):
        # 表示は [5, 3] でもソート後 "3=5" に
        html = make_html(_row_html([5, 3]))
        odds, _ = parse_odds_page(html, "odds_quinella")
        self.assertEqual(len(odds), 1)
        self.assertEqual(odds[0].combination, "3=5")

    def test_trio_sorted(self):
        html = make_html(_row_html([4, 1, 6]))
        odds, _ = parse_odds_page(html, "odds_trio")
        self.assertEqual(len(odds), 1)
        self.assertEqual(odds[0].combination, "1=4=6")

    def test_meta_dedup_same_race(self):
        """同一レースで 120 通りあっても meta は 1 行。"""
        rows = "".join(_row_html([a, b, c])
                       for a in [1,2] for b in [3,4] for c in [5,6])
        html = make_html(rows)
        odds, meta = parse_odds_page(html, "odds_trifecta")
        # 8 行の odds に対し、同一 (date, stadium, race) なので meta は 1 行
        self.assertEqual(len(odds), 8)
        self.assertEqual(len(meta), 1)

    def test_mismatched_digit_count_is_skipped(self):
        # 3連単として 2 桁の組合せが来たら落とす
        html = make_html(_row_html([1, 2]))
        odds, _ = parse_odds_page(html, "odds_trifecta")
        self.assertEqual(len(odds), 0)

    def test_invalid_bet_type(self):
        with self.assertRaises(ValueError):
            parse_odds_page(make_html(""), "odds_unknown")


class TestMetaParse(unittest.TestCase):
    def test_full_meta_fields(self):
        html = make_html(_row_html(
            [1,2,3], weekday="月", wind_dir="北東", wind_speed="3",
            wave="5", weather="曇", kimarite="差し", payout="1350",
            entry_fixed="非固定"))
        _, meta = parse_odds_page(html, "odds_trifecta")
        self.assertEqual(len(meta), 1)
        m = meta[0]
        self.assertEqual(m.weekday, "月")
        self.assertEqual(m.weather, "曇")
        self.assertEqual(m.wind_direction, "北東")
        self.assertAlmostEqual(m.wind_speed, 3.0)
        self.assertAlmostEqual(m.wave, 5.0)
        self.assertEqual(m.kimarite, "差し")
        self.assertEqual(m.payout, 1350)
        self.assertEqual(m.entry_fixed, 0)


class TestExtractTotal(unittest.TestCase):
    def test_total_extract(self):
        self.assertEqual(extract_total_count("全20,160件中 1～5000"), 20160)
        self.assertEqual(extract_total_count("全 5040 件中"), 5040)
        self.assertIsNone(extract_total_count("nothing here"))


# ================================================================
# チェックポイント
# ================================================================
class TestProgress(unittest.TestCase):
    def test_key_format(self):
        k = Progress.key("odds_trifecta", dt.date(2025, 8, 1), None, 3)
        self.assertEqual(k, "odds_trifecta:2025-08-01:ALL:3")

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "progress.json"
            pr = Progress()
            pr.mark_done("odds_trifecta", dt.date(2025, 8, 1), None, 1)
            pr.mark_failed("odds_exacta", dt.date(2025, 8, 2), None, 2, "err")
            save_progress(pr, p)
            q = load_progress(p)
            self.assertTrue(q.is_done("odds_trifecta", dt.date(2025, 8, 1), None, 1))
            self.assertIn(Progress.key("odds_exacta", dt.date(2025, 8, 2), None, 2), q.failed)

    def test_parallel_write_safety(self):
        """複数スレッドから mark_done を叩いても競合しない。"""
        pr = Progress()
        def worker(i):
            for j in range(50):
                pr.mark_done(f"odds_trifecta", dt.date(2025, 8, 1), None, i * 100 + j)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        # 8 × 50 = 400 ユニークキーが記録されるはず
        self.assertEqual(len(pr.completed), 400)


# ================================================================
# レート制限 (時刻・スリープ注入)
# ================================================================
class _Clock:
    def __init__(self, start=1700_000_000.0):
        self.t = start; self.lock = threading.Lock()
    def __call__(self): return self.t
    def advance(self, s):
        with self.lock: self.t += s


class _Sleep:
    def __init__(self, clock):
        self.clock = clock; self.calls = []; self.lock = threading.Lock()
    def __call__(self, s):
        with self.lock: self.calls.append(s)
        self.clock.advance(s)


class TestRateLimiter(unittest.TestCase):
    def test_first_call_no_wait(self):
        noon = dt.datetime(2026, 4, 19, 12, 0, 0).timestamp()
        c = _Clock(noon); s = _Sleep(c)
        r = RateLimiter(min_sleep=3, max_sleep=5, hourly_limit=10, _now_fn=c)
        self.assertEqual(r.throttle(sleep_fn=s), 0.0)

    def test_second_call_enforces_min(self):
        noon = dt.datetime(2026, 4, 19, 12, 0, 0).timestamp()
        c = _Clock(noon); s = _Sleep(c)
        r = RateLimiter(min_sleep=3, max_sleep=5, hourly_limit=10, _now_fn=c)
        r.throttle(sleep_fn=s)
        c.advance(0.1)
        slept = r.throttle(sleep_fn=s)
        self.assertGreaterEqual(slept, 2.8)
        self.assertLessEqual(slept, 5.0)

    def test_night_sleeps_until_morning(self):
        night = dt.datetime(2026, 4, 19, 23, 30, 0).timestamp()
        c = _Clock(night); s = _Sleep(c)
        r = RateLimiter(min_sleep=3, max_sleep=5, hourly_limit=10, _now_fn=c)
        r.throttle(sleep_fn=s)
        self.assertTrue(any(x >= 3600 for x in s.calls))

    def test_hourly_limit_enforced(self):
        noon = dt.datetime(2026, 4, 19, 12, 0, 0).timestamp()
        c = _Clock(noon); s = _Sleep(c)
        r = RateLimiter(min_sleep=0, max_sleep=0, hourly_limit=3, _now_fn=c)
        for _ in range(3):
            r.throttle(sleep_fn=s); c.advance(1)
        r.throttle(sleep_fn=s)
        self.assertTrue(any(x > 3000 for x in s.calls))

    def test_parallel_workers_share_limiter(self):
        """並列 N ワーカーで throttle を同時に叩いても順序化される。"""
        noon = dt.datetime(2026, 4, 19, 12, 0, 0).timestamp()
        c = _Clock(noon); s = _Sleep(c)
        r = RateLimiter(min_sleep=3, max_sleep=3, hourly_limit=10_000, _now_fn=c)
        def worker():
            for _ in range(5):
                r.throttle(sleep_fn=s)
        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads: t.start()
        for t in threads: t.join()
        # 2 ワーカー × 5 = 10 回呼んだ。うち 1 回目を除く 9 回で 3 秒待つはず。
        wait_calls = [x for x in s.calls if x >= 2.9]
        self.assertGreaterEqual(len(wait_calls), 9)


# ================================================================
# Tasks 構築
# ================================================================
class TestBuildTasks(unittest.TestCase):
    def test_all_data_types_1day(self):
        ts = build_tasks(dt.date(2025, 8, 1), dt.date(2025, 8, 1),
                         ["odds_trifecta","odds_exacta","odds_quinella","odds_trio","race_meta"])
        # 5 + 2 + 1 + 1 = 9
        self.assertEqual(len(ts), 9)
        # meta は trifecta 全ページ (5 ページ) で書く
        meta_tasks = [t for t in ts if t.write_meta]
        self.assertEqual(len(meta_tasks), 5)
        self.assertTrue(all(t.data_type == "odds_trifecta" for t in meta_tasks))
        self.assertEqual(sorted(t.page for t in meta_tasks), [1, 2, 3, 4, 5])

    def test_meta_only_covers_all_trifecta_pages(self):
        ts = build_tasks(dt.date(2025, 8, 1), dt.date(2025, 8, 1), ["race_meta"])
        # 5 ページすべて取得
        self.assertEqual(len(ts), 5)
        self.assertTrue(all(t.data_type == "odds_trifecta" and t.write_meta for t in ts))
        self.assertEqual(sorted(t.page for t in ts), [1, 2, 3, 4, 5])

    def test_phase_a1_total(self):
        d1, d2 = PHASE_RANGES["A1"]
        ts = build_tasks(d1, d2, ["odds_trifecta","odds_exacta","odds_quinella","odds_trio","race_meta"])
        # 7 日 × 9 = 63
        self.assertEqual(len(ts), 63)


# ================================================================
# CLI
# ================================================================
class TestCli(unittest.TestCase):
    def test_default_parallel_is_2(self):
        args = build_argparser().parse_args([])
        self.assertEqual(args.parallel, 2)
        self.assertEqual(args.sleep_min, 3.0)
        self.assertEqual(args.sleep_max, 5.0)

    def test_phase_presets(self):
        ns = build_argparser().parse_args(["--phase", "A1"])
        d1, d2 = resolve_date_range(ns)
        self.assertEqual((d1, d2), PHASE_RANGES["A1"])

    def test_data_types_all(self):
        self.assertEqual(len(_parse_data_types(None)), 5)

    def test_data_types_subset(self):
        out = _parse_data_types("odds_trifecta,race_meta")
        self.assertEqual(out, ["odds_trifecta", "race_meta"])

    def test_data_types_invalid(self):
        import argparse
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_data_types("bogus")

    def test_missing_range_fails(self):
        # --phase も --from-date/--to-date もないとエラー
        ns = build_argparser().parse_args([])
        with self.assertRaises(SystemExit):
            resolve_date_range(ns)

    def test_from_to_dates(self):
        ns = build_argparser().parse_args(["--from-date","2025-08-01","--to-date","2025-08-03"])
        d1, d2 = resolve_date_range(ns)
        self.assertEqual(d1, dt.date(2025, 8, 1))
        self.assertEqual(d2, dt.date(2025, 8, 3))


# ================================================================
# 並列 run() が DB を触らないこと (dry-run)
# ================================================================
class TestDryRunParallel(unittest.TestCase):
    def test_dry_run_no_progress_file(self):
        import logging
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.json"
            orig = mod.PROGRESS_PATH
            mod.PROGRESS_PATH = path
            try:
                logger = logging.getLogger("dryrun_test")
                logger.addHandler(logging.NullHandler())
                mod.run(
                    date_from=dt.date(2026, 4, 17), date_to=dt.date(2026, 4, 18),
                    data_types=["odds_trifecta","odds_exacta","race_meta"],
                    parallel=2, sleep_min=3.0, sleep_max=5.0,
                    dry_run=True, logger=logger,
                )
            finally:
                mod.PROGRESS_PATH = orig
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
