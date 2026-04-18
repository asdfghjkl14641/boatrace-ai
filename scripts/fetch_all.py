# -*- coding: utf-8 -*-
"""
ボートレースデータ取得メインスクリプト。

■ 何をするか
    - 全24会場 × 指定した日付範囲 のデータを boatrace.jp から取得し、
      Supabase (PostgreSQL) にまとめて保存する。
    - 既にDBにある (date, stadium) はスキップ (重複防止)。
    - 開催日でない会場は自動的にスキップ (get_12races が空ならスキップ)。
    - リクエスト間隔は 1.5 秒以上 (サイトへの負荷軽減)。
    - 進捗バー (tqdm) で状況を表示。
    - エラーが発生しても次のレース/会場に継続。
    - 取得ログは logs/fetch_YYYYMMDD_HHMMSS.log に記録。

■ 使い方の例
    # 昨日1日分 (GitHub Actionsの日次実行で使う形)
    python -m scripts.fetch_all --yesterday

    # 日付範囲を指定
    python -m scripts.fetch_all --start 2026-04-01 --end 2026-04-07

    # 特定の会場だけ (カンマ区切り)
    python -m scripts.fetch_all --start 2026-04-10 --end 2026-04-10 --stadiums 12,24
"""
import argparse
import logging
import os
import re
import sys
import time
import traceback
from datetime import date, datetime, timedelta
from typing import Any, Iterable

from tqdm import tqdm

# 自作スクレイパー (requests + BeautifulSoup)
from scripts.scraper import BoatraceScraper

from scripts.db import get_connection
from scripts.stadiums import ALL_STADIUM_IDS, stadium_name

# ------------------------------------------------------------
# 設定
# ------------------------------------------------------------
REQUEST_INTERVAL_SEC = 1.5          # 各HTTPリクエスト間の待機秒数 (絶対に短くしないこと)
MAX_RACES_PER_DAY = 12              # 1開催日あたりの最大レース数 (通常12)
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


# ------------------------------------------------------------
# ロギング設定 (ファイル + 標準出力)
# ------------------------------------------------------------
def setup_logging() -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(LOG_DIR, f"fetch_{ts}.log")

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # 既存ハンドラをクリア (再実行時の重複防止)
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
# ユーティリティ: タイム文字列 → 秒
# ------------------------------------------------------------
_TIME_RE = re.compile(r"(\d+)'(\d+)\"(\d)")


def time_to_sec(t: Any) -> float | None:
    """例: "1'50\"6" → 110.6 秒"""
    if t is None:
        return None
    s = str(t)
    m = _TIME_RE.search(s)
    if not m:
        return None
    mm, ss, ds = map(int, m.groups())
    return float(mm * 60 + ss + ds / 10.0)


def to_int(v: Any) -> int | None:
    try:
        if v is None or v == "":
            return None
        return int(float(v))
    except (ValueError, TypeError):
        return None


def to_float(v: Any) -> float | None:
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (ValueError, TypeError):
        return None


def to_bool(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "有", "on"):
        return True
    if s in ("false", "0", "no", "無", "off"):
        return False
    return None


# ------------------------------------------------------------
# 日付範囲の列挙
# ------------------------------------------------------------
def daterange(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


# ------------------------------------------------------------
# DB側: 既取得の (date, stadium) を取得
# ------------------------------------------------------------
def load_existing_keys(conn) -> set[tuple[date, int]]:
    """race_cards に既に登録済みの (date, stadium) の集合を返す。
    fetch_all.py 実行時の重複スキップに使う。"""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT date, stadium FROM race_cards;")
        rows = cur.fetchall()
    return {(r[0], r[1]) for r in rows}


# ------------------------------------------------------------
# DB側: 1会場分のデータをまとめてINSERT
# ------------------------------------------------------------
def bulk_insert(conn, table: str, columns: list[str], rows: list[dict]) -> int:
    """rows を table にバッチINSERTする (psycopg v3)。
    UNIQUE制約に衝突した行は ON CONFLICT DO NOTHING でスキップ。
    戻り値: executemany の rowcount 合計 (参考値)。
    """
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
        # psycopg3 は executemany をパイプラインで高速実行する
        cur.executemany(sql, values)
        inserted = cur.rowcount
    return inserted


# ------------------------------------------------------------
# 1レース分のデータ取得
#   BoatraceScraper が内部で 1.5秒以上の間隔を自動確保するため、
#   ここでの time.sleep は不要。失敗したセクションは None を入れて継続。
# ------------------------------------------------------------
def fetch_race(api: BoatraceScraper, d: date, stadium: int, race: int) -> dict:
    """1レース分の生データを取得して辞書で返す。
    キー: info / result / conditions / trifecta / winplace
    """
    out: dict[str, Any] = {}

    # A. 出走表
    try:
        out["info"] = api.get_race_info(d, stadium, race)
    except Exception as e:
        logging.warning(f"  [出走表 R{race}] 取得エラー: {e}")
        out["info"] = None

    # B. 直前情報
    try:
        out["conditions"] = api.get_just_before_info(d, stadium, race)
    except Exception as e:
        logging.warning(f"  [直前情報 R{race}] 取得エラー: {e}")
        out["conditions"] = None

    # C. 結果 (まだ締切前なら取得できない)
    try:
        out["result"] = api.get_race_result(d, stadium, race)
    except Exception as e:
        logging.warning(f"  [結果 R{race}] 取得エラー: {e}")
        out["result"] = None

    # D. 3連単オッズ
    try:
        out["trifecta"] = api.get_odds_trifecta(d, stadium, race)
    except Exception as e:
        logging.warning(f"  [3連単 R{race}] 取得エラー: {e}")
        out["trifecta"] = None

    # E. 単勝・複勝オッズ (boatrace.jpでは同じページに両方あるので1回で取得)
    try:
        out["winplace"] = api.get_odds_win_place_show(d, stadium, race)
    except Exception as e:
        logging.warning(f"  [単複オッズ R{race}] 取得エラー: {e}")
        out["winplace"] = None

    return out


# ------------------------------------------------------------
# 取得したレース生データを正規化して行リストにする
# ------------------------------------------------------------
def normalize(d: date, stadium: int, race: int, raw: dict) -> dict[str, list[dict]]:
    s_name = stadium_name(stadium)
    cards_rows: list[dict] = []
    series_rows: list[dict] = []
    results_rows: list[dict] = []
    cond_rows: list[dict] = []
    tri_rows: list[dict] = []
    win_rows: list[dict] = []
    place_rows: list[dict] = []

    # --- race_cards & current_series ---
    info = raw.get("info") or {}
    for lane in range(1, 7):
        p = info.get(f"boat{lane}")
        if not p:
            continue
        cards_rows.append({
            "date": d,
            "stadium": stadium,
            "stadium_name": s_name,
            "race_number": race,
            "lane": lane,
            "racerid": to_int(p.get("racerid")),
            "name": p.get("name"),
            "class": p.get("class"),
            "branch": p.get("branch"),
            "birthplace": p.get("birthplace"),
            "age": to_int(p.get("age")),
            "weight": to_float(p.get("weight")),
            "f": to_int(p.get("F")),
            "l": to_int(p.get("L")),
            "aveST": to_float(p.get("aveST")),
            "global_win_pt": to_float(p.get("global_win_pt")),
            "global_in2nd": to_float(p.get("global_in2nd")),
            "global_in3rd": to_float(p.get("global_in3rd")),
            "local_win_pt": to_float(p.get("local_win_pt")),
            "local_in2nd": to_float(p.get("local_in2nd")),
            "local_in3rd": to_float(p.get("local_in3rd")),
            "motor": to_int(p.get("motor")),
            "motor_in2nd": to_float(p.get("motor_in2nd")),
            "motor_in3rd": to_float(p.get("motor_in3rd")),
            "boat": to_int(p.get("boat")),
            "boat_in2nd": to_float(p.get("boat_in2nd")),
            "boat_in3rd": to_float(p.get("boat_in3rd")),
        })

        # 今節成績 (result リストを正規化)
        racer_id = to_int(p.get("racerid"))
        for res in (p.get("result") or []):
            if not res or not any(res.values()):
                continue
            series_rows.append({
                "date": d,
                "stadium": stadium,
                "stadium_name": s_name,
                "racerid": racer_id,
                "race_number": race,
                "boat_number": lane,
                "course": to_int(res.get("course")),
                "ST": to_float(res.get("ST")),
                "rank": to_int(res.get("rank")),
            })

    # --- race_results ---
    result = raw.get("result") or {}
    for r in (result.get("result") or []):
        boat_no = to_int(r.get("boat") or r.get("boat_number"))
        if boat_no is None:
            continue
        t_str = r.get("time")
        results_rows.append({
            "date": d,
            "stadium": stadium,
            "stadium_name": s_name,
            "race_number": race,
            "rank": to_int(r.get("rank")),
            "boat": boat_no,
            "racerid": to_int(r.get("racerid")),
            "name": r.get("name"),
            "time": t_str,
            "time_sec": time_to_sec(t_str),
        })

    # --- race_conditions ---
    cond = raw.get("conditions")
    if cond:
        w = cond.get("weather_information") or {}
        row = {
            "date": d,
            "stadium": stadium,
            "stadium_name": s_name,
            "race_number": race,
            "weather": w.get("weather"),
            "temperature": to_float(w.get("temperature")),
            "wind_direction": to_int(w.get("wind_direction")),
            "wind_speed": to_float(w.get("wind_speed")),
            "water_temperature": to_float(w.get("water_temperature")),
            "wave_height": to_float(w.get("wave_height")),
            "stabilizer": to_bool(cond.get("stabilizer"))
                         if cond.get("stabilizer") is not None
                         else ("安定板" in str(cond)),
        }
        for lane in range(1, 7):
            b = cond.get(f"boat{lane}") or {}
            row[f"display_time_{lane}"] = to_float(b.get("display_time"))
        cond_rows.append(row)

    # --- trifecta_odds ---
    tri = raw.get("trifecta") or {}
    for combo, v in tri.items():
        tri_rows.append({
            "date": d,
            "stadium": stadium,
            "stadium_name": s_name,
            "race_number": race,
            "combination": str(combo),
            "odds": to_float(v),
        })

    # --- win_odds / place_odds ---
    wp = raw.get("winplace") or {}
    win = wp.get("win") or {}
    place = wp.get("place_show") or wp.get("place") or {}
    for k, v in win.items():
        # vは数値または{'odds': 数値}の場合があるので両対応
        odds = v.get("odds") if isinstance(v, dict) else v
        win_rows.append({
            "date": d,
            "stadium": stadium,
            "stadium_name": s_name,
            "race_number": race,
            "boat_number": to_int(k),
            "odds": to_float(odds),
        })
    for k, v in place.items():
        odds = v.get("odds") if isinstance(v, dict) else v
        # 複勝は [下限, 上限] の配列のことがあるので、平均を保存
        if isinstance(odds, (list, tuple)) and odds:
            nums = [to_float(x) for x in odds if to_float(x) is not None]
            odds_val = sum(nums) / len(nums) if nums else None
        else:
            odds_val = to_float(odds)
        place_rows.append({
            "date": d,
            "stadium": stadium,
            "stadium_name": s_name,
            "race_number": race,
            "boat_number": to_int(k),
            "odds": odds_val,
        })

    return {
        "race_cards": cards_rows,
        "current_series": series_rows,
        "race_results": results_rows,
        "race_conditions": cond_rows,
        "trifecta_odds": tri_rows,
        "win_odds": win_rows,
        "place_odds": place_rows,
    }


# ------------------------------------------------------------
# 各テーブルのカラム順 (bulk_insert で使用)
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
    "trifecta_odds": [
        "date", "stadium", "stadium_name", "race_number", "combination", "odds",
    ],
    "win_odds": [
        "date", "stadium", "stadium_name", "race_number", "boat_number", "odds",
    ],
    "place_odds": [
        "date", "stadium", "stadium_name", "race_number", "boat_number", "odds",
    ],
    "current_series": [
        "date", "stadium", "stadium_name", "racerid", "race_number",
        "boat_number", "course", "ST", "rank",
    ],
}


# ------------------------------------------------------------
# 1会場1日分の処理
# ------------------------------------------------------------
def process_stadium_day(api: BoatraceScraper, d: date, stadium: int) -> dict[str, int]:
    """1会場1日分を取得してSupabaseに保存する。
    スクレイピング中はDB接続を張らず、INSERTの直前に開いて直後に閉じる。
    (Supabase のアイドルタイムアウトで切れるのを避けるため)"""
    s_name = stadium_name(stadium)
    logging.info(f"▶ {d} / {stadium:02d} {s_name} の取得開始")

    # 1) 開催確認 (スクレイパー内部で1.5秒間隔を確保)
    try:
        races = api.get_12races(d, stadium)
    except Exception as e:
        logging.warning(f"  開催情報取得失敗 → スキップ: {e}")
        return {}

    if not races:
        logging.info(f"  開催なし → スキップ")
        return {}

    # レース番号の候補 (races が dict なら key、list/int なら1〜12)
    if isinstance(races, dict):
        race_numbers = []
        for k in races.keys():
            # "1R", "race1", "1" 等から数字を抽出
            m = re.search(r"\d+", str(k))
            if m:
                race_numbers.append(int(m.group()))
        race_numbers = sorted(set(race_numbers))
    else:
        race_numbers = list(range(1, MAX_RACES_PER_DAY + 1))

    if not race_numbers:
        race_numbers = list(range(1, MAX_RACES_PER_DAY + 1))

    # 2) 全レース分を取得 & 正規化 (この段階ではDB接続は不要)
    buckets: dict[str, list[dict]] = {k: [] for k in COLUMNS.keys()}
    for r in race_numbers:
        try:
            raw = fetch_race(api, d, stadium, r)
            rows = normalize(d, stadium, r, raw)
            for k, v in rows.items():
                buckets[k].extend(v)
        except Exception as e:
            logging.error(f"  R{r} 処理中に想定外エラー: {e}")
            logging.error(traceback.format_exc())

    # 3) DBへまとめてINSERT (接続は毎回開いて即閉じる)
    inserted: dict[str, int] = {}
    conn = get_connection()
    try:
        for table, rows in buckets.items():
            try:
                n = bulk_insert(conn, table, COLUMNS[table], rows)
                inserted[table] = n
            except Exception as e:
                logging.error(f"  {table} INSERT失敗: {e}")
                try:
                    conn.rollback()
                except Exception:
                    pass
                inserted[table] = 0
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass

    logging.info(
        f"  ✔ {d} / {stadium:02d} {s_name}: "
        + " / ".join(f"{k}={v}" for k, v in inserted.items())
    )
    return inserted


# ------------------------------------------------------------
# コマンドライン引数
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ボートレースデータ一括取得 (Supabase保存)"
    )
    p.add_argument("--start", help="取得開始日 YYYY-MM-DD")
    p.add_argument("--end", help="取得終了日 YYYY-MM-DD (省略時は --start と同じ)")
    p.add_argument("--yesterday", action="store_true",
                   help="昨日1日分のみ取得 (--start/--end より優先)")
    p.add_argument("--stadiums", default="",
                   help="会場IDをカンマ区切りで指定 (例: 12,24)。省略時は全24会場。")
    p.add_argument("--force", action="store_true",
                   help="既にDBにあっても再取得する (通常は不要)")
    return p.parse_args()


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main() -> int:
    args = parse_args()
    logfile = setup_logging()

    # 日付範囲の確定
    if args.yesterday:
        end = date.today() - timedelta(days=1)
        start = end
    else:
        if not args.start:
            logging.error("--start または --yesterday を指定してください")
            return 2
        start = parse_date(args.start)
        end = parse_date(args.end) if args.end else start

    if start > end:
        logging.error(f"--start ({start}) が --end ({end}) より後です")
        return 2

    # 対象会場
    if args.stadiums.strip():
        stadium_ids = [int(x) for x in args.stadiums.split(",") if x.strip()]
    else:
        stadium_ids = ALL_STADIUM_IDS

    logging.info("=" * 60)
    logging.info(f"取得範囲: {start} 〜 {end}  /  会場: {stadium_ids}")
    logging.info("=" * 60)

    api = BoatraceScraper(min_interval=REQUEST_INTERVAL_SEC)

    # 既取得キーの取得は短時間で済むので、接続をすぐ閉じる
    if args.force:
        existing = set()
    else:
        conn = get_connection()
        try:
            existing = load_existing_keys(conn)
        finally:
            conn.close()
    logging.info(f"既取得済みの (date, stadium) 組合せ: {len(existing)} 件")

    # 進捗バー用に (date, stadium) のペアを列挙
    pairs = [(d, s) for d in daterange(start, end) for s in stadium_ids]

    totals: dict[str, int] = {k: 0 for k in COLUMNS.keys()}
    skipped = 0
    processed = 0
    errors = 0

    with tqdm(pairs, desc="取得中", unit="stadium-day") as bar:
        for d, stadium in bar:
            bar.set_postfix_str(f"{d} / {stadium:02d} {stadium_name(stadium)}")
            if (d, stadium) in existing:
                skipped += 1
                continue
            try:
                # DB接続はINSERT直前に張る (process_stadium_day内)
                result = process_stadium_day(api, d, stadium)
                if result:
                    processed += 1
                    for k, v in result.items():
                        totals[k] = totals.get(k, 0) + v
            except Exception as e:
                errors += 1
                logging.error(f"✖ {d} / {stadium} で致命的エラー: {e}")
                logging.error(traceback.format_exc())

    # サマリー出力
    logging.info("=" * 60)
    logging.info("取得完了サマリー")
    logging.info(f"  処理済み:   {processed} 会場日")
    logging.info(f"  スキップ:   {skipped} 会場日 (既取得/非開催)")
    logging.info(f"  エラー:     {errors} 会場日")
    logging.info(f"  INSERT件数:")
    for k, v in totals.items():
        logging.info(f"    {k:20s} {v:>8d}")
    logging.info(f"  ログ: {logfile}")
    logging.info("=" * 60)

    # エラーがあった場合でも正常終了 (ログから確認できる)。
    # 致命的な設定ミスなどは上の例外で既に落ちている。
    return 0


if __name__ == "__main__":
    sys.exit(main())
