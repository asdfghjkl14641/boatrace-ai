# -*- coding: utf-8 -*-
"""
boatrace.jp 公式サイト用 自作スクレイパー。

PyJPBoatrace の代替として、同じメソッド名で呼び出せるよう設計している:
    - get_12races(date, stadium)           開催確認 (空なら未開催)
    - get_race_info(date, stadium, race)   出走表
    - get_race_result(date, stadium, race) レース結果
    - get_just_before_info(date, stadium, race)  直前情報
    - get_odds_trifecta(date, stadium, race)     3連単オッズ
    - get_odds_win_place_show(date, stadium, race)  単勝・複勝オッズ

注意:
    - リクエスト間隔は min_interval (デフォルト1.5秒) 以上を自動で確保する。
    - HTML構造は将来変わる可能性あり。パース失敗時は None を埋めて継続する。
"""
from __future__ import annotations

import re
import time
import warnings
from datetime import date as date_cls
from typing import Any

import requests
from bs4 import BeautifulSoup
try:
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except ImportError:
    pass


BASE_URL = "https://www.boatrace.jp/owpc/pc/race"

# 公式サイトに負荷をかけないよう、ブラウザ相当のUAを付ける
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}


def _fmt_date(d: date_cls) -> str:
    return d.strftime("%Y%m%d")


def _fmt_stadium(s: int) -> str:
    return f"{s:02d}"


# ------------------------------------------------------------
# 小さなパーサ関数群 (いずれも失敗しても例外を投げず、該当値を None にする)
# ------------------------------------------------------------
def _first_match(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text)
    return m.group(1) if m else None


def _extract_numbers(text: str) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", text or "")


def _td_text(td) -> str:
    return td.get_text("\n", strip=True) if td else ""


# ------------------------------------------------------------
# メインクラス
# ------------------------------------------------------------
class BoatraceScraper:
    """boatrace.jp を直接スクレイピングしてデータを返す。"""

    def __init__(self, min_interval: float = 1.5, timeout: int = 30):
        self.min_interval = min_interval
        self.timeout = timeout
        self._last_req = 0.0
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # ------------------------------------------------------------
    # HTTPリクエスト (1.5秒以上の間隔を自動確保)
    # ------------------------------------------------------------
    def _throttle(self) -> None:
        elapsed = time.time() - self._last_req
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_req = time.time()

    def _get_soup(self, path: str, **params) -> BeautifulSoup:
        self._throttle()
        url = f"{BASE_URL}/{path}"
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        # 明示的にUTF-8を指定 (ページによってはcharset指定が欠けているため)
        resp.encoding = "utf-8"
        return BeautifulSoup(resp.text, "lxml")

    # ============================================================
    # 1. 開催確認
    # ============================================================
    def get_12races(self, d: date_cls, stadium: int) -> dict:
        """その日・その会場で開催しているか判定。
        開催していれば {"1R": {}, ..., "12R": {}} を返す。
        未開催なら {} (空dict) を返す。
        fetch_all.py は空なら skip する。
        """
        try:
            soup = self._get_soup(
                "racelist", rno=1,
                jcd=_fmt_stadium(stadium), hd=_fmt_date(d),
            )
        except requests.RequestException:
            return {}

        text = soup.get_text()
        if "開催されておりません" in text or "本日の発売はありません" in text:
            return {}

        # 出走表テーブル (選手情報のtbody) が見つかれば開催中とみなす
        rows = soup.select("table tbody tr")
        if not rows:
            return {}
        return {f"{i}R": {} for i in range(1, 13)}

    # ============================================================
    # 2. 出走表 (racelist)
    # ============================================================
    def get_race_info(self, d: date_cls, stadium: int, race: int) -> dict:
        """出走表ページ (racelist) をパース。
        戻り値: {"boat1": {...}, ..., "boat6": {...}}
        各選手dictのキー:
          racerid, name, class, branch, birthplace, age, weight,
          F, L, aveST,
          global_win_pt, global_in2nd, global_in3rd,
          local_win_pt,  local_in2nd,  local_in3rd,
          motor, motor_in2nd, motor_in3rd,
          boat,  boat_in2nd,  boat_in3rd,
          result  -> 今節成績のリスト [{"course","ST","rank"}, ...]
        """
        soup = self._get_soup(
            "racelist", rno=race,
            jcd=_fmt_stadium(stadium), hd=_fmt_date(d),
        )

        out: dict[str, Any] = {}
        # 選手のtbodyだけを選ぶ (class="is-fs12" が付いているのが各選手)。
        # ページの先頭には「締切予定時刻」のtbodyがあり、それを除外するため。
        tbodies = soup.select("tbody.is-fs12")

        for idx, tb in enumerate(tbodies[:6], start=1):
            out[f"boat{idx}"] = self._parse_racer_tbody(tb)
        return out

    def _parse_racer_tbody(self, tbody) -> dict:
        """選手1名分のtbody (<tr>×4) をdictに整形。

        boatrace.jp の実際のDOM構造 (2026年4月時点):
            tr[0]:
              td[0]  枠番                               (class=is-boatColor{lane})
              td[1]  写真+<a href="...?toban=NNNN">    (→ racerid)
              td[2]  <div>×3:
                       div[0] "NNNN / <span>X1</span>" (登番/級別)
                       div[1] "氏名" (全角空白含む)
                       div[2] "支部/出身地<br>NN歳/WW.Wkg"
              td[3]  "F1 <br>L0 <br>0.16" (F/L/平均ST)
              td[4]  全国: 勝率/2連率/3連率
              td[5]  当地: 勝率/2連率/3連率
              td[6]  モーター番号/2連率/3連率
              td[7]  ボート番号/2連率/3連率
              td[8]  空白 (rowspan=4 のプレースホルダ)
              td[9:] 今節成績の 1行目 = レース番号、class=is-boatColor{course}
            tr[1] td[:] : 今節成績 2行目 = 着順 (数字のみ)
            tr[2] td[:] : 今節成績 3行目 = ST (例 .21)
            tr[3] td[:] : 今節成績 4行目 = レース番号の再表示リンク (未使用)
        """
        p: dict[str, Any] = {
            "racerid": None, "name": None, "class": None,
            "branch": None, "birthplace": None, "age": None, "weight": None,
            "F": None, "L": None, "aveST": None,
            "global_win_pt": None, "global_in2nd": None, "global_in3rd": None,
            "local_win_pt": None, "local_in2nd": None, "local_in3rd": None,
            "motor": None, "motor_in2nd": None, "motor_in3rd": None,
            "boat": None, "boat_in2nd": None, "boat_in3rd": None,
            "result": [],
        }
        trs = tbody.find_all("tr", recursive=False)
        if not trs:
            return p
        tds = trs[0].find_all("td", recursive=False)

        # ---- td[1] : プロフィールリンクの toban=NNNN から racerid を取得 ----
        if len(tds) > 1:
            a = tds[1].find("a", href=True)
            if a:
                m = re.search(r"toban=(\d+)", a["href"])
                if m:
                    p["racerid"] = m.group(1)

        # ---- td[2] : 3つの <div> (登番/級別、氏名、支部・出身地・年齢・体重) ----
        if len(tds) > 2:
            divs = tds[2].find_all("div", recursive=False)

            # div[0]: "5325 / B1" のような形。空白や改行を正規化してから正規表現。
            if len(divs) > 0:
                raw = re.sub(r"\s+", " ", divs[0].get_text(" ", strip=True))
                m = re.match(r"(\d+)\s*/\s*(\w+)", raw)
                if m:
                    if not p["racerid"]:
                        p["racerid"] = m.group(1)
                    p["class"] = m.group(2)

            # div[1]: 氏名 (全角空白 \u3000 で苗字と名前が区切られる場合あり)
            if len(divs) > 1:
                name = divs[1].get_text(" ", strip=True)
                # 全角/半角空白の連続を半角1個に統一
                name = re.sub(r"[\u3000 ]+", " ", name).strip()
                p["name"] = name or None

            # div[2]: "大阪/大阪<br>22歳/52.0kg" → get_text('\n') で改行区切りに
            if len(divs) > 2:
                text = divs[2].get_text("\n", strip=True)
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                # lines[0] : "支部/出身地"
                if len(lines) >= 1:
                    parts = lines[0].split("/")
                    if parts:
                        p["branch"] = parts[0].strip() or None
                    if len(parts) >= 2:
                        p["birthplace"] = parts[1].strip() or None
                # lines[1] : "22歳/52.0kg"
                if len(lines) >= 2:
                    p["age"] = _first_match(r"(\d+)\s*歳", lines[1])
                    p["weight"] = _first_match(r"([\d.]+)\s*kg", lines[1])

        # ---- td[3] : F/L/平均ST (<br>区切り) ----
        if len(tds) > 3:
            text = tds[3].get_text("\n", strip=True)
            for line in (l.strip() for l in text.split("\n") if l.strip()):
                m = re.match(r"F\s*(\d+)$", line)
                if m:
                    p["F"] = m.group(1); continue
                m = re.match(r"L\s*(\d+)$", line)
                if m:
                    p["L"] = m.group(1); continue
                m = re.match(r"\d+\.\d+$", line)
                if m and p["aveST"] is None:
                    p["aveST"] = line

        # ---- td[4..7] : 各セルに数値が3個ずつ (勝率/2連率/3連率 or 番号/2連率/3連率) ----
        for idx, keys in (
            (4, ("global_win_pt", "global_in2nd", "global_in3rd")),
            (5, ("local_win_pt",  "local_in2nd",  "local_in3rd")),
            (6, ("motor", "motor_in2nd", "motor_in3rd")),
            (7, ("boat",  "boat_in2nd",  "boat_in3rd")),
        ):
            if len(tds) > idx:
                nums = _extract_numbers(tds[idx].get_text("\n", strip=True))
                if len(nums) >= 3:
                    p[keys[0]], p[keys[1]], p[keys[2]] = nums[:3]

        # ---- 今節成績: tr[0] td[9:] (レース番号+コース) + tr[1] (着順) + tr[2] (ST) ----
        # tr[0] は末尾に「今日の次レースへのリンク」列が追加されるため、
        # tr[1]/tr[2] と長さを揃える (min) ことで余分な列を除外する。
        day_tds_0 = tds[9:] if len(tds) > 9 else []
        tds_rank = trs[1].find_all("td", recursive=False) if len(trs) > 1 else []
        tds_st   = trs[2].find_all("td", recursive=False) if len(trs) > 2 else []
        n_days = min(len(day_tds_0), len(tds_rank), len(tds_st))
        day_tds_0 = day_tds_0[:n_days]

        for i, td0 in enumerate(day_tds_0):
            race_num = td0.get_text(strip=True)
            if not race_num:
                continue  # その日は未出走 (空欄)
            # class="is-boatColor{N}" の N が進入コース
            cls = " ".join(td0.get("class") or [])
            m = re.search(r"is-boatColor(\d)", cls)
            course = m.group(1) if m else None

            rank = tds_rank[i].get_text(strip=True) if i < len(tds_rank) else ""
            st   = tds_st[i].get_text(strip=True)   if i < len(tds_st)   else ""

            p["result"].append({
                "race_number": race_num,
                "course": course,
                "ST":   st   or None,
                "rank": rank or None,
            })
        return p

    # ============================================================
    # 3. レース結果 (raceresult)
    # ============================================================
    # 実HTML構造 (2026年4月確認):
    #   table[1] (class=is-w495) が着順表。7trs (1ヘッダ + 6着順行)。
    #   各着順行:
    #     td[0] : 着順 (全角数字 "１" や、"Ｆ"=フライング、"失"=失格)
    #     td[1] : 艇番 (半角 1〜6, class=is-boatColor{N})
    #     td[2] : "4738 清埜 翔子" (登番 + 半角スペース + 氏名)
    #     td[3] : レースタイム (例 1'49"3。失格時は "." や空欄)
    # ============================================================
    _FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")

    def get_race_result(self, d: date_cls, stadium: int, race: int) -> dict:
        """結果ページをパース。
        戻り値: {"result": [{"rank","boat","racerid","name","time"}, ...]}
        フライング・失格で着順が数字でない場合、rank は None になる。
        """
        soup = self._get_soup(
            "raceresult", rno=race,
            jcd=_fmt_stadium(stadium), hd=_fmt_date(d),
        )
        result: list[dict] = []
        # 着順表は「着 / 枠 / ボートレーサー / レースタイム」の4カラム固定。
        for table in soup.select("table.is-w495"):
            trs = table.find_all("tr")
            # ヘッダ行に "ボートレーサー" を含むものが目的のテーブル
            hdr_text = trs[0].get_text(" ", strip=True) if trs else ""
            if "ボートレーサー" not in hdr_text or "レースタイム" not in hdr_text:
                continue

            for tr in trs[1:]:
                tds = tr.find_all("td", recursive=False)
                if len(tds) < 4:
                    continue
                rank_raw = _td_text(tds[0]).translate(self._FULLWIDTH_DIGITS)
                boat_text = _td_text(tds[1])
                # 艇番が 1〜6 で無い行は無視 (返還艇の欄など)
                if not re.match(r"^[1-6]$", boat_text):
                    continue
                # "4738 清埜 翔子" → 先頭数字=登番、残り=氏名
                racer_cell = re.sub(r"\s+", " ",
                                    _td_text(tds[2]).replace("\u3000", " ")).strip()
                racer_id = None
                name = None
                m = re.match(r"(\d+)\s+(.+)$", racer_cell)
                if m:
                    racer_id = m.group(1)
                    name = m.group(2).strip()
                time_str = _td_text(tds[3]) or None
                # 時間らしくない文字 (".", "", "フライング" 等) は None に
                if time_str and not re.search(r"\d", time_str):
                    time_str = None
                # 着順は数字 (1〜6) だけを採用。フライング/失格等は None。
                rank = _first_match(r"(\d+)", rank_raw)

                result.append({
                    "rank": rank,
                    "boat": boat_text,
                    "racerid": racer_id,
                    "name": name,
                    "time": time_str,
                })
            if result:
                break
        return {"result": result}

    # ============================================================
    # 4. 直前情報 (beforeinfo)
    # ============================================================
    # 実HTML構造 (2026年4月確認):
    #   .weather1 配下の各ユニットが以下のクラスで区別される
    #     .is-direction        …気温 (例 "18.0℃")
    #     .is-weather          …天気 (<p class=is-weather{1-5}> + ラベル "曇り")
    #     .is-wind             …風速 (例 "1m")
    #     .is-windDirection    …風向 (<p class=is-wind{1-16}>)
    #     .is-waterTemperature …水温
    #     .is-wave             …波高 (例 "1cm")
    #
    #   展示タイムは <table class=is-w748> 内の <tbody class=is-fs12> × 6 の
    #   各 tr[0] の td[4] に入っている (例 "6.80")。
    # ============================================================
    # 天気アイコンのコード → 文字列
    _WEATHER_MAP = {"1": "晴", "2": "曇り", "3": "雨", "4": "雪", "5": "霧"}

    def get_just_before_info(self, d: date_cls, stadium: int, race: int) -> dict:
        """直前情報ページをパース。
        戻り値:
          {
            "weather_information": {
              weather, temperature, wind_direction, wind_speed,
              water_temperature, wave_height
            },
            "stabilizer": True/False,
            "boat1": {"display_time": "6.80"}, ...
          }
        wind_direction は boatrace.jp の風向アイコンの数値 (1〜16の方位コード)。
        """
        soup = self._get_soup(
            "beforeinfo", rno=race,
            jcd=_fmt_stadium(stadium), hd=_fmt_date(d),
        )

        # --- 気象情報 ---
        w: dict[str, Any] = {
            "weather": None, "temperature": None,
            "wind_direction": None, "wind_speed": None,
            "water_temperature": None, "wave_height": None,
        }
        wb = soup.select_one(".weather1")
        if wb:
            # 気温: .is-direction 内の Data
            unit = wb.select_one(".weather1_bodyUnit.is-direction")
            if unit:
                data = unit.select_one(".weather1_bodyUnitLabelData")
                if data:
                    w["temperature"] = _first_match(r"([-\d.]+)", data.get_text(strip=True))

            # 天気: .is-weather 内の画像クラス is-weatherN と、ラベル
            unit = wb.select_one(".weather1_bodyUnit.is-weather")
            if unit:
                icon = unit.select_one("[class*='is-weather']")
                if icon:
                    m = re.search(r"is-weather(\d+)", " ".join(icon.get("class", [])))
                    if m:
                        w["weather"] = self._WEATHER_MAP.get(m.group(1))
                # アイコン由来が無ければラベル文字列を使用
                if not w["weather"]:
                    lab = unit.select_one(".weather1_bodyUnitLabelTitle")
                    if lab:
                        w["weather"] = lab.get_text(strip=True) or None

            # 風速: .is-wind の Data (例 "1m")
            unit = wb.select_one(".weather1_bodyUnit.is-wind")
            if unit:
                data = unit.select_one(".weather1_bodyUnitLabelData")
                if data:
                    w["wind_speed"] = _first_match(r"([\d.]+)", data.get_text(strip=True))

            # 風向: .is-windDirection の画像クラス is-windN (N=1..16)
            unit = wb.select_one(".weather1_bodyUnit.is-windDirection")
            if unit:
                icon = unit.select_one("[class*='is-wind']")
                if icon:
                    m = re.search(r"is-wind(\d+)", " ".join(icon.get("class", [])))
                    if m:
                        w["wind_direction"] = m.group(1)

            # 水温
            unit = wb.select_one(".weather1_bodyUnit.is-waterTemperature")
            if unit:
                data = unit.select_one(".weather1_bodyUnitLabelData")
                if data:
                    w["water_temperature"] = _first_match(r"([-\d.]+)", data.get_text(strip=True))

            # 波高 (cm単位で表示されている)
            unit = wb.select_one(".weather1_bodyUnit.is-wave")
            if unit:
                data = unit.select_one(".weather1_bodyUnitLabelData")
                if data:
                    w["wave_height"] = _first_match(r"([\d.]+)", data.get_text(strip=True))

        out: dict[str, Any] = {"weather_information": w}

        # --- 安定板 ---
        # 安定板使用時は本文に "安定板使用" の表記が現れる。無使用時は何もない。
        out["stabilizer"] = ("安定板使用" in soup.get_text())

        # --- 展示タイム (6艇分) ---
        # 選手tbodyは is-fs12 クラスでフィルタ
        tbodies = soup.select("tbody.is-fs12")
        for idx in range(1, 7):
            out[f"boat{idx}"] = {"display_time": None}
        for idx, tb in enumerate(tbodies[:6], start=1):
            tr = tb.find("tr", recursive=False)
            if not tr:
                continue
            tds = tr.find_all("td", recursive=False)
            # td[4] が展示タイム (例 "6.80")。形式外ならフォールバックで全td走査。
            disp = None
            if len(tds) > 4:
                cand = _td_text(tds[4])
                if re.fullmatch(r"\d+\.\d{2}", cand):
                    disp = cand
            if disp is None:
                for td in tds:
                    t = _td_text(td)
                    if re.fullmatch(r"\d\.\d{2}", t) or re.fullmatch(r"[567]\.\d{2}", t):
                        disp = t
                        break
            out[f"boat{idx}"] = {"display_time": disp}
        return out

    # ============================================================
    # 5. 3連単オッズ (odds3t)
    # ============================================================
    # 実HTML構造 (2026年4月確認):
    #   <table> 内に 20 データ行 (1着=1,2,3,4,5,6 の6列) が並ぶ。
    #   td.oddsPoint が120個、行走査の順で並ぶ。
    #   各行 i・列 w の組合せは combos_for_w(w)[i]。
    # ============================================================
    @staticmethod
    def _build_trifecta_combos() -> list[str]:
        """boatrace.jp の td.oddsPoint が出現する順序の120通りを返す。
        並び: 行 i=0..19 × 列 w=1..6、各 (w) ごとに
        [2着 x ≠ w] × [3着 y ≠ w,x] の順で20通り。
        """
        by_w: list[list[str]] = []
        for w in range(1, 7):
            combos_w = []
            for x in range(1, 7):
                if x == w:
                    continue
                for y in range(1, 7):
                    if y == w or y == x:
                        continue
                    combos_w.append(f"{w}-{x}-{y}")
            by_w.append(combos_w)  # 5*4=20
        ordered: list[str] = []
        for i in range(20):
            for w in range(6):
                ordered.append(by_w[w][i])
        return ordered  # 120

    def get_odds_trifecta(self, d: date_cls, stadium: int, race: int) -> dict:
        """3連単オッズページをパース。
        戻り値: {"1-2-3": "19.7", "2-1-3": "108.7", ...}  (全120通り)
        """
        soup = self._get_soup(
            "odds3t", rno=race,
            jcd=_fmt_stadium(stadium), hd=_fmt_date(d),
        )
        out: dict[str, str] = {}
        cells = soup.select("td.oddsPoint")
        combos = self._build_trifecta_combos()
        if len(cells) >= len(combos):
            for combo, cell in zip(combos, cells[: len(combos)]):
                out[combo] = cell.get_text(strip=True)
        else:
            # セル数が不足する異常時は取れた分だけ詰める
            for combo, cell in zip(combos, cells):
                out[combo] = cell.get_text(strip=True)
        return out

    # ============================================================
    # 6. 単勝・複勝オッズ (oddstf)
    # ============================================================
    # 実HTML構造 (2026年4月確認):
    #   table[1] (is-w495) = 単勝、table[2] (is-w495) = 複勝
    #   各テーブルは 7tr (ヘッダ + 6艇分)。
    #   tr 内: td[0]=艇番, td[1]=氏名, td[2]=odds (class=oddsPoint)
    #   複勝は "1.1-2.6" のような下限-上限の範囲表記。
    # ============================================================
    def get_odds_win_place_show(
        self, d: date_cls, stadium: int, race: int
    ) -> dict:
        """単勝・複勝オッズページをパース。
        戻り値:
          {
            "win":   {"1": "1.1",    ..., "6": "0.0"},
            "place_show": {"1": ["1.1", "2.6"], ..., "6": ["3.1", "9.0"]}
          }
        win の値は文字列 (特払い時 "0.0" や "-" もある)。
        place_show は [下限, 上限] の配列。単一値なら1要素。
        """
        soup = self._get_soup(
            "oddstf", rno=race,
            jcd=_fmt_stadium(stadium), hd=_fmt_date(d),
        )
        win: dict[str, Any] = {}
        place: dict[str, Any] = {}
        tables = soup.select("table.is-w495")

        if len(tables) >= 1:
            for tr in tables[0].find_all("tr"):
                tds = tr.find_all("td", recursive=False)
                if len(tds) < 3:
                    continue
                boat = _td_text(tds[0])
                if not re.match(r"^[1-6]$", boat):
                    continue
                odds_td = tr.select_one("td.oddsPoint") or tds[-1]
                win[boat] = _td_text(odds_td)

        if len(tables) >= 2:
            for tr in tables[1].find_all("tr"):
                tds = tr.find_all("td", recursive=False)
                if len(tds) < 3:
                    continue
                boat = _td_text(tds[0])
                if not re.match(r"^[1-6]$", boat):
                    continue
                odds_td = tr.select_one("td.oddsPoint") or tds[-1]
                raw = _td_text(odds_td)
                # "1.1-2.6" → ['1.1', '2.6']  / "1.0" → ['1.0']
                parts = [p for p in re.split(r"\s*-\s*", raw) if re.match(r"^\d+\.?\d*$", p)]
                place[boat] = parts if parts else [raw]

        return {"win": win, "place_show": place}
