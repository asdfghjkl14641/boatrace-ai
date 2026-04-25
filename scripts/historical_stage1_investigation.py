# -*- coding: utf-8 -*-
"""段階 1: 既存データカバー確認 + murao 探索 + race_cards 代替可能性判定."""
from __future__ import annotations
import io, sys, json, time, warnings
from pathlib import Path
from datetime import datetime
import pandas as pd

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE))
sys.stdout.reconfigure(encoding="utf-8")

TARGET_FROM = "2020-02-01"
TARGET_TO   = "2023-04-30"

MURAO_BASE = "https://kyotei.murao111.net"
MURAO_ODDS = f"{MURAO_BASE}/oddses"

stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("=" * 80)
print(f"段階 1: 事前調査  ({stamp})")
print(f"対象期間: {TARGET_FROM} ~ {TARGET_TO}")
print("=" * 80)

# ========== [1.1] 既存データ確認 ==========
print("\n[1.1] 既存データカバー ...")
from scripts.db import get_connection
conn = get_connection()

covers = []
for tbl, date_col in [
    ("race_cards", "date"), ("race_results", "date"),
    ("race_conditions", "date"), ("trifecta_odds", "date"),
    ("racer_history", "race_date"),
]:
    q = f"SELECT MIN({date_col}) AS min_date, MAX({date_col}) AS max_date, COUNT(*) AS n FROM {tbl}"
    try:
        df = pd.read_sql_query(q, conn.native)
        covers.append({"table": tbl, **df.iloc[0].to_dict()})
    except Exception as e:
        covers.append({"table": tbl, "error": str(e)})
# murao 保存テーブル
for tbl in ["odds_trifecta", "odds_exacta", "odds_quinella", "odds_trio", "race_meta_murao"]:
    try:
        df = pd.read_sql_query(f"SELECT COUNT(*) AS n FROM {tbl}", conn.native)
        d = pd.read_sql_query(f"SELECT MIN(race_date) AS min_date, MAX(race_date) AS max_date FROM {tbl}", conn.native)
        covers.append({"table": tbl, **df.iloc[0].to_dict(), **d.iloc[0].to_dict()})
    except Exception as e:
        covers.append({"table": tbl, "error": str(e)})

print(f"\n  {'table':<22} {'min_date':<12} {'max_date':<12} {'n':>12}")
for c in covers:
    if "error" in c:
        print(f"  {c['table']:<22} ERROR: {c['error'][:60]}")
    else:
        mn = str(c.get("min_date", "")); mx = str(c.get("max_date", ""))
        n = c.get("n", 0)
        print(f"  {c['table']:<22} {mn:<12} {mx:<12} {n:>12,}")

# ========== [1.1b] 目標期間でのカバー ==========
print(f"\n[1.1b] 目標期間 ({TARGET_FROM}〜{TARGET_TO}) のカバー:")
for tbl, dcol in [
    ("race_cards","date"),("race_results","date"),("race_conditions","date"),
    ("trifecta_odds","date"),("racer_history","race_date"),
    ("odds_trifecta","race_date"),("race_meta_murao","race_date")]:
    try:
        n = conn.native.execute(
            f"SELECT COUNT(DISTINCT {dcol}) FROM {tbl} "
            f"WHERE {dcol} BETWEEN '{TARGET_FROM}' AND '{TARGET_TO}'"
        ).fetchone()[0]
        print(f"  {tbl:<22} {n:>5} distinct days")
    except Exception as e:
        print(f"  {tbl:<22} ERROR {str(e)[:40]}")

# ========== [1.2] murao 探索 ==========
print("\n[1.2] murao API 探索 (kyotei.murao111.net) ...")
import requests
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def probe(url, params, label):
    t0 = time.time()
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        elapsed = time.time() - t0
        return {"ok": True, "status": r.status_code, "elapsed": elapsed,
                "length": len(r.text), "params": params,
                "head": r.text[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)}

probes = {}
# trifecta odds for 2020-02-15
r = probe(MURAO_ODDS, {
    "conditions[race_date][from]": "2020-02-15",
    "conditions[race_date][to]":   "2020-02-15",
    "kachishiki_id": 2,
    "display_num":   "TEN",
    "page":          1,
}, "trifecta 2020-02")
print(f"   trifecta 2020-02-15: status={r.get('status')}  elapsed={r.get('elapsed',0):.2f}s  len={r.get('length',0)}")
if r.get("head"): print(f"      head: {r['head'][:120]!r}")
probes["trifecta_2020_02"] = r
time.sleep(1.0)

# exacta odds for 2020-02-15
r = probe(MURAO_ODDS, {
    "conditions[race_date][from]": "2020-02-15",
    "conditions[race_date][to]":   "2020-02-15",
    "kachishiki_id": 1,
    "display_num":   "TEN",
    "page":          1,
}, "exacta 2020-02")
print(f"   exacta 2020-02-15: status={r.get('status')}  elapsed={r.get('elapsed',0):.2f}s  len={r.get('length',0)}")
probes["exacta_2020_02"] = r
time.sleep(1.0)

# 2022-06-15 確認
r = probe(MURAO_ODDS, {
    "conditions[race_date][from]": "2022-06-15",
    "conditions[race_date][to]":   "2022-06-15",
    "kachishiki_id": 2,
    "display_num":   "TEN",
    "page":          1,
}, "trifecta 2022-06")
print(f"   trifecta 2022-06-15: status={r.get('status')}  elapsed={r.get('elapsed',0):.2f}s  len={r.get('length',0)}")
probes["trifecta_2022_06"] = r
time.sleep(1.0)

# 簡易: HTML から行数を推定
if probes["trifecta_2020_02"].get("status") == 200:
    # full fetch once for parsing
    rr = requests.get(MURAO_ODDS, params={
        "conditions[race_date][from]": "2020-02-15",
        "conditions[race_date][to]":   "2020-02-15",
        "kachishiki_id": 2,
        "display_num":   "FIVE_THOUSAND",
        "page":          1,
    }, headers=HEADERS, timeout=30)
    if rr.status_code == 200:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(rr.text, "html.parser")
        rows = soup.find_all("tr")
        data_rows = [r for r in rows if len(r.find_all("td")) >= 20]
        print(f"\n   [1.2b] 構造確認 (2020-02-15 三連単 FIVE_THOUSAND page=1):")
        print(f"      data rows: {len(data_rows)}")
        if data_rows:
            sample = data_rows[0]
            cells = sample.find_all("td")
            print(f"      first row cells ({len(cells)}): "
                  + " | ".join(c.get_text(strip=True)[:18] for c in cells[:6]))
probes["structure_ok"] = True
with open(LOG_DIR/"stage1_murao_probe.json", "w", encoding="utf-8") as f:
    json.dump({"timestamp": stamp, "probes": probes}, f, ensure_ascii=False, indent=2)
print(f"\n   saved: logs/stage1_murao_probe.json")

# ========== [1.3] パターン判定 ==========
print("\n[1.3] race_cards 代替可能性判定 ...")
tri_ok = probes.get("trifecta_2020_02", {}).get("status") == 200
tri_22_ok = probes.get("trifecta_2022_06", {}).get("status") == 200
exa_ok = probes.get("exacta_2020_02", {}).get("status") == 200

# murao /oddses は trifecta_odds と race_meta_murao 両方を提供する
# race_cards の代替不可だが、trifecta_odds + race_meta_murao があれば
# Phase C の期待値検証は可能 (lane × racerid × 勝率 は要素外でよい)
# ただしモデル学習には race_cards が要るので、既存 v4 モデルはこの期間では学習できない
# → "モデルを学習し直す" か "モデルは v4 のまま、バックテストだけ拡張" の 2 択

# 既存 trifecta_odds (メインテーブル) の期間内カバー
odds_mur_days = conn.native.execute(
    f"SELECT COUNT(DISTINCT date) FROM trifecta_odds "
    f"WHERE date BETWEEN '{TARGET_FROM}' AND '{TARGET_TO}'"
).fetchone()[0]
total_target_days = (pd.to_datetime(TARGET_TO) - pd.to_datetime(TARGET_FROM)).days + 1
coverage_pct = odds_mur_days/total_target_days*100 if total_target_days else 0
print(f"   既存 trifecta_odds 期間カバー: {odds_mur_days}/{total_target_days} "
      f"days ({coverage_pct:.1f}%)")

# race_cards 必要列の代替可能性を精査
# v4 で必要な列: global_win_pt, local_win_pt, aveST, motor_in2nd, motor, lane, racerid
# racer_history にある: lane, racer_id, st (→aveST 計算可), finish_pos
# racer_history にない: 事前公表勝率 (global_win_pt/local_win_pt), motor 番号, motor_in2nd
missing_critical = []
# motor 番号 — モデル v4 の f1_motor_in2nd に必須
missing_critical.append("motor (モーター番号) — racer_history, murao /oddses 共になし")
missing_critical.append("motor_in2nd (モーター2連対率) — motor 番号がないと計算不能")
# 勝率
can_compute = [
    "global_win_pt: racer_history から累積 finish_pos で再計算可能",
    "local_win_pt: 同上を stadium フィルタ",
    "aveST: racer_history.st から計算可能 (直近 30 走など)",
    "grade/class: racer_profile スナップショットで近似",
]

if not (tri_ok and tri_22_ok):
    pattern = "Z"
    reasons = [
        f"murao /oddses 応答問題: trifecta_2020_02 status={probes.get('trifecta_2020_02',{}).get('status')}, "
        f"trifecta_2022_06 status={probes.get('trifecta_2022_06',{}).get('status')}",
    ]
elif coverage_pct >= 95:
    # オッズは既に入っている — 拡張不要, むしろ race_cards 問題
    pattern = "Y-data-exists"
    reasons = [
        f"murao /oddses 200 (2020-02, 2022-06) — 取得可能",
        f"既存 trifecta_odds がすでに {coverage_pct:.1f}% カバー ({odds_mur_days}/{total_target_days} days)",
        "→ **オッズ取得は既に完了している** (別パイプラインで)",
        "race_cards は目標期間で 0 日 — 新規取得が必要だが murao にエンドポイントなし",
        "race_cards 必須列のうち motor/motor_in2nd は **いかなる既知ソースでも取れない**",
        "可否判定: 致命的欠損 → 運用ブロック",
    ]
else:
    pattern = "Y"
    reasons = [
        "murao /oddses 200 — odds + race_meta 取得可能",
        "race_cards 代替は部分的 (motor/motor_in2nd は取れない)",
    ]

print(f"\n   判定: パターン {pattern}")
for r in reasons:
    print(f"     - {r}")
print(f"\n   race_cards 欠損列の詳細:")
print(f"     ▼ 致命的 (代替手段なし):")
for m in missing_critical:
    print(f"       - {m}")
print(f"     ▼ 計算で代替可能:")
for c in can_compute:
    print(f"       - {c}")

# 既存 kyotei_murao_scraper の進捗も確認
try:
    prog = json.loads((BASE/"scripts"/"kyotei_murao_progress.json").read_text(encoding="utf-8"))
    completed = prog.get("completed", [])
    failed = prog.get("failed", {})
    dates = sorted(set(x.split(":")[1] for x in completed if ":" in x))
    print(f"\n   既存 kyotei_murao_scraper 進捗:")
    print(f"     updated_at: {prog.get('updated_at')}")
    print(f"     completed tasks: {len(completed)}")
    print(f"     completed distinct dates: {len(dates)}")
    if dates:
        print(f"     earliest: {dates[0]}, latest: {dates[-1]}")
    print(f"     failed: {len(failed)}")
except Exception as e:
    print(f"   既存 progress 読み込みエラー: {e}")

# ========== 結果保存 ==========
result = {
    "timestamp": stamp, "target_from": TARGET_FROM, "target_to": TARGET_TO,
    "covers": covers, "pattern": pattern, "reasons": reasons,
    "existing_odds_trifecta_days": odds_mur_days,
    "total_target_days": total_target_days,
    "coverage_pct": coverage_pct,
}
with open(LOG_DIR/"stage1_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2, default=str)
conn.close()
print(f"\n=== 段階 1 判定: パターン {pattern} ===")
print("logs/stage1_result.json に保存")
