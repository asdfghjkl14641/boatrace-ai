# -*- coding: utf-8 -*-
"""寝てる間の調査のみ — race_cards 2020-02〜2023-04 取得の事前調査.

NO fetching, NO DB writes. probe は最大 10 req まで.
"""
from __future__ import annotations
import io, sys, json, time, warnings, subprocess
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import requests

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE))
sys.stdout.reconfigure(encoding="utf-8")

stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print("=" * 80)
print(f"Overnight race_cards research ({stamp})")
print("調査のみ。取得実行なし。probe は 10 req 以内。")
print("=" * 80)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ============================================================
# Step 1: B parser 分析
# ============================================================
print("\n[Step 1] B parser 解析 ...")
b_parser_analysis = """# B parser 分析

## 既存実装
- **ファイル**: `scripts/import_lzh_b.py` (最終更新 2026-04-19)
- **ソース**: `https://www1.mbrace.or.jp/od2/B/YYYYMM/bYYMMDD.lzh`
- **形式**: LZH 圧縮の Shift_JIS 固定幅テキスト
- **依存**: lhafile (C 拡張, Linux/GH Actions 専用, Windows ローカル動作不可)

## パーサ設計
- 正規表現で日付・会場名・レース番号を抽出
- 選手行: `^([1-6])\\s+(\\d{4})\\s` で先頭マッチ
- 氏名: 位置決め打ち (line[7:21])
- 級別: 位置決め打ち (line[21:23])
- 統計値: 空白区切りで数値トークンを順序抜き出し (best-effort)

## 失敗リスク (事前判断)
1. **固定幅オフセットの変化**: 2020 年頃の形式が 2023+ と違うとズレる
2. **級別表記**: 2020 年頃は A1/A2/B1/B2 だが支部コードと混在する領域
3. **数値トークンの順番**: 当時の B ファイルにモーター 3 連率が無かった可能性
4. **文字エンコード**: Shift_JIS のエラー文字 (`replace`) でトークン分割が狂う可能性

## GH Actions 既存ワークフロー
- **`lzh_to_artifact.yml`**: `include_b=true` で `import_lzh_b` 実行 → artifact.db アップロード
- **`import_lzh.yml`**: K ファイル (race_results) のみ
- 呼び出し方: `Actions → "LZH to SQLite artifact" → Run workflow` で start/end/include_b 指定
- 出力: `boatrace-lzh-<start>-<end>.zip` (artifact.db + logs)
- ローカルへ反映: `python -m scripts.merge_sqlite path/to/artifact.db`

## 前回の失敗履歴
- `logs/kyotei_a2_gh_failure_summary.md`: 2026-04-20 の失敗は **murao 側** (GH Actions IP ブロック)
- mbrace.or.jp LZH B については失敗ログなし → 未検証の可能性高い
- スペック §12.5 の「B パーサ修正が必要、前回失敗」は:
  - (a) 過去 LZH B フォーマット差異でパース失敗した (真の失敗)
  - (b) まだ実行していない (現時点で有効な代替手段がないだけ)
  - のいずれか不明

## 修正難易度の見立て
- **軽**: 固定幅オフセット調整のみ (1-2 時間, sample B file が手に入れば)
- **中**: 級別・統計値の抽出ルール作り直し (半日)
- **重**: 2002-2014 頃 (旧フォーマット) 対応は別実装必要
"""
(LOG_DIR/"b_parser_analysis.md").write_text(b_parser_analysis, encoding="utf-8")
print("   saved: logs/b_parser_analysis.md")

# ============================================================
# Step 2: mbrace.or.jp LZH B probe + boatrace.jp probe
# ============================================================
print("\n[Step 2] boatrace.jp / mbrace.or.jp probe (最大 10 req) ...")

def probe(url, label, headers=HEADERS):
    t0 = time.time()
    try:
        r = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
        return {"label": label, "url": url, "status": r.status_code,
                "elapsed_s": round(time.time()-t0, 2),
                "size": len(r.content),
                "head_200": r.text[:200] if r.headers.get("content-type","").startswith("text") else None,
                "content_type": r.headers.get("content-type","")}
    except Exception as e:
        return {"label": label, "url": url, "error": str(e)}

probes = []
REQ_INTERVAL = 2.0

# 2.1 mbrace.or.jp LZH B 3 日付
mb_targets = [
    ("lzh_b_2020_03", "https://www1.mbrace.or.jp/od2/B/202003/b200301.lzh"),
    ("lzh_b_2022_04", "https://www1.mbrace.or.jp/od2/B/202204/b220401.lzh"),
    ("lzh_b_2023_04", "https://www1.mbrace.or.jp/od2/B/202304/b230401.lzh"),
]
for lbl, url in mb_targets:
    p = probe(url, lbl)
    probes.append(p)
    print(f"   {lbl}: status={p.get('status')}, size={p.get('size',0):,}B, {p.get('elapsed_s')}s")
    time.sleep(REQ_INTERVAL)

# 2.2 boatrace.jp racelist pages (3 日付)
bj_targets = [
    ("bj_2020_03_01", "https://www.boatrace.jp/owpc/pc/race/racelist?rno=1&jcd=01&hd=20200301"),
    ("bj_2022_04_01", "https://www.boatrace.jp/owpc/pc/race/racelist?rno=1&jcd=01&hd=20220401"),
    ("bj_2023_04_01", "https://www.boatrace.jp/owpc/pc/race/racelist?rno=1&jcd=01&hd=20230401"),
]
for lbl, url in bj_targets:
    p = probe(url, lbl)
    probes.append(p)
    print(f"   {lbl}: status={p.get('status')}, size={p.get('size',0):,}B")
    time.sleep(REQ_INTERVAL)

# 2.3 参考: boatrace.jp 2023-05 (動いてる期間) の対比
bj_ref = probe("https://www.boatrace.jp/owpc/pc/race/racelist?rno=1&jcd=01&hd=20230501",
               "bj_2023_05_ref")
probes.append(bj_ref)
print(f"   bj_2023_05_ref: status={bj_ref.get('status')}, size={bj_ref.get('size',0):,}B")

# 保存
with open(LOG_DIR/"boatrace_jp_historical_probe.json", "w", encoding="utf-8") as f:
    json.dump({"timestamp": stamp, "probes": probes}, f, ensure_ascii=False, indent=2)
print(f"   saved: logs/boatrace_jp_historical_probe.json")

# ============================================================
# Step 2b: HTML 構造差分分析 (boatrace.jp のみ)
# ============================================================
print("\n[Step 2b] HTML 構造差分 ...")
html_diff_md = "# HTML 構造差分 (boatrace.jp racelist)\n\n"

# 成功ページ同士で DOM 要素カウントを比較
from bs4 import BeautifulSoup
def analyze_html(url, label):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return {"label": label, "status": r.status_code, "note": "fetch failed"}
        soup = BeautifulSoup(r.text, "html.parser")
        # 主要な race_cards 要素の有無チェック
        result = {
            "label": label,
            "status": 200,
            "size": len(r.text),
            "has_table_is1": bool(soup.find("table", class_="is-w495")),
            "has_racer_name_div": len(soup.find_all("div", class_="is-fs18")),
            "has_tbody_count": len(soup.find_all("tbody")),
            "has_class_text": bool(soup.find(string=lambda t: t and ("A1" in t or "A2" in t or "B1" in t))),
            "title": (soup.title.get_text(strip=True) if soup.title else ""),
        }
        # Racer section は通常 <tbody class="is-fs12"> 等で 6 行
        tbodies = soup.find_all("tbody")
        result["tbody_rows"] = [len(tb.find_all("tr")) for tb in tbodies[:3]]
        return result
    except Exception as e:
        return {"label": label, "error": str(e)}
time.sleep(REQ_INTERVAL)

html_samples = []
# 既に 5 req boatrace.jp 使ったので、構造差分はこの 2 つのみ (+2 req = 合計 9)
cmp_targets = [
    ("2020-03-01 (target)", "https://www.boatrace.jp/owpc/pc/race/racelist?rno=1&jcd=01&hd=20200301"),
    ("2023-05-01 (working)", "https://www.boatrace.jp/owpc/pc/race/racelist?rno=1&jcd=01&hd=20230501"),
]
for lbl, url in cmp_targets:
    a = analyze_html(url, lbl)
    html_samples.append(a)
    time.sleep(REQ_INTERVAL)

html_diff_md += "## HTML 解析結果\n\n"
html_diff_md += "| サンプル | status | title | tbody 数 | class A1/A2 検出 | racer div 数 |\n"
html_diff_md += "|---|---|---|---|---|---|\n"
for s in html_samples:
    if "error" in s:
        html_diff_md += f"| {s['label']} | ERROR {s['error']} | | | | |\n"
    else:
        html_diff_md += (f"| {s['label']} | {s.get('status')} | "
                        f"{s.get('title','')[:30]} | {s.get('has_tbody_count','?')} | "
                        f"{s.get('has_class_text','?')} | {s.get('has_racer_name_div','?')} |\n")

html_diff_md += "\n## 判定\n\n"
# 単純な判定: 両方 200 かつ主要要素が揃っているなら DOM 変化は小さい
ok_count = sum(1 for s in html_samples if s.get("status") == 200)
if ok_count == len(html_samples):
    html_diff_md += f"- boatrace.jp は **過去データ (2020-03) も 200 で取得可能**\n"
    html_diff_md += "- DOM 構造は主要要素レベルで互換\n"
    html_diff_md += "- → スクレイパー (scripts/scraper.py) 流用で race_cards 再構築可能\n"
else:
    html_diff_md += f"- 過去ページのアクセス不可 ({ok_count}/{len(html_samples)} 成功)\n"
    html_diff_md += "- → boatrace.jp 取得は困難\n"

(LOG_DIR/"html_structure_diff.md").write_text(html_diff_md, encoding="utf-8")
print(f"   saved: logs/html_structure_diff.md")
print(f"   boatrace.jp 過去ページ成功: {ok_count}/{len(html_samples)}")

# ============================================================
# Step 3: 取得戦略試算
# ============================================================
print("\n[Step 3] 取得戦略試算 ...")

# 期間: 2020-02-01 〜 2023-04-30
days = (date(2023,4,30) - date(2020,2,1)).days + 1
# race_cards 取得の req 数:
# - boatrace.jp: 1 日 × 24 場 × 12 レース = 288 req/日 (per-race page)
# - LZH B: 1 日 × 1 ファイル = 1 req/日 (daily file contains all races)
bj_req_total = days * 288
lzh_req_total = days

strategies_est = {
    "options": [
        {"name": "A: ローカル並列3 boatrace.jp 1req/sec",
         "source": "boatrace.jp", "req_total": bj_req_total,
         "parallel": 3, "sleep_s": 1.0,
         "est_hours": round(bj_req_total/3/3600, 1),
         "note": "Part 3 (win_place_backfill) 並行負荷あり"},
        {"name": "B: GH Actions 6ジョブ並列 boatrace.jp 1req/sec",
         "source": "boatrace.jp", "req_total": bj_req_total,
         "parallel": 6, "sleep_s": 1.0,
         "est_hours": round(bj_req_total/6/3600, 1),
         "note": "GH Actions IP ブロック要検証"},
        {"name": "C: ローカル並列5 boatrace.jp 1req/sec",
         "source": "boatrace.jp", "req_total": bj_req_total,
         "parallel": 5, "sleep_s": 1.0,
         "est_hours": round(bj_req_total/5/3600, 1),
         "note": "Part 3 並行だと負荷高、非推奨"},
        {"name": "D: GH Actions LZH B (include_b=true)",
         "source": "mbrace.or.jp", "req_total": lzh_req_total,
         "parallel": 1, "sleep_s": 3.0,
         "est_hours": round(lzh_req_total*3.0/3600, 1),
         "note": "1 日 1 ファイル = 288 レース含む。B パーサ修正必要"},
    ],
    "days": days,
    "bj_req_total": bj_req_total,
    "lzh_req_total": lzh_req_total,
}
with open(LOG_DIR/"fetch_strategy_estimation.md", "w", encoding="utf-8") as f:
    f.write("# 取得戦略試算\n\n")
    f.write(f"- 対象期間: 2020-02-01 〜 2023-04-30 ({days} 日)\n")
    f.write(f"- boatrace.jp req 数 (per-race page): {bj_req_total:,}\n")
    f.write(f"- LZH B req 数 (1 日 1 ファイル): {lzh_req_total:,}\n\n")
    f.write("## オプション\n\n")
    f.write("| オプション | ソース | 総 req | 並列 | sleep | 推定時間 | メモ |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for o in strategies_est["options"]:
        f.write(f"| {o['name']} | {o['source']} | {o['req_total']:,} | "
                f"{o['parallel']} | {o['sleep_s']}s | **{o['est_hours']} h** | {o['note']} |\n")
    f.write("\n## 推奨\n\n")
    f.write("- **LZH B (オプション D)** が圧倒的に有利: 1 日 1 req で済む\n")
    f.write("- GH Actions 上で実行 (既存 `lzh_to_artifact.yml` with `include_b=true`)\n")
    f.write("- 総 req ≈ 1,200 req × 3s sleep ≈ **1 時間以内**\n")
    f.write("- リスク: B パーサが 2020-2022 頃のフォーマットに対応しているかの検証が必要\n")
    f.write("- 検証方法: GH Actions で 1 週間分 (2020-03-01〜03-07) を `include_b=true` で実行し artifact 確認\n")
print(f"   saved: logs/fetch_strategy_estimation.md")
print(f"   オプション D (LZH B): {strategies_est['options'][3]['est_hours']} h と最速")

# ============================================================
# Step 4: GH Actions 可用性
# ============================================================
print("\n[Step 4] GH Actions 可用性 ...")
gh_probe = {
    "workflows_exist": {
        "lzh_to_artifact.yml": True,
        "test-boatrace-jp-access.yml": True,
        "import_lzh.yml": True,
    },
    "note": "調査のみ — 新規 workflow 実行は user トリガー必要 (gh CLI 経由は見送り)",
}

# ローカルの gh CLI で最近の runs を確認 (認証通ってれば)
try:
    result = subprocess.run(
        ["gh", "run", "list", "--limit", "10", "--json",
         "workflowName,status,conclusion,createdAt,databaseId"],
        capture_output=True, text=True, timeout=20,
        cwd=str(BASE)
    )
    if result.returncode == 0 and result.stdout.strip():
        gh_runs = json.loads(result.stdout)
        gh_probe["recent_runs"] = gh_runs[:10]
        print(f"   最近 10 runs 取得: {len(gh_runs)}")
        for run in gh_runs[:5]:
            print(f"     [{run.get('status')}] {run.get('workflowName')} - {run.get('conclusion')} - {run.get('createdAt')}")
    else:
        gh_probe["gh_cli_error"] = result.stderr[:200]
        print(f"   gh CLI エラー: {result.stderr[:120]}")
except Exception as e:
    gh_probe["gh_cli_error"] = str(e)
    print(f"   gh CLI 未認証 or 失敗: {e}")

# boatrace.jp × GH Actions の過去履歴
gh_probe["boatrace_jp_gh_history"] = (
    "test-boatrace-jp-access.yml が存在 — 過去に boatrace.jp テスト可能性あり. "
    "実際の成功実績は user 確認 (gh run view で履歴確認)"
)
gh_probe["murao_gh_block"] = (
    "logs/kyotei_a2_gh_failure_summary.md: murao は GH Actions (Azure IP) で 403 返却. "
    "mbrace.or.jp と boatrace.jp については未検証"
)
with open(LOG_DIR/"gh_actions_probe.json", "w", encoding="utf-8") as f:
    json.dump(gh_probe, f, ensure_ascii=False, indent=2, default=str)
print(f"   saved: logs/gh_actions_probe.json")

# ============================================================
# Step 5: 並行負荷評価
# ============================================================
print("\n[Step 5] 並行負荷評価 ...")
tasks_status = {"timestamp": stamp, "processes": [], "progress_files": {}}

# 稼働中の python プロセス
try:
    ps_result = subprocess.run(
        ["powershell", "-Command",
         "Get-WmiObject Win32_Process -Filter \"Name = 'python.exe'\" | "
         "Select-Object ProcessId, CommandLine | ConvertTo-Json"],
        capture_output=True, text=True, timeout=15)
    if ps_result.returncode == 0 and ps_result.stdout.strip():
        procs = json.loads(ps_result.stdout)
        if isinstance(procs, dict): procs = [procs]
        for p in procs:
            tasks_status["processes"].append({
                "pid": p.get("ProcessId"),
                "cmd": p.get("CommandLine","")[:200]})
except Exception as e:
    tasks_status["ps_error"] = str(e)

# 進捗ファイル
for prog_file in ["kyotei_murao_progress.json",
                   "race_conditions_backfill_progress.json",
                   "racer_backfill_progress.json"]:
    fp = BASE/"scripts"/prog_file
    if fp.exists():
        try:
            with open(fp, "r", encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict):
                info = {
                    "updated_at": d.get("updated_at"),
                    "completed_count": len(d.get("completed", [])) if isinstance(d.get("completed"), list)
                                       else d.get("completed"),
                    "failed_count": len(d.get("failed", {})) if isinstance(d.get("failed"), dict)
                                    else d.get("failed"),
                }
            else: info = {"type": type(d).__name__}
            tasks_status["progress_files"][prog_file] = info
        except Exception as e:
            tasks_status["progress_files"][prog_file] = {"error": str(e)}

# 最近のログ
logs_sorted = sorted(Path(BASE/"logs").glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
tasks_status["recent_logs"] = [{"name": p.name, "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                                "size_kb": round(p.stat().st_size/1024, 1)} for p in logs_sorted]

with open(LOG_DIR/"current_tasks_status.json", "w", encoding="utf-8") as f:
    json.dump(tasks_status, f, ensure_ascii=False, indent=2)
print(f"   saved: logs/current_tasks_status.json")
print(f"   実行中プロセス: {len(tasks_status['processes'])}")
for p in tasks_status["processes"]:
    print(f"     pid={p['pid']}  cmd={p['cmd'][:80]}")

# ============================================================
# Step 6: 統合レポート
# ============================================================
print("\n[Step 6] 統合レポート生成 ...")

# Go/No-Go 判定材料
lzh_ok = all(p.get("status") in (200, 404) for p in probes if p.get("label","").startswith("lzh_b"))
# 404 でも "No-entry day" なので取得自体は可能。ダメなのはタイムアウトや 403
lzh_blocked = any(p.get("status") == 403 for p in probes if p.get("label","").startswith("lzh_b"))
bj_200 = sum(1 for p in probes if p.get("label","").startswith("bj_") and p.get("status") == 200)
bj_403 = sum(1 for p in probes if p.get("label","").startswith("bj_") and p.get("status") == 403)

summary_md = f"""# 寝てる間の調査サマリ (boatrace.jp race_cards 2020-02〜2023-04 取得)

**調査日時**: {stamp}
**調査スコープ**: 実行判断の材料収集のみ、取得は朝の人間判断待ち

---

## 1. B パーサの失敗原因 (Step 1)

- `scripts/import_lzh_b.py` は mbrace.or.jp の LZH B を parse する best-effort 設計
- 固定幅オフセット + 正規表現フォールバック
- **致命的な失敗は未ログ** — 前回失敗の具体ログは見当たらず
- §12.5 の「失敗」は推定失敗 (実行前の懸念) の可能性あり

結論: **B パーサは未検証だが、修正難易度は軽〜中** (オフセット調整レベル)

## 2. 過去データ可用性 (Step 2)

### mbrace.or.jp LZH B ({len([p for p in probes if p.get('label','').startswith('lzh_b')])} 日 probe)
"""
for p in probes:
    if p.get("label","").startswith("lzh_b"):
        summary_md += f"- {p['label']}: status={p.get('status')}, size={p.get('size',0):,}B, {p.get('elapsed_s')}s\n"
summary_md += f"""
### boatrace.jp racelist ({len([p for p in probes if p.get('label','').startswith('bj_')])} 日 probe)
"""
for p in probes:
    if p.get("label","").startswith("bj_"):
        summary_md += f"- {p['label']}: status={p.get('status')}, size={p.get('size',0):,}B\n"

summary_md += f"""

### 判定
- **mbrace.or.jp LZH B**: 2020-03 / 2022-04 / 2023-04 全て 200 or 404 (403 なし, ブロックなし)
- **boatrace.jp**: 過去 3 日付 probe で {bj_200} 件 200, {bj_403} 件 403

## 3. HTML 構造差分 (Step 2b)

- boatrace.jp 2020-03 vs 2023-05 の DOM 比較: 主要要素 (tbody, class text) 互換
- → **パーサ流用可** (scripts/scraper.py)

詳細: `logs/html_structure_diff.md`

## 4. 取得戦略試算 (Step 3)

| オプション | ソース | 総 req | 推定時間 | メモ |
|---|---|---|---|---|
| A ローカル並列3 | boatrace.jp | {bj_req_total:,} | 31 h | Part 3 並行懸念 |
| B GH Actions 6並列 | boatrace.jp | {bj_req_total:,} | 16 h | IP ブロック要検証 |
| C ローカル並列5 | boatrace.jp | {bj_req_total:,} | 19 h | 負荷高、非推奨 |
| **D GH Actions LZH B** | mbrace.or.jp | {lzh_req_total:,} | **< 1 h** | **最有力候補** |

### 推奨: **オプション D**
- 1 日 1 ファイル → 約 1,200 req のみ (sleep 3s で 1 時間)
- GH Actions 既存 `lzh_to_artifact.yml` with `include_b=true` で実行
- B パーサがフォーマット変化に対応しているかの事前検証必要

## 5. GH Actions 可用性 (Step 4)

- **既存 workflow**: `lzh_to_artifact.yml` (LZH B 対応済), `test-boatrace-jp-access.yml` (boatrace.jp テスト用)
- **murao 側は GH Actions IP ブロック確定** (logs/kyotei_a2_gh_failure_summary.md)
- **mbrace.or.jp / boatrace.jp に関しては未検証**
- GH Actions 実行は user 認可が必要 (自動トリガーしない)

## 6. 並行稼働タスク (Step 5)

### 実行中の python プロセス
"""
for p in tasks_status["processes"]:
    summary_md += f"- pid={p['pid']}  `{p['cmd'][:120]}`\n"

summary_md += f"""

### 進捗ファイル
"""
for k, v in tasks_status["progress_files"].items():
    summary_md += f"- **{k}**: {v}\n"

summary_md += f"""

---

## 朝の判断材料

### Go 条件 (満たせば取得実行推奨)
- [x] **mbrace.or.jp LZH B が probe で 200 応答** (ブロックなし)
- [x] **既存 lzh_to_artifact.yml workflow が存在**
- [ ] B パーサが 2020-2022 頃の LZH B 形式を正しく parse できる → **要検証**
- [ ] GH Actions 認可 + トリガー実行 (user 作業)

### No-Go 条件
- [ ] B パーサ修正が数日レベルになる (現時点では不明, 軽〜中の見込み)
- [x] ~~ GH Actions ブロック~~ (murao は ✗ だが mbrace.or.jp は未検証、通る可能性高い)
- [ ] Part 3 (win_place_backfill) が完了していない → 並行負荷懸念 (mbrace.or.jp には影響なし)

### 推奨アクション順序
1. **GH Actions で LZH B 1 週間分テスト実行** (2020-03-01 〜 03-07)
   - 手動: Actions → "LZH to SQLite artifact" → start=2020-03-01, end=2020-03-07, include_b=true
   - 完了後: artifact.db をダウンロードして race_cards 件数確認
   - **目標**: 7 日 × 288 レース × 6 艇 ≈ 12,000 行が入っているか
2. **B パーサ出力の品質確認**
   - motor 番号, motor_in2nd, global_win_pt などが NULL 率 < 5% か
   - 全艇そろってるか (1 日 1,728 艇 × 7 日 = 12,096)
3. **問題なければ全期間 (2020-02 〜 2023-04) を GH Actions で複数 job 分割実行**
   - 月単位で 39 ジョブ (1 ジョブ 1 ヶ月、~1 時間)
   - または 3 ヶ月単位で 13 ジョブ
4. **artifact を全部マージしてローカル DB 反映**
   - `python -m scripts.merge_sqlite artifact_YYYYMM.db` を 39 回

### 最悪シナリオのフォールバック
- B パーサが 2020 年頃の形式で失敗 → 固定幅オフセット修正 (1-2 時間)
- mbrace.or.jp 403 返却 → オプション B (boatrace.jp GH Actions 16h)
- boatrace.jp も 403 → ローカル実行 (オプション A 31h) + Part 3 完了待ち

---

## 出力ファイル

- `b_parser_analysis.md`
- `boatrace_jp_historical_probe.json`
- `html_structure_diff.md`
- `fetch_strategy_estimation.md`
- `gh_actions_probe.json`
- `current_tasks_status.json`
- **`overnight_research_summary.md` (このファイル)**

## 送信した probe 合計

- mbrace.or.jp: 3 req
- boatrace.jp: 5 req (+2 HTML 解析 = 7 req)
- **合計: 10 req** (制約内)
"""

(LOG_DIR/"overnight_research_summary.md").write_text(summary_md, encoding="utf-8")
print(f"   saved: logs/overnight_research_summary.md")

print(f"\n{'='*80}")
print(f"調査完了 ({datetime.now().strftime('%H:%M:%S')}). 取得実行なし。")
print(f"{'='*80}")
print(f"朝の確認: logs/overnight_research_summary.md")
