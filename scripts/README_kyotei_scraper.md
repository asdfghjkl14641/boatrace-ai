# kyotei.murao111.net 過去オッズ & レースメタ スクレイパー

本ディレクトリの `kyotei_murao_scraper.py` は、kyotei.murao111.net のオッズ検索
ページ (`/oddses`) から 4 券種 (2単・3単・2複・3複) のオッズ、およびそれに
紐づくレースメタ情報 (天気・風・波・決まり手・結果・払戻) を取得し、
`boatrace.db` (SQLite) に蓄積するスクリプトです。

---

## 1. 倫理・法令ポリシー

- 個人研究用、**運営者 OK 済み**
- 商用利用・第三者への再配布禁止 (利用規約 §5(5))
- 決済システム復旧時は遡って PREMIUM (¥9,800) を支払う意思あり
- 運営者から停止要請があれば直ちに停止、収集データは削除

### 負荷対策

| 項目 | 設定 |
|---|---|
| 並列度 | **2 ワーカー (ThreadPoolExecutor)** |
| 1 ワーカー当たり間隔 | **3〜5 秒 (乱数)** |
| 合計スループット | **約 1 req / 2 秒 (= 並列 2 × 平均 4 秒間隔)** |
| 1 時間あたり上限 | **2,400 req** (= 並列 2 × 1,200) |
| 夜間休止 | **23:00〜翌06:00 JST** |
| 429/503 対策 | 30/60/90 秒の指数バックオフ 最大 3 回 |
| 再開 | `(data_type, date, page)` 単位のチェックポイントで中断・再開 |

---

## 2. 認証

**ログイン不要**。無料会員登録もスキップ可。構造調査で全データが未ログインで取得
できることが確認されています。集計値 (回収率・的中率) だけは「計算中…」になり
ますが有料機能であり、生データ取得には無関係です。

---

## 3. 取得対象データ (`--data-types`)

| data_type | テーブル | 買い目形式 | 備考 |
|---|---|---|---|
| `odds_trifecta` | `trifecta_odds` | `"1-2-3"` | 3連単 (kachishiki_id=2) |
| `odds_exacta` | `odds_exacta` | `"1-2"` | 2連単 (kid=1) |
| `odds_quinella` | `odds_quinella` | `"1=2"` (昇順) | 2連複 (kid=3) |
| `odds_trio` | `odds_trio` | `"1=2=3"` (昇順) | 3連複 (kid=4) |
| `race_meta` | `race_meta_murao` | — | 1 レース = 1 行。天気・風・波・決まり手・結果・払戻 |

`race_meta` は `odds_trifecta` page=1 のレスポンスから派生取得します (別 req 不要)。

---

## 4. DB スキーマ (migration 004)

```sql
-- trifecta_odds: 既存テーブルに以下の列を追加
ALTER TABLE trifecta_odds ADD COLUMN odds_5min REAL;
ALTER TABLE trifecta_odds ADD COLUMN pop_5min  INTEGER;
ALTER TABLE trifecta_odds ADD COLUMN odds_1min REAL;
ALTER TABLE trifecta_odds ADD COLUMN pop_1min  INTEGER;

-- 新規 4 テーブル (同一スキーマ、ユニーク制約で冪等 upsert)
CREATE TABLE odds_exacta    (..., combination "1-2");
CREATE TABLE odds_quinella  (..., combination "1=2");
CREATE TABLE odds_trio      (..., combination "1=2=3");

-- レース単位メタ (天気・風・波・決まり手・結果・払戻)
CREATE TABLE race_meta_murao (
    date, stadium, stadium_name, race_number,
    weekday, series, grade, rank_count, rank_lineup,
    race_type, time_zone, entry_fixed, schedule_day, final_race,
    weather, wind_direction, wind_speed, wave,
    result_order, kimarite, payout,
    UNIQUE(date, stadium, race_number)
);
```

適用方法:

```bash
python -m scripts.apply_migration_004
```

※ `ALTER TABLE ADD COLUMN` は列存在チェックで冪等化済み。

---

## 5. CLI

### 5-1. フェーズプリセット

| `--phase` | 期間 | 日数 | 総 req 数 |
|---|---|---:|---:|
| `A1` | 2026-04-12 〜 2026-04-18 | 7 | **63** |
| `A2` | 2025-08-01 〜 2026-04-18 | 261 | **2,349** |
| `A3` | 2020-02-01 〜 2026-04-18 | 2,268 | **20,412** |

### 5-2. 主な引数

```
python -m scripts.kyotei_murao_scraper
    [--phase A1|A2|A3]            # 期間プリセット (--from-date / --to-date と排他)
    [--from-date YYYY-MM-DD]
    [--to-date   YYYY-MM-DD]
    [--data-types odds_trifecta,odds_exacta,odds_quinella,odds_trio,race_meta]
                                   # default=全部
    [--parallel N]                 # 1..3、default=2
    [--sleep-min 3.0]              # 最小間隔秒
    [--sleep-max 5.0]              # 最大間隔秒
    [--dry-run]                    # URL 生成のみ (HTTP/DB/progress 不触)
```

### 5-3. 推奨コマンド例

```bash
# 1) dry-run で URL 生成を確認 (触らない)
python -m scripts.kyotei_murao_scraper --dry-run --phase A1

# 2) フェーズ A-1 (1 週間テスト)
python -m scripts.kyotei_murao_scraper --phase A1 --parallel 2 \
    --sleep-min 3.0 --sleep-max 5.0

# 3) フェーズ A-2 (MVP 8.5 ヶ月)
python -m scripts.kyotei_murao_scraper --phase A2 --parallel 2

# 4) 任意期間・一部種別のみ
python -m scripts.kyotei_murao_scraper \
    --from-date 2025-08-01 --to-date 2025-08-31 \
    --data-types odds_trifecta,race_meta --parallel 2
```

### 5-4. 中断・再開

チェックポイント `scripts/kyotei_murao_progress.json` に
`{data_type}:{date}:ALL:{page}` 形式のキーで完了状況を保存。
同じコマンドを再実行すれば未完了タスクから再開します。

---

## 6. 想定実行時間

1 req あたり HTTP 0.2s + パース 3〜5s (9.5MB の HTML) と見積もると:

| フェーズ | req 数 | 並列 1 (7s/req) | 並列 2 | 夜間休止込み並列 2 |
|---|---:|---:|---:|---:|
| A-1 | 63 | 約 7 分 | **約 4 分** | 約 4 分 |
| A-2 | 2,349 | 約 4.6 h | **約 2.3 h** | **約 2〜3 h** |
| A-3 | 20,412 | 約 40 h | **約 20 h** | **約 1〜1.5 日** |

---

## 7. 単体テスト (35 件、すべてネットワーク不使用)

```bash
python -m unittest tests.test_kyotei_murao_scraper -v
```

カバー:
- URL 生成 (`kachishiki_id` のマッピング, `display_num=FIVE_THOUSAND`, stadium_id 角括弧なし)
- 4 券種の HTML パース (3連単/2連単/2連複/3連複)
- `race_meta` のパース & 重複除去 (120 組 → 1 meta 行)
- 件数抽出 `全N件中` → total
- 会場名 "24#大　村" → (24, "大村")
- 進入固定/非固定の判別
- チェックポイント I/O と **並列 8 スレッドの同時書込ストレステスト**
- `RateLimiter` (時刻注入) — 最小間隔 / 夜間休止 / 時間上限 / **並列ワーカーの相互排他**
- CLI — フェーズプリセット / data-types パース / 期間未指定エラー / dry-run
- `run()` 並列版 dry-run が progress ファイルを生成しないこと

---

## 8. モーター履歴 (A-3 前に要追加調査)

調査中の時点で `/motor_records` は GET しても空テーブルが返る挙動。
A-1 / A-2 の対象外とし、A-3 実施前に `motor_no` 指定 or form submit 挙動を
追加調査して別テーブル (`motor_history`) で取り込む予定。

---

## 9. GitHub Actions 経由で A-2 を実行する

A-2 (2025-08-01〜2026-04-18, 8.5 ヶ月) はローカル PC を 2〜3 時間占有するので、
GitHub Actions (Ubuntu ランナー) で実行するのが楽です。既存の
`lzh_to_artifact.yml` と同じ **artifact 方式** で実装してあります。

### 9-1. 実行手順

1. `.github/workflows/kyotei-murao-a2.yml` を main ブランチにプッシュ
2. GitHub の **Actions** タブ → **"kyotei.murao111 A-2 scraping"** を選択
3. **Run workflow** をクリック →
   - `from_date`: 2025-08-01 (デフォルト)
   - `to_date`: 2026-04-18 (デフォルト)
   - `parallel`: 2 (デフォルト)
   - `sleep_min` / `sleep_max`: 3.0 / 5.0 (デフォルト)
   - `data_types`: 空 (全種別取得)
4. 2〜3 時間待つ (Summary ページに経過・結果サマリが出ます)
5. 完了後、**Artifacts** セクションから
   `kyotei-murao-artifact-2025-08-01-2026-04-18.zip` をダウンロード
6. ローカルで展開して `artifact.db` を取得
7. マージ:

```bash
cd boatrace-ai
python -m scripts.merge_kyotei_murao path/to/artifact.db
```

### 9-2. ワークフローの動作

- 空の `artifact.db` を作成 → `sql/sqlite_schema.sql` + `migration 004` を適用
- `SQLITE_PATH=artifact.db` 環境変数でスクレイパーに書き先を切替
- 取得完了後 VACUUM してサイズ圧縮
- `upload-artifact@v4` で `artifact.db` とログ・`progress.json` を保存
- GH Actions タイムアウトは 350 分 (6 時間弱)。A-2 は約 1.6〜3 時間で余裕

### 9-3. ローカルへのマージ戦略

`scripts/merge_kyotei_murao.py` が以下を行います:

| テーブル | マージ方法 |
|---|---|
| `trifecta_odds` | UPSERT。既存の `odds` 列は保持しつつ、`odds_5min/pop_5min/odds_1min/pop_1min` を上書き |
| `odds_exacta/quinella/trio` | UPSERT (同上) |
| `race_meta_murao` | INSERT OR IGNORE (1 レース 1 行、変更しない) |

マージ先 DB に murao 用テーブルが無ければ `migration 004` を自動適用します。

### 9-4. トラブルシュート

| 現象 | 対処 |
|---|---|
| GH Actions タイムアウト (350 分超過) | `--parallel 3` に上げるか、期間を分割 (半年ずつ 2 回) |
| artifact.db が 2GB 超えで upload 失敗 | 期間を 2 分割して実行 (2GB 上限) |
| 429/503 が多発 | `--sleep-min 5 --sleep-max 8` に上げて再走 (冪等なので安全) |
| 途中で失敗 | ログと progress.json の artifact をダウンロードし、未完了分だけ再実行 |

---

## 10. 実行前チェックリスト

- [x] 運営者 OK 確認済み
- [x] Migration 004 適用済み (`python -m scripts.apply_migration_004`)
- [x] ユニットテスト 35 件 PASS
- [x] `--dry-run --phase A1` で URL 生成を確認
- [ ] フェーズ A-1 実行 → 結果検証 (ユーザの指示待ち)
- [ ] 問題なければフェーズ A-2
- [ ] A-3 前にモーター履歴の追加調査
