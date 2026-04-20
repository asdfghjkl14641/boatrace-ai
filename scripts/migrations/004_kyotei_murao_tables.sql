-- ============================================================
-- Migration 004: kyotei.murao111.net 取得データ向けテーブル
--
-- - trifecta_odds に 5分前/1分前/人気 カラムを追加 (既存データと併存)
-- - 2連単・2連複・3連複 の 3 つの独立テーブルを新規作成
-- - レース単位メタ情報 (天気・風・波・決まり手・結果) を 1 テーブルに集約
--
-- SQLite 用。複数回実行しても安全 (IF NOT EXISTS + ADD COLUMN は手動冪等化)。
-- ============================================================

-- ------------------------------------------------------------
-- 1) trifecta_odds の拡張
--   既存: (date, stadium, stadium_name, race_number, combination, odds)
--   追加: 5分前/1分前 のオッズ・人気
--
--   SQLite の ALTER TABLE ADD COLUMN は冪等ではないので、
--   このファイルを直接流すのではなく scripts/apply_migration_004.py から
--   「列が無ければ ADD」の形で適用する想定。
-- ------------------------------------------------------------
-- ALTER TABLE trifecta_odds ADD COLUMN odds_5min REAL;
-- ALTER TABLE trifecta_odds ADD COLUMN pop_5min  INTEGER;
-- ALTER TABLE trifecta_odds ADD COLUMN odds_1min REAL;
-- ALTER TABLE trifecta_odds ADD COLUMN pop_1min  INTEGER;

-- ------------------------------------------------------------
-- 2) 2 連単 (Exacta)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS odds_exacta (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    date         DATE    NOT NULL,
    stadium      INTEGER NOT NULL,
    stadium_name TEXT,
    race_number  INTEGER NOT NULL,
    combination  TEXT    NOT NULL,          -- "1-2" (順序あり)
    odds_5min    REAL,
    pop_5min     INTEGER,
    odds_1min    REAL,
    pop_1min     INTEGER,
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (date, stadium, race_number, combination)
);
CREATE INDEX IF NOT EXISTS idx_exacta_dsr
    ON odds_exacta (date, stadium, race_number);

-- ------------------------------------------------------------
-- 3) 2 連複 (Quinella)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS odds_quinella (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    date         DATE    NOT NULL,
    stadium      INTEGER NOT NULL,
    stadium_name TEXT,
    race_number  INTEGER NOT NULL,
    combination  TEXT    NOT NULL,          -- "1=2" (昇順)
    odds_5min    REAL,
    pop_5min     INTEGER,
    odds_1min    REAL,
    pop_1min     INTEGER,
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (date, stadium, race_number, combination)
);
CREATE INDEX IF NOT EXISTS idx_quinella_dsr
    ON odds_quinella (date, stadium, race_number);

-- ------------------------------------------------------------
-- 4) 3 連複 (Trio)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS odds_trio (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    date         DATE    NOT NULL,
    stadium      INTEGER NOT NULL,
    stadium_name TEXT,
    race_number  INTEGER NOT NULL,
    combination  TEXT    NOT NULL,          -- "1=2=3" (昇順)
    odds_5min    REAL,
    pop_5min     INTEGER,
    odds_1min    REAL,
    pop_1min     INTEGER,
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (date, stadium, race_number, combination)
);
CREATE INDEX IF NOT EXISTS idx_trio_dsr
    ON odds_trio (date, stadium, race_number);

-- ------------------------------------------------------------
-- 5) レース単位メタ情報
--   4 券種どれか 1 ページでも取れば 1 レコード入れれば良い
--   (券種ごとに重複は発生しない)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS race_meta_murao (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    date           DATE    NOT NULL,
    stadium        INTEGER NOT NULL,
    stadium_name   TEXT,
    race_number    INTEGER NOT NULL,
    weekday        TEXT,        -- 月/火/水/木/金/土/日
    series         TEXT,        -- 一般/G1 など
    grade          TEXT,        -- 一般/GⅢ 等
    rank_count     TEXT,        -- "A6B0" 等
    rank_lineup    TEXT,        -- "A1A1A1A1A1A1" 等
    race_type      TEXT,        -- 特賞|特選|選抜 等
    time_zone      TEXT,        -- ﾃﾞｲ/ﾓｰﾆﾝｸﾞ/ﾅｲﾀｰ 等
    entry_fixed    INTEGER,     -- 1: 固定 / 0: 非固定
    schedule_day   TEXT,        -- "1日目" 等 (日程)
    final_race     INTEGER,     -- その日の最終 R (セル "最終R")
    weather        TEXT,        -- 雨/晴/曇 等
    wind_direction TEXT,        -- 北/北東/… 日本語表記
    wind_speed     REAL,        -- m/s
    wave           REAL,        -- cm
    result_order   TEXT,        -- "1 5 3" (着順)
    kimarite       TEXT,        -- "逃げ"/"差し" 等
    payout         INTEGER,     -- 3連単払戻 (最初に見たレコードのもの)
    created_at     TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (date, stadium, race_number)
);
CREATE INDEX IF NOT EXISTS idx_meta_dsr
    ON race_meta_murao (date, stadium, race_number);
