-- ============================================================
-- ボートレースデータ保存用テーブル定義
-- Supabase (PostgreSQL) で実行する
--
-- 各テーブルには重複防止のためのUNIQUE制約を付与している。
-- ON CONFLICT DO NOTHING を使うことで、同じデータを
-- 2回取得してもエラーにならずスキップされる。
-- ============================================================

-- ============================================================
-- 1. race_cards (出走表)
-- ============================================================
CREATE TABLE IF NOT EXISTS race_cards (
    id            BIGSERIAL PRIMARY KEY,
    date          DATE    NOT NULL,
    stadium       INTEGER NOT NULL,
    stadium_name  TEXT,
    race_number   INTEGER NOT NULL,
    lane          INTEGER NOT NULL,      -- 枠番 (1〜6)

    racerid       INTEGER,               -- 選手登録番号
    name          TEXT,
    class         TEXT,                  -- A1/A2/B1/B2
    branch        TEXT,                  -- 支部
    birthplace    TEXT,                  -- 出身地
    age           INTEGER,
    weight        REAL,

    f             INTEGER,               -- F (フライング) 回数
    l             INTEGER,               -- L (出遅れ) 回数
    aveST         REAL,                  -- 平均スタートタイミング

    global_win_pt REAL,                  -- 全国勝率
    global_in2nd  REAL,                  -- 全国2連率
    global_in3rd  REAL,                  -- 全国3連率
    local_win_pt  REAL,                  -- 当地勝率
    local_in2nd   REAL,                  -- 当地2連率
    local_in3rd   REAL,                  -- 当地3連率

    motor         INTEGER,               -- モーター番号
    motor_in2nd   REAL,
    motor_in3rd   REAL,
    boat          INTEGER,               -- ボート番号
    boat_in2nd    REAL,
    boat_in3rd    REAL,

    created_at    TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_race_cards UNIQUE (date, stadium, race_number, lane)
);

-- ============================================================
-- 2. race_results (レース結果)
-- timeは "1'50\"6" のような文字列と、秒数値の time_sec の両方を保存
-- ============================================================
CREATE TABLE IF NOT EXISTS race_results (
    id           BIGSERIAL PRIMARY KEY,
    date         DATE    NOT NULL,
    stadium      INTEGER NOT NULL,
    stadium_name TEXT,
    race_number  INTEGER NOT NULL,

    rank         INTEGER,                -- 着順 (1〜6、失格等は NULL)
    boat         INTEGER NOT NULL,       -- 艇番 (1〜6)
    racerid      INTEGER,
    name         TEXT,
    time         TEXT,                   -- 元のタイム文字列 (例: 1'50"6)
    time_sec     REAL,                   -- 秒に変換した数値

    created_at   TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_race_results UNIQUE (date, stadium, race_number, boat)
);

-- ============================================================
-- 3. race_conditions (直前情報: 天候・水面・展示タイム)
--
-- wind_direction の数字コード (boatrace.jp の表記):
--    1=北   2=北北東  3=北東  4=東北東
--    5=東   6=東南東  7=南東  8=南南東
--    9=南  10=南南西 11=南西 12=西南西
--   13=西  14=西北西 15=北西 16=北北西
-- ============================================================
CREATE TABLE IF NOT EXISTS race_conditions (
    id                BIGSERIAL PRIMARY KEY,
    date              DATE    NOT NULL,
    stadium           INTEGER NOT NULL,
    stadium_name      TEXT,
    race_number       INTEGER NOT NULL,

    weather           TEXT,              -- 晴/曇/雨/雪 など
    temperature       REAL,              -- 気温 (℃)
    wind_direction    INTEGER,           -- 風向コード (上のコメント参照)
    wind_speed        REAL,              -- 風速 (m/s)
    water_temperature REAL,              -- 水温 (℃)
    wave_height       REAL,              -- 波高 (cm)
    stabilizer        BOOLEAN,           -- 安定板の使用有無

    display_time_1    REAL,              -- 1号艇 展示タイム
    display_time_2    REAL,
    display_time_3    REAL,
    display_time_4    REAL,
    display_time_5    REAL,
    display_time_6    REAL,

    created_at        TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_race_conditions UNIQUE (date, stadium, race_number)
);

-- ============================================================
-- 4. trifecta_odds (3連単オッズ、全120通り)
-- ============================================================
CREATE TABLE IF NOT EXISTS trifecta_odds (
    id           BIGSERIAL PRIMARY KEY,
    date         DATE    NOT NULL,
    stadium      INTEGER NOT NULL,
    stadium_name TEXT,
    race_number  INTEGER NOT NULL,
    combination  TEXT    NOT NULL,       -- 例: "1-2-3"
    odds         REAL,

    created_at   TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_trifecta_odds UNIQUE (date, stadium, race_number, combination)
);

-- ============================================================
-- 5. win_odds (単勝オッズ)
-- ============================================================
CREATE TABLE IF NOT EXISTS win_odds (
    id           BIGSERIAL PRIMARY KEY,
    date         DATE    NOT NULL,
    stadium      INTEGER NOT NULL,
    stadium_name TEXT,
    race_number  INTEGER NOT NULL,
    boat_number  INTEGER NOT NULL,       -- 艇番 (1〜6)
    odds         REAL,

    created_at   TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_win_odds UNIQUE (date, stadium, race_number, boat_number)
);

-- ============================================================
-- 6. place_odds (複勝オッズ)
-- ============================================================
CREATE TABLE IF NOT EXISTS place_odds (
    id           BIGSERIAL PRIMARY KEY,
    date         DATE    NOT NULL,
    stadium      INTEGER NOT NULL,
    stadium_name TEXT,
    race_number  INTEGER NOT NULL,
    boat_number  INTEGER NOT NULL,
    odds         REAL,

    created_at   TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_place_odds UNIQUE (date, stadium, race_number, boat_number)
);

-- ============================================================
-- 7. current_series (今節成績: race_cards の result を正規化)
-- ============================================================
CREATE TABLE IF NOT EXISTS current_series (
    id           BIGSERIAL PRIMARY KEY,
    date         DATE    NOT NULL,
    stadium      INTEGER NOT NULL,
    stadium_name TEXT,
    racerid      INTEGER NOT NULL,
    race_number  INTEGER NOT NULL,
    boat_number  INTEGER,                -- 艇番 (1〜6)
    course       INTEGER,                -- 進入コース
    ST           REAL,                   -- スタートタイミング
    rank         INTEGER,                -- 着順

    created_at   TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_current_series UNIQUE (date, stadium, racerid, race_number)
);

-- ============================================================
-- 検索を速くするための INDEX (任意、後から追加可)
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_race_cards_date_stadium      ON race_cards (date, stadium);
CREATE INDEX IF NOT EXISTS idx_race_results_date_stadium    ON race_results (date, stadium);
CREATE INDEX IF NOT EXISTS idx_race_conditions_date_stadium ON race_conditions (date, stadium);
CREATE INDEX IF NOT EXISTS idx_trifecta_odds_date_stadium   ON trifecta_odds (date, stadium);
CREATE INDEX IF NOT EXISTS idx_win_odds_date_stadium        ON win_odds (date, stadium);
CREATE INDEX IF NOT EXISTS idx_place_odds_date_stadium      ON place_odds (date, stadium);
CREATE INDEX IF NOT EXISTS idx_current_series_date_stadium  ON current_series (date, stadium);
