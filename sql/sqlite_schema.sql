-- ============================================================
-- ボートレースデータ SQLite版 (Supabase schema.sql の移植)
--
-- 変更点:
--   BIGSERIAL          → INTEGER PRIMARY KEY AUTOINCREMENT
--   REAL/TEXT/INTEGER  → そのまま (SQLiteは動的型付けだが宣言は保持)
--   BOOLEAN            → INTEGER (0/1)
--   TIMESTAMPTZ DEFAULT NOW() → TEXT DEFAULT CURRENT_TIMESTAMP
--   CONSTRAINT uq_... → UNIQUE(...) のみ (SQLiteはON CONFLICTを列指定で使う)
--
-- 参考: detect_types=PARSE_DECLTYPES で DATE は datetime.date に自動変換される。
-- ============================================================

CREATE TABLE IF NOT EXISTS race_cards (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    date          DATE    NOT NULL,
    stadium       INTEGER NOT NULL,
    stadium_name  TEXT,
    race_number   INTEGER NOT NULL,
    lane          INTEGER NOT NULL,

    racerid       INTEGER,
    name          TEXT,
    class         TEXT,
    branch        TEXT,
    birthplace    TEXT,
    age           INTEGER,
    weight        REAL,

    f             INTEGER,
    l             INTEGER,
    aveST         REAL,

    global_win_pt REAL,
    global_in2nd  REAL,
    global_in3rd  REAL,
    local_win_pt  REAL,
    local_in2nd   REAL,
    local_in3rd   REAL,

    motor         INTEGER,
    motor_in2nd   REAL,
    motor_in3rd   REAL,
    boat          INTEGER,
    boat_in2nd    REAL,
    boat_in3rd    REAL,

    created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (date, stadium, race_number, lane)
);

CREATE TABLE IF NOT EXISTS race_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    date         DATE    NOT NULL,
    stadium      INTEGER NOT NULL,
    stadium_name TEXT,
    race_number  INTEGER NOT NULL,

    rank         INTEGER,
    boat         INTEGER NOT NULL,
    racerid      INTEGER,
    name         TEXT,
    time         TEXT,
    time_sec     REAL,

    created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (date, stadium, race_number, boat)
);

-- 風向コード (boatrace.jp): 1=北 2=北北東 3=北東 ... 16=北北西
CREATE TABLE IF NOT EXISTS race_conditions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    date              DATE    NOT NULL,
    stadium           INTEGER NOT NULL,
    stadium_name      TEXT,
    race_number       INTEGER NOT NULL,

    weather           TEXT,
    temperature       REAL,
    wind_direction    INTEGER,
    wind_speed        REAL,
    water_temperature REAL,
    wave_height       REAL,
    stabilizer        INTEGER,

    display_time_1    REAL,
    display_time_2    REAL,
    display_time_3    REAL,
    display_time_4    REAL,
    display_time_5    REAL,
    display_time_6    REAL,

    created_at        TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (date, stadium, race_number)
);

CREATE TABLE IF NOT EXISTS trifecta_odds (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    date         DATE    NOT NULL,
    stadium      INTEGER NOT NULL,
    stadium_name TEXT,
    race_number  INTEGER NOT NULL,
    combination  TEXT    NOT NULL,
    odds         REAL,

    created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (date, stadium, race_number, combination)
);

CREATE TABLE IF NOT EXISTS win_odds (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    date         DATE    NOT NULL,
    stadium      INTEGER NOT NULL,
    stadium_name TEXT,
    race_number  INTEGER NOT NULL,
    boat_number  INTEGER NOT NULL,
    odds         REAL,

    created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (date, stadium, race_number, boat_number)
);

CREATE TABLE IF NOT EXISTS place_odds (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    date         DATE    NOT NULL,
    stadium      INTEGER NOT NULL,
    stadium_name TEXT,
    race_number  INTEGER NOT NULL,
    boat_number  INTEGER NOT NULL,
    odds         REAL,

    created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (date, stadium, race_number, boat_number)
);

CREATE TABLE IF NOT EXISTS current_series (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    date         DATE    NOT NULL,
    stadium      INTEGER NOT NULL,
    stadium_name TEXT,
    racerid      INTEGER NOT NULL,
    race_number  INTEGER NOT NULL,
    boat_number  INTEGER,
    course       INTEGER,
    st           REAL,
    rank         INTEGER,

    created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (date, stadium, racerid, race_number)
);

-- 検索高速化用 INDEX
CREATE INDEX IF NOT EXISTS idx_race_cards_date_stadium      ON race_cards (date, stadium);
CREATE INDEX IF NOT EXISTS idx_race_results_date_stadium    ON race_results (date, stadium);
CREATE INDEX IF NOT EXISTS idx_race_conditions_date_stadium ON race_conditions (date, stadium);
CREATE INDEX IF NOT EXISTS idx_trifecta_odds_date_stadium   ON trifecta_odds (date, stadium);
CREATE INDEX IF NOT EXISTS idx_win_odds_date_stadium        ON win_odds (date, stadium);
CREATE INDEX IF NOT EXISTS idx_place_odds_date_stadium      ON place_odds (date, stadium);
CREATE INDEX IF NOT EXISTS idx_current_series_date_stadium  ON current_series (date, stadium);
CREATE INDEX IF NOT EXISTS idx_current_series_racerid       ON current_series (racerid);
