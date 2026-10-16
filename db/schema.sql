-- Macro&Micro SQLite schema
-- PII (хэрэглэгчийн нэр, файлын нэр, зургийн binary) хадгалахгүй.

PRAGMA foreign_keys = ON;

-- Нэг upload/шинжилгээний хүсэлт
CREATE TABLE IF NOT EXISTS analyses (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    dataset    TEXT    NOT NULL,
    file_ext   TEXT    NOT NULL
);

-- Тухайн шинжилгээнд гарсан model бүрийн prediction
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id     INTEGER NOT NULL,
    model_name      TEXT    NOT NULL,
    predicted_class TEXT    NOT NULL,
    label           TEXT    NOT NULL,
    confidence      REAL    NOT NULL CHECK (confidence >= 0 AND confidence <= 100),
    decision        TEXT    NOT NULL CHECK (decision IN ('classified', 'low_confidence')),
    FOREIGN KEY (analysis_id) REFERENCES analyses (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_predictions_analysis_id
    ON predictions (analysis_id);
