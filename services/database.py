"""Database layer (SQLite).

Бүх SQL statement нь доорх тогтмол string-ууд бөгөөд утгыг зөвхөн
"?" placeholder-оор дамжуулна (parameterized query). f-string болон
string concatenation ашиглан SQL үүсгэхгүй.

Хэрэглэгчийн нэр, файлын нэр, зургийн binary зэрэг PII хадгалахгүй.
"""

import sqlite3

import config

INSERT_ANALYSIS = "INSERT INTO analyses (dataset, file_ext) VALUES (?, ?)"

INSERT_PREDICTION = """
    INSERT INTO predictions
        (analysis_id, model_name, predicted_class, label, confidence, decision)
    VALUES (?, ?, ?, ?, ?, ?)
"""

SELECT_ANALYSIS = """
    SELECT id, created_at, dataset, file_ext
    FROM analyses
    WHERE id = ?
"""

SELECT_PREDICTIONS_FOR_ANALYSIS = """
    SELECT p.id, p.analysis_id, p.model_name, p.predicted_class, p.label,
           p.confidence, p.decision
    FROM predictions AS p
    JOIN analyses AS a ON a.id = p.analysis_id
    WHERE a.id = ?
    ORDER BY p.id
"""


def get_connection(db_path=None):
    connection = sqlite3.connect(db_path or config.DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db(db_path=None):
    """db/schema.sql-ийг ажиллуулж хүснэгтүүдийг үүсгэнэ (байвал алгасна)."""
    db_path = db_path or config.DATABASE_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema = config.SCHEMA_PATH.read_text(encoding="utf-8")

    connection = get_connection(db_path)
    try:
        connection.executescript(schema)
    finally:
        connection.close()


def save_analysis(dataset, file_ext, predictions, db_path=None):
    """analyses мөр болон түүнд хамаарах predictions мөрүүдийг нэг transaction-д бичнэ.

    predictions: [{"model_name", "predicted_class", "label", "confidence", "decision"}]
    Буцаах утга: шинэ analysis id.
    """
    connection = get_connection(db_path)
    try:
        with connection:
            cursor = connection.execute(INSERT_ANALYSIS, (dataset, file_ext))
            analysis_id = cursor.lastrowid

            for prediction in predictions:
                connection.execute(
                    INSERT_PREDICTION,
                    (
                        analysis_id,
                        prediction["model_name"],
                        str(prediction["predicted_class"]),
                        prediction["label"],
                        float(prediction["confidence"]),
                        prediction["decision"],
                    ),
                )
        return analysis_id
    finally:
        connection.close()


def get_analysis(analysis_id, db_path=None):
    """Нэг шинжилгээ болон түүний prediction-уудыг буцаана (байхгүй бол None)."""
    connection = get_connection(db_path)
    try:
        analysis = connection.execute(SELECT_ANALYSIS, (analysis_id,)).fetchone()
        if analysis is None:
            return None

        rows = connection.execute(
            SELECT_PREDICTIONS_FOR_ANALYSIS, (analysis_id,)
        ).fetchall()
        result = dict(analysis)
        result["predictions"] = [dict(row) for row in rows]
        return result
    finally:
        connection.close()
