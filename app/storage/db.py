import sqlite3
from datetime import datetime
from app.config.config import DB_PATH

# Single writer connection for inserts from AsyncSaver thread (WAL allows concurrent readers elsewhere)
_writer_conn = None


def _get_writer_conn():
    global _writer_conn
    if _writer_conn is None:
        _writer_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _writer_conn.execute("PRAGMA journal_mode=WAL")
        _writer_conn.commit()
    return _writer_conn


def _close_writer_conn():
    global _writer_conn
    if _writer_conn is not None:
        _writer_conn.close()
        _writer_conn = None


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            timestamp TEXT,
            source TEXT,
            image_path TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_detection(name, source, image_path):
    conn = _get_writer_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO detections (name, timestamp, source, image_path)
        VALUES (?, ?, ?, ?)
        """,
        (name, datetime.now().isoformat(), source, image_path),
    )
    conn.commit()
    return cursor.lastrowid


def fetch_detections(filter_text=""):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    normalized_filter = (filter_text or "").strip()
    if normalized_filter:
        cursor.execute(
            """
            SELECT id, name, timestamp, image_path, source
            FROM detections
            WHERE name LIKE ?
            ORDER BY timestamp DESC
            """,
            (f"%{normalized_filter}%",),
        )
    else:
        cursor.execute(
            """
            SELECT id, name, timestamp, image_path, source
            FROM detections
            ORDER BY timestamp DESC
            """
        )

    rows = cursor.fetchall()
    conn.close()
    return rows


def delete_detection(detection_id):
    """Delete a single detection record by its id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detections WHERE id = ?", (detection_id,))
    conn.commit()
    conn.close()


def clear_detections():
    """Delete all detections and vacuum DB. Returns number of deleted rows."""
    _close_writer_conn()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detections")
    deleted_records = cursor.rowcount
    conn.commit()
    conn.close()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("VACUUM")
    conn.commit()
    conn.close()

    return deleted_records

