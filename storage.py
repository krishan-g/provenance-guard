import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "audit_log.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS submissions (
                content_id TEXT PRIMARY KEY,
                creator_id TEXT NOT NULL,
                text TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                llm_score REAL,
                style_score REAL,
                confidence REAL,
                attribution TEXT,
                label TEXT,
                status TEXT NOT NULL,
                -- populated once a creator appeals (Milestone 5)
                appeal_reasoning TEXT,
                appeal_timestamp TEXT
            )
            """
        )


def insert_submission(record):
    """Writes a new submission row (one per /submit call)."""
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO submissions (
                content_id, creator_id, text, timestamp,
                llm_score, style_score, confidence, attribution, label, status
            ) VALUES (:content_id, :creator_id, :text, :timestamp,
                      :llm_score, :style_score, :confidence, :attribution, :label, :status)
            """,
            record,
        )


def get_recent_entries(limit=20):
    """Returns the most recent submissions, newest first, for GET /log."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM submissions ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def get_submission(content_id):
    """Looks up a single submission by content_id, or None if it doesn't exist."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM submissions WHERE content_id = ?", (content_id,)
        ).fetchone()
        return dict(row) if row else None


def record_appeal(content_id, reasoning, timestamp):
    """Marks a submission as under review and attaches the creator's appeal reasoning."""
    with get_conn() as conn:
        cursor = conn.execute(
            """
            UPDATE submissions
            SET status = 'under_review', appeal_reasoning = ?, appeal_timestamp = ?
            WHERE content_id = ?
            """,
            (reasoning, timestamp, content_id),
        )
        return cursor.rowcount > 0
