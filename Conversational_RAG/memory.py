import sqlite3
from typing import List, Dict
from config import MEMORY_DB_PATH, MEMORY_WINDOW

def _get_connection():
    conn = sqlite3.connect(MEMORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def save_turn(session_id: str, user_query: str, bot_answer: str):
    with _get_connection() as conn:
        cursor = conn.cursor()
        # Get next turn_id for this session
        cursor.execute("SELECT MAX(turn_id) FROM chat_history WHERE session_id = ?", (session_id,))
        result = cursor.fetchone()
        next_turn = (result[0] or 0) + 1

        # Insert User
        cursor.execute("""
            INSERT INTO chat_history (session_id, turn_id, role, content)
            VALUES (?, ?, ?, ?)
        """, (session_id, next_turn, "user", user_query))
        
        # Insert Bot
        cursor.execute("""
            INSERT INTO chat_history (session_id, turn_id, role, content)
            VALUES (?, ?, ?, ?)
        """, (session_id, next_turn, "model", bot_answer))
        conn.commit()

def get_history(session_id: str, limit: int = MEMORY_WINDOW) -> List[Dict[str, str]]:
    """Get the last N conversational turns for a session."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        # Each turn has 2 rows (user + model). So limit * 2.
        cursor.execute("""
            SELECT role, content FROM chat_history
            WHERE session_id = ?
            ORDER BY turn_id DESC, id DESC
            LIMIT ?
        """, (session_id, limit * 2))
        rows = cursor.fetchall()
        
    # Reverse to return in chronological order
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
