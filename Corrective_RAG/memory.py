from typing import List, Dict
from database import get_pool
from config import MEMORY_WINDOW

async def save_turn_async(session_id: str, user_query: str, bot_answer: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        val = await conn.fetchval("SELECT MAX(turn_id) FROM chat_history WHERE session_id = $1", session_id)
        next_turn = (val or 0) + 1
        
        await conn.execute("""
            INSERT INTO chat_history (session_id, turn_id, role, content)
            VALUES ($1, $2, $3, $4)
        """, session_id, next_turn, "user", user_query)
        
        await conn.execute("""
            INSERT INTO chat_history (session_id, turn_id, role, content)
            VALUES ($1, $2, $3, $4)
        """, session_id, next_turn, "model", bot_answer)

async def get_history_async(session_id: str, limit: int = MEMORY_WINDOW) -> List[Dict[str, str]]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT role, content FROM chat_history
            WHERE session_id = $1
            ORDER BY turn_id DESC, id DESC
            LIMIT $2
        """, session_id, limit * 2)
        
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
