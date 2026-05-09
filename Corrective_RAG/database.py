import asyncpg
from pgvector.asyncpg import register_vector
from config import POSTGRES_DSN, EMBEDDING_DIM

_pool = None

async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        async def init(conn):
            await register_vector(conn)
        _pool = await asyncpg.create_pool(POSTGRES_DSN, init=init, min_size=1, max_size=20)
    return _pool

async def init_db():
    conn = await asyncpg.connect(POSTGRES_DSN)
    try:
        # Create extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # 1. document_chunks table
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id UUID PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding vector({EMBEDDING_DIM}),
                metadata JSONB
            );
        """)
        
        # HNSW Index for document_chunks
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx 
            ON document_chunks 
            USING hnsw (embedding vector_cosine_ops);
        """)
        
        # 2. chat_history table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                turn_id INT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 3. crag_runs table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS crag_runs (
                run_id UUID PRIMARY KEY,
                query TEXT NOT NULL,
                decision TEXT,
                web_search_triggered BOOLEAN,
                latency_ms INT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # 4. eval_cache table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_cache (
                hash_key TEXT PRIMARY KEY,
                score TEXT,
                reason TEXT
            );
        """)
        print("  [DB] Database initialized successfully.")
    finally:
        await conn.close()
