import uuid
import json
from embedder import embed_query_async
from chunker import Chunk
from database import get_pool
from config import TOP_K

async def retrieve_async(query: str, top_k: int = TOP_K) -> list[dict]:
    query_embedding = await embed_query_async(query)
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Use cosine distance <=> and order by it
        results = await conn.fetch(f"""
            SELECT chunk_id, chunk_text, metadata,
                   1 - (embedding <=> $1) AS score
            FROM document_chunks
            ORDER BY embedding <=> $1
            LIMIT $2
        """, query_embedding, top_k)
        
    chunks = []
    for r in results:
        meta = json.loads(r['metadata']) if isinstance(r['metadata'], str) else r['metadata']
        chunks.append({
            "chunk_id": str(r['chunk_id']),
            "text": r['chunk_text'],
            "metadata": meta,
            "score": float(r['score'])
        })
    return chunks

async def insert_chunks_async(chunks: list[Chunk], embeddings: list[list[float]]):
    pool = await get_pool()
    async with pool.acquire() as conn:
        records = []
        for chunk, emb in zip(chunks, embeddings):
            chunk_id = str(uuid.uuid4())
            doc_id = chunk.metadata.get("source", "unknown")
            records.append((
                chunk_id, doc_id, chunk.text, emb, json.dumps(chunk.metadata)
            ))
            
        await conn.executemany("""
            INSERT INTO document_chunks (chunk_id, doc_id, chunk_text, embedding, metadata)
            VALUES ($1::uuid, $2, $3, $4, $5::jsonb)
        """, records)
