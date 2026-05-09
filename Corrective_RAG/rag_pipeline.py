import time
import uuid
from config import DATA_DIR, TOP_K
from database import init_db, get_pool
from document_loader import load_directory
from chunker import chunk_documents
from embedder import embed_texts_async
from retriever import retrieve_async, insert_chunks_async
from evaluator import evaluate_chunks_async
from refiner import refine_chunks_async
from web_search import web_search_async
from generator import generate_async
from memory import save_turn_async, get_history_async
from enum import Enum

class GateDecision(Enum):
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    AMBIGUOUS = "AMBIGUOUS"

async def ingest_async():
    print("\n" + "=" * 60)
    print("INGESTION PIPELINE (Corrective RAG)")
    print("=" * 60)
    start = time.time()
    await init_db()
    
    print("\n[1/4] Loading documents...")
    documents = load_directory(DATA_DIR)
    if not documents:
        raise FileNotFoundError(f"No files found in {DATA_DIR}.")
        
    print("\n[2/4] Chunking documents...")
    chunks = chunk_documents(documents)
    
    print("\n[3/4] Embedding chunks...")
    texts = [c.text for c in chunks]
    embeddings = await embed_texts_async(texts)
    
    print("\n[4/4] Upserting to PostgreSQL...")
    await insert_chunks_async(chunks, embeddings)
    
    elapsed = time.time() - start
    print(f"\nIngestion complete in {elapsed:.1f}s")
    return {"documents": len(documents), "chunks": len(chunks), "time_seconds": elapsed}

def decide(graded_chunks: list[dict]) -> tuple[GateDecision, list[dict]]:
    correct = [c for c in graded_chunks if c["eval_score"] == "correct"]
    ambiguous = [c for c in graded_chunks if c["eval_score"] == "ambiguous"]
    incorrect = [c for c in graded_chunks if c["eval_score"] == "incorrect"]

    if len(correct) >= 2:
        return GateDecision.CORRECT, correct
    elif len(incorrect) == len(graded_chunks):
        return GateDecision.INCORRECT, []
    else:
        return GateDecision.AMBIGUOUS, correct + ambiguous

async def query_async(session_id: str, question: str) -> dict:
    start_time = time.time()
    await init_db()
    run_id = str(uuid.uuid4())
    
    history = await get_history_async(session_id)
    
    # 1. Retrieve
    chunks = await retrieve_async(question, top_k=TOP_K)
    
    # 2. Evaluate
    graded_chunks = await evaluate_chunks_async(question, chunks)
    decision, filtered_chunks = decide(graded_chunks)
    print(f"  [GATE] Decision: {decision.value}")
    
    final_context = ""
    web_triggered = False
    
    # 3. Branching logic
    if decision == GateDecision.CORRECT:
        final_context = await refine_chunks_async(question, filtered_chunks)
    elif decision == GateDecision.INCORRECT:
        web_triggered = True
        final_context = await web_search_async(question)
    elif decision == GateDecision.AMBIGUOUS:
        web_triggered = True
        internal_context = await refine_chunks_async(question, filtered_chunks)
        web_context = await web_search_async(question)
        final_context = internal_context + "\n\n" + web_context
        
    # 4. Generate
    answer = await generate_async(question, final_context, history)
    
    # 5. Save memory
    await save_turn_async(session_id, question, answer)
    
    # 6. Log crag_runs
    latency = int((time.time() - start_time) * 1000)
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO crag_runs (run_id, query, decision, web_search_triggered, latency_ms)
            VALUES ($1::uuid, $2, $3, $4, $5)
        """, run_id, question, decision.value, web_triggered, latency)
        
    return {
        "question": question,
        "answer": answer,
        "decision": decision.value,
        "web_used": web_triggered,
        "latency_ms": latency
    }

async def get_stats_async() -> dict:
    await init_db()
    pool = await get_pool()
    async with pool.acquire() as conn:
        vec_count = await conn.fetchval("SELECT COUNT(*) FROM document_chunks")
        run_count = await conn.fetchval("SELECT COUNT(*) FROM crag_runs")
    return {"total_vectors": vec_count, "total_runs": run_count}
