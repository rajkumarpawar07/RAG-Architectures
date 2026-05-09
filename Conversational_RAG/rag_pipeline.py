import time
from config import DATA_DIR, TOP_K
from document_loader import load_directory
from chunker import chunk_documents
from embedder import embed_texts, embed_query
from vector_store import init_collection, upsert_chunks, search, count_vectors
from memory import init_db, save_turn, get_history
from query_rewriter import rewrite_query
from generator import generate

def setup():
    """Initialize DB and Vector Store collections."""
    init_db()
    init_collection()

def ingest() -> dict:
    print("\n" + "=" * 60)
    print("INGESTION PIPELINE (Conversational RAG)")
    print("=" * 60)
    start = time.time()
    setup()

    print("\n[1/4] Loading documents...")
    documents = load_directory(DATA_DIR)
    if not documents:
        raise FileNotFoundError(f"No files found in {DATA_DIR}.")
    
    print("\n[2/4] Chunking documents...")
    chunks = chunk_documents(documents)
    
    print("\n[3/4] Embedding chunks...")
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    
    print("\n[4/4] Upserting to Qdrant...")
    upsert_chunks(embeddings, chunks)

    elapsed = time.time() - start
    stats = {"documents": len(documents), "chunks": len(chunks), "time_seconds": round(elapsed, 2)}
    
    print(f"\nIngestion complete in {elapsed:.1f}s")
    return stats

def query(session_id: str, question: str, top_k: int = TOP_K) -> dict:
    setup()
    
    # 1. Fetch History
    history = get_history(session_id)
    
    # 2. Rewrite Query (if history exists)
    standalone_query = rewrite_query(question, history)
    
    # 3. Embed Standalone Query
    query_embedding = embed_query(standalone_query)
    
    # 4. Search Qdrant
    results = search(query_embedding, top_k=top_k)
    
    # 5. Generate Answer (Passing standalone query to prompt, and history to context)
    answer = generate(standalone_query, results, history)
    
    # 6. Save Turn to Memory (Original user query, not standalone)
    save_turn(session_id, question, answer)
    
    sources = list(set([f"{r.get('source')} (score: {r.get('score'):.3f})" for r in results]))
    
    return {
        "question": question,
        "standalone_query": standalone_query,
        "answer": answer,
        "sources": sources
    }

def get_stats() -> dict:
    setup()
    return {"total_vectors_in_qdrant": count_vectors()}
