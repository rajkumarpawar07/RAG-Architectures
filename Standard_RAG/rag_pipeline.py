"""
rag_pipeline.py — Orchestrates the full RAG pipeline.

Two main flows:
  1. Ingestion: Load documents -> Chunk -> Embed -> Build FAISS index -> Save
  2. Query:    Embed query -> Search index -> Build prompt -> Generate answer
"""

import time

from document_loader import load_directory
from chunker import chunk_documents, Chunk
from embedder import embed_texts, embed_query
from vector_store import build_index, save_index, load_index, search, index_exists
from generator import generate

from config import DATA_DIR, TOP_K


def ingest() -> dict:
    """
    Run the full ingestion pipeline.

    Steps:
      1. Load all PDFs and .txt files from data/
      2. Chunk documents using recursive character splitting
      3. Embed all chunks using Gemini embedding model
      4. Build and save the FAISS index

    Returns a summary dict with stats.
    """
    print("\n" + "=" * 60)
    print("INGESTION PIPELINE")
    print("=" * 60)
    start = time.time()

    # Step 1: Load documents
    print("\n[1/4] Loading documents...")
    documents = load_directory(DATA_DIR)
    if not documents:
        raise FileNotFoundError(
            f"No supported files found in {DATA_DIR}.\n"
            f"Place .pdf or .txt files in the data/ folder and try again."
        )
    print(f"  Loaded {len(documents)} document(s)")

    # Step 2: Chunk
    print("\n[2/4] Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"  Total chunks: {len(chunks)}")

    # Step 3: Embed
    print("\n[3/4] Embedding chunks...")
    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts)
    print(f"  Embedded {len(embeddings)} chunks ({embeddings.shape[1]}D vectors)")

    # Step 4: Build and save index
    print("\n[4/4] Building FAISS index...")
    index, metadata = build_index(embeddings, chunks)
    save_index(index, metadata)

    elapsed = time.time() - start
    stats = {
        "documents": len(documents),
        "chunks": len(chunks),
        "embedding_dim": embeddings.shape[1],
        "time_seconds": round(elapsed, 2),
    }

    print(f"\n{'=' * 60}")
    print(f"Ingestion complete in {elapsed:.1f}s")
    print(f"   Documents: {stats['documents']}")
    print(f"   Chunks:    {stats['chunks']}")
    print(f"{'=' * 60}\n")

    return stats


from evaluator import evaluate_in_background
from langfuse import get_client

langfuse = get_client()

def query(question: str, top_k: int = TOP_K) -> dict:
    """
    Run the full query pipeline with Langfuse tracing.
    """
    with langfuse.start_as_current_observation(as_type="span", name="rag") as trace:
        trace_id = trace.trace_id
        
        # Step 1: Load index
        index, metadata = load_index()
    
        # Step 2: Embed query
        with langfuse.start_as_current_observation(name="embed_query", input={"question": question}) as span:
            query_embedding = embed_query(question)
    
        # Step 3: Search
        with langfuse.start_as_current_observation(name="vector_search", input={"query_embedding_dim": len(query_embedding)}) as span:
            results = search(query_embedding, index, metadata, top_k=top_k)
    
        # Step 4: Generate answer
        with langfuse.start_as_current_observation(name="generation", input={"question": question, "retrieved_chunks": results}) as span:
            answer = generate(question, results)
            span.update(output={"answer": answer})
    
        # Build source list for display
        sources = []
        contexts = []
        for r in results:
            contexts.append(r.get("text", ""))
            source_info = f"{r.get('source', '?')} (chunk {r.get('chunk_index', '?')}, score: {r.get('score', 0):.3f})"
            if source_info not in sources:
                sources.append(source_info)
    
        trace.update(
            input={"question": question},
            output={"answer": answer, "sources": sources}
        )

    # Trigger background evaluation
    if trace_id:
        evaluate_in_background(trace_id, question, contexts, answer)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": results,
    }


def get_stats() -> dict:
    """Return statistics about the current index."""
    if not index_exists():
        return {"status": "No index found. Run 'python main.py ingest' first."}

    index, metadata = load_index()

    # Count unique sources
    unique_sources = set()
    for m in metadata:
        unique_sources.add(m.get("source", "unknown"))

    return {
        "status": "ready",
        "total_vectors": index.ntotal,
        "total_chunks": len(metadata),
        "documents": sorted(unique_sources),
        "document_count": len(unique_sources),
    }
