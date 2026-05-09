"""
vector_store.py — FAISS vector store operations.

Manages building, saving, loading, and searching a FAISS index.
Uses IndexFlatIP (inner product) with L2-normalized vectors,
which is equivalent to cosine similarity -- the standard choice
for semantic search with dense embeddings.

Chunk metadata is stored alongside the index as a JSON file
for human-readability and debuggability.
"""

import json
from pathlib import Path

import faiss
import numpy as np

from config import INDEX_DIR, EMBEDDING_DIMENSION, TOP_K
from chunker import Chunk


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize each vector so that inner product == cosine similarity.

    This is a production best practice: cosine similarity is more
    robust than raw dot product for comparing text embeddings because
    it ignores magnitude differences.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms


def build_index(embeddings: np.ndarray, chunks: list[Chunk]) -> tuple[faiss.Index, list[dict]]:
    """
    Build a FAISS index from embeddings and their corresponding chunks.

    Args:
        embeddings: numpy array of shape (n, embedding_dim)
        chunks:     list of Chunk objects (same order as embeddings)

    Returns:
        (faiss_index, chunk_metadata_list)
    """
    assert len(embeddings) == len(chunks), (
        f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks"
    )

    # Normalize for cosine similarity
    normalized = _normalize(embeddings)

    # Create a flat inner-product index (exact search, no approximation)
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(normalized)

    # Extract metadata for persistence
    metadata = [chunk.metadata for chunk in chunks]
    # Also store the text in metadata for retrieval
    for i, chunk in enumerate(chunks):
        metadata[i]["text"] = chunk.text

    print(f"  [INDEX] Built FAISS index with {index.ntotal} vectors ({EMBEDDING_DIMENSION}D)")
    return index, metadata


def save_index(
    index: faiss.Index,
    metadata: list[dict],
    index_dir: Path | None = None,
) -> None:
    """
    Persist FAISS index and chunk metadata to disk.

    Files created:
    - index.faiss   -- the FAISS binary index
    - metadata.json -- chunk metadata (text + source info)
    """
    index_dir = index_dir or INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "index.faiss"
    metadata_path = index_dir / "metadata.json"

    faiss.write_index(index, str(index_path))
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  [SAVE] Index saved to: {index_dir}")
    print(f"     - index.faiss  ({index_path.stat().st_size / 1024:.1f} KB)")
    print(f"     - metadata.json ({metadata_path.stat().st_size / 1024:.1f} KB)")


def load_index(index_dir: Path | None = None) -> tuple[faiss.Index, list[dict]]:
    """
    Load a persisted FAISS index and its metadata from disk.

    Raises FileNotFoundError if the index doesn't exist.
    """
    index_dir = index_dir or INDEX_DIR
    index_path = index_dir / "index.faiss"
    metadata_path = index_dir / "metadata.json"

    if not index_path.exists():
        raise FileNotFoundError(
            f"No FAISS index found at {index_path}.\n"
            f"Run 'python main.py ingest' first to build the index."
        )

    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    print(f"  [INDEX] Loaded FAISS index: {index.ntotal} vectors")
    return index, metadata


def index_exists(index_dir: Path | None = None) -> bool:
    """Check whether a persisted index exists on disk."""
    index_dir = index_dir or INDEX_DIR
    return (index_dir / "index.faiss").exists()


def search(
    query_embedding: np.ndarray,
    index: faiss.Index,
    metadata: list[dict],
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Search the FAISS index for the top-K most similar chunks.

    Args:
        query_embedding: numpy array of shape (1, embedding_dim)
        index:           the FAISS index
        metadata:        list of chunk metadata dicts
        top_k:           number of results to return

    Returns:
        List of dicts, each containing:
        - "text": the chunk text
        - "score": cosine similarity score (0 to 1)
        - "source": source filename
        - "chunk_index": position within the source document
    """
    # Normalize query for cosine similarity
    query_normalized = _normalize(query_embedding)

    # Search
    scores, indices = index.search(query_normalized, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:  # FAISS returns -1 for padding when fewer results exist
            continue
        result = {
            **metadata[idx],
            "score": float(score),
        }
        results.append(result)

    return results
