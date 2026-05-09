import uuid
import numpy as np
from qdrant_client import QdrantClient, models

from config import QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_DIMENSION, TOP_K
from chunker import Chunk

def _get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)

def init_collection():
    client = _get_client()
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=models.Distance.COSINE
            ),
        )
        print(f"  [QDRANT] Created collection '{QDRANT_COLLECTION}'")

def upsert_chunks(embeddings: np.ndarray, chunks: list[Chunk]):
    client = _get_client()
    points = []
    for i, chunk in enumerate(chunks):
        point_id = str(uuid.uuid4())
        # Store text and all metadata
        payload = {"text": chunk.text, **chunk.metadata}
        points.append(
            models.PointStruct(
                id=point_id,
                vector=embeddings[i].tolist(),
                payload=payload
            )
        )
    
    # Upsert in batches of 100 to avoid large payload errors
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points[i:i+batch_size]
        )
    print(f"  [QDRANT] Upserted {len(points)} vectors to '{QDRANT_COLLECTION}'")

def search(query_embedding: np.ndarray, top_k: int = TOP_K) -> list[dict]:
    client = _get_client()
    if not client.collection_exists(QDRANT_COLLECTION):
        return []
    
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding[0].tolist(),
        limit=top_k,
    ).points
    
    formatted_results = []
    for res in results:
        formatted_results.append({
            "text": res.payload.get("text", ""),
            "score": res.score,
            "source": res.payload.get("source", "unknown"),
            "chunk_index": res.payload.get("chunk_index", -1)
        })
    return formatted_results

def count_vectors() -> int:
    client = _get_client()
    if not client.collection_exists(QDRANT_COLLECTION):
        return 0
    return client.count(collection_name=QDRANT_COLLECTION).count
