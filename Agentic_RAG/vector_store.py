import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
)
from google import genai
from google.genai import types
from config import settings
from chunker import Chunk

logger = logging.getLogger(__name__)

# Gemini client for embeddings
_genai_client = genai.Client(api_key=settings.GOOGLE_API_KEY)


def _embed(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed a batch of texts using gemini-embedding-001 at 768D."""
    if not texts:
        return []
    response = _genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=settings.EMBEDDING_DIM,
        ),
    )
    return [e.values for e in response.embeddings]


class VectorStore:
    def __init__(self):
        self.client = QdrantClient(url=settings.QDRANT_URL)
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if settings.QDRANT_COLLECTION not in existing:
            self.client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {settings.QDRANT_COLLECTION}")
        else:
            logger.info(f"Using existing Qdrant collection: {settings.QDRANT_COLLECTION}")

    def count(self) -> int:
        return self.client.count(collection_name=settings.QDRANT_COLLECTION).count

    def upsert(self, chunks: list[Chunk], batch_size: int = 100):
        """Embed and upsert chunks in batches. Idempotent via deterministic IDs."""
        total = len(chunks)
        logger.info(f"Upserting {total} chunks to Qdrant...")

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = _embed(texts, task_type="RETRIEVAL_DOCUMENT")

            points = [
                PointStruct(
                    id=abs(hash(c.id)) % (2**63),  # Qdrant needs uint64
                    vector=emb,
                    payload={"text": c.text, **c.metadata},
                )
                for c, emb in zip(batch, embeddings)
            ]
            self.client.upsert(
                collection_name=settings.QDRANT_COLLECTION, points=points
            )
            logger.info(f"  Upserted batch {i // batch_size + 1} ({len(batch)} chunks)")

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Embed query and return top-k results with text and metadata."""
        k = top_k or settings.TOP_K
        if self.count() == 0:
            return []

        query_vec = _embed([query], task_type="RETRIEVAL_QUERY")[0]
        hits = self.client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_vec,
            limit=k,
            with_payload=True,
        )
        return [
            {
                "text": h.payload.get("text", ""),
                "source": h.payload.get("source", "unknown"),
                "score": h.score,
            }
            for h in hits
        ]


_store_instance: "VectorStore | None" = None


def get_store() -> "VectorStore":
    global _store_instance
    if _store_instance is None:
        _store_instance = VectorStore()
    return _store_instance


# Backward-compatible alias — only connects when first accessed
class _LazyStore:
    def __getattr__(self, name):
        return getattr(get_store(), name)


store = _LazyStore()
