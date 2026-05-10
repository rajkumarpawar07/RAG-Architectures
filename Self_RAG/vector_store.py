import chromadb
from config import settings
from llm_client import embed_texts
import logging

logger = logging.getLogger("self_rag")

class VectorStore:
    def __init__(self):
        # Persistent local storage
        self.client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name="self_rag_docs",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB initialized collection='self_rag_docs' path='{settings.CHROMA_DIR}'")

    def index_exists(self) -> bool:
        return self.collection.count() > 0
        
    def get_stats(self) -> dict:
        return {
            "total_chunks": self.collection.count()
        }

    def upsert_chunks(self, chunks: list):
        """
        Embeds and upserts chunks into ChromaDB.
        Idempotent operation because chunks have deterministic hashes.
        """
        if not chunks:
            return
            
        batch_size = 100
        total_chunks = len(chunks)
        
        logger.info(f"Upserting chunks to ChromaDB total={total_chunks}")
        
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i+batch_size]
            texts = [c.text for c in batch_chunks]
            
            # Embed with RETRIEVAL_DOCUMENT
            embeddings = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
            
            ids = [c.id for c in batch_chunks]
            metadatas = [c.metadata for c in batch_chunks]
            
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            logger.info(f"Upserted batch start={i} count={len(batch_chunks)}")
            
    def search(self, query: str, top_k: int = None) -> list[dict]:
        """
        Embeds query and searches the collection.
        """
        if top_k is None:
            top_k = settings.TOP_K
            
        if not self.index_exists():
            logger.warning("Search attempted on empty index")
            return []
            
        # Embed with RETRIEVAL_QUERY
        query_embedding = embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
            
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]  # cosine distance
            })
            
        return formatted_results

store = VectorStore()
