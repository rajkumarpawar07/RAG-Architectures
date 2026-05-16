import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, DataType, Property
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from config import settings
import logging
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Custom Langchain Embeddings wrapper using the official google-genai SDK
class GenAIEmbeddings(Embeddings):
    def __init__(self):
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        # Using the known working string from Agentic RAG
        self.model = "gemini-embedding-001"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            ),
        )
        return [e.values for e in response.embeddings]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768,
            ),
        )
        return response.embeddings[0].values

embeddings = GenAIEmbeddings()

class VectorStoreManager:
    def __init__(self):
        # Connect to Weaviate local Docker instance (v4 client)
        self.client = weaviate.connect_to_local(
            port=8080,
            grpc_port=50051
        )
        self.index_name = settings.WEAVIATE_INDEX_NAME
        self._ensure_collection()
        
        # Initialize Langchain wrapper
        self.vector_store = WeaviateVectorStore(
            client=self.client,
            index_name=self.index_name,
            text_key="text",
            embedding=embeddings
        )

    def _ensure_collection(self):
        """Create the Weaviate collection if it doesn't exist."""
        if not self.client.collections.exists(self.index_name):
            logger.info(f"Creating Weaviate collection: {self.index_name}")
            self.client.collections.create(
                name=self.index_name,
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="chunk_id", data_type=DataType.TEXT),
                ],
            )
            
    def get_retriever(self, k: int):
        return self.vector_store.as_retriever(search_kwargs={"k": k})

    def upsert(self, chunks: list[Document]):
        """Upsert documents into Weaviate using the Langchain wrapper."""
        if not chunks:
            return
            
        # Extract UUIDs based on chunk_ids for deterministic insertion
        uuids = [
            weaviate.util.generate_uuid5(chunk.metadata["chunk_id"]) 
            for chunk in chunks
        ]
        
        self.vector_store.add_documents(chunks, uuids=uuids)
        logger.info(f"Upserted {len(chunks)} chunks into Weaviate.")
        
    def count(self) -> int:
        """Return the total number of objects in the collection."""
        collection = self.client.collections.get(self.index_name)
        return len(collection)

    def close(self):
        self.client.close()

# Provide a lazy singleton to avoid connecting on import if Weaviate is offline
_store_instance = None

def get_store() -> VectorStoreManager:
    global _store_instance
    if _store_instance is None:
        _store_instance = VectorStoreManager()
    return _store_instance

class _LazyStore:
    def __getattr__(self, name):
        return getattr(get_store(), name)

store = _LazyStore()
