import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

class Settings(BaseModel):
    # Gemini
    GOOGLE_API_KEY: str = Field(default=os.getenv("GOOGLE_API_KEY", ""))

    # Qdrant
    QDRANT_URL: str = Field(default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    QDRANT_COLLECTION: str = Field(default=os.getenv("QDRANT_COLLECTION", "agentic_rag_docs"))

    # Tavily
    TAVILY_API_KEY: str = Field(default=os.getenv("TAVILY_API_KEY", ""))

    # Langsmith (read by LangChain SDK automatically, but stored here for clarity)
    LANGSMITH_API_KEY: str = Field(default=os.getenv("LANGSMITH_API_KEY", ""))
    LANGSMITH_PROJECT: str = Field(default=os.getenv("LANGSMITH_PROJECT", "agentic-rag"))

    # Chunking
    CHUNK_SIZE: int = Field(default=800)
    CHUNK_OVERLAP: int = Field(default=150)

    # Retrieval
    TOP_K: int = Field(default=5)
    EMBEDDING_DIM: int = Field(default=768)

    # Agent
    MAX_ITERATIONS: int = Field(default=5)

    # Models
    LLM_MODEL: str = Field(default="gemini-3.1-flash-lite")
    EMBEDDING_MODEL: str = Field(default="models/text-embedding-004")

settings = Settings()
