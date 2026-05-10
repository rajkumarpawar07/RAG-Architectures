import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

class Settings(BaseModel):
    GEMINI_API_KEY: str = Field(default=os.getenv("GEMINI_API_KEY", ""))
    CHROMA_DIR: str = Field(default=str(CHROMA_DIR))
    
    CHUNK_SIZE: int = Field(default=800)
    CHUNK_OVERLAP: int = Field(default=150)
    TOP_K: int = Field(default=5)
    
    RELEVANCE_THRESHOLD: int = Field(default=3)
    USEFULNESS_THRESHOLD: int = Field(default=3)
    
    MAX_REVISIONS: int = Field(default=3)
    MAX_RETRIEVAL_LOOPS: int = Field(default=2)
    
    EMBEDDING_DIM: int = Field(default=768)

settings = Settings()
