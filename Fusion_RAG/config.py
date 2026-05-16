from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os
from dotenv import load_dotenv

# Load explicitly from .env if running scripts directly
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Absolute path to the data directory
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    LANGSMITH_TRACING: str = "true"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "FusionRAG"
    
    # Weaviate Settings
    WEAVIATE_URL: str = "http://localhost:8080"
    WEAVIATE_INDEX_NAME: str = "FusionRAGDocs"
    
    # Retrieval & Pipeline Settings
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    EMBEDDING_MODEL: str = "models/embedding-001"
    LLM_MODEL: str = "gemini-3.1-flash-lite"
    
    # RRF specific
    NUM_QUERIES: int = 4
    TOP_K_PER_QUERY: int = 4
    FINAL_TOP_K: int = 5

    model_config = SettingsConfigDict(
        env_file=str(env_path), 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

settings = Settings()

# Ensure Langsmith tracing is enabled in env
os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGSMITH_TRACING
os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
