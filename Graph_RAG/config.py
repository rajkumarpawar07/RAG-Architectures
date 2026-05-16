from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os
from dotenv import load_dotenv

# Load explicitly from .env if running scripts directly
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    
    # Neo4j Settings
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    
    # LangSmith Tracing
    LANGSMITH_TRACING: str = "true"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "GraphRAG"

    # Chunking
    CHUNK_SIZE: int = 500 # Smaller chunks are better for graph extraction
    CHUNK_OVERLAP: int = 50

    # Models
    LLM_MODEL: str = "gemini-3.1-flash-lite"

    model_config = SettingsConfigDict(
        env_file=str(env_path), 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

settings = Settings()

# Ensure variables are set in env
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
if settings.LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGSMITH_TRACING
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
