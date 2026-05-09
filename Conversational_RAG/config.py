import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MEMORY_DB_PATH = BASE_DIR / "memory.db"

def _load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        if "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"\''))

_load_env(BASE_DIR / ".env")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Embedding
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 3072
EMBEDDING_BATCH_SIZE = 100

# LLM
LLM_MODEL = "gemini-3.1-flash-lite"

# Qdrant
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "conversational_rag"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ". ", ", ", " ", ""]

# Retrieval & Memory
TOP_K = 5
MEMORY_WINDOW = 5
