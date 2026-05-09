import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

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
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN", 
    "postgresql://postgres:postgres@localhost:5433/rag_db"
)

# Models
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
LLM_MODEL = "gemini-3.1-flash-lite"
GRADER_MODEL = "gemini-3.1-flash-lite"

# Params
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5
MEMORY_WINDOW = 5
