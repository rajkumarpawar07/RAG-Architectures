"""
config.py — Central configuration for the Standard RAG pipeline.

All tuneable parameters live here. Modify these to experiment
with different chunking sizes, models, retrieval depths, etc.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "faiss_index"


def _load_env(env_path: Path) -> None:
    """Load key=value pairs from a .env file into os.environ."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_env(BASE_DIR / ".env")

# ──────────────────────────────────────────────
# Gemini
# ──────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# Embedding
EMBEDDING_MODEL = "gemini-embedding-001"  
EMBEDDING_DIMENSION = 3072
EMBEDDING_BATCH_SIZE = 100               

# LLM
LLM_MODEL = "gemini-3.1-flash-lite"

# ──────────────────────────────────────────────
# Chunking  (Recursive Character Text Splitting)
# ──────────────────────────────────────────────
CHUNK_SIZE = 1000          # Max characters per chunk
CHUNK_OVERLAP = 200        # Overlap between consecutive chunks
SEPARATORS = [             # Tried in order — coarsest first
    "\n\n",                # Paragraph boundaries
    "\n",                  # Line boundaries
    ". ",                  # Sentence boundaries
    ", ",                  # Clause boundaries
    " ",                   # Word boundaries
    "",                    # Character-level (last resort)
]

# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
TOP_K = 5                  # Number of chunks to retrieve per query
