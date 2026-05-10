from google import genai
from google.genai import types
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from config import settings
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("self_rag")

client = genai.Client(api_key=settings.GEMINI_API_KEY)

# Simple in-memory cache instead of diskcache for now
cache = {}

# Models
FLASH_MODEL = "gemini-3.1-flash-lite"
PRO_MODEL = "gemini-1.5-pro"
EMBEDDING_MODEL = "gemini-embedding-001"

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(5),
    reraise=True
)
def generate_structured(prompt: str, schema_class, use_pro: bool = False, temperature: float = 0.0):
    """Generate structured JSON output based on a Pydantic schema using Gemini."""
    model = PRO_MODEL if use_pro else FLASH_MODEL
    
    logger.debug(f"Calling structured model model={model} schema={schema_class.__name__}")
    
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema_class,
            temperature=temperature
        )
    )
    
    return schema_class.model_validate_json(response.text)

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(5),
    reraise=True
)
def generate_text(prompt: str, use_pro: bool = False, temperature: float = 0.7) -> str:
    """Generate standard unstructured text."""
    model = PRO_MODEL if use_pro else FLASH_MODEL
    logger.debug(f"Calling text model model={model}")
    
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature)
    )
    return response.text

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(5),
    reraise=True
)
def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Generate embeddings truncated to 768 dimensions."""
    if not texts:
        return []
        
    logger.debug(f"Generating embeddings count={len(texts)} task_type={task_type}")
    
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=settings.EMBEDDING_DIM
        )
    )
    return [e.values for e in response.embeddings]

def get_cached_is_rel(query: str, chunk_id: str):
    """Retrieve cached relevance score if available."""
    key = hashlib.md5(f"{query}_{chunk_id}".encode("utf-8")).hexdigest()
    return cache.get(key)

def set_cached_is_rel(query: str, chunk_id: str, result, expire: int = 86400):
    """Cache relevance score."""
    key = hashlib.md5(f"{query}_{chunk_id}".encode("utf-8")).hexdigest()
    cache[key] = result
