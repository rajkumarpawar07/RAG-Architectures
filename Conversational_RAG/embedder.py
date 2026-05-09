import time
import numpy as np
from google import genai
from config import GOOGLE_API_KEY, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

def _get_client() -> genai.Client:
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
    return genai.Client(api_key=GOOGLE_API_KEY)

def _embed_batch(client: genai.Client, texts: list[str], task_type: str, max_retries: int = 5) -> list[list[float]]:
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config={"task_type": task_type},
            )
            return [e.values for e in response.embeddings]
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "50" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = 2 ** attempt + 2
                print(f"  [WAIT] Rate limited, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise
    raise RuntimeError(f"Failed to embed after {max_retries} retries")

def embed_texts(texts: list[str]) -> np.ndarray:
    client = _get_client()
    all_embeddings = []
    total_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        batch_num = i // EMBEDDING_BATCH_SIZE + 1
        print(f"  [EMBED] Batch {batch_num}/{total_batches} ({len(batch)} texts)")
        embeddings = _embed_batch(client, batch, task_type="RETRIEVAL_DOCUMENT")
        all_embeddings.extend(embeddings)
    return np.array(all_embeddings, dtype=np.float32)

def embed_query(query: str) -> np.ndarray:
    client = _get_client()
    embeddings = _embed_batch(client, [query], task_type="RETRIEVAL_QUERY")
    return np.array(embeddings, dtype=np.float32)
