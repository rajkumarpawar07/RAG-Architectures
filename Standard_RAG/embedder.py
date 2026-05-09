"""
embedder.py — Gemini embedding integration.

Uses the google-genai SDK to embed texts with the text-embedding-004
model. Supports separate task types for documents vs queries (a
production best practice that improves retrieval quality).

Handles batching (API limit of 100 texts per request) and includes
exponential backoff for rate limiting.
"""

import time

import numpy as np
from google import genai

from config import GOOGLE_API_KEY, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, EMBEDDING_DIMENSION


def _get_client() -> genai.Client:
    """Initialize and return a Gemini client."""
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is not set.\n"
            "Get your key at: https://aistudio.google.com/apikey\n"
            "Then set it:  set GOOGLE_API_KEY=your_key_here  (Windows)"
        )
    return genai.Client(api_key=GOOGLE_API_KEY)


def _embed_batch(
    client: genai.Client,
    texts: list[str],
    task_type: str,
    max_retries: int = 5,
) -> list[list[float]]:
    """
    Embed a single batch of texts with exponential backoff.

    Retries on rate-limit (429) and server errors (5xx).
    """
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config={
                    "task_type": task_type,
                },
            )
            return [e.values for e in response.embeddings]

        except Exception as e:
            error_str = str(e)
            # Retry on rate limit or server errors
            if "429" in error_str or "500" in error_str or "503" in error_str:
                wait_time = 2 ** attempt
                print(f"  [WAIT] Rate limited, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise

    raise RuntimeError(f"Failed to embed after {max_retries} retries")


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of document texts for indexing.

    Uses task_type="RETRIEVAL_DOCUMENT" which optimizes the
    embeddings for being *retrieved* (as opposed to being the query).

    Returns a numpy array of shape (n_texts, embedding_dim).
    """
    client = _get_client()
    all_embeddings: list[list[float]] = []

    # Process in batches (Gemini API limit)
    total_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        batch_num = i // EMBEDDING_BATCH_SIZE + 1
        print(f"  [EMBED] Batch {batch_num}/{total_batches} ({len(batch)} texts)")
        embeddings = _embed_batch(client, batch, task_type="RETRIEVAL_DOCUMENT")
        all_embeddings.extend(embeddings)

    return np.array(all_embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query text for searching.

    Uses task_type="RETRIEVAL_QUERY" which optimizes the embedding
    for matching against document embeddings.

    Returns a numpy array of shape (1, embedding_dim).
    """
    client = _get_client()
    embeddings = _embed_batch(client, [query], task_type="RETRIEVAL_QUERY")
    return np.array(embeddings, dtype=np.float32)
