import asyncio
import numpy as np
from google import genai
from config import GOOGLE_API_KEY, EMBEDDING_MODEL

def _get_client() -> genai.Client:
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
    return genai.Client(api_key=GOOGLE_API_KEY)

async def embed_texts_async(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    client = _get_client()
    
    # Run the sync genai client in an executor for true async behavior
    def _embed():
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config={"task_type": task_type, "output_dimensionality": 768},
        )
        return [e.values for e in response.embeddings]

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed)

async def embed_query_async(query: str) -> list[float]:
    embeddings = await embed_texts_async([query], task_type="RETRIEVAL_QUERY")
    return embeddings[0]
