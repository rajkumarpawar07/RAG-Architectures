"""
generator.py — Gemini LLM generation with RAG prompt engineering.

Builds a well-structured prompt that includes retrieved context
with source citations, then calls the Gemini LLM to generate
a grounded answer.
"""

from google import genai

from config import GOOGLE_API_KEY, LLM_MODEL


SYSTEM_PROMPT = (
    "You are a helpful, accurate assistant that answers questions "
    "based ONLY on the provided context.\n\n"
    "RULES:\n"
    "1. Answer the question using ONLY the information in the CONTEXT section below.\n"
    "2. If the context contains the answer, provide a clear and comprehensive response.\n"
    "3. ALWAYS cite which source(s) you used (e.g., [Source: filename.pdf]).\n"
    "4. If the context does NOT contain enough information, say: "
    "\"I don't have enough information in the provided documents to answer this.\"\n"
    "5. Do NOT make up information or use knowledge outside the provided context.\n"
    "6. Keep your answer concise but complete."
)


def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """Build the full RAG prompt with retrieved context and user query."""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk.get("source", "unknown")
        chunk_idx = chunk.get("chunk_index", "?")
        score = chunk.get("score", 0.0)
        text = chunk.get("text", "")
        context_parts.append(
            f"[Source {i}: {source}, Chunk {chunk_idx}] (relevance: {score:.3f})\n{text}"
        )

    context = "\n\n---\n\n".join(context_parts)
    return f"CONTEXT:\n{context}\n\nQUESTION: {query}"


def generate(query: str, retrieved_chunks: list[dict]) -> str:
    """Generate an answer using Gemini LLM given query and retrieved context."""
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is not set.\n"
            "Get your key at: https://aistudio.google.com/apikey"
        )

    import time

    client = genai.Client(api_key=GOOGLE_API_KEY)
    prompt = build_prompt(query, retrieved_chunks)

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "temperature": 0.3,
                    "max_output_tokens": 2048,
                },
            )
            return response.text

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = 2 ** attempt + 5  # Generous wait for rate limits
                print(f"  [WAIT] Rate limited, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise

    raise RuntimeError(f"Failed to generate after {max_retries} retries due to rate limiting")
