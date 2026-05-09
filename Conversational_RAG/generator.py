import time
from google import genai
from config import GOOGLE_API_KEY, LLM_MODEL

SYSTEM_PROMPT = (
    "You are a helpful, accurate conversational assistant.\n"
    "You have been provided with CONTEXT from a knowledge base.\n"
    "Answer the user's question using ONLY the information in the CONTEXT.\n"
    "If the context does NOT contain enough information, say: "
    "\"I don't have enough information in the provided documents to answer this.\"\n"
    "ALWAYS cite the source documents (e.g., [Source: filename.pdf]) when providing facts.\n"
)

def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk.get("source", "unknown")
        score = chunk.get("score", 0.0)
        text = chunk.get("text", "")
        context_parts.append(
            f"[Source: {source}] (relevance: {score:.3f})\n{text}"
        )
    context = "\n\n---\n\n".join(context_parts)
    return f"CONTEXT:\n{context}\n\nLATEST QUESTION: {query}"

def generate(query: str, retrieved_chunks: list[dict], history: list[dict] = None) -> str:
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    client = genai.Client(api_key=GOOGLE_API_KEY)
    
    # We pass the history to the LLM directly via contents array
    contents = []
    
    # Add history
    if history:
        for msg in history:
            contents.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})
    
    # Add the current prompt (which includes the retrieved context and user query)
    current_prompt = build_prompt(query, retrieved_chunks)
    contents.append({"role": "user", "parts": [{"text": current_prompt}]})

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=contents,
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
                wait_time = 2 ** attempt + 3
                print(f"  [WAIT] Rate limited, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise

    raise RuntimeError("Failed to generate due to rate limiting")
