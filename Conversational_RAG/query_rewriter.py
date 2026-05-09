import time
from google import genai
from config import GOOGLE_API_KEY, LLM_MODEL

REWRITE_PROMPT = """You are an expert query rewriter for a search system.
Given the following conversation history and the user's latest query, rewrite the user's query so that it is a standalone, fully contextualized query that can be understood without the history.
If the latest query is already fully standalone and doesn't rely on previous context (like pronouns 'it', 'he', 'that'), just return the original query exactly as is.
DO NOT answer the query. Only return the standalone rewritten query text.

Conversation History:
{history}

Latest User Query: {query}

Standalone Query:"""

def rewrite_query(query: str, history: list[dict]) -> str:
    """Rewrite query based on chat history to resolve pronouns and context."""
    if not history:
        return query
        
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    # Format history into a string
    history_str = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    prompt = REWRITE_PROMPT.format(history=history_str.strip(), query=query)
    client = genai.Client(api_key=GOOGLE_API_KEY)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config={"temperature": 0.0}  # Deterministic rewriting
            )
            rewritten = response.text.strip()
            print(f"  [REWRITE] '{query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                time.sleep(2 ** attempt + 2)
            else:
                raise

    return query  # Fallback to original if rewrite fails
