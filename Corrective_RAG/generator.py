import asyncio
from google import genai
from config import GOOGLE_API_KEY, LLM_MODEL

SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question using ONLY the provided context.
If the context doesn't contain the answer, say "I don't have enough information".
Always cite your sources using [Source: <name>] notation."""

async def generate_async(query: str, context: str, history: list[dict] = None) -> str:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    
    contents = []
    if history:
        for msg in history:
            contents.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})
            
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}"
    contents.append({"role": "user", "parts": [{"text": user_prompt}]})
    
    def _call_llm():
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=contents,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "temperature": 0.1,
                "max_output_tokens": 2048
            }
        )
        return response.text
        
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _call_llm)
