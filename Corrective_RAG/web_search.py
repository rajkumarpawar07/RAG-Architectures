import asyncio
from google import genai
from tavily import AsyncTavilyClient
from config import GOOGLE_API_KEY, TAVILY_API_KEY, LLM_MODEL

REWRITE_PROMPT = """Rewrite this query for a web search engine.
Make it specific, add constraints, remove conversational filler.
Output only the rewritten query string, nothing else.

Original: {original}"""

async def rewrite_query_async(query: str) -> str:
    client = genai.Client(api_key=GOOGLE_API_KEY)
    def _call_llm():
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=REWRITE_PROMPT.format(original=query),
            config={"temperature": 0.0}
        )
        return response.text.strip()
        
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _call_llm)

async def web_search_async(query: str) -> str:
    if not TAVILY_API_KEY:
        print("  [WARN] TAVILY_API_KEY not set. Skipping web search.")
        return ""
        
    rewritten = await rewrite_query_async(query)
    print(f"  [WEB SEARCH] Rewrote query to: '{rewritten}'")
    
    client = AsyncTavilyClient(api_key=TAVILY_API_KEY)
    response = await client.search(query=rewritten, max_results=3, search_depth="advanced")
    
    context_parts = []
    for r in response.get("results", []):
        context_parts.append(f"[Web: {r.get('url')}]\n{r.get('content')}")
        
    return "\n\n---\n\n".join(context_parts)
