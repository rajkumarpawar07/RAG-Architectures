from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from config import settings
import os

os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY

_tavily = TavilySearch(max_results=5)


@tool
def web_search(query: str) -> str:
    """
    Search the web for real-time, up-to-date, or external information that is
    NOT available in the internal knowledge base. Use this for current events,
    live data, regulations, pricing, or any information that changes frequently.
    """
    try:
        results = _tavily.invoke(query)
        if not results:
            return "No web search results found."

        parts = []
        for i, r in enumerate(results, 1):
            url = r.get("url", "unknown")
            content = r.get("content", "")
            parts.append(f"[Web Result {i}] (URL: {url})\n{content}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"Web search failed: {str(e)}"
