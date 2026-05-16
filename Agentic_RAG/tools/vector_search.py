from langchain_core.tools import tool
from vector_store import store


@tool
def vector_search(query: str) -> str:
    """
    Search the internal knowledge base (Qdrant vector database) for document chunks
    relevant to the query. Use this tool when the user's question can be answered
    from indexed private or proprietary documents.
    """
    results = store.search(query)
    if not results:
        return "No relevant documents found in the internal knowledge base."

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[Result {i}] (Source: {r['source']}, Score: {r['score']:.3f})\n{r['text']}"
        )
    return "\n\n---\n\n".join(parts)
