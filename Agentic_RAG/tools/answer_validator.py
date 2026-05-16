from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings

_llm = ChatGoogleGenerativeAI(
    model=settings.LLM_MODEL,
    google_api_key=settings.GOOGLE_API_KEY,
    temperature=0.0,
)


@tool
def validate_sufficiency(query: str, gathered_context: str) -> str:
    """
    Evaluate whether the gathered context and evidence is sufficient to produce
    a complete, accurate, and grounded answer to the user's original query.
    Call this tool after you have gathered information from vector_search or
    web_search and want to verify you have enough evidence before generating
    the final answer.
    Returns either 'SUFFICIENT' or 'INSUFFICIENT: <reason and what is still missing>'.
    """
    prompt = f"""You are a strict evidence evaluator.

Assess whether the gathered context below contains enough information to fully and accurately answer the user's query.

User Query: {query}

Gathered Context:
{gathered_context}

Respond with EXACTLY one of these two options:
1. SUFFICIENT
2. INSUFFICIENT: <brief explanation of what specific information is still missing>

Your assessment:"""

    response = _llm.invoke(prompt)
    return response.content.strip()
