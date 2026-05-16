from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings

_llm = ChatGoogleGenerativeAI(
    model=settings.LLM_MODEL,
    google_api_key=settings.GOOGLE_API_KEY,
    temperature=0.0,
)


@tool
def decompose_query(complex_query: str) -> str:
    """
    Break a complex, multi-part, or ambiguous user question into a list of
    focused, self-contained sub-questions. Use this tool FIRST when the user's
    query contains multiple parts, comparisons, or requires information from
    multiple domains (e.g., regulatory + technical + financial).
    Returns a numbered list of sub-questions to guide subsequent tool calls.
    """
    prompt = f"""You are an expert query analyst. 
    
Break the following complex question into 2-4 focused, self-contained sub-questions.
Each sub-question should target a specific piece of information needed to fully answer the original query.
Return ONLY the numbered sub-questions, nothing else.

Complex Query: {complex_query}

Sub-questions:"""

    response = _llm.invoke(prompt)
    return response.content
