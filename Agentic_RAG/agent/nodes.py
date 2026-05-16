import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import ToolMessage
from config import settings
from tools import ALL_TOOLS
from agent.state import AgentState

logger = logging.getLogger(__name__)

# Single LLM instance bound to all tools (enables tool calling)
_llm = ChatGoogleGenerativeAI(
    model=settings.LLM_MODEL,
    google_api_key=settings.GOOGLE_API_KEY,
    temperature=0.0,
)
_llm_with_tools = _llm.bind_tools(ALL_TOOLS)

SYSTEM_PROMPT = """You are an expert research agent with access to four tools:

1. decompose_query   — Break complex, multi-part queries into focused sub-questions.
2. vector_search     — Search private internal documents in the knowledge base.
3. web_search        — Search the web for real-time or external information.
4. validate_sufficiency — Check if gathered evidence is enough to answer the query.

Strategy:
- For complex/multi-part queries: call decompose_query FIRST.
- For factual or document-specific queries: call vector_search.
- For real-time, regulatory, or external information: call web_search.
- After gathering evidence: call validate_sufficiency.
- If INSUFFICIENT: gather more evidence before stopping.
- If SUFFICIENT or iterations are exhausted: stop using tools.

Always cite your sources. Be thorough but concise."""


def agent_node(state: AgentState) -> dict:
    """
    The ReAct brain. Receives the full message history and decides
    which tool to call next, or stops if evidence is sufficient.
    """
    logger.info(f"[AgentNode] Iteration {state['iterations'] + 1}")

    # Build message list: system prompt + history + current query if first iteration
    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    # If first iteration, inject the user query
    if state["iterations"] == 0:
        msgs.append(HumanMessage(content=state["query"]))

    response = _llm_with_tools.invoke(msgs)

    # Extract any new context from tool observations in the message history
    new_context = state.get("gathered_context", "")

    return {
        "messages": [response],
        "iterations": state["iterations"] + 1,
        "gathered_context": new_context,
    }


def tool_output_node(state: AgentState) -> dict:
    """
    After tool execution, extract tool outputs to update gathered_context and sources.
    This node runs after LangGraph's ToolNode executes the actual tool calls.
    """
    gathered = state.get("gathered_context", "")
    new_sources = []

    # Find the most recent ToolMessages and accumulate their content
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            content = msg.content or ""
            # Extract URLs from web search results as sources
            for line in content.split("\n"):
                if line.startswith("(URL:") or "URL:" in line:
                    url = line.split("URL:")[-1].strip().rstrip(")")
                    if url:
                        new_sources.append(url)
            # Append tool output to gathered context
            gathered = gathered + "\n\n" + content if gathered else content
            break  # Only process the most recent ToolMessage in this pass

    return {
        "gathered_context": gathered,
        "sources": new_sources,
    }


def generator_node(state: AgentState) -> dict:
    """
    Dedicated final synthesis node — separate from the ReAct loop.
    Given the original query + all accumulated context + sources,
    produces a grounded, cited final answer.
    """
    logger.info("[GeneratorNode] Synthesizing final answer")

    sources_str = "\n".join(f"- {s}" for s in set(state.get("sources", [])))
    context = state.get("gathered_context", "No context gathered.")

    prompt = f"""You are a precise research synthesizer.

Using ONLY the gathered evidence below, produce a comprehensive, well-structured answer
to the user's query. Include inline citations where possible.
If information is from web results, mention the source URL.
If evidence is insufficient, clearly state what is unknown.

User Query: {state['query']}

Gathered Evidence:
{context}

Sources:
{sources_str if sources_str else "No external sources."}

Final Answer:"""

    response = _llm.invoke([HumanMessage(content=prompt)])
    return {"final_answer": response.content}
