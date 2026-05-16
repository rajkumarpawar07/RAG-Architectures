from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage
from config import settings
from tools import ALL_TOOLS
from agent.state import AgentState
from agent.nodes import agent_node, tool_output_node, generator_node

import logging
logger = logging.getLogger(__name__)


def should_continue(state: AgentState) -> str:
    """
    Conditional edge after agent_node:
    - If the last message has tool calls AND we haven't hit max iterations → 'tools'
    - Otherwise → 'generator'
    """
    last_msg = state["messages"][-1]
    iterations = state["iterations"]

    if isinstance(last_msg, AIMessage) and last_msg.tool_calls and iterations < settings.MAX_ITERATIONS:
        logger.info(f"[Router] Tool call detected → routing to tools (iter {iterations})")
        return "tools"
    else:
        logger.info(f"[Router] No tool call or max iterations → routing to generator")
        return "generator"


def build_graph():
    """Assemble and compile the LangGraph agent."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("accumulate", tool_output_node)   # accumulate context after tool execution
    graph.add_node("generator", generator_node)

    # Entry point
    graph.set_entry_point("agent")

    # After agent → decide: run tools or generate
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "generator": "generator"},
    )

    # After tools execute → accumulate context → back to agent for next ReAct step
    graph.add_edge("tools", "accumulate")
    graph.add_edge("accumulate", "agent")

    # Generator is terminal
    graph.add_edge("generator", END)

    compiled = graph.compile()
    logger.info("LangGraph agent compiled successfully")
    return compiled


# Singleton graph instance
agent_graph = build_graph()
