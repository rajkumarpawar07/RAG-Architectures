from typing import TypedDict, Annotated
import operator


class AgentState(TypedDict):
    # The user's original query — never mutated
    query: str

    # Full ReAct message thread (HumanMessage, AIMessage, ToolMessage, ...)
    # Annotated with operator.add so LangGraph APPENDS on each node instead of overwriting
    messages: Annotated[list, operator.add]

    # Accumulated text evidence gathered from all tool calls
    gathered_context: str

    # Citation sources accumulated across iterations
    sources: Annotated[list, operator.add]

    # How many ReAct iterations have completed
    iterations: int

    # Whether validate_sufficiency returned SUFFICIENT
    is_sufficient: bool

    # The final synthesized answer
    final_answer: str
