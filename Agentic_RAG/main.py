import sys
import io
import typer
import logging
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.prompt import Prompt
from rich.markdown import Markdown
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

from config import settings, DATA_DIR

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="agentic-rag",
    help="Agentic RAG - an autonomous research agent powered by LangGraph + Gemini.",
    add_completion=False,
)
console = Console(highlight=False)


# --- Helpers ---

TOOL_ICONS = {
    "decompose_query": "[cyan][DECOMPOSE][/cyan]",
    "vector_search":   "[blue][VECTOR][/blue]",
    "web_search":      "[magenta][WEB][/magenta]",
    "validate_sufficiency": "[green][VALIDATE][/green]",
}


def _render_thought_panel(trace: list[str]) -> Panel:
    content = "\n".join(trace) if trace else "[dim]Waiting for agent...[/dim]"
    return Panel(content, title="[yellow]Agent Reasoning Trace[/yellow]", border_style="yellow", padding=(0, 1))


def _run_agent_with_stream(query: str) -> tuple[dict, list[str]]:
    """Stream the agent graph and collect trace + final state."""
    from agent.graph import agent_graph

    initial_state = {
        "query": query,
        "messages": [HumanMessage(content=query)],
        "gathered_context": "",
        "sources": [],
        "iterations": 0,
        "is_sufficient": False,
        "final_answer": "",
    }

    trace_lines: list[str] = []
    final_state: dict = {}

    with Live(
        _render_thought_panel(trace_lines),
        console=console,
        refresh_per_second=4,
        vertical_overflow="visible",
    ) as live:
        for step in agent_graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_output in step.items():

                if node_name == "agent":
                    msgs = node_output.get("messages", [])
                    for msg in msgs:
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                name = tc.get("name", "unknown")
                                args = tc.get("args", {})
                                icon = TOOL_ICONS.get(name, "[dim][TOOL][/dim]")
                                arg_str = ", ".join(
                                    f"{k}={repr(v)[:60]}" for k, v in args.items()
                                )
                                trace_lines.append(f"  {icon} [bold]{name}[/bold]({arg_str})")
                        elif isinstance(msg, AIMessage) and not msg.tool_calls:
                            trace_lines.append(
                                "  [dim]Agent reasoning complete - moving to synthesis[/dim]"
                            )

                elif node_name == "accumulate":
                    ctx = node_output.get("gathered_context", "")
                    new_sources = node_output.get("sources", [])
                    trace_lines.append(f"  [dim]Context updated ({len(ctx)} chars total)[/dim]")
                    for s in new_sources:
                        trace_lines.append(f"  [dim]Source: {s}[/dim]")

                elif node_name == "generator":
                    trace_lines.append("  [green]Synthesizing final answer...[/green]")
                    final_state.update(node_output)

                live.update(_render_thought_panel(trace_lines))

    return final_state, trace_lines


# --- Commands ---

@app.command()
def ingest(
    data_dir: str = typer.Argument(str(DATA_DIR), help="Directory of documents to ingest"),
):
    """Parse documents with Docling, chunk, embed, and upsert to Qdrant."""
    from document_loader import load_directory
    from chunker import chunk_documents
    from vector_store import store

    console.print(Panel.fit("[bold blue]Agentic RAG - Document Ingestion[/bold blue]"))

    with console.status("[bold green]Parsing documents with Docling..."):
        docs = load_directory(Path(data_dir))

    if not docs:
        console.print("[red]No supported documents found.[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Loaded[/green] [bold]{len(docs)}[/bold] documents.")

    with console.status("[bold green]Chunking..."):
        chunks = chunk_documents(docs, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    console.print(f"[green]Created[/green] [bold]{len(chunks)}[/bold] chunks.")

    with console.status(
        f"[bold green]Embedding & upserting to Qdrant (collection: {settings.QDRANT_COLLECTION})..."
    ):
        store.upsert(chunks)

    console.print(f"[green]Upserted to Qdrant.[/green] Total chunks: [bold]{store.count()}[/bold]")
    console.print("[bold green]Ingestion complete![/bold green]")


@app.command()
def query(
    question: str = typer.Argument(..., help="The question to ask the agent"),
):
    """Run a single one-shot query through the Agentic RAG pipeline."""
    console.print(Panel(f"[bold cyan]Query:[/bold cyan] {question}"))
    final_state, _ = _run_agent_with_stream(question)

    answer = final_state.get("final_answer", "No answer produced.")
    console.print()
    console.print(Panel(Markdown(answer), title="[green]Final Answer[/green]", border_style="green"))


@app.command()
def chat():
    """Start an interactive REPL session with the Agentic RAG agent."""
    console.print(
        Panel.fit(
            "[bold cyan]Agentic RAG[/bold cyan] - Interactive Research Session\n"
            "[dim]Type your question and press Enter. Type 'exit' or 'quit' to stop.[/dim]",
            border_style="cyan",
        )
    )

    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if user_input.strip().lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye![/dim]")
            break

        if not user_input.strip():
            continue

        console.print()
        final_state, _ = _run_agent_with_stream(user_input)
        answer = final_state.get("final_answer", "The agent could not produce an answer.")

        console.print()
        console.print(Panel(Markdown(answer), title="[green]Agent Answer[/green]", border_style="green"))

        sources = list(set(final_state.get("sources", [])))
        if sources:
            console.print("[dim]Sources:[/dim]")
            for s in sources:
                console.print(f"  {s}")


@app.command()
def stats():
    """Show vector store and agent configuration statistics."""
    from vector_store import store

    table = Table(
        title="Agentic RAG - System Stats", show_header=True, header_style="bold magenta"
    )
    table.add_column("Parameter", style="dim")
    table.add_column("Value")

    table.add_row("Qdrant Collection", settings.QDRANT_COLLECTION)
    table.add_row("Total Indexed Chunks", str(store.count()))
    table.add_row("Embedding Dimensions", str(settings.EMBEDDING_DIM))
    table.add_row("LLM Model", settings.LLM_MODEL)
    table.add_row("Top-K Retrieval", str(settings.TOP_K))
    table.add_row("Max Agent Iterations", str(settings.MAX_ITERATIONS))
    table.add_row("Langsmith Project", settings.LANGSMITH_PROJECT)

    console.print(table)


if __name__ == "__main__":
    app()
