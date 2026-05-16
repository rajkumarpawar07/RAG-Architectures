import typer
import logging
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from config import settings, DATA_DIR

logging.basicConfig(level=logging.WARNING)  
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="graph-rag",
    help="Graph RAG - Entity extraction and relationship traversal with Neo4j.",
    add_completion=False,
)
console = Console(highlight=False)

@app.command()
def ingest(
    data_dir: str = typer.Argument(str(DATA_DIR), help="Directory of documents to ingest"),
):
    """Extract entities and relationships from documents and load into Neo4j."""
    from document_loader import load_directory
    from chunker import chunk_documents
    from graph_builder import graph_store

    console.print(Panel.fit("[bold blue]Graph RAG - Knowledge Graph Construction[/bold blue]"))

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
        "[bold green]Extracting Entities & Relationships (LLMGraphTransformer) -> Neo4j..."
    ):
        graph_store.extract_and_store(chunks)

    console.print("[bold green]Knowledge Graph Construction Complete![/bold green]")

@app.command()
def query(
    question: str = typer.Argument(..., help="The question to ask the Graph pipeline"),
):
    """Run a query through the GraphRAG pipeline."""
    from pipeline import run_graph_rag
    
    console.print(Panel(f"[bold cyan]Original Query:[/bold cyan] {question}"))
    
    with console.status("[bold green]Traversing Graph & Generating Answer..."):
        try:
            result = run_graph_rag(question)
            answer = result.get("result", "No answer generated.")
        except Exception as e:
            console.print(f"[bold red]Error during GraphRAG:[/bold red] {e}")
            raise typer.Exit(1)

    console.print("\n")
    console.print(Panel(Markdown(answer), title="[green]Final Answer[/green]", border_style="green"))


@app.command()
def stats():
    """Show Neo4j Graph statistics."""
    from graph_builder import graph_store

    console.print(Panel.fit("[bold blue]Neo4j Graph Schema[/bold blue]"))
    
    # Try to print the graph schema
    try:
        graph_store.graph.refresh_schema()
        schema = graph_store.graph.get_schema
        console.print(schema)
    except Exception as e:
        console.print(f"[red]Could not retrieve schema. Is Neo4j running?[/red] Error: {e}")

if __name__ == "__main__":
    app()
