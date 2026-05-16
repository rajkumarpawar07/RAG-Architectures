import typer
import logging
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from config import settings, DATA_DIR

logging.basicConfig(level=logging.WARNING)  
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="fusion-rag",
    help="Fusion RAG - Multi-query expansion and Reciprocal Rank Fusion.",
    add_completion=False,
)
console = Console(highlight=False)


@app.command()
def ingest(
    data_dir: str = typer.Argument(str(DATA_DIR), help="Directory of documents to ingest"),
):
    """Parse documents with Docling, chunk, embed, and upsert to Weaviate."""
    from document_loader import load_directory
    from chunker import chunk_documents
    from vector_store import store

    console.print(Panel.fit("[bold blue]Fusion RAG - Document Ingestion[/bold blue]"))

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
        f"[bold green]Embedding & upserting to Weaviate (collection: {settings.WEAVIATE_INDEX_NAME})..."
    ):
        store.upsert(chunks)

    console.print(f"[green]Upserted to Weaviate.[/green] Total chunks: [bold]{store.count()}[/bold]")
    console.print("[bold green]Ingestion complete![/bold green]")
    store.close()


@app.command()
def query(
    question: str = typer.Argument(..., help="The question to ask the Fusion pipeline"),
):
    """Run a single one-shot query through the Fusion RAG pipeline."""
    from pipeline import run_fusion_rag
    from vector_store import store
    
    console.print(Panel(f"[bold cyan]Original Query:[/bold cyan] {question}"))
    
    with console.status("[bold green]Running Fusion RAG Pipeline (Expansion -> Retrieval -> RRF -> Generate)..."):
        result = run_fusion_rag(question)
        
    console.print("\n[bold magenta]Generated Query Variations:[/bold magenta]")
    for i, q in enumerate(result["expanded_queries"], 1):
        console.print(f"  {i}. {q}")
        
    console.print(f"\n[bold yellow]Top {settings.FINAL_TOP_K} Documents after RRF Fusion:[/bold yellow]")
    for i, doc in enumerate(result["fused_docs"], 1):
        source = doc.metadata.get("source", "unknown")
        snippet = doc.page_content.replace("\n", " ")[:150] + "..."
        console.print(f"  [dim]{i}. [{source}] {snippet}[/dim]")

    console.print("\n")
    console.print(Panel(Markdown(result["answer"]), title="[green]Final Answer[/green]", border_style="green"))
    
    store.close()


@app.command()
def stats():
    """Show vector store and configuration statistics."""
    from vector_store import store

    table = Table(
        title="Fusion RAG - System Stats", show_header=True, header_style="bold magenta"
    )
    table.add_column("Parameter", style="dim")
    table.add_column("Value")

    table.add_row("Weaviate Collection", settings.WEAVIATE_INDEX_NAME)
    table.add_row("Total Indexed Chunks", str(store.count()))
    table.add_row("Number of Queries Expanded", str(settings.NUM_QUERIES))
    table.add_row("Top K per Query", str(settings.TOP_K_PER_QUERY))
    table.add_row("Final Top K (Fused)", str(settings.FINAL_TOP_K))
    table.add_row("LangSmith Project", settings.LANGSMITH_PROJECT)

    console.print(table)
    store.close()


if __name__ == "__main__":
    app()
