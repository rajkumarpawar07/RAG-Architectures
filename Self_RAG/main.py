import typer
from rich.console import Console
from rich.panel import Panel
from pathlib import Path
from config import settings, DATA_DIR

app = typer.Typer(help="Self-RAG CLI")
console = Console()

def setup_logging(verbose: bool):
    import logging
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)


@app.command()
def ingest(
    data_dir: str = typer.Argument(str(DATA_DIR), help="Directory containing documents to ingest"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """Parse, chunk, embed, and index documents into ChromaDB."""
    setup_logging(verbose)
    
    from document_loader import load_directory
    from chunker import chunk_documents
    from vector_store import store
    
    console.print(Panel.fit("Starting Ingestion Pipeline", style="bold blue"))
    
    # 1. Load
    with console.status("[bold green]Parsing documents with Docling...") as status:
        documents = load_directory(Path(data_dir))
        if not documents:
            console.print("[red]No supported documents found.[/red]")
            raise typer.Exit(1)
        console.print(f"✅ Loaded {len(documents)} documents.")
        
    # 2. Chunk
    with console.status("[bold green]Chunking documents (Recursive Character Split)...") as status:
        chunks = chunk_documents(documents, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        console.print(f"✅ Created {len(chunks)} chunks.")
        
    # 3. Upsert
    with console.status(f"[bold green]Embedding and upserting to ChromaDB (dim={settings.EMBEDDING_DIM})...") as status:
        store.upsert_chunks(chunks)
        console.print(f"✅ Upserted chunks to vector store.")
        
    console.print("[bold green]Ingestion complete![/bold green]")

@app.command()
def stats():
    """Show vector store statistics."""
    from vector_store import store
    stats = store.get_stats()
    console.print(Panel(f"Total chunks in ChromaDB: [bold]{stats['total_chunks']}[/bold]", title="Stats"))

@app.command()
def query(
    question: str = typer.Argument(..., help="The query to ask the Self-RAG system"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """Run a query through the Self-RAG pipeline."""
    setup_logging(verbose)
    
    import asyncio
    from self_rag import run_self_rag_async
    
    console.print(Panel(f"[bold cyan]Query:[/bold cyan] {question}"))
    
    status_text = ""
    with console.status("[yellow]Initializing...[/yellow]") as status:
        def update_progress(msg: str, emoji: str):
            status.update(f"[{emoji}] {msg}")
            
        loop = asyncio.get_event_loop()
        state = loop.run_until_complete(run_self_rag_async(question, progress_callback=update_progress))
        
    console.print("\n[bold green]Final Answer:[/bold green]")
    console.print(Panel(state.final_answer, border_style="green"))
    
    if verbose:
        console.print("\n[bold magenta]Token Trace (Reflection Log):[/bold magenta]")
        for token in state.token_trace:
            console.print(token)

@app.command(name="eval")
def evaluate(
    test_file: str = typer.Argument(..., help="Path to JSONL file containing test cases"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """Run the evaluation harness on a set of test cases."""
    setup_logging(verbose)
    
    import asyncio
    from eval_harness import run_eval_async
    
    path = Path(test_file)
    if not path.exists():
        console.print(f"[red]Error: File {test_file} not found.[/red]")
        raise typer.Exit(1)
        
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_eval_async(path))

if __name__ == "__main__":
    app()
