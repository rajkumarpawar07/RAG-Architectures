"""
main.py — CLI entry point for the Standard RAG pipeline.

Commands:
  python main.py ingest          Ingest all documents from data/ folder
  python main.py query "..."     One-shot query
  python main.py chat            Interactive chat loop
  python main.py stats           Show index statistics
"""

import argparse
import sys
import os

# Fix Windows console encoding for emoji/unicode output
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rag_pipeline import ingest, query, get_stats


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest documents from the data/ folder."""
    try:
        stats = ingest()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Ingestion failed: {e}")
        sys.exit(1)


def cmd_query(args: argparse.Namespace) -> None:
    """Run a single query against the RAG pipeline."""
    question = args.question
    if not question:
        print("[ERROR] Please provide a question. Example:")
        print('   python main.py query "What is this document about?"')
        sys.exit(1)

    try:
        result = query(question)
        _print_result(result)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Query failed: {e}")
        sys.exit(1)


def cmd_chat(args: argparse.Namespace) -> None:
    """Start an interactive chat session."""
    print("\n" + "=" * 60)
    print("RAG Chat - Interactive Mode")
    print("=" * 60)
    print("Type your questions below. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        try:
            result = query(question)
            _print_result(result)
        except FileNotFoundError as e:
            print(f"\n[ERROR] {e}")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}\n")


def cmd_stats(args: argparse.Namespace) -> None:
    """Display index statistics."""
    stats = get_stats()

    print("\n" + "=" * 60)
    print("Index Statistics")
    print("=" * 60)

    if stats.get("status") != "ready":
        print(f"\n  {stats['status']}\n")
        return

    print(f"  Status:     {stats['status']}")
    print(f"  Vectors:    {stats['total_vectors']}")
    print(f"  Chunks:     {stats['total_chunks']}")
    print(f"  Documents:  {stats['document_count']}")
    print(f"\n  Sources:")
    for doc in stats["documents"]:
        print(f"    - {doc}")
    print()


def _print_result(result: dict) -> None:
    """Pretty-print a query result."""
    print(f"\n{'-' * 60}")
    print(f"Answer:\n")
    print(result["answer"])
    print(f"\n{'-' * 60}")
    print("Sources:")
    for src in result["sources"]:
        print(f"   * {src}")
    print(f"{'-' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standard RAG Pipeline - From-Scratch Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py ingest              Load & index documents from data/
  python main.py query "Who is X?"   Ask a question
  python main.py chat                Start interactive chat
  python main.py stats               Show index info
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ingest
    subparsers.add_parser("ingest", help="Ingest documents from data/ folder")

    # query
    query_parser = subparsers.add_parser("query", help="Run a single query")
    query_parser.add_argument("question", type=str, help="The question to ask")

    # chat
    subparsers.add_parser("chat", help="Start interactive chat session")

    # stats
    subparsers.add_parser("stats", help="Show index statistics")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "ingest": cmd_ingest,
        "query": cmd_query,
        "chat": cmd_chat,
        "stats": cmd_stats,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
