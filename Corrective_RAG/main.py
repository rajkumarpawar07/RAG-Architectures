import argparse
import sys
import os
import asyncio

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rag_pipeline import ingest_async, query_async, get_stats_async

def cmd_ingest(args):
    try:
        asyncio.run(ingest_async())
    except Exception as e:
        print(f"\n[ERROR] Ingestion failed: {e}")
        sys.exit(1)

def cmd_query(args):
    question = args.question
    session_id = args.session
    if not question:
        print("[ERROR] Please provide a question.")
        sys.exit(1)

    try:
        result = asyncio.run(query_async(session_id, question))
        _print_result(result)
    except Exception as e:
        print(f"\n[ERROR] Query failed: {e}")
        sys.exit(1)

def cmd_chat(args):
    session_id = args.session
    print("\n" + "=" * 60)
    print(f"Corrective RAG Chat - Session: {session_id}")
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
            result = asyncio.run(query_async(session_id, question))
            print(f"\nBot:\n{result['answer']}\n")
            print(f"  [Debug] Decision: {result['decision']} | Latency: {result['latency_ms']}ms")
        except Exception as e:
            print(f"\n[ERROR] {e}\n")

def cmd_stats(args):
    stats = asyncio.run(get_stats_async())
    print("\n" + "=" * 60)
    print("Index Statistics")
    print("=" * 60)
    print(f"  Total Vectors in PostgreSQL: {stats['total_vectors']}")
    print(f"  Total CRAG Runs Logged: {stats['total_runs']}")
    print()

def _print_result(result: dict) -> None:
    print(f"\n{'-' * 60}")
    print(f"Answer:\n")
    print(result["answer"])
    print(f"\n{'-' * 60}")
    print(f"Decision Gate : {result['decision']}")
    print(f"Web Triggered : {result['web_used']}")
    print(f"Latency       : {result['latency_ms']} ms")
    print(f"{'-' * 60}\n")

def main():
    parser = argparse.ArgumentParser(description="Corrective RAG Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("ingest", help="Ingest documents from data/ folder")

    query_parser = subparsers.add_parser("query", help="Run a single query")
    query_parser.add_argument("question", type=str, help="The question to ask")
    query_parser.add_argument("--session", type=str, default="default", help="Session ID for memory")

    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    chat_parser.add_argument("--session", type=str, default="default", help="Session ID for memory")

    subparsers.add_parser("stats", help="Show index statistics")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {"ingest": cmd_ingest, "query": cmd_query, "chat": cmd_chat, "stats": cmd_stats}
    commands[args.command](args)

if __name__ == "__main__":
    main()
