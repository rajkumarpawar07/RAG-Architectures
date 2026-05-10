import json
import time
import asyncio
from pathlib import Path
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from self_rag import run_self_rag_async

console = Console()

class EvalTestCase(BaseModel):
    question: str
    expected_answer: str

def load_test_cases(file_path: Path) -> list[EvalTestCase]:
    cases = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            data = json.loads(line)
            cases.append(EvalTestCase(**data))
    return cases

async def run_eval_async(test_file: Path):
    cases = load_test_cases(test_file)
    console.print(f"Loaded {len(cases)} test cases from {test_file.name}")
    
    results = []
    
    for i, case in enumerate(cases):
        console.print(f"\n[bold cyan]Evaluating [{i+1}/{len(cases)}]:[/bold cyan] {case.question}")
        start_time = time.time()
        
        # Run state machine
        state = await run_self_rag_async(case.question)
        
        latency = time.time() - start_time
        
        # Analyze trace
        is_sup_calls = [t for t in state.token_trace if t["node"] == "IsSup"]
        is_rel_calls = [t for t in state.token_trace if t["node"] == "IsRel"]
        is_use_calls = [t for t in state.token_trace if t["node"] == "IsUse"]
        
        fully_supported = any(call.get("support_level") == "FULLY_SUPPORTED" for call in is_sup_calls)
        
        rel_precision = 0
        if is_rel_calls:
            last_rel = is_rel_calls[-1]
            retrieved = last_rel.get("total_retrieved", 0)
            relevant = last_rel.get("total_relevant", 0)
            if retrieved > 0:
                rel_precision = relevant / retrieved
                
        use_score = 0
        if is_use_calls:
            use_score = is_use_calls[-1].get("score", 0)
            
        results.append({
            "question": case.question,
            "latency": latency,
            "revisions": state.revision_count,
            "fully_supported": fully_supported,
            "rel_precision": rel_precision,
            "usefulness_score": use_score,
            "retrieval_loops": state.retrieval_loop_count
        })
        
        console.print(f"  Latency: {latency:.2f}s | Revisions: {state.revision_count} | Supported: {fully_supported}")
        
    # Aggregate Metrics
    total = len(results)
    hallucination_rate = 1.0 - (sum(1 for r in results if r["fully_supported"]) / total) if total else 0
    avg_rel_precision = sum(r["rel_precision"] for r in results) / total if total else 0
    avg_usefulness = sum(r["usefulness_score"] for r in results) / total if total else 0
    avg_latency = sum(r["latency"] for r in results) / total if total else 0
    avg_revisions = sum(r["revisions"] for r in results) / total if total else 0
    
    table = Table(title="Self-RAG Evaluation Report", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    
    table.add_row("Total Test Cases", str(total))
    table.add_row("Hallucination Rate (1 - Fully Supported)", f"{hallucination_rate:.1%}")
    table.add_row("Avg Relevance Precision", f"{avg_rel_precision:.1%}")
    table.add_row("Avg Usefulness Score (1-5)", f"{avg_usefulness:.2f}")
    table.add_row("Mean Latency", f"{avg_latency:.2f}s")
    table.add_row("Mean Revisions per Query", f"{avg_revisions:.2f}")
    
    console.print("\n")
    console.print(table)
