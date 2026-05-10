import asyncio
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
import logging

from vector_store import store
from llm_client import generate_text, get_cached_is_rel, set_cached_is_rel
from evaluators import (
    check_needs_retrieval, check_relevance_async, check_groundedness,
    check_usefulness, rewrite_query, SupportLevel
)
from config import settings

logger = logging.getLogger("self_rag")
log_dir = Path("~/.selfrag/logs/").expanduser()
log_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class PipelineState:
    original_query: str
    current_query: str
    retrieved_chunks: list[dict] = field(default_factory=list)
    relevant_chunks: list[dict] = field(default_factory=list)
    draft_answer: str = ""
    final_answer: str = ""
    revision_count: int = 0
    retrieval_loop_count: int = 0
    token_trace: list[dict] = field(default_factory=list)

def log_reflection(state: PipelineState, node: str, decision: dict):
    logger.info(f"Reflection | node={node} | decision={decision}")
    state.token_trace.append({"node": node, **decision})
    
    # Audit trail JSONL
    log_file = log_dir / "audit_trail.jsonl"
    entry = {
        "timestamp": time.time(),
        "query": state.original_query,
        "node": node,
        "decision": decision
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def generate_draft(query: str, chunks: list[dict], unsupported_claims: list[str] = None) -> str:
    context_str = "\n\n".join([f"Source: {c['metadata'].get('source')} (Chunk {c['metadata'].get('chunk_index')})\n{c['text']}" for c in chunks])
    
    prompt = f"""
    Answer the user's question using ONLY the provided context.
    Cite the sources using their names.
    
    Question: {query}
    
    Context:
    {context_str}
    """
    
    if unsupported_claims:
        prompt += "\n\nCRITICAL INSTRUCTION: Your previous answer contained the following unsupported claims. You MUST completely remove or fix them to be strictly grounded in the context:\n"
        for claim in unsupported_claims:
            prompt += f"- {claim}\n"
            
    return generate_text(prompt, use_pro=False)

async def run_self_rag_async(query: str, progress_callback=None) -> PipelineState:
    state = PipelineState(original_query=query, current_query=query)
    
    if progress_callback: progress_callback("Deciding if retrieval is needed...", "🤔")
    decision = check_needs_retrieval(query)
    log_reflection(state, "IsRet", decision.model_dump())
    
    if not decision.needs_retrieval:
        if progress_callback: progress_callback("Answering from parametric knowledge...", "🧠")
        state.final_answer = generate_text(query, use_pro=False)
        return state

    while state.retrieval_loop_count < settings.MAX_RETRIEVAL_LOOPS:
        state.retrieval_loop_count += 1
        
        # 1. Retrieve
        if progress_callback: progress_callback(f"Retrieving documents (Loop {state.retrieval_loop_count})...", "🔎")
        chunks = store.search(state.current_query, top_k=settings.TOP_K)
        state.retrieved_chunks = chunks
        
        if not chunks:
            state.final_answer = "No relevant information found in the database."
            return state

        # 2. Filter Relevance (Parallel)
        if progress_callback: progress_callback("Filtering relevant chunks (parallel)...", "⚖️")
        
        async def evaluate_chunk(c):
            cached = get_cached_is_rel(state.current_query, c["id"])
            if cached:
                return c, cached
            
            res = await check_relevance_async(state.current_query, c["text"])
            set_cached_is_rel(state.current_query, c["id"], res.model_dump())
            return c, res.model_dump()
            
        eval_tasks = [evaluate_chunk(c) for c in chunks]
        eval_results = await asyncio.gather(*eval_tasks)
        
        state.relevant_chunks = [c for c, res in eval_results if res.get("is_relevant") and res.get("score", 0) >= settings.RELEVANCE_THRESHOLD]
        
        log_reflection(state, "IsRel", {"total_retrieved": len(chunks), "total_relevant": len(state.relevant_chunks)})
        
        if not state.relevant_chunks:
            if progress_callback: progress_callback("No relevant chunks found. Rewriting query via HyDE...", "🔄")
            rewrite = rewrite_query(state.original_query)
            log_reflection(state, "RewriteQuery", rewrite.model_dump())
            state.current_query = rewrite.hypothetical_document  # Use HyDE
            continue

        # 3. Generate & Revise Loop
        state.revision_count = 0
        unsupported = None
        context_str = "\n\n".join([c['text'] for c in state.relevant_chunks])
        
        while state.revision_count <= settings.MAX_REVISIONS:
            if progress_callback: progress_callback(f"Generating draft (Revision {state.revision_count})...", "✍️")
            state.draft_answer = generate_draft(state.original_query, state.relevant_chunks, unsupported)
            
            if progress_callback: progress_callback("Checking groundedness (Hallucination check)...", "🛡️")
            groundedness = check_groundedness(state.draft_answer, context_str)
            log_reflection(state, "IsSup", groundedness.model_dump())
            
            if groundedness.support_level == SupportLevel.FULLY:
                break
                
            unsupported = groundedness.unsupported_claims
            state.revision_count += 1
            if progress_callback: progress_callback(f"Hallucination detected! Revising answer...", "⚠️")

        # 4. Usefulness Check
        if progress_callback: progress_callback("Evaluating final usefulness...", "🎯")
        usefulness = check_usefulness(state.original_query, state.draft_answer)
        log_reflection(state, "IsUse", usefulness.model_dump())
        
        if usefulness.is_useful and usefulness.score >= settings.USEFULNESS_THRESHOLD:
            state.final_answer = state.draft_answer
            return state
        else:
            if progress_callback: progress_callback("Answer not useful. Rewriting query via HyDE...", "🔄")
            rewrite = rewrite_query(state.original_query)
            log_reflection(state, "RewriteQuery", rewrite.model_dump())
            state.current_query = rewrite.hypothetical_document
            
    # Max retrieval loops exceeded
    state.final_answer = state.draft_answer if state.draft_answer else "Failed to find a useful answer after maximum retries."
    return state
