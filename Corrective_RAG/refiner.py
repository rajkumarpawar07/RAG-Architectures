import asyncio
from chunker import _split_into_sentences
from evaluator import grade_chunk_async

async def refine_chunks_async(query: str, chunks: list[dict]) -> str:
    """Takes correct/ambiguous chunks, splits to sentences, grades them, and merges passing sentences."""
    all_strips = []
    for chunk in chunks:
        sentences = _split_into_sentences(chunk["text"])
        for s in sentences:
            all_strips.append((chunk, s))
            
    tasks = [grade_chunk_async(query, strip) for _, strip in all_strips]
    scores = await asyncio.gather(*tasks)
    
    kept_strips = []
    for (chunk, strip), res in zip(all_strips, scores):
        if res["score"] in ["correct", "ambiguous"]:
            kept_strips.append(strip)
            
    return "\n\n".join(kept_strips)
