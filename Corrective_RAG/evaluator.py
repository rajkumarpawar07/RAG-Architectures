import hashlib
import json
import asyncio
from google import genai
from config import GOOGLE_API_KEY, GRADER_MODEL
from database import get_pool

GRADER_PROMPT = """You are a relevance evaluator. Given a user query and a document chunk,
output a JSON object with:
  - "score": one of "correct", "ambiguous", "incorrect"
  - "confidence": float 0.0-1.0
  - "reason": one sentence explaining the score

Only output valid JSON. No markdown formatting, no extra text.

User query: {query}
Document chunk: {chunk_text}"""

async def grade_chunk_async(query: str, chunk_text: str) -> dict:
    hash_input = f"{query}:::{chunk_text}".encode('utf-8')
    hash_key = hashlib.sha256(hash_input).hexdigest()
    
    pool = await get_pool()
    async with pool.acquire() as conn:
        cached = await conn.fetchrow("SELECT score, reason FROM eval_cache WHERE hash_key = $1", hash_key)
        if cached:
            return {"score": cached["score"], "reason": cached["reason"], "confidence": 1.0}

    prompt = GRADER_PROMPT.format(query=query, chunk_text=chunk_text)
    client = genai.Client(api_key=GOOGLE_API_KEY)
    
    def _call_llm():
        response = client.models.generate_content(
            model=GRADER_MODEL,
            contents=prompt,
            config={"temperature": 0.0}
        )
        return response.text.strip()
    
    loop = asyncio.get_event_loop()
    raw_response = await loop.run_in_executor(None, _call_llm)
    
    try:
        if raw_response.startswith("```json"):
            raw_response = raw_response.strip("```json\n").strip("\n```")
        elif raw_response.startswith("```"):
            raw_response = raw_response.strip("```\n").strip("\n```")
        result = json.loads(raw_response)
        
        score = result.get("score", "incorrect").lower()
        if score not in ["correct", "ambiguous", "incorrect"]:
            score = "incorrect"
            
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO eval_cache (hash_key, score, reason)
                VALUES ($1, $2, $3)
                ON CONFLICT (hash_key) DO NOTHING
            """, hash_key, score, result.get("reason", ""))
            
        return result
    except Exception as e:
        print(f"  [EVAL ERROR] {e} - Raw: {raw_response}")
        return {"score": "incorrect", "confidence": 0.0, "reason": "Failed to parse JSON"}

async def evaluate_chunks_async(query: str, chunks: list[dict]) -> list[dict]:
    tasks = [grade_chunk_async(query, chunk["text"]) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    
    graded = []
    for chunk, res in zip(chunks, results):
        graded.append({
            **chunk,
            "eval_score": res["score"],
            "eval_reason": res["reason"]
        })
    return graded
