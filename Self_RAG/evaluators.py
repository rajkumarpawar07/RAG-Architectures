from pydantic import BaseModel
from enum import Enum
from llm_client import generate_structured
import asyncio

class RetrievalDecision(BaseModel):
    needs_retrieval: bool
    reason: str

class RelevanceResult(BaseModel):
    is_relevant: bool
    score: int  # 1-5
    reason: str

class SupportLevel(str, Enum):
    FULLY = "FULLY_SUPPORTED"
    PARTIAL = "PARTIALLY_SUPPORTED"
    NONE = "NO_SUPPORT"

class GroundednessResult(BaseModel):
    support_level: SupportLevel
    unsupported_claims: list[str]
    reason: str

class UsefulnessResult(BaseModel):
    is_useful: bool
    score: int  # 1-5
    reason: str

class QueryRewriteResult(BaseModel):
    hypothetical_document: str
    new_query: str
    reason: str

def check_needs_retrieval(query: str) -> RetrievalDecision:
    prompt = f"""
    You are an expert router. Determine if the following user query requires retrieving 
    external factual documents or if it can be answered using general parametric knowledge.
    Queries about specific individuals, company policies, proprietary information, or recent events need retrieval.
    General facts like "What is 2+2?" or "What is a dog?" do not.
    
    Query: {query}
    """
    return generate_structured(prompt, RetrievalDecision, use_pro=False)

async def check_relevance_async(query: str, chunk_text: str) -> RelevanceResult:
    prompt = f"""
    You are a strict relevance evaluator. Evaluate if the provided document chunk contains 
    information that is relevant to answering the user's query.
    Score it from 1 to 5, where 1 is completely irrelevant and 5 is highly relevant.
    Consider it relevant (is_relevant=True) if the score is >= 3.
    
    Query: {query}
    Document Chunk: {chunk_text}
    """
    # Wrap synchronous Gemini call in asyncio.to_thread for parallel execution
    return await asyncio.to_thread(generate_structured, prompt, RelevanceResult, False)

def check_groundedness(draft_answer: str, retrieved_context: str) -> GroundednessResult:
    prompt = f"""
    You are a strict fact-checker. Determine if the drafted answer is supported by the provided context.
    - FULLY_SUPPORTED: All claims in the answer are found in the context.
    - PARTIALLY_SUPPORTED: Some claims are found, but others are fabricated or unsupported.
    - NO_SUPPORT: The answer is entirely fabricated or ignores the context.
    
    If not FULLY_SUPPORTED, list the exact unsupported claims.
    
    Context:
    {retrieved_context}
    
    Draft Answer:
    {draft_answer}
    """
    return generate_structured(prompt, GroundednessResult, use_pro=True)

def check_usefulness(query: str, final_answer: str) -> UsefulnessResult:
    prompt = f"""
    Evaluate if the final answer genuinely and effectively answers the user's original query.
    Score from 1 to 5. Consider it useful (is_useful=True) if score is >= 3.
    If the answer says "I don't have enough information", it is NOT useful (is_useful=False).
    
    Query: {query}
    Final Answer: {final_answer}
    """
    return generate_structured(prompt, UsefulnessResult, use_pro=False)

def rewrite_query(original_query: str) -> QueryRewriteResult:
    prompt = f"""
    The original query failed to retrieve useful information. 
    Write a hypothetical document that would perfectly answer the query (HyDE approach). 
    Also, write a new, improved search query that is more specific.
    
    Original Query: {original_query}
    """
    return generate_structured(prompt, QueryRewriteResult, use_pro=False)
