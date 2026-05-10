import os
import asyncio
import threading
from langfuse import get_client
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from config import LLM_MODEL, EMBEDDING_MODEL

# Langfuse client
langfuse = get_client()

# Setup Ragas metrics with Gemini models
_llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
_embeddings = GoogleGenerativeAIEmbeddings(model=f"models/{EMBEDDING_MODEL}")

ragas_llm = LangchainLLMWrapper(_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(_embeddings)

metrics = [
    Faithfulness(llm=ragas_llm),
    ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
]

async def _score_with_ragas_async(trace_id: str, query: str, contexts: list[str], answer: str):
    """Run Ragas evaluation and push scores to Langfuse asynchronously."""
    # Add minor delay so main thread finishes printing the answer first
    await asyncio.sleep(1)
    print(f"\n[Observability] Started background Ragas evaluation for trace...")
    
    sample = SingleTurnSample(
        user_input=query,
        retrieved_contexts=contexts,
        response=answer,
    )
    
    for metric in metrics:
        try:
            score = await metric.single_turn_ascore(sample)
            print(f"\n[Observability] {metric.name} score: {score:.3f}")
            
            # Send to Langfuse
            langfuse.score(
                trace_id=trace_id,
                name=metric.name,
                value=score
            )
        except Exception as e:
            print(f"\n[Observability ERROR] Failed to calculate {metric.name}: {e}")
            
    # Ensure all events are flushed to Langfuse
    langfuse.flush()
    print(f"[Observability] Finished evaluation.")

def evaluate_in_background(trace_id: str, query: str, contexts: list[str], answer: str):
    """Fire and forget wrapper to run the evaluation in a separate thread."""
    def run_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_score_with_ragas_async(trace_id, query, contexts, answer))
        loop.close()
        
    thread = threading.Thread(target=run_loop)
    thread.daemon = False # Set False so script waits for eval to finish before exiting completely
    thread.start()
