from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from config import settings
from vector_store import store
import logging

logger = logging.getLogger(__name__)

# Setup LLM
llm = ChatGoogleGenerativeAI(
    model=settings.LLM_MODEL,
    temperature=0.2
)

# ---------------------------------------------------------
# 1. Query Expansion
# ---------------------------------------------------------
query_expansion_template = """You are an AI assistant tasked with generating multiple search queries to maximize recall.
Generate {num_queries} distinct, highly relevant search queries based on the original user query.
These queries will be used to retrieve documents from a vector database.

Original Query: {question}

Output the queries separated by newlines, with NO numbering or bullet points. Just the raw queries."""

query_expansion_prompt = PromptTemplate(
    input_variables=["question", "num_queries"],
    template=query_expansion_template
)

def parse_queries(text: str) -> list[str]:
    # Split by newline and filter empty
    queries = [q.strip() for q in text.split('\n') if q.strip()]
    return queries

generate_queries_chain = (
    query_expansion_prompt 
    | llm 
    | StrOutputParser() 
    | RunnableLambda(parse_queries)
)

# ---------------------------------------------------------
# 2 & 3. Parallel Retrieval and Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------
def reciprocal_rank_fusion(results: list[list[Document]], k=60) -> list[Document]:
    """
    Reciprocal Rank Fusion (RRF).
    Takes a list of document lists (one list per query).
    Computes RRF score = sum(1 / (rank + k))
    """
    fused_scores = {}
    doc_map = {}
    
    for docs in results:
        for rank, doc in enumerate(docs):
            # Unique identifier for deduplication
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:100])
            
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                doc_map[doc_id] = doc
                
            fused_scores[doc_id] += 1 / (rank + k)
            
    # Sort by RRF score descending
    reranked_docs = [
        doc_map[doc_id] for doc_id, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    
    return reranked_docs[:settings.FINAL_TOP_K]

def get_fused_documents(question: str) -> dict:
    """Takes original query, expands it, retrieves for all, and fuses them."""
    logger.info("Generating query variations...")
    expanded_queries = generate_queries_chain.invoke({
        "question": question, 
        "num_queries": settings.NUM_QUERIES
    })
    
    # Include original query
    all_queries = [question] + expanded_queries
    logger.info(f"Generated {len(expanded_queries)} variations. Retrieving across {len(all_queries)} total queries in parallel...")
    
    retriever = store.get_retriever(k=settings.TOP_K_PER_QUERY)
    
    # Parallel retrieval via LCEL's .map()
    retrieval_results = retriever.map().invoke(all_queries)
    
    logger.info("Applying Reciprocal Rank Fusion (RRF)...")
    fused_docs = reciprocal_rank_fusion(retrieval_results)
    
    return {
        "question": question,
        "expanded_queries": expanded_queries,
        "fused_docs": fused_docs
    }

# ---------------------------------------------------------
# 4. Final Generation
# ---------------------------------------------------------
generation_template = """You are a helpful research assistant. Answer the user's question based ONLY on the following fused context.
If you cannot answer the question based on the context, say "I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:"""

generation_prompt = ChatPromptTemplate.from_template(generation_template)

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"--- Document ---\nSource: {d.metadata.get('source', 'unknown')}\n{d.page_content}" for d in docs)

def generate_answer(inputs: dict) -> str:
    """Executes the final generation step using the fused docs."""
    fused_docs = inputs["fused_docs"]
    question = inputs["question"]
    context = format_docs(fused_docs)
    
    logger.info("Generating final answer...")
    chain = generation_prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

def run_fusion_rag(question: str) -> dict:
    """End to end pipeline execution."""
    fusion_data = get_fused_documents(question)
    answer = generate_answer(fusion_data)
    
    return {
        "expanded_queries": fusion_data["expanded_queries"],
        "fused_docs": fusion_data["fused_docs"],
        "answer": answer
    }
