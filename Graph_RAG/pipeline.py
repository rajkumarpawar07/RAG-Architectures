from langchain_neo4j import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings
from graph_builder import graph_store

def get_qa_chain():
    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        temperature=0
    )
    
    # Create the Graph Cypher QA Chain
    chain = GraphCypherQAChain.from_llm(
        cypher_llm=llm,
        qa_llm=llm,
        graph=graph_store.graph,
        verbose=True,
        allow_dangerous_requests=True # Required to run generated cypher queries against the DB
    )
    return chain

def run_graph_rag(question: str) -> dict:
    """Executes the GraphRAG pipeline for a given question."""
    chain = get_qa_chain()
    
    # The chain outputs 'result' which is the final LLM string.
    # It prints its internal intermediate steps (Cypher query generation) due to verbose=True.
    response = chain.invoke({"query": question})
    
    return response
