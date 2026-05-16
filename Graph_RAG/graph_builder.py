import logging
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from config import settings

logger = logging.getLogger(__name__)

class GraphStoreManager:
    def __init__(self):
        # Initialize connection to Neo4j
        self.graph = Neo4jGraph(
            url=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
            enhanced_schema=True
        )
        
        # Initialize Gemini LLM for extraction
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=0, # Use 0 temperature for deterministic entity extraction
        )
        
        # Initialize Graph Transformer
        self.transformer = LLMGraphTransformer(
            llm=self.llm,
        )

    def extract_and_store(self, chunks: list[Document]):
        """Convert chunks to GraphDocuments and insert to Neo4j."""
        if not chunks:
            return
            
        logger.info(f"Extracting nodes and edges from {len(chunks)} chunks... This may take a while.")
        
        # Convert documents to GraphDocuments (Nodes & Relationships)
        graph_documents = self.transformer.convert_to_graph_documents(chunks)
        
        logger.info(f"Extracted {len(graph_documents)} graph documents. Inserting into Neo4j...")
        
        # Add to Neo4j
        self.graph.add_graph_documents(
            graph_documents, 
            baseEntityLabel=True, 
            include_source=True
        )
        
        # Refresh schema after insertion
        self.graph.refresh_schema()
        logger.info("Graph insertion complete.")
        
    def query(self, cypher_query: str):
        """Execute a raw cypher query against the DB."""
        return self.graph.query(cypher_query)

# Provide a lazy singleton to avoid connecting on import if Neo4j is offline
_store_instance = None

def get_graph() -> GraphStoreManager:
    global _store_instance
    if _store_instance is None:
        _store_instance = GraphStoreManager()
    return _store_instance

class _LazyGraph:
    def __getattr__(self, name):
        return getattr(get_graph(), name)

graph_store = _LazyGraph()
