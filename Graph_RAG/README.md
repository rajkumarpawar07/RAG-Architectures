# Graph RAG: The Relationship Reasoner

A pure Graph-based Retrieval-Augmented Generation (GraphRAG) architecture built entirely from scratch. While vector databases retrieve text based on semantic similarity ("what looks similar"), this architecture retrieves information based on deterministic paths ("what is connected, and how"). 

This project extracts entities and explicit relationships from unstructured text to construct a robust Knowledge Graph in **Neo4j**, enabling multi-hop causal reasoning that standard Vector RAGs struggle with.

---

## 🧠 Architecture

```mermaid
flowchart TD
    %% Styling
    classDef llm fill:#8A2BE2,stroke:#4B0082,stroke-width:2px,color:#fff,rx:5px
    classDef db fill:#008B8B,stroke:#006400,stroke-width:2px,color:#fff,rx:5px
    classDef doc fill:#FF8C00,stroke:#B8860B,stroke-width:2px,color:#fff,rx:5px
    classDef user fill:#DC143C,stroke:#8B0000,stroke-width:2px,color:#fff,rx:5px
    classDef logic fill:#4169E1,stroke:#00008B,stroke-width:2px,color:#fff,rx:5px

    %% Graph Construction Pipeline
    subgraph Offline [Phase 1: Knowledge Graph Construction]
        direction LR
        PDF[Raw Documents PDF]:::doc --> Loader[Docling Loader]:::logic
        Loader --> Chunks[Chunking]:::logic
        Chunks --> Gemini1{Gemini LLM\nEntity Extractor}:::llm
        Gemini1 -- Extracts --> Nodes[Nodes & Edges\n(Person)-[WORKED_AT]->(Org)]:::logic
        Nodes --> Neo4j[(Neo4j Graph Database)]:::db
    end

    %% Graph Query Pipeline
    subgraph Online [Phase 2: Graph Traversal & QA]
        direction TD
        Query([User Question]):::user --> GraphChain[GraphCypherQAChain]:::logic
        GraphChain --> Gemini2{Gemini LLM\nCypher Generator}:::llm
        Gemini2 -- Translates to Cypher --> Neo4j
        Neo4j -- Subgraph/Paths --> Gemini3{Gemini LLM\nSynthesis}:::llm
        Gemini3 --> Answer([Final Answer]):::user
    end

    Offline -.- Online
```

---

## ⚙️ How It Works

### The Ambiguity of Vectors vs. The Certainty of Graphs
Standard RAG relies on Vector Similarity. If you ask, *"Which company did the applicant work for before joining Accenture?"*, a vector search will blindly retrieve the chunks where "Accenture" and "work" are mentioned. It struggles with temporal relationships or causal links.

**Graph RAG** solves this by converting unstructured text into a deterministic, explicitly linked network of Nodes and Relationships:

1. **Information Extraction (`LLMGraphTransformer`)**: We feed the unstructured text to an LLM, asking it to strictly extract entities. It turns the sentence *"Rajkumar worked at Cogent E Services"* into a concrete graph relationship: `(Rajkumar:Person)-[:WORKED_AT]->(Cogent E Services:Organization)`.
2. **Cypher Generation (`GraphCypherQAChain`)**: When the user asks a question, the LLM translates the natural language into a database query (Cypher).
3. **Graph Traversal**: The system hits Neo4j to follow the exact relationship pathways, pulling out the exact deterministic facts.
4. **Answer Generation**: The explicit relationship paths are fed back to the LLM to generate a conversational answer.

## 🛠️ Tech Stack
*   **Orchestration**: `LangChain` & `langchain-neo4j`
*   **Graph Database**: `Neo4j v5` (Dockerized with APOC plugins)
*   **LLM**: `Google Gemini 3.1 Flash Lite`
*   **Parser**: `Docling`
*   **CLI**: `Typer` & `Rich`

---

## 🚀 Setup & Execution

### 1. Prerequisites
You need Docker installed to run the Neo4j database locally.

### 2. Environment Setup
Install dependencies and configure your environment:
```bash
pip install -r requirements.txt
```
Create a `.env` file in this directory with your API keys:
```env
GOOGLE_API_KEY="your_api_key_here"

# Optional: LangSmith Tracing
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="your_langsmith_key"
LANGSMITH_PROJECT="GraphRAG"
```

### 3. Spin up Neo4j
We use Docker Compose to spin up Neo4j with the `APOC` (Awesome Procedures On Cypher) plugin enabled.
```bash
docker-compose up -d
```

### 4. Build the Graph (Ingestion)
Drop a PDF into the `data/` folder and run the ingest command. Be patient! The LLM reads every chunk to extract nodes and relationships.
```bash
python main.py ingest
```

### 5. Traverse the Graph (Query)
Ask a multi-hop or relationship-based question:
```bash
python main.py query "Where has Rajkumar Pawar worked?"
```

### 6. View Graph Statistics
You can inspect the generated graph schema:
```bash
python main.py stats
```
