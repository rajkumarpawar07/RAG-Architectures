# Graph RAG: The Relationship Reasoner

A pure Graph-based Retrieval-Augmented Generation (GraphRAG) architecture built entirely from scratch. While vector databases retrieve text based on semantic similarity ("what looks similar"), this architecture retrieves information based on deterministic paths ("what is connected, and how"). 

This project extracts entities and explicit relationships from unstructured text to construct a robust Knowledge Graph in **Neo4j**, enabling multi-hop causal reasoning that standard Vector RAGs struggle with.

---

## 🧠 Architecture

<img width="1692" height="930" alt="ChatGPT Image May 17, 2026, 12_46_01 AM" src="https://github.com/user-attachments/assets/171fdb04-8758-422e-83d3-30317839022a" />

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
