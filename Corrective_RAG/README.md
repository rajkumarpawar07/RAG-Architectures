<h1 align="center">Corrective RAG (CRAG)</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Google%20Gemini-SDK-4285F4?style=flat-square&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/PostgreSQL-pgvector-336791?style=flat-square&logo=postgresql" />
  <img src="https://img.shields.io/badge/Tavily-Web%20Search-FF6F00?style=flat-square" />
  <img src="https://img.shields.io/badge/AsyncIO-Fast-blueviolet?style=flat-square" />
</p>

<p align="center">
  A highly resilient RAG pipeline that evaluates internal document quality via a "Decision Gate",<br/>
  falling back to live web search when necessary to drastically reduce hallucinations.<br/>
  Part of the <a href="https://github.com/rajkumarpawar07/RAG-Architectures"><strong>RAG-Architectures</strong></a> collection.
</p>

---

## The Problem This Solves

Traditional RAG blindly trusts the vector database. If a user asks a question and the DB retrieves irrelevant information, the LLM will try to generate an answer anyway—resulting in high-confidence hallucinations.

**Corrective RAG (CRAG)** fixes this by grading the retrieved documents before generating an answer.

---

## Features

- **Retrieval Evaluator (Decision Gate)**: A lightweight LLM grades retrieved chunks as `Correct`, `Ambiguous`, or `Incorrect`.
- **Knowledge Refinement**: `Correct` chunks are split into sentences. Only sentences that directly answer the query are kept, filtering out noise.
- **Web Fallback (Tavily)**: If internal documents are `Incorrect` or `Ambiguous`, CRAG triggers a live web search to fetch accurate real-world data.
- **Query Rewriting**: Before searching the web, the system rewrites vague queries into highly optimized search queries.
- **Fully Asynchronous**: Uses `asyncio` and `asyncpg` to run LLM grading calls in parallel, dropping latency significantly.
- **Universal PostgreSQL**: Uses a single Postgres container for Vector Search (`pgvector`), Memory (`chat_history`), LLM Caching (`eval_cache`), and Observability (`crag_runs`).

---

## Architecture

```mermaid
flowchart TD
    %% Query Pipeline
    subgraph Query["💬 Corrective RAG Query Pipeline"]
        E["User Query"] --> F["Fetch History\n(Postgres Memory)"]
        F --> G["Rewrite Query (LLM)"]
        G --> H["Search PostgreSQL\n(Vector Similarity)"]
        
        H --> I{"Retrieval Evaluator\n(Grader LLM)"}
        
        %% Correct Branch
        I -->|Correct| J["Knowledge Refinement\n(Decompose, Filter, Recombine)"]
        
        %% Incorrect Branch
        I -->|Incorrect| K["Tavily Web Search\n(Live Knowledge)"]
        
        %% Ambiguous Branch
        I -->|Ambiguous| L["Merge Refined Internal\n+ Web Search"]
        J -.-> L
        K -.-> L
        
        %% Generation
        J --> M["Generator LLM"]
        K --> M
        L --> M
        
        M --> N["Answer + Sources"]
        
        %% Post Generation
        N -.->|"Save Turn"| O[("Postgres\nchat_history")]
        N -.->|"Log Metrics"| P[("Postgres\ncrag_runs")]
    end
```

---

## Getting Started

### 1. Start PostgreSQL (with pgvector)

Run the database container locally (we map to port 5433 to avoid conflicts with native Postgres installations):

```bash
docker run --name pgvector-crag \
  -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=rag_db \
  -p 5433:5432 -d pgvector/pgvector:pg16
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup API Keys

Create a `.env` file in the `Corrective_RAG/` directory:

```env
GOOGLE_API_KEY="your_gemini_api_key"
TAVILY_API_KEY="your_tavily_api_key"
POSTGRES_DSN="postgresql://postgres:postgres@localhost:5433/rag_db"
```

---

## Usage

### Ingest Documents

Drop your PDFs or HTML files into the `data/` folder:

```bash
python main.py ingest
```

### Run a Query

Ask a specific question:

```bash
python main.py query "What is the price of Bitcoin today?"
```

**Example Output (Web Fallback Triggered):**
```text
[GATE] Decision: INCORRECT
[WEB SEARCH] Rewrote query to: 'current bitcoin price USD live market data'

------------------------------------------------------------
Answer:
The live Bitcoin price today is $80,876.36 USD [Source: https://coinmarketcap.com/currencies/bitcoin/].

------------------------------------------------------------
Decision Gate : INCORRECT
Web Triggered : True
Latency       : 35588 ms
------------------------------------------------------------
```

### Check Observability Logs

Check how many queries fell back to web search:

```bash
python main.py stats
```
*(All runs are logged to the `crag_runs` table in PostgreSQL).*

---

## Part of RAG-Architectures

```text
RAG-Architectures/
├── Standard_RAG/
├── Conversational_RAG/
├── Corrective_RAG_(CRAG)/  ◀ You are here
├── Adaptive_RAG/
├── Self-RAG/
├── Fusion_RAG/
├── HyDE/
├── Agentic_RAG/
└── Graph_RAG/
```

🔗 [View the full collection →](https://github.com/rajkumarpawar07/RAG-Architectures)
