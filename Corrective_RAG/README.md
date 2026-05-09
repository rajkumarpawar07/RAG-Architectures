<h1 align="center">Corrective RAG (CRAG)</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Google%20Gemini-SDK-4285F4?style=flat-square&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/PostgreSQL-pgvector-336791?style=flat-square&logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/Tavily-Web%20Search-FF6F00?style=flat-square" />
  <img src="https://img.shields.io/badge/AsyncIO-Parallel%20Grading-blueviolet?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
</p>

<p align="center">
  A highly resilient RAG pipeline with a <strong>Decision Gate</strong> that grades retrieved documents<br/>
  and falls back to live web search when internal knowledge is insufficient —<br/>
  drastically reducing hallucinations at the source.<br/>
  Part of the <a href="https://github.com/rajkumarpawar07/RAG-Architectures"><strong>RAG-Architectures</strong></a> collection.
</p>

---

## The Problem This Solves

Standard RAG blindly trusts the vector database. If retrieved chunks are irrelevant to the query, the LLM still attempts to generate an answer — producing high-confidence hallucinations from low-quality context.

**Corrective RAG (CRAG)** intercepts this failure point with a **Retrieval Evaluator** that grades retrieved documents *before* they reach the generator. Based on the verdict — `Correct`, `Ambiguous`, or `Incorrect` — the pipeline either refines internal knowledge, falls back to live web search, or blends both. The LLM only generates from sources that have been validated.

---

## Features

**Decision Gate (Retrieval Evaluator)**
A lightweight LLM grader classifies each retrieval as `Correct`, `Ambiguous`, or `Incorrect` before any answer is generated — acting as a quality checkpoint between retrieval and generation.

**Knowledge Refinement**
When chunks pass as `Correct`, they are decomposed into individual sentences. Only sentences that directly answer the query are retained, filtering out noise before passing context to the generator.

**Web Fallback via Tavily**
When internal documents score `Incorrect` or `Ambiguous`, the pipeline triggers a live Tavily web search to fetch accurate, up-to-date information — seamlessly substituting or supplementing internal knowledge.

**Query Rewriting for Web Search**
Before hitting the web, vague or conversational queries are rewritten into optimized search queries by an LLM, maximizing the relevance of web results.

**Fully Asynchronous Pipeline**
Uses `asyncio` and `asyncpg` to run LLM grading calls in parallel across retrieved chunks, significantly reducing end-to-end latency.

**Unified PostgreSQL Backend**
A single Postgres container handles four concerns — vector search (`pgvector`), conversation memory (`chat_history`), LLM response caching (`eval_cache`), and observability (`crag_runs`) — eliminating the need for multiple infrastructure dependencies.

---

## Architecture

<img width="1774" height="887" alt="ChatGPT Image May 10, 2026, 01_03_46 AM" src="https://github.com/user-attachments/assets/a295a435-08d6-4691-95b4-858dfe1aa663" />


---

## Project Structure

```
Corrective_RAG/
├── data/                   # Drop your PDFs and HTML files here
├── config.py               # Central config: models, DB DSN, top-k, thresholds
├── document_loader.py      # PDF and HTML document parsing
├── chunker.py              # Recursive character text splitting
├── embedder.py             # Gemini embeddings with batching
├── vector_store.py         # pgvector operations: upsert and similarity search
├── evaluator.py            # Decision Gate: async LLM grading (Correct / Ambiguous / Incorrect)
├── refiner.py              # Knowledge Refinement: sentence-level filtering
├── web_search.py           # Tavily web search + query rewriting
├── memory.py               # Postgres chat_history: save and retrieve turns
├── generator.py            # Gemini LLM: final answer generation
├── rag_pipeline.py         # Orchestrates all pipeline branches
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
└── .env                    # API keys (not committed)
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- A Google Gemini API key — get one at [Google AI Studio](https://aistudio.google.com/apikey)
- A Tavily API key — get one at [tavily.com](https://tavily.com)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Installation

**1. Start PostgreSQL with pgvector**

Port `5433` is used to avoid conflicts with local Postgres installations:

```bash
docker run --name pgvector-crag \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=rag_db \
  -p 5433:5432 -d \
  pgvector/pgvector:pg16
```

**2. Clone the repository and navigate to the module**

```bash
git clone https://github.com/rajkumarpawar07/RAG-Architectures.git
cd RAG-Architectures/Corrective_RAG
```

**3. Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Configure your API keys**

Create a `.env` file in the `Corrective_RAG/` directory:

```env
GOOGLE_API_KEY="your_gemini_api_key"
TAVILY_API_KEY="your_tavily_api_key"
POSTGRES_DSN="postgresql://postgres:postgres@localhost:5433/rag_db"
```

---

## Usage

### Ingest Documents

Place your `.pdf` or `.html` files into the `data/` folder, then run:

```bash
python main.py ingest
```

### Query

```bash
python main.py query "What is the price of Bitcoin today?"
```

**Example output when web fallback is triggered:**

```
[GATE] Decision: INCORRECT
[WEB SEARCH] Rewrote query to: 'current bitcoin price USD live market data'

------------------------------------------------------------
Answer:
The live Bitcoin price today is $80,876.36 USD.
[Source: https://coinmarketcap.com/currencies/bitcoin/]

------------------------------------------------------------
Decision Gate : INCORRECT
Web Triggered : True
Latency       : 35588 ms
------------------------------------------------------------
```

### View Observability Logs

```bash
python main.py stats
```

All pipeline runs are logged to the `crag_runs` table in PostgreSQL, including gate decisions, web fallback triggers, and latency metrics.

---

## Configuration

All pipeline parameters live in `config.py`.

| Parameter          | Default                                           | Description                                          |
|--------------------|---------------------------------------------------|------------------------------------------------------|
| `TOP_K`            | `5`                                               | Chunks retrieved per query                           |
| `MEMORY_WINDOW`    | `5`                                               | Conversation turns included in each prompt           |
| `POSTGRES_DSN`     | `postgresql://postgres:postgres@localhost:5433/rag_db` | PostgreSQL connection string                    |
| Embedding model    | `gemini-embedding-001`                            | Gemini embedding model                               |
| LLM                | `gemini-2.5-flash-lite`                           | Swap for `gemini-2.0-flash`, `gemini-2.5-flash`, etc. |

---

## Tech Stack

| Component              | Library / Service                            |
|------------------------|----------------------------------------------|
| Embeddings             | `gemini-embedding-001` via `google-genai`    |
| Vector Database        | PostgreSQL + `pgvector` via Docker           |
| Conversation Memory    | PostgreSQL (`chat_history` table)            |
| Observability          | PostgreSQL (`crag_runs` table)               |
| LLM Grader + Generator | `gemini-2.5-flash-lite` via `google-genai`   |
| Web Search             | [Tavily](https://tavily.com)                 |
| Async Runtime          | `asyncio` + `asyncpg`                        |
| Config Management      | `python-dotenv`                              |
| Language               | Python 3.9+                                  |

---

## How It Differs from Previous Modules

| Capability               | Standard RAG | Conversational RAG | Corrective RAG |
|--------------------------|:------------:|:------------------:|:--------------:|
| Multi-turn memory        | ✗            | ✓                  | ✓              |
| Query rewriting          | ✗            | ✓                  | ✓              |
| Retrieval quality grading| ✗            | ✗                  | ✓              |
| Web search fallback      | ✗            | ✗                  | ✓              |
| Knowledge refinement     | ✗            | ✗                  | ✓              |
| Async parallel grading   | ✗            | ✗                  | ✓              |
| Observability logging    | ✗            | ✗                  | ✓              |
| Vector backend           | FAISS        | Qdrant             | PostgreSQL + pgvector |

---

## Part of RAG-Architectures

This module extends Conversational RAG by introducing retrieval evaluation, knowledge refinement, and web fallback.

```
RAG-Architectures/
├── Standard_RAG/
├── Conversational_RAG/
├── Corrective_RAG/      ◀ You are here
├── Adaptive_RAG/
├── Self_RAG/
├── Fusion_RAG/
├── HyDE/
├── Agentic_RAG/
└── Graph_RAG/
```

🔗 [View the full collection →](https://github.com/rajkumarpawar07/RAG-Architectures)

---

## Contributing

Contributions, issues, and suggestions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: describe your change"`
4. Push and open a Pull Request

---

## License

MIT License — see the [LICENSE](../../LICENSE) file for details.

---

<p align="center">Built by <a href="https://github.com/rajkumarpawar07">Rajkumar Pawar</a></p>
