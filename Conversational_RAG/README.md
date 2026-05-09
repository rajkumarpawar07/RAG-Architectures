# Conversational RAG

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Google%20Gemini-SDK-4285F4?style=flat-square&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Qdrant-Vector%20DB-E5005A?style=flat-square" />
  <img src="https://img.shields.io/badge/SQLite-Memory-003B57?style=flat-square" />
  <img src="https://img.shields.io/badge/PyMuPDF-Fast%20Parser-FFD43B?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
</p>

<p align="center">
  A stateful RAG pipeline that remembers conversation history and dynamically rewrites queries<br/>
  to solve the <strong>"context blindness"</strong> problem of standard RAG systems.<br/>
  Part of the <a href="https://github.com/rajkumarpawar07/RAG-Architectures"><strong>RAG-Architectures</strong></a> collection.
</p>

---

## The Problem This Solves

In a standard RAG setup, every query is treated in isolation. If a user asks a follow-up like *"How much does it cost?"*, the system fails — it doesn't know what *"it"* refers to when searching the vector database. Each turn is context-blind.

This architecture fixes that with two additions:

- **Stateful Memory** (SQLite) — stores the last N conversation turns, persisted to disk and resumable via session IDs
- **Query Rewriting** (LLM) — uses conversation history to transform vague follow-ups into fully contextualized, standalone queries before hitting the vector database

The result: a natural, human-like chat experience grounded in your documents.

---

## Features

**Conversational Memory**
Uses `sqlite3` to persist and retrieve the last N turns natively on disk. Sessions are identified by unique IDs, so conversations can be paused and resumed at any time.

**LLM-Powered Query Rewriting**
A dedicated zero-temperature generation step rewrites contextual follow-ups (e.g., *"How much does it cost?"*) into fully resolved queries (e.g., *"What is the price of the Enterprise Plan?"*) before retrieval — dramatically improving vector search accuracy.

**High-Performance Vector Search**
Powered by **Qdrant** running locally via Docker — exact cosine similarity matching with a clean client interface for upsert and search operations.

**Fast PDF Ingestion**
Uses **PyMuPDF (`fitz`)** for rapid, lightweight text extraction during the ingestion phase.

**Context-Aware Generation**
The final LLM is fed three inputs simultaneously: retrieved chunks, the rewritten standalone query, and full conversation history — ensuring answers are both document-grounded and conversationally coherent.

---

## Architecture

```
╔══════════════════════════════════════════════════════════════╗
║                      INGESTION PIPELINE                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  📂 data/  (PDFs, TXTs)                                      ║
║       │                                                      ║
║       ▼                                                      ║
║  📑 PyMuPDF Parser  ─────────────  Fast text extraction      ║
║       │                                                      ║
║       ▼                                                      ║
║  ✂️  Recursive Chunker                                        ║
║       │                                                      ║
║       ▼                                                      ║
║  🔢 Gemini Embedder  ────────────  Batched                   ║
║       │                                                      ║
║       ▼                                                      ║
║  🗃️  Qdrant Vector DB  ──────────  Local via Docker          ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║                  CONVERSATIONAL QUERY PIPELINE               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ❓ User Query                                               ║
║       │                                                      ║
║       ├── Has History? ──Yes──▶ Fetch Last N Turns (SQLite)  ║
║       │                               │                      ║
║       │                               ▼                      ║
║       │                        Query Rewriter (LLM)          ║
║       │                               │                      ║
║       └────────── No ─────────────────┘                      ║
║                                       │                      ║
║                                       ▼                      ║
║                            Embed Standalone Query            ║
║                                       │                      ║
║                                       ▼                      ║
║                            Search Qdrant → Top-K Chunks      ║
║                                       │                      ║
║                                       ▼                      ║
║                   Prompt Builder  ←───┤                      ║
║                   (Chunks + Query + History)                  ║
║                                       │                      ║
║                                       ▼                      ║
║                          Gemini LLM Generator                ║
║                                       │                      ║
║                                       ▼                      ║
║                          Answer + Source Citations           ║
║                                       │                      ║
║                                       ▼                      ║
║                     Save Q&A Pair ──▶ SQLite Memory DB       ║
╚══════════════════════════════════════════════════════════════╝
```
<img width="1774" height="887" alt="ChatGPT Image May 9, 2026, 11_07_41 PM" src="https://github.com/user-attachments/assets/7a7b2726-f78e-4e83-aa25-bb3531d9ee81" />

---

## Project Structure

```
Conversational_RAG/
├── data/                   # Drop your PDFs and .txt files here
├── memory.db               # Auto-created: SQLite DB for chat history
├── config.py               # Central config: models, paths, top-k, memory window
├── document_loader.py      # Fast PDF text extraction using PyMuPDF
├── chunker.py              # Recursive character text splitting
├── embedder.py             # Gemini embeddings with batching
├── vector_store.py         # Qdrant client: upsert and search operations
├── memory.py               # SQLite: save and retrieve conversation history
├── query_rewriter.py       # LLM-based contextual query rewriting
├── generator.py            # Gemini LLM: final answer generation
├── rag_pipeline.py         # Orchestrates ingestion + conversational flows
├── main.py                 # CLI entry point with session ID support
├── requirements.txt        # Python dependencies
└── .env                    # API key (not committed)
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- A Google Gemini API key — get one at [Google AI Studio](https://aistudio.google.com/apikey)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (to run Qdrant locally)

### Installation

**1. Start the Qdrant Vector Database**

Ensure Docker is running, then launch the Qdrant container:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**2. Clone the repository and navigate to the module**
```bash
git clone https://github.com/rajkumarpawar07/RAG-Architectures.git
cd RAG-Architectures/Conversational_RAG
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

**5. Configure your API key**

Create a `.env` file in the `Conversational_RAG/` directory:
```env
GOOGLE_API_KEY="your_api_key_here"
```

---

## Usage

### Ingest Documents

Place your `.pdf` or `.txt` files into the `data/` folder, then run:

```bash
python main.py ingest
```

Parses, chunks, embeds, and upserts all documents into your local Qdrant instance.

---

### Interactive Chat (With Memory)

Start a continuous, stateful Q&A session. The system remembers your previous questions within the session.

```bash
python main.py chat --session "my_chat_1"
```

Type `quit` or `exit` to end the session. Resume it later by passing the same `--session` ID.

---

### Query — Single Question

```bash
python main.py query "Who is Rajkumar Pawar?" --session "my_chat_1"
```

Appends to the specified session's history so context carries forward across calls.

---

### View Index Statistics

```bash
python main.py stats
```

Displays the number of vectors currently stored in Qdrant.

---

## Configuration

All pipeline parameters live in `config.py`.

| Parameter         | Default                  | Description                                              |
|-------------------|--------------------------|----------------------------------------------------------|
| `MEMORY_WINDOW`   | `5`                      | Number of past Q&A turns included in each prompt         |
| `CHUNK_SIZE`      | `1000`                   | Characters per chunk                                     |
| `CHUNK_OVERLAP`   | `200`                    | Overlap between consecutive chunks                       |
| `TOP_K`           | `5`                      | Number of chunks retrieved per query                     |
| `QDRANT_URL`      | `http://localhost:6333`  | Qdrant server URL                                        |
| `QDRANT_COLLECTION` | `rag_docs`             | Qdrant collection name                                   |
| LLM               | `gemini-2.5-flash-lite`  | Swap for `gemini-2.0-flash`, `gemini-2.5-flash`, etc.   |

---

## Tech Stack

| Component         | Library / Model                            |
|-------------------|--------------------------------------------|
| PDF Parsing       | PyMuPDF (`fitz`)                           |
| Embeddings        | `gemini-embedding-001` via `google-genai`  |
| Vector Database   | [Qdrant](https://qdrant.tech) via Docker   |
| Conversation Memory | `sqlite3` (stdlib)                       |
| Query Rewriting   | Gemini LLM (zero temperature)              |
| LLM Generator     | `gemini-2.5-flash-lite` via `google-genai` |
| Config Management | `python-dotenv`                            |
| Language          | Python 3.9+                                |

---

## How It Differs from Standard RAG

| Capability                  | Standard RAG | Conversational RAG |
|-----------------------------|--------------|--------------------|
| Multi-turn memory           | ✗            | ✓ (SQLite)         |
| Follow-up question handling | ✗            | ✓ (Query rewriting)|
| Session persistence         | ✗            | ✓ (Session IDs)    |
| Vector database             | FAISS        | Qdrant             |
| PDF parser                  | Docling      | PyMuPDF            |

---

## Part of RAG-Architectures

This module builds on the Standard RAG pipeline by introducing conversational state and query rewriting.

```
RAG-Architectures/
├── Standard_RAG/
├── Conversational_RAG/  ◀ You are here
├── HyDE_RAG/
├── Corrective_RAG/
├── Agentic_RAG/
├── Graph_RAG/
└── Hybrid_RAG/
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

<p align="center">Built by <a href="https://github.com/rajkumarpawar07">Raj Kumar Pawar</a></p>
