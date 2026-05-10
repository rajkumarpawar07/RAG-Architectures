<h1 align="center">Self-Reflective RAG (Self-RAG)</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Google%20Gemini-SDK-4285F4?style=flat-square&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-E34F26?style=flat-square" />
  <img src="https://img.shields.io/badge/Pydantic-Structured%20Outputs-E92063?style=flat-square" />
  <img src="https://img.shields.io/badge/AsyncIO-Parallel%20Eval-blueviolet?style=flat-square" />
  <img src="https://img.shields.io/badge/Typer%20%2B%20Rich-CLI-00C7B7?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
</p>

<p align="center">
  A RAG pipeline with a built-in conscience — the LLM evaluates its own retrievals,<br/>
  fact-checks its own answers, and rewrites its own queries when it falls short.<br/>
  Part of the <a href="https://github.com/rajkumarpawar07/RAG-Architectures"><strong>RAG-Architectures</strong></a> collection.
</p>

---

## The Problem This Solves

Standard RAG has no self-awareness — it retrieves blindly, generates unconditionally, and never questions whether the answer it produced is actually grounded. The result is confident-sounding hallucinations that are hard to detect.

**Self-RAG** replaces the linear retrieve-then-generate pipeline with a **pure Python state machine** where the LLM acts as its own critic at five distinct checkpoints. It can skip retrieval entirely for trivial queries, drop irrelevant chunks before generation, detect unsupported claims in its own output, loop to correct hallucinations, and rewrite its query from scratch using HyDE if the answer doesn't actually help the user.

---

## The 5 Reflection Gates

```
[IsRet] → [IsRel] → Generate → [IsSup] → [Revise] → [IsUse]
```

**Gate 1 — IsRet: Retrieval Decision**
Does this query actually require searching the database? Factual lookups skip retrieval entirely (e.g. *"What is 2+2"* never touches the vector store), saving latency and avoiding irrelevant context injection.

**Gate 2 — IsRel: Relevance Filter** *(Parallel)*
Retrieved chunks are evaluated concurrently and scored 1–5 for relevance. Any chunk scoring below 3 is dropped before generation. If no chunks survive, HyDE is triggered immediately.

**Gate 3 — IsSup: Groundedness Check**
After a draft answer is generated, the model acts as a strict fact-checker against the retrieved context, assigning one of three verdicts: `FULLY_SUPPORTED`, `PARTIALLY_SUPPORTED`, or `NO_SUPPORT`.

**Gate 4 — Revise: Hallucination Correction Loop**
If `IsSup` finds unsupported claims, the draft is returned to the generator with explicit instructions to remove fabricated content. This loop runs up to **3 times** before escalating.

**Gate 5 — IsUse: Usefulness Check + HyDE Rewrite**
If the final grounded answer doesn't actually address the user's question, the system uses **HyDE** (Hypothetical Document Embeddings) — writing a hypothetical ideal answer, embedding it, and triggering a completely fresh retrieval loop with a better semantic signal.

---

## Architecture

<img width="1086" height="1448" alt="ChatGPT Image May 10, 2026, 09_22_17 PM" src="https://github.com/user-attachments/assets/1e2f952d-cfcb-4a62-8940-9ff4714ab446" />

---

## Project Structure

```
Self_RAG/
├── data/                   # Drop your PDFs, DOCX, and HTML files here
├── chroma_db/              # Auto-created: persistent ChromaDB vector store
├── config.py               # Central config: thresholds, loop limits, chunk sizes
├── document_loader.py      # Docling-based layout-aware document parsing
├── chunker.py              # Recursive character text splitting
├── embedder.py             # Gemini embeddings (768D truncated)
├── vector_store.py         # ChromaDB operations: upsert and cosine search
├── gates/
│   ├── is_ret.py           # Gate 1: Retrieval decision
│   ├── is_rel.py           # Gate 2: Async relevance scoring
│   ├── is_sup.py           # Gate 3: Groundedness check
│   ├── revise.py           # Gate 4: Hallucination correction
│   └── is_use.py           # Gate 5: Usefulness check + HyDE rewrite
├── generator.py            # Gemini LLM: draft and final answer generation
├── rag_pipeline.py         # State machine orchestration
├── evaluator.py            # Batch eval harness (hallucination rate, latency)
├── main.py                 # Typer CLI entry point
├── requirements.txt        # Python dependencies
└── .env                    # API key (not committed)
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- A Google Gemini API key — get one at [Google AI Studio](https://aistudio.google.com/apikey)

### Installation

**1. Clone the repository and navigate to the module**

```bash
git clone https://github.com/rajkumarpawar07/RAG-Architectures.git
cd RAG-Architectures/Self_RAG
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure your API key**

Create a `.env` file in the `Self_RAG/` directory:

```env
GEMINI_API_KEY="your_google_ai_studio_key"
```

---

## Usage

### Ingest Documents

Place your `.pdf`, `.docx`, or `.html` files into the `data/` folder, then run:

```bash
python main.py ingest
```

Parses with Docling, chunks recursively, embeds with 768D truncation, and upserts to ChromaDB.

---

### Query — Standard

Ask a question and watch the reflection gates execute in real-time via the Rich terminal UI:

```bash
python main.py query "Who is Rajkumar Pawar and what are his skills?"
```

---

### Query — Verbose Mode

Add `-v` to print the full JSON trace of every reflection token as the state machine runs:

```bash
python main.py query "Who is Rajkumar Pawar?" --verbose
```

---

### Evaluation Harness

Run a batch `.jsonl` file of `{"question": "...", "expected_answer": "..."}` pairs to measure hallucination rate, relevance precision, and latency across your configuration:

```bash
python main.py eval test_cases.jsonl
```

---

### View Index Statistics

```bash
python main.py stats
```

Displays the number of chunks currently indexed in ChromaDB.

---

## Configuration

All thresholds and loop limits live in `config.py`.

| Parameter              | Default | Description                                             |
|------------------------|---------|---------------------------------------------------------|
| `CHUNK_SIZE`           | `800`   | Characters per chunk                                    |
| `CHUNK_OVERLAP`        | `150`   | Overlap between consecutive chunks                      |
| `RELEVANCE_THRESHOLD`  | `3`     | Minimum IsRel score (out of 5) to keep a chunk          |
| `USEFULNESS_THRESHOLD` | `3`     | Minimum IsUse score (out of 5) to accept a final answer |
| `MAX_REVISIONS`        | `3`     | Maximum hallucination correction loops (Gate 4)         |
| `MAX_RETRIEVAL_LOOPS`  | `2`     | Maximum HyDE-triggered retrieval retries (Gate 5)       |

---

## Tech Stack

| Component            | Library / Model                                         |
|----------------------|---------------------------------------------------------|
| Fast Classifier LLM  | `gemini-2.5-flash-lite` via `google-genai`              |
| Generator LLM        | `gemini-1.5-pro` via `google-genai`                     |
| Embeddings           | `gemini-embedding-001` (768D truncated) via `google-genai` |
| Vector Store         | [ChromaDB](https://www.trychroma.com) (cosine similarity)|
| Document Parsing     | [Docling](https://github.com/DS4SD/docling)             |
| Structured Outputs   | Pydantic models via `response_schema`                   |
| Async Runtime        | `asyncio` (parallel Gate 2 evaluation)                  |
| CLI + Terminal UI    | [Typer](https://typer.tiangolo.com) + [Rich](https://rich.readthedocs.io) |
| Config Management    | `python-dotenv`                                         |
| Language             | Python 3.9+                                             |

---

## How It Differs from Previous Modules

| Capability                     | Standard RAG | Conversational RAG | Corrective RAG | Self-RAG |
|--------------------------------|:------------:|:------------------:|:--------------:|:--------:|
| Multi-turn memory              | ✗            | ✓                  | ✓              | ✗        |
| Query rewriting                | ✗            | ✓                  | ✓              | ✓ (HyDE) |
| Retrieval quality grading      | ✗            | ✗                  | ✓              | ✓        |
| Web search fallback            | ✗            | ✗                  | ✓              | ✗        |
| Retrieval skip for simple queries | ✗         | ✗                  | ✗              | ✓        |
| Hallucination detection        | ✗            | ✗                  | ✗              | ✓        |
| Self-correction loop           | ✗            | ✗                  | ✗              | ✓        |
| HyDE re-retrieval              | ✗            | ✗                  | ✗              | ✓        |
| Dual LLM (speed + accuracy)    | ✗            | ✗                  | ✗              | ✓        |
| Batch evaluation harness       | ✗            | ✗                  | ✗              | ✓        |
| Vector backend                 | FAISS        | Qdrant             | pgvector       | ChromaDB |

---

## Part of RAG-Architectures

This module represents the most autonomous pipeline in the collection — the LLM critiques and corrects its own outputs without external intervention.

```
RAG-Architectures/
├── Standard_RAG/
├── Conversational_RAG/
├── Corrective_RAG/
├── Self_RAG/            ◀ You are here
├── Adaptive_RAG/
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
