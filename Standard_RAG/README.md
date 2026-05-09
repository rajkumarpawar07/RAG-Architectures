# Standard RAG - Retrieval-Augmented Generation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Google%20Gemini-SDK-4285F4?style=flat-square&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-FF6F00?style=flat-square" />
  <img src="https://img.shields.io/badge/Docling-PDF%20Parser-6A0DAD?style=flat-square" />
  <img src="https://img.shields.io/badge/Dependencies-%3C%205-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
</p>

<p align="center">
  A transparent, from-scratch implementation of a Standard RAG pipeline — built to be understood, not abstracted away.<br/>
  Part of the <a href="https://github.com/rajkumarpawar07/RAG-Architectures"><strong>RAG-Architectures</strong></a> collection.
</p>

---

## Why This Exists

Most RAG tutorials reach for LangChain or LlamaIndex. While powerful, those frameworks hide the mechanics that matter most — chunking logic, embedding strategies, prompt structure, index design. This project builds the full pipeline in **fewer than 5 dependencies**, giving you complete visibility and control over every component.

---

## Features

**Intelligent PDF Parsing**
Uses [Docling](https://github.com/DS4SD/docling) instead of naive text extraction — preserving tables, headers, and document structure with high fidelity.

**Production-Grade Chunking**
Recursive Character Text Splitting maintains semantic coherence across paragraph and sentence boundaries, the current industry standard.

**State-of-the-Art Embeddings**
Powered by `gemini-embedding-001` (3072-dimensional vectors) via the official `google-genai` SDK, with task-specific types (`RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`) that improve semantic matching by 5–10%.

**Efficient Vector Search**
`faiss-cpu` with exact inner product search (cosine similarity via L2 normalization) delivers sub-millisecond retrieval.

**Grounded Generation**
`gemini-2.5-flash-lite` generates answers strictly grounded in retrieved context, with source citations enforced through prompt engineering.

**Resilient API Calls**
Exponential backoff handles rate limits (HTTP 429) automatically throughout the pipeline.

---

## Architecture

<img width="1536" height="823" alt="ChatGPT Image May 9, 2026, 08_48_48 PM" src="https://github.com/user-attachments/assets/e81eb559-4aab-4795-b066-66b086679676" />

---

## Project Structure

```
Standard_RAG/
├── data/                   # Drop your PDFs and .txt files here
├── faiss_index/            # Auto-created: persisted FAISS index + metadata
├── config.py               # Central config: models, chunk size, paths, top-k
├── document_loader.py      # PDF (Docling) and plain-text file loading
├── chunker.py              # Recursive character text splitting
├── embedder.py             # Gemini embeddings with batching & exponential backoff
├── vector_store.py         # FAISS index: build, save, load, search
├── generator.py            # Prompt engineering + Gemini LLM generation
├── rag_pipeline.py         # Orchestrates ingestion and query flows
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
└── .env                    # API key (not committed)
```

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- A Google Gemini API key — get one at [Google AI Studio](https://aistudio.google.com/apikey)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/rajkumarpawar07/RAG-Architectures.git
cd RAG-Architectures/Standard_RAG
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

> **Note:** On first run, Docling may download OCR models — this takes a minute or two.

**4. Configure your API key**

Create a `.env` file in the `Standard_RAG/` directory:
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

Parses, chunks, embeds, and indexes all documents into a persisted FAISS index.

---

### Query — Single Question

```bash
python main.py query "What are the main skills mentioned in the resume?"
```

Returns a grounded answer with source citations from your indexed documents.

---

### Query — Interactive Chat

```bash
python main.py chat
```

Starts a continuous Q&A session. Type `quit` or `exit` to end.

---

### View Index Statistics

```bash
python main.py stats
```

Displays the number of indexed documents and chunks in the current vector store.

---

## Configuration

All pipeline parameters live in `config.py` — no digging through framework internals.

| Parameter       | Default                 | Description                                               |
|-----------------|-------------------------|-----------------------------------------------------------|
| `CHUNK_SIZE`    | `1000`                  | Characters per chunk                                      |
| `CHUNK_OVERLAP` | `200`                   | Overlap between consecutive chunks                        |
| `TOP_K`         | `5`                     | Number of chunks retrieved per query                      |
| Embedding model | `gemini-embedding-001`  | Swap for any Gemini-compatible embedding model            |
| LLM             | `gemini-2.5-flash-lite` | Swap for `gemini-2.0-flash`, `gemini-2.0-pro-exp`, etc.  |

---

## Tech Stack

| Component         | Library / Model                              |
|-------------------|----------------------------------------------|
| PDF Parsing       | [Docling](https://github.com/DS4SD/docling)  |
| Embeddings        | `gemini-embedding-001` via `google-genai`    |
| Vector Search     | `faiss-cpu`                                  |
| LLM               | `gemini-2.5-flash-lite` via `google-genai`   |
| Config Management | `python-dotenv`                              |
| Language          | Python 3.9+                                  |

---

## Part of RAG-Architectures

This module is the baseline in a broader collection of RAG variants. Start here to understand the fundamentals before exploring advanced patterns.

```
RAG-Architectures/
├── Standard_RAG/        ◀ You are here
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
