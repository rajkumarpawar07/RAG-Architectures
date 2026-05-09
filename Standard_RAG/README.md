# 🔍 Standard RAG — Retrieval-Augmented Generation
 
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-0.1%2B-green?style=for-the-badge&logo=chainlink&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/OpenAI-GPT--4-412991?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>
<p align="center">
  A clean, production-ready implementation of the <strong>Standard RAG (Retrieval-Augmented Generation)</strong> pipeline — part of the <a href="https://github.com/rajkumarpawar07/RAG-Architectures">RAG-Architectures</a> collection.
</p>
---
A professional, from-scratch implementation of a standard Retrieval-Augmented Generation (RAG) pipeline. This project is designed as an educational reference and a production-ready starting point, avoiding heavy frameworks to clearly demonstrate how core RAG concepts work under the hood.

## 🌟 Key Features

- **High-Quality PDF Extraction:** Uses [Docling](https://github.com/DS4SD/docling) to intelligently parse PDFs, preserving document structure (tables, headers) better than naive extractors.
- **Production-Grade Chunking:** Implements Recursive Character Text Splitting (the industry standard) to preserve semantic coherence across paragraphs and sentences.
- **State-of-the-Art Models:** Powered by the official `google-genai` SDK:
  - **Embeddings:** `gemini-embedding-001` (3072-dimensional vectors).
  - **LLM:** `gemini-3.1-flash-lite` for fast, cost-effective, and highly capable text generation.
- **Task-Specific Embeddings:** Uses Gemini's `RETRIEVAL_DOCUMENT` and `RETRIEVAL_QUERY` task types to optimize semantic matching by 5-10%.
- **Efficient Vector Search:** Utilizes `faiss-cpu` with exact inner product (cosine similarity via L2 normalization) for sub-millisecond retrieval.
- **Robust Error Handling:** Implements exponential backoff for API rate limits (e.g., HTTP 429).
- **Source Citations:** Prompt engineering strictly enforces that the LLM grounds its answers in the provided context and cites its sources.

---

## 🏗️ Architecture Overview

The system is divided into two primary flows:

### 1. Ingestion Pipeline
Documents (PDFs, TXTs) are loaded from the `data/` directory → Parsed & Extracted → Chunked recursively → Embedded in batches via Gemini → Indexed using FAISS → Persisted to disk along with metadata.

### 2. Query Pipeline
User question → Embedded via Gemini → FAISS vector search retrieves Top-K chunks → Prompt builder injects context → Gemini LLM generates a grounded answer with citations.

---

## 📂 Project Structure

```text
Standard_RAG/
├── data/                   # Drop your PDFs and .txt files here
├── faiss_index/            # Auto-created: persisted FAISS index + metadata
├── config.py               # Central configuration (models, chunking, paths)
├── document_loader.py      # Loads PDFs (via Docling) and text files
├── chunker.py              # Recursive character text splitting
├── embedder.py             # Gemini embeddings with batching & backoff
├── vector_store.py         # FAISS index operations (build, save, search)
├── generator.py            # Gemini LLM prompt engineering & generation
├── rag_pipeline.py         # Orchestrates ingestion and query flows
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables (API Key)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A Google Gemini API Key. Get yours at [Google AI Studio](https://aistudio.google.com/apikey).

### Installation

1. Clone or download this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your Google API key:
   ```env
   GOOGLE_API_KEY="your_api_key_here"
   ```

### 1. Ingest Documents

Place your `.pdf` and `.txt` files into the `data/` folder, then run:

```bash
python main.py ingest
```

*Note: On the first run, Docling may download OCR models which can take a minute or two.*

### 2. Query the Index

Ask a single question and get an answer with source citations:

```bash
python main.py query "What are the main skills mentioned in the resume?"
```

### 3. Interactive Chat

Start a continuous Q&A session:

```bash
python main.py chat
```
*(Type `quit` or `exit` to end the session)*

### 4. View Statistics

Check how many documents and chunks are currently in your vector store:

```bash
python main.py stats
```

---

## ⚙️ Configuration (`config.py`)

You can easily tune the pipeline by modifying `config.py`:

- **Chunking:** Adjust `CHUNK_SIZE` (default 1000) and `CHUNK_OVERLAP` (default 200).
- **Retrieval:** Change `TOP_K` (default 5) to retrieve more or fewer chunks per query.
- **Models:** Swap out the Gemini embedding model or LLM (e.g., change to `gemini-2.0-flash` or `gemini-2.0-pro-exp`).

---

## 💡 Why from scratch?

Many developers rely heavily on frameworks like LangChain or LlamaIndex. While powerful, they can obscure the underlying mechanics of RAG. This project provides a transparent, easy-to-debug, and highly customizable alternative using less than 5 dependencies. You have complete control over the chunking logic, prompts, and index structure.
