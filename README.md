<h1 align="center">RAG Architectures</h1>

<p>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Google%20Gemini-SDK-4285F4?style=flat-square&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Architectures-4%20built%20%7C%204%20coming-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/Framework-From%20Scratch-FF6F00?style=flat-square" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" />
</p>

<p align="center">
  A collection of advanced <strong>Retrieval-Augmented Generation (RAG)</strong> pipelines built from scratch.<br/>
  Designed to be understood, not abstracted away behind heavy frameworks.
</p>

---

## 🌟 Why This Repository Exists

Most RAG tutorials reach for heavy frameworks like LangChain or LlamaIndex. While powerful, those frameworks hide the mechanics that matter most — chunking logic, embedding strategies, evaluation gates, and state machines. 

This repository builds the evolution of RAG systems **from scratch**, giving you complete visibility and control over every component. By reading this code, you will actually understand how state-of-the-art RAG pipelines work under the hood.

---

## 📚 Architectures

### 1. [Standard RAG](./Standard_RAG)
A transparent implementation of the baseline RAG pipeline.
*   **The Problem:** LLMs hallucinate because they lack private or up-to-date knowledge.
*   **The Solution:** Retrieve relevant documents via vector search and inject them into the LLM context.
*   **Tech Stack:** `FAISS`, `Docling` (for high-fidelity PDF parsing), `Google GenAI SDK`, `Langfuse` (Tracing), `Ragas` (Evaluation).

### 2. [Conversational RAG](./Conversational_RAG)
A stateful RAG pipeline that remembers conversation history.
*   **The Problem:** Standard RAG treats every query in isolation ("context blindness" for follow-up questions).
*   **The Solution:** Introduces **Stateful Memory** to store chat turns and **LLM Query Rewriting** to resolve contextual pronouns into standalone search queries.
*   **Tech Stack:** `Qdrant` (via Docker), `SQLite` (Memory), `PyMuPDF`.

### 3. [Corrective RAG (CRAG)](./Corrective_RAG)
A highly resilient pipeline with retrieval evaluation and web fallback.
*   **The Problem:** Standard RAG blindly trusts the vector DB, leading to hallucinations if the retrieved context is irrelevant.
*   **The Solution:** Intercepts retrieval with an **LLM Decision Gate** (Correct, Ambiguous, Incorrect). Filters irrelevant sentences and falls back to **Live Web Search** when internal knowledge fails.
*   **Tech Stack:** `PostgreSQL + pgvector`, `Tavily Web Search API`, `asyncio` (Parallel Grading).

### 4. [Self-Reflective RAG (Self-RAG)](./Self_RAG)
The ultimate self-correcting RAG state machine that critiques its own reasoning.
*   **The Problem:** Even with good retrieval, models can misinterpret context, hallucinate claims, or fail to directly answer the user's prompt.
*   **The Solution:** A pure Python loop with **5 Reflection Gates** (`IsRet`, `IsRel`, `IsSup`, `Revise`, `IsUse`). If it detects hallucinations, it rewrites its own draft. If the final answer is useless, it generates a hypothetical ideal document (`HyDE`) and retrieves again.
*   **Tech Stack:** `ChromaDB`, `Pydantic Structured Outputs` (Native enforcement), Dual Gemini Models (`Flash Lite` & `Pro`), `Typer` & `Rich` (Beautiful CLI inner monologue).

### 5. [Agentic RAG](./Agentic_RAG)
An autonomous research agent powered by a LangGraph ReAct loop.
*   **The Problem:** Complex queries cannot be solved with a single vector database lookup. They require multi-step reasoning, evidence synthesis, and real-time external data.
*   **The Solution:** An autonomous agent that dynamically plans, routes searches between internal private documents (Qdrant) and the public web (Tavily), and validates its own evidence before generating an answer.
*   **Tech Stack:** `LangGraph`, `Qdrant`, `Tavily Web Search API`, `LangSmith` (Observability), `Docling`.

---

## 🚀 Getting Started

Each architecture is completely self-contained with its own `README.md`, `requirements.txt`, and dependencies. 

To explore a specific architecture, simply navigate to its folder:

```bash
cd Standard_RAG
# Follow the README instructions in that folder
```

All projects currently utilize the **Google Gemini API** (`gemini-1.5-pro` and `gemini-3.1-flash-lite`) via the official `google-genai` SDK for low latency and high quality reasoning.

---

## Capability Matrix

| Capability                    | Standard | Conversational | Corrective | Self-RAG |
|-------------------------------|:--------:|:--------------:|:----------:|:--------:|
| Document Q&A                  | ✓        | ✓              | ✓          | ✓        |
| Multi-turn memory             | ✗        | ✓              | ✓          | ✗        |
| Query rewriting               | ✗        | ✓              | ✓          | ✓ (HyDE) |
| Retrieval quality grading     | ✗        | ✗              | ✓          | ✓        |
| Web search fallback           | ✗        | ✗              | ✓          | ✗        |
| Retrieval skip (trivial Q)    | ✗        | ✗              | ✗          | ✓        |
| Hallucination detection       | ✗        | ✗              | ✗          | ✓        |
| Self-correction loop          | ✗        | ✗              | ✗          | ✓        |
| HyDE re-retrieval             | ✗        | ✗              | ✗          | ✓        |
| Dual LLM (speed + accuracy)   | ✗        | ✗              | ✗          | ✓        |
| Observability logging         | ✗        | ✗              | ✓          | ✗        |
| Batch evaluation harness      | ✗        | ✗              | ✗          | ✓        |
| Vector backend                | FAISS    | Qdrant         | pgvector   | ChromaDB |

---

## 🛠️ Roadmap / Upcoming Architectures

*   [ ] **Adaptive RAG**: Dynamically routes queries to different RAG strategies (e.g., QA vs Summarization) based on intent.
*   [ ] **Fusion RAG**: Reciprocal Rank Fusion (RRF) for blending results from multiple retrieval strategies.
*   [ ] **Graph RAG**: Using Knowledge Graphs to answer global, multi-hop queries over entire document corpuses.

---

## 🤝 Contributing

Contributions, issues, and suggestions are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: describe your change"`
4. Push and open a Pull Request

---

## 📜 License

MIT License — see the [LICENSE](LICENSE) file for details.

<p align="center">Built by <a href="https://github.com/rajkumarpawar07">Rajkumar Pawar</a></p>
