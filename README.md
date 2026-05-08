# 🤖 Vertex AI Docs Assistant

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about Google Cloud Vertex AI, grounded in the official documentation. Built as a personal project to demonstrate applied LLM engineering.

🚀 **Live Demo:** https://vertex-rag.streamlit.app  
📁 **GitHub:** https://github.com/thomas-code-lab/vertex-rag

---

## What it does

Ask natural language questions about Vertex AI and get accurate, sourced answers — pulled directly from the official GCP documentation. Every answer includes citations to the exact doc pages used, so you can verify the response yourself.

Example questions:
- *"What is Vertex AI and what problems does it solve?"*
- *"What's the difference between online and batch prediction?"*
- *"How does grounding work in Vertex AI generative models?"*
- *"What models are available in the Vertex AI Model Garden?"*

---

## Architecture

```
Phase 1 — Ingestion (runs once at startup)
GCP Docs pages → BeautifulSoup (clean HTML) → RecursiveCharacterTextSplitter
→ sentence-transformers (embed) → ChromaDB (store vectors on disk)

Phase 2 — Query (every question)
User question → sentence-transformers (embed) → ChromaDB MMR retrieval (top-6 chunks)
→ Groq / Llama 3 (generate answer) → Streamlit (display answer + source URLs)
```

### Key design decisions

**Why RAG instead of fine-tuning?**  
Vertex AI docs are updated frequently. RAG lets us re-ingest updated pages in seconds. Fine-tuning would require retraining for every doc update and still wouldn't guarantee factual accuracy.

**Why MMR retrieval?**  
Maximal Marginal Relevance adds diversity to retrieved chunks — it avoids returning 6 near-identical chunks and ensures broader coverage of the answer space.

**Why sentence-transformers locally instead of an API embedder?**  
Keeps the project fully free with no API calls for embeddings. The `all-MiniLM-L6-v2` model (~90MB) is fast on CPU and accurate enough for technical documentation retrieval.

**Hallucination mitigation:**  
- System prompt: *"Answer ONLY from context, say I don't know otherwise"*
- Source citations shown for every answer so users can verify
- top-k=6 with MMR for diverse, relevant context
- Low LLM temperature (0.1) for factual, deterministic responses

---

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| UI | Streamlit | Fast to build, easy to deploy |
| Orchestration | LangChain | Chains retrieval + generation |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free, runs locally, no API needed |
| Vector store | ChromaDB | Simple, file-based, no server needed |
| LLM | Groq API (Llama 3) | Free tier, fast inference |
| Deployment | Streamlit Community Cloud | Free, native Streamlit support |

**Total cost: $0** — fully free stack.

---

## Local Setup

**1. Clone and install**
```bash
git clone https://github.com/thomas-code-lab/vertex-rag
cd vertex-rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Get a free Groq API key**

Sign up at https://console.groq.com — no credit card needed.

```bash
export GROQ_API_KEY=your_key_here
```

**3. Ingest the Vertex AI docs (runs automatically on first start)**
```bash
python ingest.py
```

Fetches ~14 Vertex AI documentation pages, chunks them into 391 pieces, embeds with sentence-transformers, and saves to ChromaDB. Takes ~2 minutes on first run.

**4. Start the app**
```bash
streamlit run app.py
```

Open http://localhost:8501

---

## Indexed Vertex AI Topics

- Vertex AI overview & platform concepts
- Generative AI & Gemini models
- Text embeddings
- Grounding & RAG on Vertex AI
- Model training
- Online & batch prediction
- Model Garden
- Vertex AI Pipelines
- Feature Store
- Prompt engineering

---

## RAG Pipeline Details

### Chunking strategy
Documents are split using `RecursiveCharacterTextSplitter` with:
- Chunk size: 500 tokens
- Overlap: 50 tokens (prevents information loss at boundaries)
- Split order: paragraphs → sentences → words (preserves semantic units)

### Retrieval
- Embedding model: `all-MiniLM-L6-v2` (384-dimensional vectors)
- Search type: MMR (Maximal Marginal Relevance)
- top-k: 6 chunks retrieved per query
- fetch-k: 20 candidates before MMR re-ranking

### Generation
- Model: Llama 3 via Groq API
- Temperature: 0.1 (factual, low randomness)
- System prompt enforces context-only answers with "I don't know" fallback
