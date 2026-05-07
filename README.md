# Vertex AI Docs RAG Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about
Google Cloud Vertex AI — grounded in the official documentation.

Built as a personal project to demonstrate applied LLM engineering skills.

## Architecture

```
User question
     │
     ▼
Streamlit UI (app.py)
     │
     ▼
RAG Chain (rag_chain.py)
     ├── ChromaDB  ← top-6 relevant chunks retrieved by vector similarity
     └── Groq API  ← Llama 3 generates answer from retrieved context
```

**Ingestion pipeline (runs once):**
GCP Docs pages → text extraction → recursive chunking (500 tokens, 50 overlap)
→ sentence-transformers embeddings → ChromaDB vector store

**Query pipeline (every question):**
Question → embed → ChromaDB MMR retrieval → Groq Llama 3 → answer + source URLs

## Tech Stack

| Component | Tool |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector store | ChromaDB |
| LLM | Groq API (Llama 3 8B) |
| Deployment | HuggingFace Spaces |

**Cost: $0** — all components are free, no credit card required.

## Local Setup

**1. Clone and install**
```bash
git clone <your-repo>
cd vertex-rag
pip install -r requirements.txt
```

**2. Get a free Groq API key**

Sign up at https://console.groq.com — no credit card needed.

```bash
export GROQ_API_KEY=your_key_here
```

**3. Ingest the Vertex AI docs (run once)**
```bash
python ingest.py
```
This fetches ~15 Vertex AI documentation pages, chunks them, embeds them,
and saves everything to `./chroma_db/`. Takes ~2 minutes on first run
(downloads the embedding model). Subsequent runs are faster.

**4. Start the app**
```bash
streamlit run app.py
```

Open http://localhost:8501

## Deploy to HuggingFace Spaces

1. Create a new Space at https://huggingface.co/spaces
   - SDK: Streamlit
   - Visibility: Public

2. Add your secret:
   - Go to Settings → Variables and secrets
   - Add `GROQ_API_KEY` as a secret

3. Push the code:
```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push space main
```

**Note:** Run `ingest.py` locally first and commit the `chroma_db/` folder,
or add an ingestion step to the Space startup.

## Design Decisions

**Why RAG instead of fine-tuning?**
Vertex AI docs are updated frequently. RAG lets us re-ingest updated pages
in seconds. Fine-tuning would require retraining for every doc update.

**Why MMR retrieval?**
Maximal Marginal Relevance adds diversity to retrieved chunks — it avoids
returning 6 near-identical chunks and ensures broader coverage of the answer.

**Why sentence-transformers locally instead of an API embedder?**
Keeps the project fully free with no API calls for embeddings. The
all-MiniLM-L6-v2 model (~90MB) is fast on CPU and accurate enough for
technical documentation retrieval.

**Hallucination mitigation:**
- System prompt: "Answer ONLY from context, say I don't know otherwise"
- Source citations shown for every answer
- top-k=6 with MMR for diverse, relevant context
- Low LLM temperature (0.1) for factual responses

## Example Questions

- What is Vertex AI and what problems does it solve?
- How do I deploy a model to a Vertex AI endpoint?
- What's the difference between online and batch prediction?
- How does grounding work in Vertex AI generative models?
- What models are available in the Vertex AI Model Garden?
