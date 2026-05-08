"""
rag_chain.py — The RAG pipeline.

What it does on every query:
  1. Embeds the user question into a vector
  2. Retrieves top-6 most relevant chunks from ChromaDB
  3. Builds a prompt with those chunks as context
  4. Calls Groq (Llama 3) to generate a grounded answer
  5. Returns the answer + source URLs
"""

import os
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 6  # number of chunks to retrieve

# ── System prompt — the most important guardrail ───────────────────────────────
# "Answer ONLY from context" forces the LLM to stay grounded.
# "Say I don't know" gives it a safe exit instead of hallucinating.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions about 
Google Cloud Vertex AI based ONLY on the provided documentation context.

Rules:
- Answer using ONLY the information in the context below.
- If the context does not contain enough information to answer, say:
  "I don't have enough information in the Vertex AI docs I've indexed to answer this. 
   Please check https://cloud.google.com/vertex-ai/docs directly."
- Always cite which source URLs you used at the end of your answer.
- Be concise and precise. Use bullet points where appropriate.

Context:
{context}
"""

HUMAN_PROMPT = "Question: {question}"


def load_retriever():
    """Load ChromaDB and return a retriever. Called once at app startup."""
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"ChromaDB not found at '{CHROMA_DIR}'. "
            "Please run: python ingest.py"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    # search_type="mmr" (Maximal Marginal Relevance) adds diversity —
    # it avoids returning 6 chunks that all say the same thing.
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 20},
    )

    return retriever


def build_chain(retriever):
    """Build and return the RAG chain. Called once at app startup."""
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. "
            "Get a free key at https://console.groq.com and set it:\n"
            "  export GROQ_API_KEY=your_key_here"
        )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,        # low temp = more factual, less creative
        max_tokens=1024,
        groq_api_key=groq_api_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    return llm, prompt, retriever


def ask(question: str, llm, prompt, retriever) -> dict:
    """
    Run the full RAG pipeline for a question.

    Returns:
        {
            "answer": str,
            "sources": list[str],   # unique source URLs used
            "chunks": list[str],    # raw retrieved chunks (for debugging)
            "latency_ms": int,      # end-to-end latency
        }
    """
    start = time.time()

    # Step 1: Retrieve relevant chunks
    docs = retriever.invoke(question)

    if not docs:
        return {
            "answer": "I couldn't find any relevant information in the indexed Vertex AI docs.",
            "sources": [],
            "chunks": [],
            "latency_ms": int((time.time() - start) * 1000),
        }

    # Step 2: Build context string from chunks
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Chunk {i} from {source}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)

    # Step 3: Call Groq LLM
    messages = prompt.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    answer = response.content

    # Step 4: Extract unique source URLs
    sources = list(dict.fromkeys(
        doc.metadata.get("source", "") for doc in docs
        if doc.metadata.get("source")
    ))

    latency_ms = int((time.time() - start) * 1000)

    return {
        "answer": answer,
        "sources": sources,
        "chunks": [doc.page_content for doc in docs],
        "latency_ms": latency_ms,
    }
