"""
ingest.py — Run this ONCE before starting the app.

What it does:
  1. Fetches Vertex AI documentation pages from GCP docs
  2. Splits text into overlapping chunks (~500 tokens)
  3. Embeds each chunk using sentence-transformers (free, local)
  4. Stores everything in ChromaDB on disk

Usage:
  python ingest.py
"""

import os
import time
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Vertex AI doc URLs to ingest ──────────────────────────────────────────────
# Focused on the most commonly asked-about Vertex AI topics.
# Add more URLs here anytime — just re-run ingest.py.
URLS = [
    # Overview
    "https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform",
    # Generative AI
    "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/overview",
    "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models",
    "https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/send-chat-prompts-gemini",
    "https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
    # RAG / Grounding
    "https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview",
    # Model deployment / prediction
    "https://cloud.google.com/vertex-ai/docs/predictions/overview",
    "https://cloud.google.com/vertex-ai/docs/predictions/get-online-predictions",
    "https://cloud.google.com/vertex-ai/docs/predictions/get-batch-predictions",
    # Training
    "https://cloud.google.com/vertex-ai/docs/training/overview",
    # Model Garden
    "https://cloud.google.com/vertex-ai/docs/start/explore-models",
    # Pipelines
    "https://cloud.google.com/vertex-ai/docs/pipelines/introduction",
    # Feature Store
    "https://cloud.google.com/vertex-ai/docs/featurestore/latest/overview",
    # Prompt engineering
    "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/introduction-prompt-design",
]

CHROMA_DIR = "./chroma_db"   # where ChromaDB saves its files
EMBED_MODEL = "all-MiniLM-L6-v2"  # small, fast, free — runs locally


def fetch_page(url: str) -> str:
    """Fetch a GCP doc page and return clean text (no nav/footer)."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (RAG-project/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # GCP docs put the main content in <article> or devsite-article
        main = (
            soup.find("article")
            or soup.find("devsite-article")
            or soup.find("main")
            or soup.find("div", class_="devsite-article-body")
        )

        if main:
            # Remove nav, code tabs, feedback widgets
            for tag in main.find_all(["nav", "aside", "footer", "script", "style"]):
                tag.decompose()
            text = main.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # Collapse excessive blank lines
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines)

    except Exception as e:
        print(f"  ⚠️  Could not fetch {url}: {e}")
        return ""


def main():
    print("🚀 Starting Vertex AI docs ingestion...\n")

    # ── Step 1: Fetch all pages ───────────────────────────────────────────────
    documents = []
    metadatas = []

    for url in URLS:
        print(f"📄 Fetching: {url}")
        text = fetch_page(url)
        if text:
            documents.append(text)
            metadatas.append({"source": url})
            print(f"   ✅ {len(text):,} chars")
        else:
            print(f"   ❌ Skipped (empty or error)")
        time.sleep(0.5)  # be polite to GCP servers

    print(f"\n✅ Fetched {len(documents)} pages\n")

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    # RecursiveCharacterTextSplitter splits on paragraphs first, then sentences,
    # then words — preserving semantic units better than blind character splits.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # ~500 tokens per chunk
        chunk_overlap=50,     # 50-token overlap so nothing is lost at boundaries
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    all_metas = []

    for doc_text, meta in zip(documents, metadatas):
        chunks = splitter.split_text(doc_text)
        all_chunks.extend(chunks)
        all_metas.extend([meta] * len(chunks))

    print(f"✂️  Created {len(all_chunks)} chunks from {len(documents)} pages")
    print(f"   Average chunk size: {sum(len(c) for c in all_chunks) // len(all_chunks)} chars\n")

    # ── Step 3: Embed + store in ChromaDB ─────────────────────────────────────
    print(f"🔢 Loading embedding model: {EMBED_MODEL}")
    print("   (downloads ~90MB on first run, cached after that)\n")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("💾 Storing embeddings in ChromaDB...")

    # Chroma.from_texts embeds all chunks and persists to disk in one call
    vectorstore = Chroma.from_texts(
        texts=all_chunks,
        embedding=embeddings,
        metadatas=all_metas,
        persist_directory=CHROMA_DIR,
    )

    vectorstore.persist()

    print(f"\n🎉 Done! {len(all_chunks)} chunks stored in '{CHROMA_DIR}/'")
    print("   You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
