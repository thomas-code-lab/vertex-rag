"""
app.py — Streamlit chat UI for the Vertex AI RAG Chatbot.

Run locally:
  streamlit run app.py

Deploy to HuggingFace Spaces:
  - Create a new Space (Streamlit type)
  - Push this repo via git
  - Add GROQ_API_KEY as a Space secret
"""

import streamlit as st
from rag_chain import load_retriever, build_chain, ask

import os
if not os.path.exists("./chroma_db"):
    import subprocess
    subprocess.run(["python", "ingest.py"], check=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vertex AI Docs Assistant",
    page_icon="🤖",
    layout="centered",
)

# ── Load RAG chain (cached so it only runs once per session) ──────────────────
@st.cache_resource(show_spinner="Loading Vertex AI knowledge base...")
def get_chain():
    retriever = load_retriever()
    llm, prompt, retriever = build_chain(retriever)
    return llm, prompt, retriever


# ── App header ────────────────────────────────────────────────────────────────
st.title("🤖 Vertex AI Docs Assistant")
st.caption(
    "Ask anything about Google Cloud Vertex AI. "
    "Answers are grounded in the official documentation."
)

# Show which docs are indexed in the sidebar
with st.sidebar:
    st.header("📚 Indexed Topics")
    st.markdown("""
    This assistant has knowledge of:
    - Vertex AI overview & concepts
    - Generative AI & Gemini models
    - Text embeddings
    - Grounding & RAG on Vertex
    - Model training
    - Online & batch prediction
    - Model Garden
    - Vertex AI Pipelines
    - Feature Store
    - Prompt engineering
    """)
    st.divider()
    st.caption("Built with LangChain · ChromaDB · Groq · Llama 3")
    st.caption("Deployed on HuggingFace Spaces")

    # Debug toggle — shows retrieved chunks
    show_chunks = st.toggle("Show retrieved chunks", value=False)

# ── Suggested questions ───────────────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "What is Vertex AI and what problems does it solve?",
    "How do I deploy a model to a Vertex AI endpoint?",
    "What's the difference between online and batch prediction?",
    "How does grounding work in Vertex AI generative models?",
    "What models are available in the Vertex AI Model Garden?",
]

# ── Session state for chat history ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain_loaded" not in st.session_state:
    st.session_state.chain_loaded = False

# ── Load chain ────────────────────────────────────────────────────────────────
try:
    llm, prompt, retriever = get_chain()
    st.session_state.chain_loaded = True
except FileNotFoundError:
    st.error(
        "⚠️ Knowledge base not found. Please run `python ingest.py` first.",
        icon="🚨"
    )
    st.stop()
except EnvironmentError as e:
    st.error(str(e), icon="🔑")
    st.stop()

# ── Show suggested questions if no chat yet ───────────────────────────────────
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(1)
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state["pending_question"] = q
            st.rerun()

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 Sources used"):
                for src in msg["sources"]:
                    st.markdown(f"- [{src}]({src})")

        # Show chunks if debug mode is on
        if show_chunks and msg["role"] == "assistant" and msg.get("chunks"):
            with st.expander(f"🔍 Retrieved chunks ({len(msg['chunks'])})"):
                for i, chunk in enumerate(msg["chunks"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                    st.divider()

        # Show latency
        if msg["role"] == "assistant" and msg.get("latency_ms"):
            st.caption(f"⚡ {msg['latency_ms']}ms")

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.session_state.pop("pending_question", None)
if not user_input:
    user_input = st.chat_input("Ask about Vertex AI...")
if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching docs and generating answer..."):
            result = ask(user_input, llm, prompt, retriever)

        st.markdown(result["answer"])

        # Sources
        if result["sources"]:
            with st.expander("📎 Sources used"):
                for src in result["sources"]:
                    st.markdown(f"- [{src}]({src})")

        # Debug chunks
        if show_chunks and result["chunks"]:
            with st.expander(f"🔍 Retrieved chunks ({len(result['chunks'])})"):
                for i, chunk in enumerate(result["chunks"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                    st.divider()

        st.caption(f"⚡ {result['latency_ms']}ms")

    # Store in history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "chunks": result["chunks"],
        "latency_ms": result["latency_ms"],
    })
