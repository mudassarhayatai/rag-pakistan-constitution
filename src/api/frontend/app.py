"""
Section 10: Streamlit frontend.

Thin UI on top of the FastAPI backend (Section 8) -- calls /health and
/query over HTTP rather than reimplementing the pipeline, so the API
stays the single source of truth. A CLI, a Slack bot, or another
frontend could hit the same backend without duplicating any logic.

Run alongside the API (two separate processes):
    uvicorn src.api.main:app --port 8000          # terminal 1
    streamlit run src/frontend/app.py              # terminal 2
"""

import os
import yaml
import streamlit as st
from pathlib import Path
from api_client import APIError, ask_question, check_health  # same-directory import



CONFIG_PATH = Path("config.yml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

API_BASE_URL_DEFAULT = config.get("api_base_url_default", "http://localhost:8000")

st.set_page_config(page_title="Constitution of Pakistan — RAG", page_icon="📜", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.title("⚙️ Settings")
    api_url = st.text_input("API URL", value=API_BASE_URL_DEFAULT).rstrip("/")
    st.caption("Change this if the backend isn't running on localhost:8000")

    st.divider()
    try:
        health = check_health(api_url)
        status_icon = "🟢" if health.get("status") == "ok" else "🟡"
        st.markdown(f"{status_icon} **Status**: {health.get('status')}")
        st.markdown(f"**Dense search**: {'✅ enabled' if health.get('dense_search_enabled') else '❌ BM25-only'}")
        st.markdown(f"**Reranking**: {'✅ enabled' if health.get('rerank_enabled') else '❌ disabled'}")
    except APIError as e:
        st.error(f"Backend unreachable: {e}")

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()

st.title("📜 Constitution of Pakistan — Q&A")
st.caption(
    "Answers are grounded in the Constitution's text with Article citations. "
    "If the retrieved context doesn't cover your question, the system is instructed to say so rather than guess."
)

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        if entry["sources"] or entry["faithfulness"]:
            with st.expander(f"Sources & diagnostics ({entry['latency_ms']}ms)"):
                for source in entry["sources"]:
                    tag = "🔍 retrieved" if source["source"] == "retrieved" else f"🔗 {source['source']}"
                    score = f" · rerank score {source['rerank_score']:.3f}" if source.get("rerank_score") is not None else ""
                    st.markdown(f"- **Article {source['article_number']}** — {tag}{score}")

                faithfulness = entry["faithfulness"]
                if faithfulness.get("ungrounded_citations"):
                    st.warning(
                        "⚠️ This answer cites Article(s) not found in the retrieved context: "
                        + ", ".join(faithfulness["ungrounded_citations"])
                        + " — possible hallucination."
                    )
                elif faithfulness.get("has_citation"):
                    st.success("✅ All citations in this answer are grounded in the retrieved context.")
                elif faithfulness:
                    st.info("No Article citations found in this answer.")

question = st.chat_input("Ask a question about the Constitution of Pakistan...")
if question:
    with st.spinner("Searching the Constitution and generating an answer..."):
        try:
            result = ask_question(api_url, question)
            st.session_state.history.append({
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
                "faithfulness": result["faithfulness"],
                "latency_ms": result["latency_ms"],
            })
        except APIError as e:
            st.session_state.history.append({
                "question": question,
                "answer": f"⚠️ {e}",
                "sources": [],
                "faithfulness": {},
                "latency_ms": 0,
            })
    st.rerun()