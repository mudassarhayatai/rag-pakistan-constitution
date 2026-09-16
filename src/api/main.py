"""
Section 8: FastAPI serving layer.

Wraps the retrieval -> rerank -> generation pipeline (Sections 5-7) behind
an HTTP API. The one production-relevant thing this adds beyond the CLI
scripts: models and clients are loaded ONCE at startup (lifespan) and
reused across every request, not reloaded per-call. Reloading a ~2GB
embedding model per request would make this unusable in practice, and a
demo that doesn't do this isn't actually demonstrating serving competence.

Usage (once Section 4's Qdrant collection is populated and Ollama is running):
    uvicorn src.api.main:app --reload --port 8000
    visit http://localhost:8000/docs 

    curl -X POST http://localhost:8000/query \
        -H "Content-Type: application/json" \
        -d '{"question": "What does Article 9A say?"}'

"""

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import yaml


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root -> makes `src` importable
 
from src.retrieval.hybrid_search import HybridRetriever, EMBEDDING_MODEL_NAME
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.generate import build_prompt, OllamaGenerator, check_citation_faithfulness


# ---------------------------------------------------------------------------
# Config -- environment-variable driven, 
# ---------------------------------------------------------------------------

CONFIG_PATH = Path("config.yml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

CHUNKS_PATH = config.get("chunks_path", "child_chunks.json")
PARENTS_PATH = config.get("parents_path", "parent_chunks.json")
QDRANT_PATH = config.get("qdrant_path", "qdrant_local")
QDRANT_URL = config.get("qdrant_url")
COLLECTION = config.get("collection", "constitution_pk")
OLLAMA_HOST = config.get("ollama_host", "http://localhost:11434")
OLLAMA_MODEL = config.get("ollama_model", "qwen3.5:0.8b")
DEFAULT_CANDIDATE_K = int(config.get("candidate_k", 10))
DEFAULT_TOP_K = int(config.get("top_k", 5))




# CHUNKS_PATH = os.environ.get("RAG_CHUNKS_PATH", "child_chunks.json")
# PARENTS_PATH = os.environ.get("RAG_PARENTS_PATH", "parent_chunks.json")
# QDRANT_PATH = os.environ.get("RAG_QDRANT_PATH", "qdrant_local")
# QDRANT_URL = os.environ.get("RAG_QDRANT_URL")  # if set, overrides QDRANT_PATH (local mode)
# COLLECTION = os.environ.get("RAG_COLLECTION", "constitution_pk")
# OLLAMA_HOST = os.environ.get("RAG_OLLAMA_HOST", "http://localhost:11434")
# OLLAMA_MODEL = os.environ.get("RAG_OLLAMA_MODEL", "qwen3.5:0.8b")
# DEFAULT_CANDIDATE_K = int(os.environ.get("RAG_CANDIDATE_K", "10"))
# DEFAULT_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))


# ---------------------------------------------------------------------------
# App state -- populated once in lifespan(), reused across every request
# ---------------------------------------------------------------------------

class AppState:
    retriever: Optional[HybridRetriever] = None
    reranker: Optional[CrossEncoderReranker] = None
    generator: Optional[OllamaGenerator] = None
    parents_by_id: dict = {}
    dense_search_enabled: bool = False
    rerank_enabled: bool = False


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        child_chunks = json.load(f)
    with open(PARENTS_PATH, encoding="utf-8") as f:
        parent_chunks = json.load(f)
    state.parents_by_id = {p["parent_id"]: p for p in parent_chunks}

    qdrant_client = None
    embedding_model = None
    try:
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer

        qdrant_client = QdrantClient(url=QDRANT_URL) if QDRANT_URL else QdrantClient(path=QDRANT_PATH)
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        state.dense_search_enabled = True
    except ImportError as e:
        print(f"[startup] Dense search unavailable ({e}) -- serving BM25-only")

    state.retriever = HybridRetriever(
        child_chunks, parent_chunks,
        qdrant_client=qdrant_client, collection=COLLECTION, embedding_model=embedding_model,
    )

    try:
        state.reranker = CrossEncoderReranker()
        state.rerank_enabled = True
    except ImportError as e:
        print(f"[startup] Reranker unavailable ({e}) -- skipping rerank stage")

    state.generator = OllamaGenerator(model=OLLAMA_MODEL, host=OLLAMA_HOST)

    print(f"[startup] Ready. dense_search={state.dense_search_enabled} rerank={state.rerank_enabled}")
    yield
    # No explicit teardown: Qdrant local mode and in-process models are
    # cleaned up on process exit; there's no open connection to close for
    # the Ollama client (plain HTTP request per call, nothing persistent).


app = FastAPI(title="Constitution of Pakistan RAG API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    candidate_k: Optional[int] = None
    top_k: Optional[int] = None


class SourceArticle(BaseModel):
    article_number: str
    source: str
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceArticle]
    faithfulness: dict
    dense_search_used: bool
    rerank_used: bool
    latency_ms: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Check this before debugging anything else -- if startup silently
    left retriever unset, every /query call will 503 with a much more
    confusing trail to follow."""
    return {
        "status": "ok" if state.retriever is not None else "not_ready",
        "dense_search_enabled": state.dense_search_enabled,
        "rerank_enabled": state.rerank_enabled,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if state.retriever is None:
        raise HTTPException(status_code=503, detail="Service not ready -- check /health")

    start = time.monotonic()
    candidate_k = request.candidate_k or DEFAULT_CANDIDATE_K
    top_k = request.top_k or DEFAULT_TOP_K

    candidates = state.retriever.search(request.question, top_k=candidate_k)
    if not candidates:
        raise HTTPException(status_code=404, detail="Please ask a question related to the Constitution of Pakistan.")

    if state.rerank_enabled:
        results = state.reranker.rerank(request.question, candidates, top_k=top_k)
    else:
        results = candidates[:top_k]

    system_prompt, user_prompt = build_prompt(request.question, results, state.parents_by_id)

    try:
        answer = state.generator.generate(system_prompt, user_prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Generation backend unavailable: {e}")

    faithfulness = check_citation_faithfulness(answer, results)
    latency_ms = int((time.monotonic() - start) * 1000)

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=[
            SourceArticle(article_number=r["article_number"], source=r["source"], rerank_score=r.get("rerank_score"))
            for r in results
        ],
        faithfulness=faithfulness,
        dense_search_used=state.dense_search_enabled,
        rerank_used=state.rerank_enabled,
        latency_ms=latency_ms,
    )



