"""
Section 6: Reranking.

Hybrid retrieval (Section 5) casts a reasonably wide net but its ranking
is still approximate -- BM25 and embedding similarity are both cheap and
fast compared to a cross-encoder that actually reads the query and each
candidate together. Reranking trades speed for precision on a small
candidate set (re-scoring the top 10-20 hybrid hits, never the whole
corpus -- a cross-encoder is too slow to run over everything).

# Pipeline:

#     BM25 + Dense Retrieval
#              ↓
#             RRF
#              ↓
#        Candidate set
#              ↓
#        Cross-Encoder
#              ↓
#          Final ranking
#              

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (open weights) via sentence-transformers'
CrossEncoder.

Usage:
python -m src.retrieval.reranker
"""

import json
from typing import Callable, Optional


def rerank_scored(
    query: str,
    candidates: list[dict],
    score_fn: Callable[[str, str], float],
    top_k: Optional[int] = None,
) -> list[dict]:
    """Score every candidate against the query with `score_fn` and return
    them sorted by score descending, each annotated with `rerank_score`."""

    scored = [
        {
            **c,
            "rerank_score": score_fn(query, c["text"])
        }
        for c in candidates
    ]

    scored.sort(
        key=lambda c: c["rerank_score"],
        reverse=True
    )

    return scored[:top_k] if top_k else scored


class CrossEncoderReranker:
    """Wrapper around sentence-transformers CrossEncoder."""

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: Optional[str] = None):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            model_name or self.MODEL_NAME
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:

        pairs = [
            (query, c["text"])
            for c in candidates
        ]

        scores = self.model.predict(pairs)

        scored = [
            {
                **c,
                "rerank_score": float(s)
            }
            for c, s in zip(candidates, scores)
        ]

        scored.sort(
            key=lambda c: c["rerank_score"],
            reverse=True
        )

        return scored[:top_k] if top_k else scored


if __name__ == "__main__":

    from src.retrieval.hybrid_search import (
        HybridRetriever,
        EMBEDDING_MODEL_NAME
    )

    # ==========================================================
    # Configuration
    # ==========================================================

    CHUNKS_PATH = r"data\processed\child_chunks.json"
    PARENTS_PATH = r"data\processed\parent_chunks.json"

    QUERY = "What does Article 25 say about equality of citizens?"

    # Number of results retrieved from hybrid search
    CANDIDATE_K = 10

    # Number of results kept after reranking
    TOP_K = 5

    # Qdrant collection
    COLLECTION = "constitution_pk"

    # Local Qdrant database path
    QDRANT_PATH = "qdrant_local"

    # Set True to completely skip Qdrant and embeddings
    BM25_ONLY = False

    # ==========================================================
    # LOAD CHUNKS
    # ==========================================================

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        child_chunks = json.load(f)

    with open(PARENTS_PATH, encoding="utf-8") as f:
        parent_chunks = json.load(f)

    # ==========================================================
    # INITIALIZE DENSE SEARCH
    # ==========================================================

    qdrant_client = None
    embedding_model = None

    if not BM25_ONLY:

        try:
            from qdrant_client import QdrantClient
            from sentence_transformers import SentenceTransformer

            qdrant_client = QdrantClient(
                path=QDRANT_PATH
            )

            embedding_model = SentenceTransformer(
                EMBEDDING_MODEL_NAME
            )

            print(
                "Dense search enabled for retrieval stage "
                "(Qdrant + embedding model loaded)"
            )

        except ImportError as e:

            print(
                f"qdrant-client/sentence-transformers "
                f"not available ({e}) -- "
                "retrieval falling back to BM25-only"
            )

    # ==========================================================
    # HYBRID RETRIEVAL
    # ==========================================================

    retriever = HybridRetriever(
        child_chunks,
        parent_chunks,
        qdrant_client=qdrant_client,
        collection=COLLECTION,
        embedding_model=embedding_model,
    )

    candidates = retriever.search(
        QUERY,
        top_k=CANDIDATE_K
    )

    print(
        f"\nRetrieved {len(candidates)} candidates "
        "from hybrid search."
    )

    # ==========================================================
    # CROSS-ENCODER RERANKING
    # ==========================================================

    try:

        reranker = CrossEncoderReranker()

        results = reranker.rerank(
            QUERY,
            candidates,
            top_k=TOP_K
        )

        print(
            f"Reranked with "
            f"{CrossEncoderReranker.MODEL_NAME}"
        )

    except ImportError as e:

        print(
            f"sentence-transformers not available ({e}) "
            "-- skipping rerank, "
            "showing retrieval order"
        )

        results = candidates[:TOP_K]

    # ==========================================================
    # DISPLAY RESULTS
    # ==========================================================

    print(f"\nResults for: {QUERY!r}")

    for r in results:

        score_str = (
            f" score={r['rerank_score']:.4f}"
            if "rerank_score" in r
            else ""
        )

        print(
            f"  [{r['source']}]{score_str} "
            f"Article {r['article_number']}: "
            f"{r['text'][:100]}..."
        )
        
    if qdrant_client is not None:
        qdrant_client.close()