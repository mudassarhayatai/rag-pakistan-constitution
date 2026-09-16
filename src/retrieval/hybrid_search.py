"""
Section 5: Hybrid retrieval (BM25 + dense vector search, fused) with
cross-reference expansion.

Pure vector search misses exact-term queries (article numbers, specific
legal terms) that don't need to be *semantically* similar to match -- they
need to literally contain the term. BM25 catches those. Fusing both
rankings via Reciprocal Rank Fusion gets the benefit of each without
having to pick one over the other.

Cross-reference expansion pulls in Articles that a top-ranked result
explicitly references (Section 2's data) -- guaranteed complete legal
context instead of hoping semantic similarity surfaces the referenced
Article too.


Usage:
python -m src.retrieval.hybrid_search

"""

import json
import math
import re
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

# Hard-coded paths
CHUNKS_PATH = "data/processed/child_chunks.json"
PARENTS_PATH = "data/processed/parent_chunks.json"
QDRANT_PATH = "qdrant_local"

# Query
QUERY = "	Who must give consent before a Money Bill can be introduced in the Provincial Assembly?"

# Retrieval settings
TOP_K = 5
COLLECTION = "constitution_pk"

# False = BM25 + Dense + RRF
# True  = BM25 only
BM25_ONLY = False

# Embedding model MUST match the model used during indexing
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Dense relevance threshold
DEFAULT_DENSE_SCORE_THRESHOLD = 0.6

# BM25 THRESHOLD
BM25_SCORE_THRESHOLD = 9.0


# ============================================================
# BM25
# ============================================================

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset("""
a an and are as at be by for from has have he in is it its of on that the
to was were will with what who whom when where why how do does did this
these those i you we they them his her their but or if not no so
""".split())


def tokenize(text: str) -> list[str]:
    return [
        t
        for t in _TOKEN_RE.findall(text.lower())
        if t not in _STOPWORDS
    ]


class BM25:

    def __init__(
        self,
        doc_ids: list[str],
        documents: list[str],
        k1: float = 1.5,
        b: float = 0.75
    ):

        self.k1 = k1
        self.b = b
        self.doc_ids = doc_ids

        self.doc_tokens = [
            tokenize(doc)
            for doc in documents
        ]

        self.doc_lengths = [
            len(toks)
            for toks in self.doc_tokens
        ]

        self.avgdl = (
            sum(self.doc_lengths) / len(self.doc_lengths)
            if self.doc_lengths
            else 0.0
        )

        self.term_freqs = [
            Counter(toks)
            for toks in self.doc_tokens
        ]

        # Document frequency
        df = Counter()

        for toks in self.doc_tokens:
            for term in set(toks):
                df[term] += 1

        n_docs = max(len(documents), 1)

        self.idf = {
            term: math.log(
                (n_docs - freq + 0.5)
                / (freq + 0.5)
                + 1
            )
            for term, freq in df.items()
        }

    def score(
        self,
        query_tokens: list[str],
        doc_index: int
    ) -> float:

        score = 0.0

        doc_len = self.doc_lengths[doc_index]
        freqs = self.term_freqs[doc_index]

        for term in query_tokens:

            if term not in freqs:
                continue

            idf = self.idf.get(term, 0.0)
            f = freqs[term]

            denom = (
                f
                + self.k1
                * (
                    1
                    - self.b
                    + self.b
                    * doc_len
                    / (self.avgdl or 1)
                )
            )

            score += (
                idf
                * (f * (self.k1 + 1))
                / denom
            )

        return score

    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: float = 5.0
    ) -> list[tuple[str, float]]:

        print()
        print("=" * 70)
        print("BM25 SEARCH")
        print("=" * 70)

        print(f"Query: {query}")

        query_tokens = tokenize(query)

        print(f"Query tokens: {query_tokens}")
        print(f"Documents searched: {len(self.doc_ids)}")
        print(f"Requested top-k: {top_k}")
        print(f"BM25 score threshold: {score_threshold}")

        scored = [
            (
                self.doc_ids[i],
                self.score(query_tokens, i)
            )
            for i in range(len(self.doc_ids))
        ]

        # --------------------------------------------------------
        # Remove documents below BM25 threshold
        # --------------------------------------------------------

        scored = [
            s
            for s in scored
            if s[1] >= score_threshold
        ]

        scored.sort(
            key=lambda x: x[1],
            reverse=True
        )

        results = scored[:top_k]

        print()
        print(
            f"BM25 documents above threshold: "
            f"{len(scored)}"
        )

        print(
            f"BM25 returned: "
            f"{len(results)}"
        )

        if not results:

            print(
                f"No BM25 results with score >= "
                f"{score_threshold}"
            )

        for rank, (doc_id, score) in enumerate(
            results,
            start=1
        ):

            print(
                f"  {rank}. "
                f"{doc_id} "
                f"| BM25 score = {score:.4f}"
            )

        return results


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60
) -> list[tuple[str, float]]:

    print()
    print("=" * 70)
    print("RECIPROCAL RANK FUSION")
    print("=" * 70)

    print(f"RRF constant k: {k}")
    print(f"Number of ranking lists: {len(ranked_lists)}")

    scores: dict[str, float] = {}

    for list_number, ranked in enumerate(
        ranked_lists,
        start=1
    ):

        print()
        print(f"Ranking list {list_number}")
        print("-" * 50)

        for rank, doc_id in enumerate(
            ranked,
            start=1
        ):

            contribution = 1.0 / (k + rank)

            scores[doc_id] = (
                scores.get(doc_id, 0.0)
                + contribution
            )

            print(
                f"  Rank {rank}: "
                f"{doc_id} "
                f"| contribution = {contribution:.6f}"
            )

    fused = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print()
    print("FINAL RRF RANKING")
    print("-" * 50)

    for rank, (doc_id, score) in enumerate(
        fused,
        start=1
    ):

        print(
            f"  {rank}. "
            f"{doc_id} "
            f"| RRF score = {score:.6f}"
        )

    return fused


# ============================================================
# CROSS-REFERENCE EXPANSION
# ============================================================

def expand_with_references(
    top_chunk_ids: list[str],
    chunks_by_id: dict[str, dict],
    parents_by_id: dict[str, dict],
    max_extra: int = 3,
) -> list[dict]:

    print()
    print("=" * 70)
    print("CROSS-REFERENCE EXPANSION")
    print("=" * 70)

    print(f"Input chunks: {len(top_chunk_ids)}")
    print(f"Maximum extra reference articles: {max_extra}")

    results = []
    seen_articles = set()

    # --------------------------------------------------------
    # Add directly retrieved chunks
    # --------------------------------------------------------

    print()
    print("DIRECTLY RETRIEVED ARTICLES")
    print("-" * 50)

    for chunk_id in top_chunk_ids:

        chunk = chunks_by_id.get(chunk_id)

        if not chunk:
            print(
                f"  WARNING: chunk not found: {chunk_id}"
            )
            continue

        article_num = chunk["article_number"]

        if article_num in seen_articles:
            print(
                f"  Skipping duplicate Article {article_num}"
            )
            continue

        seen_articles.add(article_num)

        results.append(
            {
                "chunk_id": chunk_id,
                "article_number": article_num,
                "text": chunk["raw_text"],
                "source": "retrieved",
            }
        )

        print(
            f"  Added Article {article_num} "
            f"| chunk={chunk_id}"
        )

    # --------------------------------------------------------
    # Add referenced articles
    # --------------------------------------------------------

    extras_added = 0

    print()
    print("REFERENCED ARTICLES")
    print("-" * 50)

    for chunk_id in top_chunk_ids:

        if extras_added >= max_extra:
            print(
                f"  Reached max_extra limit: {max_extra}"
            )
            break

        chunk = chunks_by_id.get(chunk_id)

        if not chunk:
            continue

        article_num = chunk["article_number"]

        parent = parents_by_id.get(
            chunk["parent_id"],
            {}
        )

        references = parent.get(
            "references",
            []
        )

        print(
            f"  Article {article_num} references: "
            f"{references}"
        )

        for ref_article in references:

            if extras_added >= max_extra:
                break

            if ref_article in seen_articles:
                print(
                    f"    Article {ref_article} "
                    f"already present - skipping"
                )
                continue

            ref_parent = parents_by_id.get(
                f"article_{ref_article}"
            )

            if not ref_parent:
                print(
                    f"    Article {ref_article} "
                    f"parent not found"
                )
                continue

            seen_articles.add(ref_article)

            results.append(
                {
                    "chunk_id": None,
                    "article_number": ref_article,
                    "text": ref_parent.get("text"),
                    "source": (
                        f"cross_reference "
                        f"(via Article {article_num})"
                    ),
                }
            )

            extras_added += 1

            print(
                f"    + Added referenced Article "
                f"{ref_article}"
            )

    print()
    print(
        f"Direct articles: "
        f"{len(top_chunk_ids)}"
    )

    print(
        f"Extra reference articles: "
        f"{extras_added}"
    )

    print(
        f"Total final results: "
        f"{len(results)}"
    )

    return results


# ============================================================
# HYBRID RETRIEVER
# ============================================================

class HybridRetriever:

    def __init__(
        self,
        child_chunks: list[dict],
        parent_chunks: list[dict],
        qdrant_client=None,
        collection: str = "constitution_pk",
        embedding_model=None,
        dense_score_threshold: float = DEFAULT_DENSE_SCORE_THRESHOLD,
    ):

        print()
        print("=" * 70)
        print("INITIALIZING RETRIEVER")
        print("=" * 70)

        self.chunks_by_id = {
            c["chunk_id"]: c
            for c in child_chunks
        }

        self.parents_by_id = {
            p["parent_id"]: p
            for p in parent_chunks
        }

        print(
            f"Child chunks: "
            f"{len(child_chunks)}"
        )

        print(
            f"Parent chunks: "
            f"{len(parent_chunks)}"
        )

        self.bm25 = BM25(
            doc_ids=[
                c["chunk_id"]
                for c in child_chunks
            ],
            documents=[
                c["text"]
                for c in child_chunks
            ]
        )

        self.qdrant_client = qdrant_client
        self.collection = collection
        self.embedding_model = embedding_model
        self.dense_score_threshold = (
            dense_score_threshold
        )

        print(
            f"Qdrant enabled: "
            f"{self.qdrant_client is not None}"
        )

        print(
            f"Embedding model enabled: "
            f"{self.embedding_model is not None}"
        )

        print(
            f"Collection: "
            f"{self.collection}"
        )

        print(
            f"Dense threshold: "
            f"{self.dense_score_threshold}"
        )

    # ========================================================
    # DENSE SEARCH
    # ========================================================

    def _dense_search(
        self,
        query: str,
        top_k: int
    ) -> list[str]:

        print()
        print("=" * 70)
        print("DENSE VECTOR SEARCH")
        print("=" * 70)

        print(f"Query: {query}")
        print(f"Top-K: {top_k}")

        if (
            self.qdrant_client is None
            or self.embedding_model is None
        ):

            print(
                "Dense search is disabled."
            )

            return []

        print(
            "Generating query embedding..."
        )

        query_vector = (
            self.embedding_model
            .encode(
                query,
                normalize_embeddings=True
            )
            .tolist()
        )

        print(
            f"Embedding dimension: "
            f"{len(query_vector)}"
        )

        print(
            f"Searching Qdrant collection: "
            f"{self.collection}"
        )

        print(
            f"Score threshold: "
            f"{self.dense_score_threshold}"
        )

        response = (
            self.qdrant_client
            .query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
                score_threshold=(
                    self.dense_score_threshold
                ),
            )
        )

        print(
            f"Dense results returned: "
            f"{len(response.points)}"
        )

        results = []

        for rank, point in enumerate(
            response.points,
            start=1
        ):

            chunk_id = (
                point.payload["chunk_id"]
            )

            score = point.score

            results.append(chunk_id)

            article_number = (
                point.payload.get(
                    "article_number",
                    "unknown"
                )
            )

            print(
                f"  {rank}. "
                f"{chunk_id} "
                f"| Article {article_number} "
                f"| score = {score:.4f}"
            )

        return results

    # ========================================================
    # MAIN SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 10,
        max_extra_from_refs: int = 3
    ) -> list[dict]:

        print()
        print()
        print("#" * 70)
        print("RETRIEVAL PIPELINE")
        print("#" * 70)

        print(f"Query: {query}")
        print(f"Top-K: {top_k}")

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        bm25_results = self.bm25.search(
            query,
            top_k=top_k,
            score_threshold=BM25_SCORE_THRESHOLD
        )

        bm25_ranked = [
            doc_id
            for doc_id, _ in bm25_results
        ]

        # ----------------------------------------------------
        # BM25 relevance guardrail
        # ----------------------------------------------------

        if not bm25_ranked:

            print()
            print("=" * 70)
            print("BM25 RELEVANCE GUARDRAIL")
            print("=" * 70)

            print(
                f"No BM25 result reached the minimum "
                f"score of {BM25_SCORE_THRESHOLD}."
            )

            print(
                "Query is considered irrelevant "
                "for this knowledge base."
            )

            print(
                "Skipping dense retrieval."
            )

            print(
                "Skipping RRF."
            )

            print(
                "Skipping cross-reference expansion."
            )

            print(
                "Returning empty retrieval context."
            )

            return []

        # ----------------------------------------------------
        # Dense
        # ----------------------------------------------------

        dense_ranked = []

        if not BM25_ONLY:

            dense_ranked = self._dense_search(
                query,
                top_k=top_k
            )

        else:

            print()
            print("=" * 70)
            print("DENSE SEARCH SKIPPED")
            print("=" * 70)
            print(
                "BM25_ONLY = True"
            )

        # ----------------------------------------------------
        # Display rankings
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("BM25 RANKING")
        print("=" * 70)

        if bm25_ranked:

            for rank, doc_id in enumerate(
                bm25_ranked,
                start=1
            ):

                print(
                    f"  {rank}. {doc_id}"
                )

        else:

            print("  No BM25 results.")

        print()
        print("=" * 70)
        print("DENSE RANKING")
        print("=" * 70)

        if dense_ranked:

            for rank, doc_id in enumerate(
                dense_ranked,
                start=1
            ):

                print(
                    f"  {rank}. {doc_id}"
                )

        else:

            print("  No dense results.")

        # ----------------------------------------------------
        # Relevance guardrail
        # ----------------------------------------------------

        if (
            not bm25_ranked
            and not dense_ranked
        ):

            print()
            print(
                "No BM25 or dense results found."
            )

            print(
                "Returning empty result."
            )

            return []

        # ----------------------------------------------------
        # BM25-only
        # ----------------------------------------------------

        if BM25_ONLY:

            print()
            print("=" * 70)
            print("BM25-ONLY MODE")
            print("=" * 70)

            top_ids = bm25_ranked[:top_k]

        # ----------------------------------------------------
        # Hybrid BM25 + Dense
        # ----------------------------------------------------

        else:

            ranked_lists = [
                r
                for r in (
                    bm25_ranked,
                    dense_ranked
                )
                if r
            ]

            print()
            print("=" * 70)
            print("HYBRID RETRIEVAL")
            print("=" * 70)

            print(
                "Combining BM25 + Dense "
                "using RRF..."
            )

            fused = reciprocal_rank_fusion(
                ranked_lists
            )

            top_ids = [
                doc_id
                for doc_id, _ in fused[:top_k]
            ]

        # ----------------------------------------------------
        # Top results
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("TOP RESULTS")
        print("=" * 70)

        for rank, doc_id in enumerate(
            top_ids,
            start=1
        ):

            chunk = self.chunks_by_id.get(
                doc_id
            )

            if chunk:

                print(
                    f"  {rank}. "
                    f"{doc_id} "
                    f"| Article "
                    f"{chunk['article_number']}"
                )

            else:

                print(
                    f"  {rank}. "
                    f"{doc_id}"
                )

        # ----------------------------------------------------
        # Cross-reference expansion
        # ----------------------------------------------------

        results = expand_with_references(
            top_ids,
            self.chunks_by_id,
            self.parents_by_id,
            max_extra=max_extra_from_refs
        )

        # ----------------------------------------------------
        # Final results
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("FINAL RETRIEVAL RESULTS")
        print("=" * 70)

        for rank, result in enumerate(
            results,
            start=1
        ):

            article = result[
                "article_number"
            ]

            source = result[
                "source"
            ]

            print(
                f"  {rank}. "
                f"Article {article} "
                f"| {source}"
            )

        print()
        print("#" * 70)
        print("RETRIEVAL COMPLETE")
        print("#" * 70)

        return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("#" * 70)
    print("CONSTITUTION RAG RETRIEVER")
    print("#" * 70)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print()
    print("CONFIGURATION")
    print("-" * 70)

    print(f"Chunks path : {CHUNKS_PATH}")
    print(f"Parents path: {PARENTS_PATH}")
    print(f"Qdrant path : {QDRANT_PATH}")
    print(f"Collection  : {COLLECTION}")
    print(f"Query       : {QUERY}")
    print(f"Top-K       : {TOP_K}")
    print(f"BM25 only   : {BM25_ONLY}")
    print(
        f"Embedding   : "
        f"{EMBEDDING_MODEL_NAME}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    with open(
        CHUNKS_PATH,
        encoding="utf-8"
    ) as f:

        child_chunks = json.load(f)

    print(
        f"Loaded child chunks: "
        f"{len(child_chunks)}"
    )

    with open(
        PARENTS_PATH,
        encoding="utf-8"
    ) as f:

        parent_chunks = json.load(f)

    print(
        f"Loaded parent chunks: "
        f"{len(parent_chunks)}"
    )

    # --------------------------------------------------------
    # Initialize Qdrant + embedding model
    # --------------------------------------------------------

    qdrant_client = None
    embedding_model = None

    if not BM25_ONLY:

        print()
        print("=" * 70)
        print("INITIALIZING DENSE RETRIEVAL")
        print("=" * 70)

        try:

            from qdrant_client import QdrantClient
            from sentence_transformers import SentenceTransformer

            print(
                "Connecting to local Qdrant..."
            )

            qdrant_client = QdrantClient(
                path=QDRANT_PATH
            )

            print(
                "Loading embedding model..."
            )

            embedding_model = (
                SentenceTransformer(
                    EMBEDDING_MODEL_NAME
                )
            )

            print(
                "Dense search enabled."
            )

        except ImportError as e:

            print(
                f"Could not load dense "
                f"retrieval dependencies: {e}"
            )

            print(
                "Falling back to BM25-only."
            )

            BM25_ONLY = True

    else:

        print()
        print(
            "BM25_ONLY = True"
        )

        print(
            "Qdrant and embedding model "
            "will not be loaded."
        )

    # --------------------------------------------------------
    # Create retriever
    # --------------------------------------------------------

    retriever = HybridRetriever(
        child_chunks,
        parent_chunks,
        qdrant_client=qdrant_client,
        collection=COLLECTION,
        embedding_model=embedding_model,
    )

    # --------------------------------------------------------
    # Run retrieval
    # --------------------------------------------------------

    results = retriever.search(
        QUERY,
        top_k=TOP_K
    )

    # --------------------------------------------------------
    # Print actual text
    # --------------------------------------------------------

    print()
    print()
    print("#" * 70)
    print("RETRIEVED CONTEXT")
    print("#" * 70)

    if not results:

        print("No results found.")

    else:

        for i, result in enumerate(
            results,
            start=1
        ):

            print()
            print(
                f"[{i}] "
                f"Article "
                f"{result['article_number']}"
            )

            print(
                f"Source: "
                f"{result['source']}"
            )

            print("-" * 70)

            text = result.get(
                "text",
                ""
            )

            print(text)

    print()
    print("#" * 70)
    print("DONE")
    print("#" * 70)

    if qdrant_client is not None:
        qdrant_client.close()