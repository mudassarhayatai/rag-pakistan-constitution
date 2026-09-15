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


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:

    def __init__(
        self,
        doc_ids: list[str],
        documents: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self.doc_ids = doc_ids

        self.doc_tokens = [tokenize(doc) for doc in documents]

        self.doc_lengths = [
            len(tokens)
            for tokens in self.doc_tokens
        ]

        self.avgdl = (
            sum(self.doc_lengths) / len(self.doc_lengths)
            if self.doc_lengths
            else 0.0
        )

        self.term_freqs = [
            Counter(tokens)
            for tokens in self.doc_tokens
        ]

        # Document frequency
        df = Counter()

        for tokens in self.doc_tokens:
            for term in set(tokens):
                df[term] += 1

        n_docs = max(len(documents), 1)

        self.idf = {
            term: math.log(
                (n_docs - freq + 0.5) /
                (freq + 0.5)
                + 1
            )
            for term, freq in df.items()
        }

        print("\n" + "=" * 70)
        print("BM25 INDEX CREATED")
        print("=" * 70)

        print(f"Number of documents : {len(documents)}")
        print(f"Average document len: {self.avgdl:.2f} tokens")
        print(f"k1                  : {self.k1}")
        print(f"b                   : {self.b}")
        print(f"Unique terms        : {len(self.idf)}")

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
                    + self.b * doc_len / (self.avgdl or 1)
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
        top_k: int = 10
    ) -> list[tuple[str, float]]:

        print("\n" + "=" * 70)
        print("BM25 SEARCH")
        print("=" * 70)

        print(f"Query: {query}")

        query_tokens = tokenize(query)

        print(f"Query tokens: {query_tokens}")

        # -------------------------------------------------------
        # Show IDF for query terms
        # -------------------------------------------------------

        print("\nQuery term IDF:")

        for term in query_tokens:

            if term in self.idf:

                print(
                    f"  {term:<15} "
                    f"IDF = {self.idf[term]:.4f}"
                )

            else:

                print(
                    f"  {term:<15} "
                    f"NOT FOUND in corpus"
                )

        # -------------------------------------------------------
        # Score every document
        # -------------------------------------------------------

        scored = []

        for i in range(len(self.doc_ids)):

            score = self.score(
                query_tokens,
                i
            )

            if score > 0:

                scored.append(
                    (
                        self.doc_ids[i],
                        score
                    )
                )

        # Sort highest score first
        scored.sort(
            key=lambda x: x[1],
            reverse=True
        )

        results = scored[:top_k]

        # -------------------------------------------------------
        # Display BM25 ranking
        # -------------------------------------------------------

        print("\nBM25 TOP RESULTS:")

        for rank, (doc_id, score) in enumerate(
            results,
            start=1
        ):

            print(
                f"  Rank {rank}: "
                f"{doc_id:<20} "
                f"BM25 Score = {score:.4f}"
            )

        print(
            f"\nBM25 candidates with score > 0: "
            f"{len(scored)}"
        )

        return results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60
) -> list[tuple[str, float]]:

    print("\n" + "=" * 70)
    print("RECIPROCAL RANK FUSION (RRF)")
    print("=" * 70)

    print(f"RRF constant k = {k}")

    scores: dict[str, float] = {}

    # Keep track of where each document appeared
    contributions = {}

    for list_number, ranked in enumerate(
        ranked_lists,
        start=1
    ):

        print(
            f"\nRanking list #{list_number}"
        )

        for rank, doc_id in enumerate(
            ranked,
            start=1
        ):

            contribution = 1.0 / (k + rank)

            scores[doc_id] = (
                scores.get(doc_id, 0.0)
                + contribution
            )

            if doc_id not in contributions:
                contributions[doc_id] = []

            contributions[doc_id].append(
                (
                    rank,
                    contribution
                )
            )

            print(
                f"  {doc_id:<20} "
                f"rank={rank:<3} "
                f"contribution={contribution:.6f}"
            )

    # Sort by RRF score
    fused = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nFINAL RRF SCORES:")

    for rank, (doc_id, score) in enumerate(
        fused,
        start=1
    ):

        print(
            f"  Rank {rank}: "
            f"{doc_id:<20} "
            f"RRF = {score:.6f}"
        )

    return fused


# ---------------------------------------------------------------------------
# Cross-reference expansion
# ---------------------------------------------------------------------------

def expand_with_references(
    top_chunk_ids: list[str],
    chunks_by_id: dict[str, dict],
    parents_by_id: dict[str, dict],
    max_extra: int = 3,
) -> list[dict]:

    print("\n" + "=" * 70)
    print("CROSS-REFERENCE EXPANSION")
    print("=" * 70)

    print(
        f"Starting with top chunk IDs: "
        f"{top_chunk_ids}"
    )

    results = []

    seen_articles = set()

    # -------------------------------------------------------
    # Add directly retrieved chunks
    # -------------------------------------------------------

    print("\nDirectly retrieved chunks:")

    for chunk_id in top_chunk_ids:

        chunk = chunks_by_id.get(chunk_id)

        if not chunk:

            print(
                f"  {chunk_id}: NOT FOUND"
            )

            continue

        article_num = chunk["article_number"]

        if article_num in seen_articles:

            print(
                f"  Article {article_num}: "
                f"already seen -> skip"
            )

            continue

        seen_articles.add(article_num)

        print(
            f"  Article {article_num} "
            f"<- chunk {chunk_id}"
        )

        results.append(
            {
                "chunk_id": chunk_id,
                "article_number": article_num,
                "text": chunk["raw_text"],
                "source": "retrieved",
            }
        )

    # -------------------------------------------------------
    # Add referenced articles
    # -------------------------------------------------------

    extras_added = 0

    print("\nLooking for cross-references...")

    for chunk_id in top_chunk_ids:

        if extras_added >= max_extra:
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
            f"\nArticle {article_num} "
            f"references: {references}"
        )

        for ref_article in references:

            if extras_added >= max_extra:
                break

            if ref_article in seen_articles:

                print(
                    f"  Article {ref_article}: "
                    f"already included -> skip"
                )

                continue

            ref_parent = parents_by_id.get(
                f"article_{ref_article}"
            )

            if not ref_parent:

                print(
                    f"  Article {ref_article}: "
                    f"parent not found"
                )

                continue

            print(
                f"  + Adding referenced "
                f"Article {ref_article}"
            )

            seen_articles.add(
                ref_article
            )

            results.append(
                {
                    "chunk_id": None,
                    "article_number": ref_article,
                    "text": ref_parent.get("text"),
                    "source":
                        f"cross_reference "
                        f"(via Article {article_num})",
                }
            )

            extras_added += 1

    print(
        f"\nCross-reference articles added: "
        f"{extras_added}"
    )

    return results


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:

    def __init__(
        self,
        child_chunks: list[dict],
        parent_chunks: list[dict],
        qdrant_client=None,
        collection: str = "constitution_pk",
        embedding_model=None,
    ):

        print("\n" + "=" * 70)
        print("INITIALIZING HYBRID RETRIEVER")
        print("=" * 70)

        print(
            f"Child chunks : {len(child_chunks)}"
        )

        print(
            f"Parent chunks: {len(parent_chunks)}"
        )

        self.chunks_by_id = {
            c["chunk_id"]: c
            for c in child_chunks
        }

        self.parents_by_id = {
            p["parent_id"]: p
            for p in parent_chunks
        }

        print(
            f"Chunk ID mapping created: "
            f"{len(self.chunks_by_id)}"
        )

        print(
            f"Parent ID mapping created: "
            f"{len(self.parents_by_id)}"
        )

        # -------------------------------------------------------
        # Build BM25
        # -------------------------------------------------------

        self.bm25 = BM25(
            doc_ids=[
                c["chunk_id"]
                for c in child_chunks
            ],
            documents=[
                c["text"]
                for c in child_chunks
            ],
        )

        self.qdrant_client = qdrant_client
        self.collection = collection
        self.embedding_model = embedding_model

        print("\nDense search:")

        if self.qdrant_client is not None:
            print("  Qdrant: ENABLED")
        else:
            print("  Qdrant: DISABLED")

        if self.embedding_model is not None:
            print(
                "  Embedding model: ENABLED"
            )
        else:
            print(
                "  Embedding model: DISABLED"
            )

    # -----------------------------------------------------------
    # Dense search
    # -----------------------------------------------------------

    def _dense_search(
        self,
        query: str,
        top_k: int
    ) -> list[str]:

        print("\n" + "=" * 70)
        print("DENSE VECTOR SEARCH")
        print("=" * 70)

        if (
            self.qdrant_client is None
            or self.embedding_model is None
        ):

            print(
                "Dense search disabled."
            )

            return []

        print(
            f"Query: {query}"
        )

        # -------------------------------------------------------
        # Convert query into embedding
        # -------------------------------------------------------

        query_vector = (
            self.embedding_model.encode(
                query,
                normalize_embeddings=True
            )
            .tolist()
        )

        print(
            f"Query vector dimension: "
            f"{len(query_vector)}"
        )

        print(
            "First 10 vector values:"
        )

        print(
            query_vector[:10]
        )

        # -------------------------------------------------------
        # Search Qdrant
        # -------------------------------------------------------

        print(
            f"\nSearching Qdrant collection: "
            f"{self.collection}"
        )

        print(
            f"Top K requested: {top_k}"
        )

        response = (
            self.qdrant_client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
        )

        print(
            f"\nQdrant returned "
            f"{len(response.points)} points"
        )

        # -------------------------------------------------------
        # Show dense ranking
        # -------------------------------------------------------

        dense_ids = []

        print("\nDENSE TOP RESULTS:")

        for rank, point in enumerate(
            response.points,
            start=1
        ):

            chunk_id = point.payload.get(
                "chunk_id"
            )

            dense_ids.append(
                chunk_id
            )

            print(
                f"  Rank {rank}: "
                f"{chunk_id:<20} "
                f"Similarity = {point.score:.6f}"
            )

        return dense_ids

    # -----------------------------------------------------------
    # Main search
    # -----------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        max_extra_from_refs: int = 3,
    ) -> list[dict]:

        print("\n")
        print("#" * 70)
        print("HYBRID SEARCH START")
        print("#" * 70)

        print(
            f"\nQUERY:\n{query}"
        )

        # -------------------------------------------------------
        # BM25
        # -------------------------------------------------------

        bm25_results = self.bm25.search(
            query,
            top_k=top_k
        )

        bm25_ranked = [
            doc_id
            for doc_id, score
            in bm25_results
        ]

        print(
            "\nBM25 ranking IDs:"
        )

        print(
            bm25_ranked
        )

        # -------------------------------------------------------
        # Dense
        # -------------------------------------------------------

        dense_ranked = self._dense_search(
            query,
            top_k=top_k
        )

        print(
            "\nDense ranking IDs:"
        )

        print(
            dense_ranked
        )

        # -------------------------------------------------------
        # RRF
        # -------------------------------------------------------

        ranked_lists = [
            ranking
            for ranking in (
                bm25_ranked,
                dense_ranked
            )
            if ranking
        ]

        print("\nRanking lists sent to RRF:")

        for i, ranking in enumerate(
            ranked_lists,
            start=1
        ):

            print(
                f"  List {i}: {ranking}"
            )

        if ranked_lists:

            fused = reciprocal_rank_fusion(
                ranked_lists
            )

        else:

            fused = []

        # -------------------------------------------------------
        # Select top IDs
        # -------------------------------------------------------

        top_ids = [
            doc_id
            for doc_id, score
            in fused[:top_k]
        ]

        if not top_ids:

            top_ids = bm25_ranked

        print(
            "\nFINAL IDs AFTER RRF:"
        )

        for rank, doc_id in enumerate(
            top_ids,
            start=1
        ):

            print(
                f"  Rank {rank}: {doc_id}"
            )

        # -------------------------------------------------------
        # Cross-reference expansion
        # -------------------------------------------------------

        results = expand_with_references(
            top_ids,
            self.chunks_by_id,
            self.parents_by_id,
            max_extra=max_extra_from_refs,
        )

        print("\n" + "=" * 70)
        print("HYBRID SEARCH FINISHED")
        print("=" * 70)

        return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ===========================================================
    # Configuration
    # ===========================================================

    CHUNKS_PATH = r"data\processed\child_chunks.json"
    PARENTS_PATH = r"data\processed\parent_chunks.json"

    QUERY = (
        "What does Article 25 say "
        "about equality of citizens?"
    )

    TOP_K = 5

    COLLECTION = "constitution_pk"

    QDRANT_PATH = "qdrant_local"

    BM25_ONLY = False

    # ===========================================================
    # Load child chunks
    # ===========================================================

    print("\n" + "#" * 70)
    print("LOADING DATA")
    print("#" * 70)

    with open(
        CHUNKS_PATH,
        encoding="utf-8"
    ) as f:

        child_chunks = json.load(f)

    print(
        f"Loaded child chunks: "
        f"{len(child_chunks)}"
    )

    # ===========================================================
    # Load parent chunks
    # ===========================================================

    with open(
        PARENTS_PATH,
        encoding="utf-8"
    ) as f:

        parent_chunks = json.load(f)

    print(
        f"Loaded parent chunks: "
        f"{len(parent_chunks)}"
    )

    # ===========================================================
    # Initialize Qdrant + embedding model
    # ===========================================================

    qdrant_client = None
    embedding_model = None

    if not BM25_ONLY:

        print("\n" + "#" * 70)
        print("LOADING DENSE SEARCH")
        print("#" * 70)

        try:

            from qdrant_client import QdrantClient
            from sentence_transformers import SentenceTransformer

            print(
                f"Opening Qdrant: "
                f"{QDRANT_PATH}"
            )

            qdrant_client = QdrantClient(
                path=QDRANT_PATH
            )

            print(
                f"Loading embedding model: "
                f"{EMBEDDING_MODEL_NAME}"
            )

            embedding_model = SentenceTransformer(
                EMBEDDING_MODEL_NAME
            )

            print(
                f"Embedding dimension: "
                f"{embedding_model.get_embedding_dimension()}"
            )

            collection_info = (
                qdrant_client.get_collection(
                    COLLECTION
                )
            )

            print(
                f"Qdrant collection: "
                f"{COLLECTION}"
            )

            print(
                f"Collection points: "
                f"{collection_info.points_count}"
            )

            print(
                "\nDense search ENABLED"
            )

        except ImportError as e:

            print(
                f"Could not load dense search: "
                f"{e}"
            )

            print(
                "Falling back to BM25-only"
            )

    # ===========================================================
    # Create retriever
    # ===========================================================

    retriever = HybridRetriever(
        child_chunks,
        parent_chunks,
        qdrant_client=qdrant_client,
        collection=COLLECTION,
        embedding_model=embedding_model,
    )

    # ===========================================================
    # Search
    # ===========================================================

    results = retriever.search(
        QUERY,
        top_k=TOP_K
    )

    # ===========================================================
    # Final results
    # ===========================================================

    print("\n" + "#" * 70)
    print("FINAL RETRIEVAL RESULTS")
    print("#" * 70)

    print(
        f"\nResults for:\n{QUERY}\n"
    )

    for rank, r in enumerate(
        results,
        start=1
    ):

        print(
            f"\nResult #{rank}"
        )

        print(
            f"Source         : "
            f"{r['source']}"
        )

        print(
            f"Article        : "
            f"{r['article_number']}"
        )

        print(
            f"Chunk ID       : "
            f"{r['chunk_id']}"
        )

        print(
            f"Text           : "
            f"{r['text'][:300]}..."
        )

    print("\n" + "#" * 70)
    print("DONE")
    print("#" * 70)

    if qdrant_client is not None:
        qdrant_client.close()