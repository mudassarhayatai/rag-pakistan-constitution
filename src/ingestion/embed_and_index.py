"""
Section 4: Embedding + loading into Qdrant.

Embeds every child chunk (from Section 3) with an open-source model and
loads the vectors into a Qdrant collection, denormalizing enough parent
Article data into the payload that a retrieval hit can assemble full
context with zero extra lookups.

Fully open-source, no API keys: BAAI/bge-small-en-v1.5 runs locally via
sentence-transformers, and Qdrant's local on-disk mode needs no server
process at all (pass --qdrant-path). Swap to --qdrant-url once you want a
real deployed instance (e.g. the Docker container from docker-compose.yml).


Usage:
python -m src.ingestion.embed_and_index
"""

import argparse
import json
from typing import Iterable, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # bge-m3's dense output dimension 


def load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def batched(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embed_chunks(model: SentenceTransformer, chunks: list[dict], batch_size: int = 32) -> list[list[float]]:

    vectors: list[list[float]] = []
    texts = [c["text"] for c in chunks]
    for batch in batched(texts, batch_size):
        batch_vectors = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        vectors.extend(v.tolist() for v in batch_vectors)
    return vectors


def ensure_collection(client: QdrantClient, collection: str, dim: int) -> None:
    if client.collection_exists(collection):
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def build_points(chunks: list[dict], vectors: list[list[float]], parents_by_id: dict[str, dict]) -> list[PointStruct]:
    """One Qdrant point per child chunk. The payload denormalizes the
    parent Article's text and metadata directly onto each point -- a
    retrieval hit needs zero follow-up reads to assemble full context,
    trading a little storage for a lot of runtime simplicity."""
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        parent = parents_by_id.get(chunk["parent_id"], {})
        payload = {
            "chunk_id": chunk["chunk_id"],
            "article_number": chunk["article_number"],
            "clause_number": chunk.get("clause_number"),
            "child_text": chunk["raw_text"],
            "parent_id": chunk["parent_id"],
            "parent_text": parent.get("text"),
            "article_title": parent.get("title"),
            "part": parent.get("part"),
            "part_title": parent.get("part_title"),
            "chapter": parent.get("chapter"),
            "chapter_title": parent.get("chapter_title"),
            "references": parent.get("references", []),
            "amendment_history": parent.get("amendment_history", []),
            "source_pages": parent.get("source_pages", []),
        }
        points.append(PointStruct(id=i, vector=vector, payload=payload))
    return points


def run(
    chunks_path: str,
    parents_path: str,
    collection: str,
    qdrant_path: Optional[str],
    qdrant_url: Optional[str],
) -> int:
    chunks = load_json(chunks_path)
    parents = load_json(parents_path)
    parents_by_id = {p["parent_id"]: p for p in parents}

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} (open weights, runs fully local)")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    actual_dim = model.get_sentence_embedding_dimension()
    if actual_dim != EMBEDDING_DIM:
        print(f"WARNING: model dim {actual_dim} != EMBEDDING_DIM constant {EMBEDDING_DIM}; using {actual_dim}")

    print(f"Embedding {len(chunks)} chunks...")
    vectors = embed_chunks(model, chunks)

    client = QdrantClient(url=qdrant_url) if qdrant_url else QdrantClient(path=qdrant_path or "data/qdrant_local")
    ensure_collection(client, collection, dim=len(vectors[0]))

    points = build_points(chunks, vectors, parents_by_id)
    client.upsert(collection_name=collection, points=points)

    print(f"Indexed {len(points)} points into Qdrant collection '{collection}'")
    return len(points)


if __name__ == "__main__":


    chunks_path = r"data\processed\child_chunks.json"
    parents_path = r"data\processed\parent_chunks.json"

    collection = "constitution_pk"

    # Local Qdrant database
    qdrant_path = r"qdrant_local"

    # Set to None when using local Qdrant
    qdrant_url = None

    # ============================================================

    run(
        chunks_path=chunks_path,
        parents_path=parents_path,
        collection=collection,
        qdrant_path=qdrant_path,
        qdrant_url=qdrant_url,
    )